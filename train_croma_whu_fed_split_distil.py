"""
Federated Split Learning + Distillation 分割训练脚本（单卡串行模拟）

架构划分：
- 服务器端 (Server)：cross_encoder + 分割头 (seg_head) + 两个小辅助投影模型 proj_OM / proj_RM
- 客户端 (Client，共 NUM_CLIENTS 个)：每个客户端持有 radar_encoder + optical_encoder

训练流程（每个 epoch 视为一轮）：
1. 本轮开始时，各客户端从上一轮聚合得到的全局客户端模型同步参数
2. 服务器依次与每个客户端建立连接，完成连接阶段训练：
   - 客户端前向传播，得到 radar_encodings 和 optical_encodings，发送给服务器
   - 服务器经 cross_encoder 融合得到 joint_encodings，再经 seg_head 得到分割 logits，
     计算交叉熵损失并更新服务器参数（cross_encoder + seg_head）
   - 服务器同时训练两个小辅助模型：proj_OM(optical_encodings->joint_encodings)、
     proj_RM(radar_encodings->joint_encodings)，以 joint_encodings 为标签做 MSE 蒸馏
   - 服务器不再向客户端传递梯度，客户端编码器在连接阶段不更新
3. 客户端与服务器断开连接后（单卡环境，紧接其后）：
   - 将 proj_OM、proj_RM 以及两份 seg_head 复制到客户端
   - 客户端用本地数据集训练两个模型：
       optical_encoder -> proj_OM -> seg_head
       radar_encoder  -> proj_RM -> seg_head
     共 local_epochs 个周期，从而更新本地编码器
4. 所有客户端都完成一轮后，对多个客户端模型（编码器）参数做 FedAvg 聚合，形成全局客户端模型
5. 使用全局客户端模型 + 服务器在验证集上验证
6. 进入下一轮：各客户端使用全局客户端模型开始训练

数据集随机平均分配到各个客户端（轮询切分，保证每个客户端样本数基本一致）。
"""

import os
import argparse
import copy
import random
from collections import OrderedDict
from datetime import datetime
import time
import csv
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from datasets import (
    WHUOptSarPatchDataset,
    BigEarthNetDataset,
    Houston2013PatchDataset,
    CLASS_NAMES,
)
from pretrain_croma import CROMA


class SplitClient(nn.Module):
    """客户端模型：部署 radar_encoder + optical_encoder。

    负责本地数据的编码，将 radar_encodings 与 optical_encodings 发送给服务器；
    在断开连接后，使用服务器复制下来的 proj_OM/proj_RM 与 seg_head 做本地训练。
    """

    def __init__(self, optical_encoder: nn.Module, radar_encoder: nn.Module,
                 attn_bias: torch.Tensor):
        super().__init__()
        self.optical_encoder = optical_encoder
        self.radar_encoder = radar_encoder
        self.register_buffer("attn_bias", attn_bias)

    def encode(self, optical_imgs: torch.Tensor, radar_imgs: torch.Tensor):
        """客户端前向：分别编码光学与雷达图像。

        Returns:
            optical_encodings: [B, N, D]
            radar_encodings: [B, N, D]
        """
        attn_bias = self.attn_bias.to(optical_imgs.device)
        optical_encodings = self.optical_encoder(
            imgs=optical_imgs, attn_bias=attn_bias, mask_info=None
        )
        radar_encodings = self.radar_encoder(
            imgs=radar_imgs, attn_bias=attn_bias, mask_info=None
        )
        return optical_encodings, radar_encodings


class Projector(nn.Module):
    """小辅助投影模型：把单模态编码映射到 joint_encodings 空间。

    proj_OM 以 optical_encodings 为输入、proj_RM 以 radar_encodings 为输入，
    均以 joint_encodings 为标签进行 MSE 蒸馏训练。输入输出形状均为 [B, N, D]。
    """

    def __init__(self, dim: int, hidden_mult: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, int(dim * hidden_mult)),
            nn.GELU(),
            nn.Linear(int(dim * hidden_mult), dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D] -> [B, N, D]
        return self.net(x)


class ServerModel(nn.Module):
    """服务器模型：cross_encoder + 分割头 (seg_head) + 两个辅助投影模型 proj_OM/proj_RM。

    接收客户端发送的 radar_encodings 与 optical_encodings，
    完成跨模态融合与像素级分割（交叉熵损失），并训练两个投影模型做蒸馏。
    """

    def __init__(self, cross_encoder: nn.Module, encoder_dim: int,
                 num_patches: int, num_classes: int):
        super().__init__()
        self.cross_encoder = cross_encoder
        self.encoder_dim = encoder_dim
        self.num_patches = num_patches

        # 假设 patch 在空间上是正方形网格
        self.h_patches = int(num_patches ** 0.5)
        self.w_patches = int(num_patches ** 0.5)

        self.seg_head = nn.Sequential(
            nn.Conv2d(encoder_dim, encoder_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(encoder_dim, num_classes, kernel_size=1),
        )

        # 两个小辅助投影模型（蒸馏到单模态编码）
        self.proj_OM = Projector(dim=encoder_dim)
        self.proj_RM = Projector(dim=encoder_dim)

    def forward(self, radar_encodings: torch.Tensor, optical_encodings: torch.Tensor,
                attn_bias: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        """joint 模式：融合编码并生成分割预测（用于验证）。

        Args:
            radar_encodings: [B, N, D] 客户端发送的雷达编码
            optical_encodings: [B, N, D] 客户端发送的光学编码
            attn_bias: 注意力偏置
            output_size: (H, W) 分割输出尺寸

        Returns:
            logits: [B, num_classes, H, W]
        """
        joint_encodings = self.cross_encoder(
            x=radar_encodings, context=optical_encodings, alibi=attn_bias
        )  # [B, N, D]

        b, n, d = joint_encodings.shape
        assert n == self.h_patches * self.w_patches, \
            "num_patches 与 h_patches*w_patches 不一致，请检查 image_size/patch_size 设置"

        # [B, N, D] -> [B, D, H_p, W_p]
        feat = (
            joint_encodings.view(b, self.h_patches, self.w_patches, d)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        logits_low = self.seg_head(feat)  # [B, num_classes, H_p, W_p]

        # 上采样到原图大小进行像素级分割
        H, W = output_size
        logits = F.interpolate(logits_low, size=(H, W), mode="bilinear", align_corners=False)
        return logits


class ClientLocalModel(nn.Module):
    """客户端本地模型：encoder -> proj -> seg_head。

    断开连接后由客户端在本地数据上训练，用于更新本地编码器。
    """

    def __init__(self, encoder: nn.Module, proj: nn.Module, seg_head: nn.Module,
                 num_patches: int, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.proj = proj
        self.seg_head = seg_head
        self.num_patches = num_patches
        self.h_patches = int(num_patches ** 0.5)
        self.w_patches = int(num_patches ** 0.5)

    def forward(self, imgs: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        """单模态图像 -> 编码 -> 投影 -> 分割 logits [B, num_classes, H, W]。"""
        encodings = self.encoder(
            imgs=imgs, attn_bias=attn_bias, mask_info=None
        )  # [B, N, D]
        proj_out = self.proj(encodings)  # [B, N, D]

        b, n, d = proj_out.shape
        assert n == self.h_patches * self.w_patches, \
            "num_patches 与 h_patches*w_patches 不一致，请检查 image_size/patch_size 设置"

        feat = (
            proj_out.view(b, self.h_patches, self.w_patches, d)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        logits_low = self.seg_head(feat)

        H, W = imgs.shape[-2:]
        logits = F.interpolate(logits_low, size=(H, W), mode="bilinear", align_corners=False)
        return logits


def parse_args():
    parser = argparse.ArgumentParser(
        description="Federated Split Learning + Distillation 分割训练 - 单卡串行模拟 "
                    "(服务器 cross_encoder+seg_head+proj_OM/proj_RM, "
                    "客户端 radar_encoder+optical_encoder, 不回传梯度)"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/featurize/data/whu-opt-sar-dataset",
        help="数据根目录",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["whu-opt-sar", "bigearthnet", "houston2013"],
        default="whu-opt-sar",
        help="选择数据集：whu-opt-sar / bigearthnet / houston2013",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="单个裁剪 patch 的空间尺寸，等于数据集的 patch_size",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4, help="客户端本地训练学习率")
    parser.add_argument("--lr_server", type=float, default=1e-4, help="服务器（cross_encoder+分割头+投影模型）学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--encoder_dim", type=int, default=192)
    parser.add_argument("--encoder_layers", type=int, default=6)
    parser.add_argument("--attention_heads", type=int, default=16)
    parser.add_argument("--vit_patch_size", type=int, default=8, help="ViT patch size")
    parser.add_argument("--decoder_dim", type=int, default=512)
    parser.add_argument("--decoder_layers", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument(
        "--num_clients",
        type=int,
        default=4,
        help="客户端数量（训练时维护的客户端个数）",
    )
    parser.add_argument(
        "--local_epochs",
        type=int,
        default=5,
        help="客户端与服务器断开后，在本地数据集上训练的周期数",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="数据集随机分配到各客户端的随机种子",
    )
    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        default="",
        help="预训练 CROMA checkpoint 路径（为空则随机初始化模型）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../CROMA_checkpoint/croma_whu_fed_split_distil_checkpoints",
    )
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument(
        "--num_ratio",
        type=float,
        default=1.0,
        help="使用多少比例的 patch（<1 代表子集，>1 代表有放回扩增）",
    )
    parser.add_argument(
        "--stride_ratio",
        type=float,
        default=0.9,
        help="滑动窗口步长与 image_size 的比例（whu-opt-sar 使用）",
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=0.0, help=">0 时启用梯度裁剪"
    )
    parser.add_argument(
        "--save_final_only",
        action="store_true",
        help="只在训练全部完成后保存一次模型（默认每个 epoch 保存一次）",
    )
    parser.add_argument(
        "--freeze_encoders",
        action="store_true",
        default=False,
        help="冻结所有客户端的 radar_encoder 和 optical_encoder 参数，仅训练服务器与本地头",
    )
    return parser.parse_args()


def create_loaders(args) -> Tuple[List[DataLoader], DataLoader, Optional[int]]:
    """构建训练集并按 num_clients 随机均分，构建验证集。

    Returns:
        client_train_loaders: 每个客户端的训练 DataLoader
        val_loader: 验证 DataLoader
        inferred_num_patches: 从真实样本推断的 patch 数（可能为 None）
    """
    if args.dataset == "whu-opt-sar":
        train_set = WHUOptSarPatchDataset(
            root_dir=args.data_root,
            split="train",
            patch_size=args.image_size,
            stride_ratio=args.stride_ratio,
            num_ratio=args.num_ratio,
        )
        val_set = WHUOptSarPatchDataset(
            root_dir=args.data_root,
            split="val",
            patch_size=args.image_size,
            stride_ratio=args.stride_ratio,
            num_ratio=1.0,
        )
    elif args.dataset == "bigearthnet":
        train_set = BigEarthNetDataset(
            root=args.data_root,
            split="train",
            ratio=args.num_ratio,
        )
        val_set = BigEarthNetDataset(
            root=args.data_root,
            split="validation",
            ratio=1.0,
        )
    else:  # houston2013
        train_set = Houston2013PatchDataset(
            root_dir=args.data_root,
            split="train",
            patch_size=args.image_size,
            stride=args.image_size,
        )
        val_set = Houston2013PatchDataset(
            root_dir=args.data_root,
            split="val",
            patch_size=args.image_size,
            stride=args.image_size,
        )

    # ========== 数据集随机平均分配到各客户端 ==========
    rng = random.Random(args.seed)
    all_indices = list(range(len(train_set)))
    rng.shuffle(all_indices)

    client_train_loaders = []
    for c in range(args.num_clients):
        # 轮询切分：每个客户端拿到的样本数量最多相差 1，保证"平均分配"
        client_indices = all_indices[c::args.num_clients]
        client_subset = Subset(train_set, client_indices)
        client_loader = DataLoader(
            client_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        client_train_loaders.append(client_loader)

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # 从真实样本推断 patch 数，避免与 image_size 不一致
    inferred_num_patches = None
    try:
        sample = train_set[0]
        optical_sample = sample[0]
        _, H, W = optical_sample.shape
        inferred_num_patches = (H // args.vit_patch_size) * (W // args.vit_patch_size)
    except Exception:
        inferred_num_patches = None

    return client_train_loaders, val_loader, inferred_num_patches


def build_models(args, device, inferred_num_patches=None):
    """构建客户端模型集合、服务器模型与 attn_bias。

    Returns:
        clients: List[SplitClient]，每个客户端持有独立的 radar_encoder + optical_encoder
        server: ServerModel（cross_encoder + 分割头 + proj_OM/proj_RM）
        attn_bias: 注意力偏置
        num_patches: patch 数
    """
    if inferred_num_patches is None:
        assert (
            args.image_size % args.vit_patch_size == 0
        ), "image_size 必须能被 vit_patch_size 整除"
        num_patches = (args.image_size // args.vit_patch_size) ** 2
    else:
        num_patches = inferred_num_patches

    if args.dataset == "whu-opt-sar":
        opt_ch = 4
        radar_ch = 1
    elif args.dataset == "bigearthnet":
        opt_ch = 10
        radar_ch = 2
    else:  # houston2013
        opt_ch = 144
        radar_ch = 1

    # 构建与预训练阶段相同配置的 CROMA 以加载权重
    croma = CROMA(
        patch_size=args.vit_patch_size,
        encoder_dim=args.encoder_dim,
        encoder_layers=args.encoder_layers,
        attention_heads=args.attention_heads,
        decoder_dim=args.decoder_dim,
        decoder_layers=args.decoder_layers,
        total_channels=opt_ch + radar_ch,
        num_patches=num_patches,
        opt_channels=opt_ch,
        radar_channels=radar_ch,
    )

    # 加载预训练权重（若提供了 checkpoint 路径）
    if args.pretrained_ckpt:
        ckpt = torch.load(args.pretrained_ckpt, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = croma.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[Warning] Missing keys when loading CROMA: {missing}")
        if unexpected:
            print(f"[Warning] Unexpected keys when loading CROMA: {unexpected}")
    else:
        print("[Info] pretrained_ckpt 为空，模型将随机初始化")

    attn_bias = croma.attn_bias

    # 每个客户端持有独立的编码器副本
    clients = []
    for _ in range(args.num_clients):
        client = SplitClient(
            optical_encoder=copy.deepcopy(croma.optical_encoder),
            radar_encoder=copy.deepcopy(croma.radar_encoder),
            attn_bias=attn_bias,
        ).to(device)
        clients.append(client)

    # 服务器持有 cross_encoder + 分割头 + 两个投影模型
    server = ServerModel(
        cross_encoder=croma.cross_encoder,
        encoder_dim=args.encoder_dim,
        num_patches=num_patches,
        num_classes=args.num_classes,
    ).to(device)

    return clients, server, attn_bias, num_patches


def server_connect_step(client: SplitClient, server: ServerModel,
                        optical: torch.Tensor, sar: torch.Tensor, labels: torch.Tensor,
                        server_optimizer: torch.optim.Optimizer,
                        proj_optimizer: torch.optim.Optimizer,
                        criterion: nn.Module, device: torch.device,
                        max_grad_norm: float = 0.0):
    """客户端与服务器连接阶段的单步训练（模拟通信）。

    流程：
    1. 客户端前向，得到 radar_encodings 与 optical_encodings（编码器不更新）
    2. 发送激活值到服务器（模拟通信，切断计算图）
    3. 服务器前向：cross_encoder -> joint_encodings -> seg_head -> logits
    4. 计算交叉熵损失，更新服务器参数（cross_encoder + seg_head）
    5. 以 joint_encodings 为标签，MSE 训练 proj_OM / proj_RM
    6. 服务器不向客户端传递梯度

    Returns:
        loss_ce: 当前 batch 的交叉熵损失
        loss_proj: 当前 batch 的投影蒸馏损失
        forward_bytes: 前向传输字节数（激活值）
    """
    optical = optical.to(device, non_blocking=True)
    sar = sar.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    H, W = optical.shape[-2:]

    # ===== 阶段 1: 客户端前向（无梯度回传，编码器不更新） =====
    client.eval()
    optical_encodings, radar_encodings = client.encode(optical, sar)

    # ===== 阶段 2: 激活值传输到服务器（模拟通信） =====
    optical_act = optical_encodings.detach().requires_grad_(True)
    radar_act = radar_encodings.detach().requires_grad_(True)
    forward_bytes = (optical_act.numel() + radar_act.numel()) * optical_act.element_size()

    # ===== 阶段 3: 服务器前向 =====
    server.train()
    attn_bias = client.attn_bias.to(device)
    joint_encodings = server.cross_encoder(
        x=radar_act, context=optical_act, alibi=attn_bias
    )  # [B, N, D]

    b, n, d = joint_encodings.shape
    feat = (
        joint_encodings.view(b, server.h_patches, server.w_patches, d)
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    logits_low = server.seg_head(feat)
    logits = F.interpolate(logits_low, size=(H, W), mode="bilinear", align_corners=False)
    loss_ce = criterion(logits, labels)

    # ===== 阶段 4: 更新服务器 (cross_encoder + seg_head) =====
    server_optimizer.zero_grad(set_to_none=True)
    loss_ce.backward()
    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(server.parameters(), max_grad_norm)
    server_optimizer.step()

    # ===== 阶段 5: 训练两个投影模型（以 joint_encodings 为标签做蒸馏） =====
    joint_target = joint_encodings.detach()
    loss_proj = (
        F.mse_loss(server.proj_OM(optical_act), joint_target)
        + F.mse_loss(server.proj_RM(radar_act), joint_target)
    )
    proj_optimizer.zero_grad(set_to_none=True)
    loss_proj.backward()
    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(
            list(server.proj_OM.parameters()) + list(server.proj_RM.parameters()),
            max_grad_norm,
        )
    proj_optimizer.step()

    return loss_ce.item(), loss_proj.item(), forward_bytes


def client_local_train(args, client: SplitClient, server: ServerModel,
                       loader: DataLoader, device: torch.device,
                       epoch: int, client_id: int, start_time: float) -> float:
    """客户端与服务器断开连接后的本地训练（单卡环境，紧接在连接阶段之后）。

    将 proj_OM、proj_RM 和两份 seg_head 复制到客户端，
    用本地数据集训练两个模型：
      - optical_encoder -> proj_OM -> seg_head
      - radar_encoder  -> proj_RM -> seg_head

    Returns:
        avg_loss: 两个本地模型按像素加权的平均交叉熵损失
    """
    # ===== 复制辅助模型与分割头到客户端 =====
    proj_OM_local = copy.deepcopy(server.proj_OM).to(device)
    proj_RM_local = copy.deepcopy(server.proj_RM).to(device)
    seg_head_OM = copy.deepcopy(server.seg_head).to(device)
    seg_head_RM = copy.deepcopy(server.seg_head).to(device)

    model_OM = ClientLocalModel(
        encoder=client.optical_encoder,
        proj=proj_OM_local,
        seg_head=seg_head_OM,
        num_patches=server.num_patches,
        num_classes=args.num_classes,
    ).to(device)
    model_RM = ClientLocalModel(
        encoder=client.radar_encoder,
        proj=proj_RM_local,
        seg_head=seg_head_RM,
        num_patches=server.num_patches,
        num_classes=args.num_classes,
    ).to(device)

    optimizer_OM = torch.optim.AdamW(
        model_OM.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    optimizer_RM = torch.optim.AdamW(
        model_RM.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    if args.dataset in {"bigearthnet", "houston2013"}:
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
    else:
        criterion = nn.CrossEntropyLoss()

    attn_bias = client.attn_bias.to(device)
    total_loss = 0.0
    total_pixels = 0

    for local_epoch in range(args.local_epochs):
        for optical, sar, labels in loader:
            optical = optical.to(device, non_blocking=True)
            sar = sar.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # 模型 A: optical_encoder -> proj_OM -> seg_head
            model_OM.train()
            logits_OM = model_OM(optical, attn_bias)
            loss_OM = criterion(logits_OM, labels)
            optimizer_OM.zero_grad(set_to_none=True)
            loss_OM.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model_OM.parameters(), args.max_grad_norm)
            optimizer_OM.step()

            # 模型 B: radar_encoder -> proj_RM -> seg_head
            model_RM.train()
            logits_RM = model_RM(sar, attn_bias)
            loss_RM = criterion(logits_RM, labels)
            optimizer_RM.zero_grad(set_to_none=True)
            loss_RM.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model_RM.parameters(), args.max_grad_norm)
            optimizer_RM.step()

            total_loss += (loss_OM.item() + loss_RM.item()) * labels.numel()
            total_pixels += labels.numel()

    avg_loss = total_loss / max(1, total_pixels)

    elapsed = time.time() - start_time
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    print(
        f"[{elapsed_str}] Epoch {epoch} | Client {client_id+1} 本地训练 "
        f"({args.local_epochs} epoch, {len(loader.dataset)} 样本) | 平均损失: {avg_loss:.4f}"
    )
    return avg_loss


def train_one_round(args, clients: List[SplitClient], server: ServerModel,
                    server_optimizer: torch.optim.Optimizer,
                    proj_optimizer: torch.optim.Optimizer,
                    client_loaders: List[DataLoader], criterion: nn.Module,
                    device: torch.device, epoch: int, start_time: float):
    """一轮训练：服务器依次与每个客户端建立连接，断开后客户端立即本地训练。

    Returns:
        avg_loss: 按像素加权的连接阶段平均交叉熵损失
        total_forward_bytes: 通信量统计（激活值上传）
    """
    total_loss = 0.0
    total_pixels = 0
    total_forward_bytes = 0

    for client_id in range(args.num_clients):
        client = clients[client_id]
        loader = client_loaders[client_id]

        if len(loader) == 0:
            print(f"[Epoch {epoch}] Client {client_id+1}/{args.num_clients}: 本地数据为空，跳过")
            continue

        # ===== 连接阶段：训练服务器 + 两个投影模型 =====
        conn_loss = 0.0
        conn_pixels = 0
        for step, (optical, sar, labels) in enumerate(loader):
            loss_ce, loss_proj, fwd_bytes = server_connect_step(
                client=client,
                server=server,
                optical=optical,
                sar=sar,
                labels=labels,
                server_optimizer=server_optimizer,
                proj_optimizer=proj_optimizer,
                criterion=criterion,
                device=device,
                max_grad_norm=args.max_grad_norm,
            )

            pixels = labels.numel()
            conn_loss += loss_ce * pixels
            conn_pixels += pixels
            total_forward_bytes += fwd_bytes

            if (step + 1) % args.log_interval == 0:
                elapsed = time.time() - start_time
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                print(
                    f"[{elapsed_str}] Epoch {epoch} | Client {client_id+1}/{args.num_clients} | "
                    f"Step {step+1}/{len(loader)} | CE: {loss_ce:.4f} | Proj: {loss_proj:.4f}"
                )

        conn_avg = conn_loss / max(1, conn_pixels)
        total_loss += conn_loss
        total_pixels += conn_pixels
        print(
            f"[Epoch {epoch}] Client {client_id+1}/{args.num_clients} 连接阶段完成 | "
            f"平均 CE 损失: {conn_avg:.4f}"
        )

        # ===== 断开连接：客户端本地训练（紧接其后） =====
        client_local_train(
            args, client, server, loader, device, epoch, client_id, start_time
        )

    avg_loss = total_loss / max(1, total_pixels)
    return avg_loss, total_forward_bytes


def aggregate_client_models(clients: List[SplitClient], device: torch.device) -> OrderedDict:
    """对多个客户端模型参数做 FedAvg 聚合，得到全局客户端模型 state_dict。"""
    num_clients = len(clients)
    global_state = OrderedDict()
    with torch.no_grad():
        first_state = clients[0].state_dict()
        for key in first_state.keys():
            stacked = torch.stack(
                [clients[i].state_dict()[key].float().to(device) for i in range(num_clients)]
            )
            global_state[key] = stacked.mean(dim=0)
    return global_state


def sync_clients_to_global(clients: List[SplitClient], global_state: OrderedDict):
    """本轮开始时，让各客户端从全局客户端模型同步参数。"""
    for client in clients:
        client.load_state_dict(global_state)


@torch.no_grad()
def evaluate(args, global_client: SplitClient, server: ServerModel,
             val_loader: DataLoader, criterion: nn.Module, device: torch.device,
             epoch: int, start_time: float):
    """使用全局客户端模型 + 服务器在验证集上评估。"""
    global_client.eval()
    server.eval()

    if args.dataset == "houston2013":
        total_loss = 0.0
        num_pixels = 0
        num_eval_classes = args.num_classes
        conf = torch.zeros((num_eval_classes, num_eval_classes), device=device)

        for optical, sar, labels in val_loader:
            optical = optical.to(device, non_blocking=True)
            sar = sar.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            H, W = optical.shape[-2:]
            optical_encodings, radar_encodings = global_client.encode(optical, sar)
            logits = server(
                radar_encodings=radar_encodings,
                optical_encodings=optical_encodings,
                attn_bias=global_client.attn_bias.to(device),
                output_size=(H, W),
            )
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)

            valid = labels != -1
            if valid.any():
                y_true = labels[valid].long()
                y_pred = preds[valid].long()
                in_range = (
                    (y_true >= 0) & (y_true < num_eval_classes)
                    & (y_pred >= 0) & (y_pred < num_eval_classes)
                )
                y_true = y_true[in_range]
                y_pred = y_pred[in_range]
                if y_true.numel() > 0:
                    idx = y_true * num_eval_classes + y_pred
                    conf += torch.bincount(
                        idx,
                        minlength=num_eval_classes * num_eval_classes,
                    ).reshape(num_eval_classes, num_eval_classes)

            total_loss += loss.item() * labels.numel()
            num_pixels += labels.numel()

        avg_loss = total_loss / max(1, num_pixels)

        total = conf.sum()
        if total > 0:
            diag = torch.diag(conf).float()
            oa = (diag.sum() / total).item()

            per_class_total = conf.sum(dim=1)
            valid_cls = per_class_total > 0
            class_acc = torch.zeros_like(per_class_total, dtype=torch.float32)
            class_acc[valid_cls] = diag[valid_cls] / per_class_total[valid_cls].float()
            aa = class_acc[valid_cls].mean().item() if valid_cls.any() else 0.0

            row_marginal = conf.sum(dim=1)
            col_marginal = conf.sum(dim=0)
            pe = (row_marginal * col_marginal).sum() / (total * total)
            po = diag.sum() / total
            kappa = ((po - pe) / (1 - pe)).item() if float(1 - pe) > 1e-12 else 0.0
        else:
            oa, aa, kappa = 0.0, 0.0, 0.0

        elapsed = time.time() - start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        print(
            f"[{elapsed_str}] [Val] Epoch {epoch} | Loss: {avg_loss:.4f} | "
            f"OA: {oa:.4f} | AA: {aa:.4f} | Kappa: {kappa:.4f}"
        )
        return avg_loss, {"oa": oa, "aa": aa, "kappa": kappa}

    # whu-opt-sar / bigearthnet：计算 mIoU
    total_loss = 0.0
    num_pixels = 0
    num_classes = args.num_classes
    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)

    for optical, sar, labels in val_loader:
        optical = optical.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        H, W = optical.shape[-2:]
        optical_encodings, radar_encodings = global_client.encode(optical, sar)
        logits = server(
            radar_encodings=radar_encodings,
            optical_encodings=optical_encodings,
            attn_bias=global_client.attn_bias.to(device),
            output_size=(H, W),
        )
        loss = criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)  # [B, H, W]
        for c in range(num_classes):
            pred_c = preds == c
            label_c = labels == c
            intersection[c] += (pred_c & label_c).sum()
            union[c] += (pred_c | label_c).sum()

        total_loss += loss.item() * labels.numel()
        num_pixels += labels.numel()

    avg_loss = total_loss / max(1, num_pixels)

    iou_per_class = intersection / (union + 1e-6)
    valid = union > 0
    miou = iou_per_class[valid].mean().item() if valid.any() else 0.0

    elapsed = time.time() - start_time
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    print(f"[{elapsed_str}] [Val] Epoch {epoch} | Loss: {avg_loss:.4f} | mIoU: {miou:.4f}")
    return avg_loss, {"miou": miou}


def save_checkpoint(global_client: SplitClient, server: ServerModel,
                    server_optimizer: torch.optim.Optimizer,
                    proj_optimizer: torch.optim.Optimizer,
                    args, epoch: int, run_dir: str, last_ckpt_path: Optional[str] = None):
    """保存全局客户端模型、服务器模型与优化器状态。"""
    os.makedirs(run_dir, exist_ok=True)

    if last_ckpt_path is not None and os.path.exists(last_ckpt_path):
        try:
            os.remove(last_ckpt_path)
        except OSError:
            pass

    ckpt_path = os.path.join(run_dir, f"fed_split_distil_checkpoint_epoch_{epoch}.pt")
    torch.save(
        {
            "epoch": epoch,
            "global_client_state_dict": global_client.state_dict(),
            "server_state_dict": server.state_dict(),
            "server_optimizer_state_dict": server_optimizer.state_dict(),
            "proj_optimizer_state_dict": proj_optimizer.state_dict(),
            "args": vars(args),
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")
    return ckpt_path


def main():
    args = parse_args()

    # 按数据集覆盖类别数
    if args.dataset == "bigearthnet":
        num_be_classes = len(CLASS_NAMES)
        if args.num_classes != num_be_classes:
            print(f"[Info] BigEarthNet: 覆盖 num_classes 从 {args.num_classes} 到 {num_be_classes}")
            args.num_classes = num_be_classes
    elif args.dataset == "houston2013":
        num_houston_classes = 15
        if args.num_classes != num_houston_classes:
            print(f"[Info] Houston2013: 覆盖 num_classes 从 {args.num_classes} 到 {num_houston_classes}")
            args.num_classes = num_houston_classes

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 60)
    print("Federated Split Learning + Distillation 配置:")
    print(f"  - 服务器: cross_encoder + seg_head + proj_OM/proj_RM")
    print(f"  - 客户端: radar_encoder + optical_encoder, 共 {args.num_clients} 个")
    print(f"  - 本地训练周期: {args.local_epochs}")
    print(f"  - 数据集: {args.dataset} | Device: {device}")
    print("=" * 60)

    client_loaders, val_loader, inferred_num_patches = create_loaders(args)
    shard_sizes = [len(loader.dataset) for loader in client_loaders]
    print(
        f"客户端数据划分: {shard_sizes} | "
        f"验证样本: {len(val_loader.dataset)}"
    )

    clients, server, attn_bias, num_patches = build_models(
        args, device, inferred_num_patches=inferred_num_patches
    )

    # 冻结所有客户端的编码器（可选）
    if args.freeze_encoders:
        for client in clients:
            for param in client.parameters():
                param.requires_grad = False
        trainable = sum(p.numel() for p in server.parameters() if p.requires_grad)
        total = sum(p.numel() for p in clients[0].parameters()) * args.num_clients \
            + sum(p.numel() for p in server.parameters())
        print(
            f"[Freeze] 所有客户端编码器已冻结 | "
            f"可训练参数（服务器）: {trainable:,} / 总参数: {total:,}"
        )

    # 服务器优化器：仅 cross_encoder + seg_head
    server_optimizer = torch.optim.AdamW(
        list(server.cross_encoder.parameters()) + list(server.seg_head.parameters()),
        lr=args.lr_server,
        weight_decay=args.weight_decay,
    )
    # 投影模型优化器：proj_OM + proj_RM
    proj_optimizer = torch.optim.AdamW(
        list(server.proj_OM.parameters()) + list(server.proj_RM.parameters()),
        lr=args.lr_server,
        weight_decay=args.weight_decay,
    )

    # 用于验证的全局客户端模型（独立实例，仅用于验证）
    global_client = SplitClient(
        optical_encoder=copy.deepcopy(clients[0].optical_encoder),
        radar_encoder=copy.deepcopy(clients[0].radar_encoder),
        attn_bias=attn_bias,
    ).to(device)

    # 初始化全局客户端模型（初始各客户端为同一预训练权重，聚合结果即预训练权重）
    global_state = aggregate_client_models(clients, device)
    sync_clients_to_global(clients, global_state)
    global_client.load_state_dict(global_state)

    # 损失函数：bigearthnet / houston2013 存在 -1 忽略类
    if args.dataset in {"bigearthnet", "houston2013"}:
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
    else:
        criterion = nn.CrossEntropyLoss()

    # 运行目录和指标文件
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, run_timestamp)
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "epoch_metrics.csv")

    start_time = time.time()
    last_ckpt_path = None
    total_comm_bytes = 0

    for epoch in range(1, args.epochs + 1):
        # ===== 1. 本轮开始：各客户端从全局客户端模型同步参数 =====
        sync_clients_to_global(clients, global_state)

        # ===== 2. 服务器依次连接各客户端：连接阶段训练 + 断开后本地训练 =====
        train_loss, fwd_bytes = train_one_round(
            args, clients, server, server_optimizer, proj_optimizer,
            client_loaders, criterion, device, epoch, start_time,
        )
        epoch_comm_bytes = fwd_bytes
        total_comm_bytes += epoch_comm_bytes

        # ===== 3. 聚合客户端模型 -> 全局客户端模型 =====
        global_state = aggregate_client_models(clients, device)
        global_client.load_state_dict(global_state)

        # ===== 4. 使用全局客户端模型 + 服务器在验证集上验证 =====
        val_loss, val_metrics = evaluate(
            args, global_client, server, val_loader, criterion, device, epoch, start_time
        )

        # 记录指标
        file_exists = os.path.exists(metrics_path)
        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            if args.dataset == "houston2013":
                if not file_exists:
                    writer.writerow([
                        "epoch",
                        "train_loss",
                        "val_loss",
                        "val_OA",
                        "val_AA",
                        "val_kappa",
                        "round_comm_MB",
                        "total_comm_MB",
                    ])
                writer.writerow([
                    epoch,
                    float(train_loss),
                    float(val_loss),
                    float(val_metrics["oa"]),
                    float(val_metrics["aa"]),
                    float(val_metrics["kappa"]),
                    epoch_comm_bytes / (1024 * 1024),
                    total_comm_bytes / (1024 * 1024),
                ])
            else:
                if not file_exists:
                    writer.writerow([
                        "epoch",
                        "train_loss",
                        "val_loss",
                        "val_mIoU",
                        "round_comm_MB",
                        "total_comm_MB",
                    ])
                writer.writerow([
                    epoch,
                    float(train_loss),
                    float(val_loss),
                    float(val_metrics["miou"]),
                    epoch_comm_bytes / (1024 * 1024),
                    total_comm_bytes / (1024 * 1024),
                ])

        if not args.save_final_only:
            last_ckpt_path = save_checkpoint(
                global_client, server, server_optimizer, proj_optimizer,
                args, epoch, run_dir, last_ckpt_path,
            )

    if args.save_final_only:
        last_ckpt_path = save_checkpoint(
            global_client, server, server_optimizer, proj_optimizer,
            args, args.epochs, run_dir, last_ckpt_path,
        )

    print("=" * 60)
    print(f"训练完成! 总通信量: {total_comm_bytes / (1024 * 1024 * 1024):.2f} GB")
    print("=" * 60)


if __name__ == "__main__":
    main()
