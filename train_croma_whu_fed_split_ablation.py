"""
Federated Split Learning 消融实验版本（单卡串行模拟）

与 fed_split_distil 版本的区别（消融点）：
1. 去掉投影层 proj_OM/proj_RM：客户端本地模型不再有投影层，
   分割头直接对 optical_encoder / radar_encoder 的输出进行分割。
2. 客户端各自维护自己的分割头：分割头不再每个训练轮从服务器端复制，
   而是在构建时初始化一次，之后每个客户端在本地持久更新、互不共享。
3. 服务器端不再训练投影模型，仅保留 cross_encoder + seg_head（joint 路径与验证）。

架构划分：
- 服务器端 (Server)：cross_encoder + 分割头 (seg_head)
- 客户端 (Client，共 NUM_CLIENTS 个)：
    - optical_encoder + radar_encoder（参与 FedAvg 聚合，形成全局客户端模型）
    - seg_head_OM / seg_head_RM（各自维护的本地分割头，不参与聚合）

训练流程（每个 epoch 视为一轮）：
1. 本轮开始时，各客户端从上一轮聚合得到的全局客户端模型（编码器）同步参数
2. 服务器依次与每个客户端建立连接（连接阶段）：
   - 客户端前向传播，得到 radar_encodings 和 optical_encodings，发送给服务器
   - 服务器经 cross_encoder 融合 + seg_head 分割，计算交叉熵损失并更新服务器参数
   - 服务器不向客户端传递梯度
3. 客户端与服务器断开连接后（单卡环境，紧接其后）：
   - 客户端用本地数据集训练两个模型（无投影层）：
       optical_encoder -> seg_head_OM
       radar_encoder  -> seg_head_RM
     共 local_epochs 个周期，更新本地编码器与各自维护的本地分割头
4. 所有客户端都完成一轮后，对客户端编码器参数做 FedAvg 聚合，形成全局客户端模型
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
    """客户端模型（消融版）：radar_encoder + optical_encoder + 各自维护的本地分割头。

    - optical_encoder / radar_encoder：参与 FedAvg 聚合（全局客户端模型）
    - seg_head_OM / seg_head_RM：客户端各自维护的本地分割头，不参与聚合，
      构建时初始化一次，之后在本地训练中持久更新
    """

    def __init__(self, optical_encoder: nn.Module, radar_encoder: nn.Module,
                 seg_head_OM: nn.Module, seg_head_RM: nn.Module,
                 attn_bias: torch.Tensor, num_patches: int):
        super().__init__()
        self.optical_encoder = optical_encoder
        self.radar_encoder = radar_encoder
        self.seg_head_OM = seg_head_OM  # optical 分支本地分割头
        self.seg_head_RM = seg_head_RM  # radar 分支本地分割头
        self.register_buffer("attn_bias", attn_bias)
        self.num_patches = num_patches

        # 假设 patch 在空间上是正方形网格
        self.h_patches = int(num_patches ** 0.5)
        self.w_patches = int(num_patches ** 0.5)

    def encode(self, optical_imgs: torch.Tensor, radar_imgs: torch.Tensor):
        """客户端前向：分别编码光学与雷达图像（连接阶段使用，编码器不更新）。

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


class ServerModel(nn.Module):
    """服务器模型（消融版）：cross_encoder + 分割头 (seg_head)。

    接收客户端发送的 radar_encodings 与 optical_encodings，
    完成跨模态融合与像素级分割（交叉熵损失）。无投影模型。
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

    def forward(self, radar_encodings: torch.Tensor, optical_encodings: torch.Tensor,
                attn_bias: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        """joint 模式：融合编码并生成分割预测（用于验证）。"""
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
    """客户端本地模型（消融版）：encoder -> seg_head（无投影层）。

    分割头直接对 encoder 的输出进行分割。
    """

    def __init__(self, encoder: nn.Module, seg_head: nn.Module, num_patches: int):
        super().__init__()
        self.encoder = encoder
        self.seg_head = seg_head
        self.num_patches = num_patches
        self.h_patches = int(num_patches ** 0.5)
        self.w_patches = int(num_patches ** 0.5)

    def forward(self, imgs: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        """单模态图像 -> 编码 -> 分割 logits [B, num_classes, H, W]（无投影层）。"""
        encodings = self.encoder(
            imgs=imgs, attn_bias=attn_bias, mask_info=None
        )  # [B, N, D]

        b, n, d = encodings.shape
        assert n == self.h_patches * self.w_patches, \
            "num_patches 与 h_patches*w_patches 不一致，请检查 image_size/patch_size 设置"

        feat = (
            encodings.view(b, self.h_patches, self.w_patches, d)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        logits_low = self.seg_head(feat)

        H, W = imgs.shape[-2:]
        logits = F.interpolate(logits_low, size=(H, W), mode="bilinear", align_corners=False)
        return logits


def parse_args():
    parser = argparse.ArgumentParser(
        description="Federated Split Learning 消融实验 - 单卡串行模拟 "
                    "(去掉投影层, 分割头直接对编码输出分割, 客户端各自维护本地分割头)"
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
    parser.add_argument("--lr_server", type=float, default=1e-4, help="服务器（cross_encoder+分割头）学习率")
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
        default="../CROMA_checkpoint/croma_whu_fed_split_ablation_checkpoints",
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
        help="冻结所有客户端的 radar_encoder 和 optical_encoder 参数，仅训练服务器与本地分割头",
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
            num_ratio=args.num_ratio,
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
        clients: List[SplitClient]，每个客户端持有编码器 + 各自维护的本地分割头
        server: ServerModel（cross_encoder + 分割头）
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

    # 分割头模板（与训练脚本中 seg_head 相同的结构）
    seg_head_template = nn.Sequential(
        nn.Conv2d(args.encoder_dim, args.encoder_dim, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(args.encoder_dim, args.num_classes, kernel_size=1),
    )

    # 每个客户端：编码器（聚合）+ 各自维护的本地分割头
    # 本地分割头仅在构建时初始化一次，之后各客户端在本地训练中持久更新，不再从服务器复制
    clients = []
    for _ in range(args.num_clients):
        client = SplitClient(
            optical_encoder=copy.deepcopy(croma.optical_encoder),
            radar_encoder=copy.deepcopy(croma.radar_encoder),
            seg_head_OM=copy.deepcopy(seg_head_template),
            seg_head_RM=copy.deepcopy(seg_head_template),
            attn_bias=attn_bias,
            num_patches=num_patches,
        ).to(device)
        clients.append(client)

    # 服务器持有 cross_encoder + 分割头（无投影模型）
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
                        criterion: nn.Module, device: torch.device,
                        max_grad_norm: float = 0.0):
    """客户端与服务器连接阶段的单步训练（模拟通信，无投影模型、无梯度回传）。

    流程：
    1. 客户端前向，得到 radar_encodings 与 optical_encodings（编码器不更新）
    2. 发送激活值到服务器（模拟通信，切断计算图）
    3. 服务器前向：cross_encoder -> joint_encodings -> seg_head -> logits
    4. 计算交叉熵损失，更新服务器参数（cross_encoder + seg_head）

    Returns:
        loss: 当前 batch 的交叉熵损失
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
    logits = server(
        radar_encodings=radar_act,
        optical_encodings=optical_act,
        attn_bias=attn_bias,
        output_size=(H, W),
    )  # [B, num_classes, H, W]
    loss = criterion(logits, labels)

    # ===== 阶段 4: 更新服务器 (cross_encoder + seg_head) =====
    server_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(server.parameters(), max_grad_norm)
    server_optimizer.step()

    return loss.item(), forward_bytes


def client_local_train(args, client: SplitClient,
                       loader: DataLoader, device: torch.device,
                       epoch: int, client_id: int, start_time: float) -> float:
    """客户端与服务器断开连接后的本地训练（单卡环境，紧接在连接阶段之后）。

    使用客户端各自维护的本地分割头（不从服务器复制）训练两个模型（无投影层）：
      - optical_encoder -> seg_head_OM
      - radar_encoder  -> seg_head_RM

    Returns:
        avg_loss: 两个本地模型按像素加权的平均交叉熵损失
    """
    # 本地模型直接使用客户端自身维护的分割头
    model_OM = ClientLocalModel(
        encoder=client.optical_encoder,
        seg_head=client.seg_head_OM,
        num_patches=client.num_patches,
    ).to(device)
    model_RM = ClientLocalModel(
        encoder=client.radar_encoder,
        seg_head=client.seg_head_RM,
        num_patches=client.num_patches,
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

            # 模型 A: optical_encoder -> seg_head_OM（无投影层）
            model_OM.train()
            logits_OM = model_OM(optical, attn_bias)
            loss_OM = criterion(logits_OM, labels)
            optimizer_OM.zero_grad(set_to_none=True)
            loss_OM.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model_OM.parameters(), args.max_grad_norm)
            optimizer_OM.step()

            # 模型 B: radar_encoder -> seg_head_RM（无投影层）
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

        # ===== 连接阶段：训练服务器 =====
        conn_loss = 0.0
        conn_pixels = 0
        for step, (optical, sar, labels) in enumerate(loader):
            loss, fwd_bytes = server_connect_step(
                client=client,
                server=server,
                optical=optical,
                sar=sar,
                labels=labels,
                server_optimizer=server_optimizer,
                criterion=criterion,
                device=device,
                max_grad_norm=args.max_grad_norm,
            )

            pixels = labels.numel()
            conn_loss += loss * pixels
            conn_pixels += pixels
            total_forward_bytes += fwd_bytes

            if (step + 1) % args.log_interval == 0:
                elapsed = time.time() - start_time
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                print(
                    f"[{elapsed_str}] Epoch {epoch} | Client {client_id+1}/{args.num_clients} | "
                    f"Step {step+1}/{len(loader)} | Loss: {loss:.4f}"
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
            args, client, loader, device, epoch, client_id, start_time
        )

    avg_loss = total_loss / max(1, total_pixels)
    return avg_loss, total_forward_bytes


def is_aggregatable_key(key: str) -> bool:
    """判断 key 是否参与 FedAvg 聚合（仅编码器与 attn_bias；本地分割头不聚合）。"""
    return (
        key.startswith("optical_encoder.")
        or key.startswith("radar_encoder.")
        or key == "attn_bias"
    )


def aggregate_client_models(clients: List[SplitClient], device: torch.device) -> OrderedDict:
    """对多个客户端的编码器参数做 FedAvg 聚合，得到全局客户端模型 state_dict。

    客户端各自维护的本地分割头（seg_head_OM/seg_head_RM）不参与聚合。
    """
    num_clients = len(clients)
    global_state = OrderedDict()
    with torch.no_grad():
        first_state = clients[0].state_dict()
        for key in first_state.keys():
            if not is_aggregatable_key(key):
                continue  # 跳过本地分割头
            stacked = torch.stack(
                [clients[i].state_dict()[key].float().to(device) for i in range(num_clients)]
            )
            global_state[key] = stacked.mean(dim=0)
    return global_state


def sync_clients_to_global(clients: List[SplitClient], global_state: OrderedDict):
    """本轮开始时，让各客户端从全局客户端模型（编码器）同步参数。

    本地分割头不包含在 global_state 中，因此用 strict=False 加载，各自保留。
    """
    for client in clients:
        client.load_state_dict(global_state, strict=False)


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


def save_checkpoint(global_state: OrderedDict, server: ServerModel,
                    server_optimizer: torch.optim.Optimizer,
                    args, epoch: int, run_dir: str, last_ckpt_path: Optional[str] = None):
    """保存全局客户端模型（编码器聚合状态）、服务器模型与优化器状态。"""
    os.makedirs(run_dir, exist_ok=True)

    if last_ckpt_path is not None and os.path.exists(last_ckpt_path):
        try:
            os.remove(last_ckpt_path)
        except OSError:
            pass

    ckpt_path = os.path.join(run_dir, f"fed_split_ablation_checkpoint_epoch_{epoch}.pt")
    torch.save(
        {
            "epoch": epoch,
            "global_client_state_dict": global_state,  # 编码器聚合状态
            "server_state_dict": server.state_dict(),
            "server_optimizer_state_dict": server_optimizer.state_dict(),
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
    print("Federated Split Learning 消融实验配置:")
    print(f"  - 服务器: cross_encoder + seg_head")
    print(f"  - 客户端: radar_encoder + optical_encoder + 各自维护的本地分割头, 共 {args.num_clients} 个")
    print(f"  - 本地训练周期: {args.local_epochs} | 无投影层")
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
            for param in client.optical_encoder.parameters():
                param.requires_grad = False
            for param in client.radar_encoder.parameters():
                param.requires_grad = False
        trainable = sum(p.numel() for p in server.parameters() if p.requires_grad)
        total = sum(p.numel() for p in clients[0].parameters()) * args.num_clients \
            + sum(p.numel() for p in server.parameters())
        print(
            f"[Freeze] 所有客户端编码器已冻结 | "
            f"可训练参数（服务器）: {trainable:,} / 总参数: {total:,}"
        )

    # 服务器优化器：cross_encoder + seg_head
    server_optimizer = torch.optim.AdamW(
        list(server.cross_encoder.parameters()) + list(server.seg_head.parameters()),
        lr=args.lr_server,
        weight_decay=args.weight_decay,
    )

    # 用于验证的全局客户端模型（独立实例，仅用于验证，分割头不参与验证）
    global_client = copy.deepcopy(clients[0]).to(device)

    # 初始化全局客户端模型（编码器聚合，本地分割头不聚合）
    global_state = aggregate_client_models(clients, device)
    sync_clients_to_global(clients, global_state)
    global_client.load_state_dict(global_state, strict=False)

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
        # ===== 1. 本轮开始：各客户端从全局客户端模型（编码器）同步参数 =====
        sync_clients_to_global(clients, global_state)

        # ===== 2. 服务器依次连接各客户端：连接阶段训练 + 断开后本地训练 =====
        train_loss, fwd_bytes = train_one_round(
            args, clients, server, server_optimizer,
            client_loaders, criterion, device, epoch, start_time,
        )
        epoch_comm_bytes = fwd_bytes
        total_comm_bytes += epoch_comm_bytes

        # ===== 3. 聚合客户端编码器 -> 全局客户端模型 =====
        global_state = aggregate_client_models(clients, device)
        global_client.load_state_dict(global_state, strict=False)

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
                global_state, server, server_optimizer,
                args, epoch, run_dir, last_ckpt_path,
            )

    if args.save_final_only:
        last_ckpt_path = save_checkpoint(
            global_state, server, server_optimizer,
            args, args.epochs, run_dir, last_ckpt_path,
        )

    print("=" * 60)
    print(f"训练完成! 总通信量: {total_comm_bytes / (1024 * 1024 * 1024):.2f} GB")
    print("=" * 60)


if __name__ == "__main__":
    main()
