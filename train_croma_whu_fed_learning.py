"""
Federated Learning (FedAvg) 分割训练脚本（单卡串行模拟）

架构：
- 客户端 (Client，共 NUM_CLIENTS 个)：每个客户端持有完整模型
    optical_encoder + radar_encoder + cross_encoder (joint encoder) + 分割头 (seg_head)
- 无独立服务器模型：全局模型由各客户端参数聚合得到

训练流程（每个 epoch 视为一轮）：
1. 本轮开始时，各客户端从上一轮聚合得到的全局模型同步参数
2. 服务器（协调者）依次连接每个客户端，各客户端在本地数据集上训练 local_epochs 个周期：
   - 客户端完整前向：optical/radar 编码 -> cross_encoder 融合 -> seg_head 分割
   - 计算交叉熵损失，本地反向传播并更新完整模型参数
3. 所有客户端都完成训练后，对多个客户端模型（完整模型）参数做 FedAvg 聚合，形成全局模型
4. 使用全局模型在验证集上验证
5. 进入下一轮：各客户端使用全局模型开始训练

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


class ClientModel(nn.Module):
    """联邦学习客户端模型：部署完整模型。

    包含 optical_encoder + radar_encoder + cross_encoder (joint encoder) + 分割头 (seg_head)。
    每个客户端独立持有完整模型副本，在本地数据上训练后参与 FedAvg 聚合。
    """

    def __init__(self, optical_encoder: nn.Module, radar_encoder: nn.Module,
                 cross_encoder: nn.Module, seg_head: nn.Module,
                 attn_bias: torch.Tensor, num_patches: int):
        super().__init__()
        self.optical_encoder = optical_encoder
        self.radar_encoder = radar_encoder
        self.cross_encoder = cross_encoder
        self.seg_head = seg_head
        self.register_buffer("attn_bias", attn_bias)
        self.num_patches = num_patches

        # 假设 patch 在空间上是正方形网格
        self.h_patches = int(num_patches ** 0.5)
        self.w_patches = int(num_patches ** 0.5)

    def forward(self, optical: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        """joint 模式：完整前向，返回 [B, num_classes, H, W] 的分割 logits。"""
        attn_bias = self.attn_bias.to(optical.device)

        optical_encodings = self.optical_encoder(
            imgs=optical, attn_bias=attn_bias, mask_info=None
        )  # [B, N, D]
        radar_encodings = self.radar_encoder(
            imgs=sar, attn_bias=attn_bias, mask_info=None
        )  # [B, N, D]

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
        logits_low = self.seg_head(feat)

        # 上采样到原图大小进行像素级分割
        H, W = optical.shape[-2:]
        logits = F.interpolate(logits_low, size=(H, W), mode="bilinear", align_corners=False)
        return logits


def parse_args():
    parser = argparse.ArgumentParser(
        description="Federated Learning (FedAvg) 分割训练 - 单卡串行模拟 "
                    "(每个客户端持有 optical_encoder+radar_encoder+cross_encoder+seg_head 完整模型)"
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
        default=10,
        help="每个客户端每轮本地训练的周期数",
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
        default="../CROMA_checkpoint/croma_whu_fed_learning_checkpoints",
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
        help="冻结所有客户端的 radar_encoder 和 optical_encoder 参数，仅训练 cross_encoder 和分割头",
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
    """构建 num_clients 个完整客户端模型与 attn_bias。

    Returns:
        clients: List[ClientModel]，每个客户端持有完整模型副本
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

    # 每个客户端持有完整模型副本
    clients = []
    for _ in range(args.num_clients):
        client = ClientModel(
            optical_encoder=copy.deepcopy(croma.optical_encoder),
            radar_encoder=copy.deepcopy(croma.radar_encoder),
            cross_encoder=copy.deepcopy(croma.cross_encoder),
            seg_head=copy.deepcopy(seg_head_template),
            attn_bias=attn_bias,
            num_patches=num_patches,
        ).to(device)
        clients.append(client)

    return clients, attn_bias, num_patches


def client_train_epoch(model: ClientModel, loader: DataLoader,
                       optimizer: torch.optim.Optimizer, criterion: nn.Module,
                       device: torch.device, args, epoch: int,
                       client_id: int, local_epoch: int, start_time: float) -> float:
    """单个客户端本地训练一个周期（完整模型前向/反向）。

    Returns:
        avg_loss: 按像素加权的平均交叉熵损失
    """
    model.train()
    total_loss = 0.0
    num_pixels = 0

    for step, (optical, sar, labels) in enumerate(loader):
        optical = optical.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(optical, sar)  # [B, num_classes, H, W]
        loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * labels.numel()
        num_pixels += labels.numel()

        if (step + 1) % args.log_interval == 0:
            elapsed = time.time() - start_time
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            print(
                f"[{elapsed_str}] Epoch {epoch} | Client {client_id+1}/{args.num_clients} | "
                f"LocalEpoch {local_epoch+1}/{args.local_epochs} | "
                f"Step {step+1}/{len(loader)} | Loss: {loss.item():.4f}"
            )

    return total_loss / max(1, num_pixels)


def train_one_round(args, clients: List[ClientModel],
                    client_optimizers: List[torch.optim.Optimizer],
                    client_loaders: List[DataLoader], criterion: nn.Module,
                    device: torch.device, epoch: int, start_time: float):
    """一轮训练：协调者依次连接各客户端，每个客户端本地训练 local_epochs 个周期。

    Returns:
        avg_loss: 按像素加权的平均训练损失
    """
    total_loss = 0.0
    total_pixels = 0

    for client_id in range(args.num_clients):
        model = clients[client_id]
        optimizer = client_optimizers[client_id]
        loader = client_loaders[client_id]

        if len(loader) == 0:
            print(f"[Epoch {epoch}] Client {client_id+1}/{args.num_clients}: 本地数据为空，跳过")
            continue

        client_loss = 0.0
        client_pixels = 0

        for local_epoch in range(args.local_epochs):
            local_loss = client_train_epoch(
                model, loader, optimizer, criterion, device,
                args, epoch, client_id, local_epoch, start_time,
            )
            # 近似按像素加权累计（每个本地周期样本量相同）
            client_loss += local_loss
            client_pixels += 1

        client_avg = client_loss / max(1, client_pixels)
        total_loss += client_avg * len(loader.dataset)
        total_pixels += len(loader.dataset)
        print(
            f"[Epoch {epoch}] Client {client_id+1}/{args.num_clients} 本地训练完成 | "
            f"本地样本: {len(loader.dataset)} | 平均损失: {client_avg:.4f}"
        )

    avg_loss = total_loss / max(1, total_pixels)
    return avg_loss


def aggregate_client_models(clients: List[ClientModel], device: torch.device) -> OrderedDict:
    """对多个客户端完整模型参数做 FedAvg 聚合，得到全局模型 state_dict。"""
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


def sync_clients_to_global(clients: List[ClientModel], global_state: OrderedDict):
    """本轮开始时，让各客户端从全局模型同步参数。"""
    for client in clients:
        client.load_state_dict(global_state)


@torch.no_grad()
def evaluate(args, global_model: ClientModel, val_loader: DataLoader,
             criterion: nn.Module, device: torch.device,
             epoch: int, start_time: float):
    """使用全局模型在验证集上评估。"""
    global_model.eval()

    if args.dataset == "houston2013":
        total_loss = 0.0
        num_pixels = 0
        num_eval_classes = args.num_classes
        conf = torch.zeros((num_eval_classes, num_eval_classes), device=device)

        for optical, sar, labels in val_loader:
            optical = optical.to(device, non_blocking=True)
            sar = sar.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = global_model(optical, sar)
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

        logits = global_model(optical, sar)
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


def save_checkpoint(global_model: ClientModel, args, epoch: int,
                    run_dir: str, last_ckpt_path: Optional[str] = None):
    """保存全局模型状态。"""
    os.makedirs(run_dir, exist_ok=True)

    if last_ckpt_path is not None and os.path.exists(last_ckpt_path):
        try:
            os.remove(last_ckpt_path)
        except OSError:
            pass

    ckpt_path = os.path.join(run_dir, f"fed_checkpoint_epoch_{epoch}.pt")
    torch.save(
        {
            "epoch": epoch,
            "global_client_state_dict": global_model.state_dict(),
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
    print("Federated Learning (FedAvg) 配置:")
    print(f"  - 每个客户端: optical_encoder + radar_encoder + cross_encoder + seg_head")
    print(f"  - 客户端数量: {args.num_clients} | 本地训练周期: {args.local_epochs}")
    print(f"  - 数据集: {args.dataset} | Device: {device}")
    print("=" * 60)

    client_loaders, val_loader, inferred_num_patches = create_loaders(args)
    shard_sizes = [len(loader.dataset) for loader in client_loaders]
    print(
        f"客户端数据划分: {shard_sizes} | "
        f"验证样本: {len(val_loader.dataset)}"
    )

    clients, attn_bias, num_patches = build_models(
        args, device, inferred_num_patches=inferred_num_patches
    )

    # 冻结所有客户端的编码器（可选）
    if args.freeze_encoders:
        for client in clients:
            for param in client.optical_encoder.parameters():
                param.requires_grad = False
            for param in client.radar_encoder.parameters():
                param.requires_grad = False
        trainable = sum(
            p.numel() for p in clients[0].parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in clients[0].parameters()) * args.num_clients
        print(
            f"[Freeze] 所有客户端编码器已冻结 | "
            f"可训练参数（每客户端 cross_encoder+seg_head）: {trainable:,} / 总参数: {total:,}"
        )

    # 每个客户端一个优化器
    client_optimizers = [
        torch.optim.AdamW(c.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for c in clients
    ]

    # 用于验证的全局模型（独立实例，仅用于验证）
    global_model = copy.deepcopy(clients[0]).to(device)

    # 初始化全局模型（初始各客户端为同一预训练权重，聚合结果即预训练权重）
    global_state = aggregate_client_models(clients, device)
    sync_clients_to_global(clients, global_state)
    global_model.load_state_dict(global_state)

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

    for epoch in range(1, args.epochs + 1):
        # ===== 1. 本轮开始：各客户端从全局模型同步参数 =====
        sync_clients_to_global(clients, global_state)

        # ===== 2. 依次连接各客户端，本地训练 local_epochs 个周期 =====
        train_loss = train_one_round(
            args, clients, client_optimizers, client_loaders,
            criterion, device, epoch, start_time,
        )

        # ===== 3. 聚合客户端模型 -> 全局模型 =====
        global_state = aggregate_client_models(clients, device)
        global_model.load_state_dict(global_state)

        # ===== 4. 使用全局模型在验证集上验证 =====
        val_loss, val_metrics = evaluate(
            args, global_model, val_loader, criterion, device, epoch, start_time
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
                    ])
                writer.writerow([
                    epoch,
                    float(train_loss),
                    float(val_loss),
                    float(val_metrics["oa"]),
                    float(val_metrics["aa"]),
                    float(val_metrics["kappa"]),
                ])
            else:
                if not file_exists:
                    writer.writerow([
                        "epoch",
                        "train_loss",
                        "val_loss",
                        "val_mIoU",
                    ])
                writer.writerow([
                    epoch,
                    float(train_loss),
                    float(val_loss),
                    float(val_metrics["miou"]),
                ])

        if not args.save_final_only:
            last_ckpt_path = save_checkpoint(
                global_model, args, epoch, run_dir, last_ckpt_path
            )

    if args.save_final_only:
        last_ckpt_path = save_checkpoint(
            global_model, args, args.epochs, run_dir, last_ckpt_path
        )

    print("=" * 60)
    print("训练完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
