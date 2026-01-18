import os
import argparse
from datetime import datetime
import time
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets import WHUOptSarPatchDataset
from pretrain_croma import CROMA

OPT_CHANNELS = 4
SAR_CHANNELS = 1


class CROMASegmentation(nn.Module):
    """使用预训练好的 CROMA 作为编码器的分割模型。"""

    def __init__(self, croma_model: CROMA, num_classes: int):
        super().__init__()
        self.croma = croma_model
        dim = self.croma.encoder_dim

        # 假设 patch 在空间上是正方形网格
        self.h_patches = int(self.croma.num_patches ** 0.5)
        self.w_patches = int(self.croma.num_patches ** 0.5)

        self.seg_head = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, num_classes, kernel_size=1),
        )

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        # imgs: [B, C, H, W], C = OPT_CHANNELS + SAR_CHANNELS
        device = imgs.device
        radar_imgs = imgs[:, self.croma.opt_channels :, ...]
        optical_imgs = imgs[:, : self.croma.opt_channels, ...]

        attn_bias = self.croma.attn_bias.to(device)

        # 不使用 mask，直接提取完整 patch 编码
        radar_encodings = self.croma.radar_encoder(
            imgs=radar_imgs, attn_bias=attn_bias, mask_info=None
        )  # [B, N, D]
        optical_encodings = self.croma.optical_encoder(
            imgs=optical_imgs, attn_bias=attn_bias, mask_info=None
        )  # [B, N, D]

        joint_encodings = self.croma.cross_encoder(
            x=radar_encodings, context=optical_encodings, alibi=attn_bias
        )  # [B, N, D]

        b, n, d = joint_encodings.shape
        assert (
            n == self.h_patches * self.w_patches
        ), "num_patches 与 h_patches*w_patches 不一致，请检查 image_size/patch_size 设置"

        # [B, N, D] -> [B, D, H_p, W_p]
        feat = (
            joint_encodings.view(b, self.h_patches, self.w_patches, d)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        logits_low = self.seg_head(feat)  # [B, num_classes, H_p, W_p]

        # 上采样到原图大小进行像素级分割
        H, W = imgs.shape[-2:]
        logits = F.interpolate(
            logits_low, size=(H, W), mode="bilinear", align_corners=False
        )
        return logits


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune CROMA on WHU Opt-SAR for segmentation"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="../whu-opt-sar",
        help="WHU 光学-SAR 数据根目录，包含 optical/sar/lbl 子目录",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="单个裁剪 patch 的空间尺寸，等于 WHUOptSarPatchDataset.patch_size",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--encoder_dim", type=int, default=768)
    parser.add_argument("--encoder_layers", type=int, default=12)
    parser.add_argument("--attention_heads", type=int, default=16)
    parser.add_argument("--decoder_dim", type=int, default=512)
    parser.add_argument("--decoder_layers", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        required=True,
        help="预训练 CROMA checkpoint 路径 (train_croma_whu_pretrain.py 保存的 .pt 文件)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../CROMA_checkpoint/croma_whu_segmentation_checkpoints",
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
        help="滑动窗口步长与 image_size 的比例",
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=0.0, help=">0 时启用梯度裁剪"
    )
    return parser.parse_args()


def create_loaders(args, rank, world_size, distributed):
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

    if distributed:
        train_sampler = DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def build_model(args, device):
    assert (
        args.image_size % 8 == 0
    ), "image_size 必须能被 8 整除，以适配 CROMA 的 patch_size=8"
    num_patches = (args.image_size // 8) ** 2

    # 构建与预训练阶段相同配置的 CROMA
    croma = CROMA(
        patch_size=8,
        encoder_dim=args.encoder_dim,
        encoder_layers=args.encoder_layers,
        attention_heads=args.attention_heads,
        decoder_dim=args.decoder_dim,
        decoder_layers=args.decoder_layers,
        total_channels=OPT_CHANNELS + SAR_CHANNELS,
        num_patches=num_patches,
        opt_channels=OPT_CHANNELS,
        radar_channels=SAR_CHANNELS,
    )

    # 加载预训练权重
    ckpt = torch.load(args.pretrained_ckpt, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = croma.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Warning] Missing keys when loading CROMA: {missing}")
    if unexpected:
        print(f"[Warning] Unexpected keys when loading CROMA: {unexpected}")

    croma.to(device)

    seg_model = CROMASegmentation(croma_model=croma, num_classes=args.num_classes)
    seg_model.to(device)

    return seg_model, num_patches


def train_one_epoch(model, train_loader, optimizer, device, args, epoch, rank, world_size, start_time):
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    num_pixels = 0

    for step, (optical, sar, labels) in enumerate(train_loader):
        optical = optical.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)  # [B, H, W]

        imgs = torch.cat([optical, sar], dim=1)

        logits = model(imgs)  # [B, num_classes, H, W]
        loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.max_grad_norm and args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        batch_pixels = labels.numel()
        total_loss += loss.item() * batch_pixels
        num_pixels += batch_pixels

        if rank == 0 and (step + 1) % args.log_interval == 0:
            elapsed = time.time() - start_time
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            print(
                f"[{elapsed_str}] Epoch {epoch} | Step {step+1}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / max(1, num_pixels)
    return avg_loss


def evaluate(model, val_loader, device, args, epoch, rank, world_size, start_time):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    num_pixels = 0
    num_classes = args.num_classes
    # 按类别累计交并集，用于计算 mIoU
    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for optical, sar, labels in val_loader:
            optical = optical.to(device, non_blocking=True)
            sar = sar.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            imgs = torch.cat([optical, sar], dim=1)

            logits = model(imgs)
            loss = criterion(logits, labels)

            # 计算本 batch 的 IoU 累计量
            preds = torch.argmax(logits, dim=1)  # [B, H, W]
            for c in range(num_classes):
                pred_c = preds == c
                label_c = labels == c
                intersection[c] += (pred_c & label_c).sum()
                union[c] += (pred_c | label_c).sum()

            batch_pixels = labels.numel()
            total_loss += loss.item() * batch_pixels
            num_pixels += batch_pixels

    avg_loss = total_loss / max(1, num_pixels)

    # 计算每类 IoU 和 mIoU
    iou_per_class = intersection / (union + 1e-6)
    valid = union > 0
    if valid.any():
        miou = iou_per_class[valid].mean().item()
    else:
        miou = 0.0

    if rank == 0:
        elapsed = time.time() - start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        print(f"[{elapsed_str}] [Val] Epoch {epoch} | Loss: {avg_loss:.4f} | mIoU: {miou:.4f}")

    return avg_loss, miou


def save_checkpoint(model, optimizer, args, epoch, rank, run_dir, last_ckpt_path=None):
    """在运行级别目录下保存当前 epoch 的模型，并删除上一个 epoch 的模型。"""
    if rank != 0:
        return last_ckpt_path

    os.makedirs(run_dir, exist_ok=True)

    if last_ckpt_path is not None and os.path.exists(last_ckpt_path):
        try:
            os.remove(last_ckpt_path)
        except OSError:
            pass

    ckpt_path = os.path.join(run_dir, f"seg_checkpoint_epoch_{epoch}.pt")
    model_to_save = model.module if isinstance(model, DDP) else model
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")
    return ckpt_path


def init_distributed():
    """初始化分布式环境（torchrun 启动时）。"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        distributed = True
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        distributed = False

    return rank, world_size, local_rank, distributed


def main():
    args = parse_args()
    rank, world_size, local_rank, distributed = init_distributed()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if rank == 0:
        print(f"Distributed: {distributed}, world_size: {world_size}, device: {device}")

    train_loader, val_loader = create_loaders(args, rank, world_size, distributed)
    if rank == 0:
        print(
            f"Train patches: {len(train_loader.dataset)}, Val patches: {len(val_loader.dataset)}"
        )

    model, _ = build_model(args, device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 本次运行独立目录和指标文件
    if rank == 0:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.output_dir, run_timestamp)
        os.makedirs(run_dir, exist_ok=True)
        metrics_path = os.path.join(run_dir, "epoch_metrics.csv")
    else:
        run_dir = None
        metrics_path = None

    start_time = time.time()
    last_ckpt_path = None

    for epoch in range(1, args.epochs + 1):
        if distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, args, epoch, rank, world_size, start_time
        )

        val_loss, val_miou = evaluate(
            model, val_loader, device, args, epoch, rank, world_size, start_time
        )

        if rank == 0:
            file_exists = os.path.exists(metrics_path)
            with open(metrics_path, "a", newline="") as f:
                writer = csv.writer(f)
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
                    float(val_miou),
                ])

        last_ckpt_path = save_checkpoint(
            model, optimizer, args, epoch, rank, run_dir, last_ckpt_path
        )


if __name__ == "__main__":
    main()
