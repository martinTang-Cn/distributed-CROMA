import os
import argparse
from datetime import datetime
import time
import csv

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets import WHUOptSarPatchDataset, BigEarthNetDataset, Houston2013PatchDataset
from pretrain_croma import CROMA, get_mask

OPT_CHANNELS = 4  # 光谱通道数 
SAR_CHANNELS = 1    # SAR通道数 


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain CROMA on WHU Opt-SAR patches (no labels)")
    parser.add_argument("--data_root", type=str, default="/home/featurize/data/Houston2013",
                        help="WHU 光学-SAR 数据根目录，包含 optical/sar/lbl 子目录")
    parser.add_argument("--image_size", type=int, default=256,
                        help="WHU数据集中裁剪出来的图像尺寸，等于 WHUOptSarPatchDataset.patch_size")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--mask_ratio_radar", type=float, default=0.75,
                        help="雷达分支 MAE 掩码比例")
    parser.add_argument("--mask_ratio_optical", type=float, default=0.75,
                        help="光学分支 MAE 掩码比例")
    parser.add_argument("--encoder_dim", type=int, default=768)
    parser.add_argument("--encoder_layers", type=int, default=6)
    parser.add_argument("--attention_heads", type=int, default=16)
    parser.add_argument("--decoder_dim", type=int, default=512)
    parser.add_argument("--decoder_layers", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="../CROMA_checkpoint/croma_whu_pretrain_checkpoints")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--num_ratio", type=float, default=1.0,
                        help="使用多少比例的训练集（<1 代表子集，>1 代表有放回扩增）")
    parser.add_argument("--stride_ratio", type=float, default=0.9,
                        help="滑动窗口步长与 image_size 的比例")
    parser.add_argument("--max_grad_norm", type=float, default=0.0,
                        help=">0 时启用梯度裁剪")
    parser.add_argument("--dataset", type=str, choices=["whu", "bigearthnet", "houston2013"], default="whu",
                        help="选择使用的数据集：'whu' 使用 WHUOptSarPatchDataset，'bigearthnet' 使用 BigEarthNetDataset，'houston2013' 使用 Houston2013PatchDataset")
    parser.add_argument("--resume_checkpoint", type=str, default="",
                        help="可选：提供未完成训练的 checkpoint 路径，从该 epoch+1 继续训练")
    return parser.parse_args()


def create_loaders(args, rank, world_size, distributed):
    # 支持 WHU、BigEarthNet 和 Houston2013 三种数据集
    if args.dataset == "whu":
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

    # 尝试从数据集中推断实际的 patch 数，以避免与 args.image_size 不一致
    inferred_num_patches = None
    try:
        sample = train_set[0]
        # sample 返回 (optical, sar, ...) 其中 optical shape = (C, H, W)
        optical_sample = sample[0]
        _, H, W = optical_sample.shape
        inferred_num_patches = (H // 8) * (W // 8)
    except Exception:
        inferred_num_patches = None

    return train_loader, val_loader, inferred_num_patches


def build_model(args, device, inferred_num_patches=None):
    # Prefer inferred_num_patches (from actual dataset samples) to avoid mismatch
    if inferred_num_patches is None:
        assert args.image_size % 8 == 0, "image_size 必须能被 8 整除，以适配 CROMA 的 patch_size=8"
        num_patches = (args.image_size // 8) ** 2
    else:
        num_patches = inferred_num_patches
    # 根据所选数据集设置光学与雷达通道数
    if args.dataset == "whu":
        opt_ch = 4
        radar_ch = 1
    elif args.dataset == "bigearthnet":
        opt_ch = 10
        radar_ch = 2
    else:  # houston2013
        opt_ch = 10
        radar_ch = 1

    model = CROMA(
        patch_size=8,
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
    model.to(device)
    return model, num_patches


def train_one_epoch(model, train_loader, optimizer, device, num_patches, args, epoch, rank, world_size, start_time):
    model.train()
    total_loss = 0.0
    total_contrast = 0.0
    total_mae = 0.0
    num_steps = 0

    for step, (optical, sar, _) in enumerate(train_loader):
        optical = optical.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)

        imgs = torch.cat([optical, sar], dim=1)

        bsz = imgs.size(0)
        radar_mask_info = get_mask(
            bsz,
            seq_len=num_patches,
            device=device,
            mask_ratio=args.mask_ratio_radar,
        )
        optical_mask_info = get_mask(
            bsz,
            seq_len=num_patches,
            device=device,
            mask_ratio=args.mask_ratio_optical,
        )

        contrast_loss, mae_loss = model(
            imgs=imgs,
            radar_mask_info=radar_mask_info,
            optical_mask_info=optical_mask_info,
            rank=rank,
            world_size=world_size,
        )
        loss = contrast_loss + mae_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.max_grad_norm and args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * bsz
        total_contrast += contrast_loss.item() * bsz
        total_mae += mae_loss.item() * bsz
        num_steps += bsz

        if rank == 0 and (step + 1) % args.log_interval == 0:
            elapsed = time.time() - start_time
            elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))
            print(
                f"[{elapsed_str}] Epoch {epoch} | Step {step+1}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f} | Contrast: {contrast_loss.item():.4f} | MAE: {mae_loss.item():.4f}"
            )

    avg_loss = total_loss / max(1, num_steps)
    avg_contrast = total_contrast / max(1, num_steps)
    avg_mae = total_mae / max(1, num_steps)
    return avg_loss, avg_contrast, avg_mae


def evaluate(model, val_loader, device, num_patches, args, epoch, rank, world_size, start_time):
    model.eval()
    total_loss = 0.0
    total_contrast = 0.0
    total_mae = 0.0
    num_steps = 0

    with torch.no_grad():
        for optical, sar, _ in val_loader:
            optical = optical.to(device, non_blocking=True)
            sar = sar.to(device, non_blocking=True)
            imgs = torch.cat([optical, sar], dim=1)

            bsz = imgs.size(0)
            radar_mask_info = get_mask(
                bsz,
                seq_len=num_patches,
                device=device,
                mask_ratio=args.mask_ratio_radar,
            )
            optical_mask_info = get_mask(
                bsz,
                seq_len=num_patches,
                device=device,
                mask_ratio=args.mask_ratio_optical,
            )

            contrast_loss, mae_loss = model(
                imgs=imgs,
                radar_mask_info=radar_mask_info,
                optical_mask_info=optical_mask_info,
                rank=rank,
                world_size=world_size,
            )
            loss = contrast_loss + mae_loss

            total_loss += loss.item() * bsz
            total_contrast += contrast_loss.item() * bsz
            total_mae += mae_loss.item() * bsz
            num_steps += bsz

    avg_loss = total_loss / max(1, num_steps)
    avg_contrast = total_contrast / max(1, num_steps)
    avg_mae = total_mae / max(1, num_steps)

    if rank == 0:
        elapsed = time.time() - start_time
        elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))
        print(
            f"[{elapsed_str}] [Val] Epoch {epoch} | Loss: {avg_loss:.4f} | "
            f"Contrast: {avg_contrast:.4f} | MAE: {avg_mae:.4f}"
        )
    return avg_loss, avg_contrast, avg_mae



def save_checkpoint(model, optimizer, args, epoch, rank, run_dir, last_ckpt_path=None):
    """在运行级别目录下保存当前 epoch 的模型，并删除上一个 epoch 的模型。

    仅在 rank==0 上实际执行文件写入，其他 rank 直接返回 last_ckpt_path。
    """
    if rank != 0:
        return last_ckpt_path

    os.makedirs(run_dir, exist_ok=True)

    # 删除上一 epoch 的 checkpoint，保证目录内始终只有最新模型
    if last_ckpt_path is not None and os.path.exists(last_ckpt_path):
        try:
            os.remove(last_ckpt_path)
        except OSError:
            pass

    ckpt_path = os.path.join(run_dir, f"checkpoint_epoch_{epoch}.pt")
    # 兼容 DDP 与非 DDP
    model_to_save = model.module if isinstance(model, DDP) else model
    torch.save({
        "epoch": epoch,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")
    return ckpt_path



def init_distributed():
    """初始化分布式环境（torchrun 启动时）。

    返回: rank, world_size, local_rank, distributed(bool)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        distributed = True
    else:
        # 单进程单卡/CPU 回退
        rank = 0
        world_size = 1
        local_rank = 0
        distributed = False

    return rank, world_size, local_rank, distributed


def load_checkpoint_if_needed(model, optimizer, args, device, rank):
    """按需加载 checkpoint，返回 start_epoch 和 checkpoint 路径。"""
    if not args.resume_checkpoint:
        return 1, None

    ckpt_path = args.resume_checkpoint
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"resume checkpoint 不存在: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    last_epoch = int(checkpoint.get("epoch", 0))
    start_epoch = last_epoch + 1

    if rank == 0:
        print(f"Resumed from checkpoint: {ckpt_path}")
        print(f"Resume start epoch: {start_epoch}")

    return start_epoch, ckpt_path


def main():
    args = parse_args()
    rank, world_size, local_rank, distributed = init_distributed()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if rank == 0:
        print(f"Distributed: {distributed}, world_size: {world_size}, device: {device}")

    train_loader, val_loader, inferred_num_patches = create_loaders(args, rank, world_size, distributed)
    if rank == 0:
        print(f"Train patches: {len(train_loader.dataset)}, Val patches: {len(val_loader.dataset)}")
        if inferred_num_patches is not None:
            print(f"Inferred num_patches from data: {inferred_num_patches}")

    model, num_patches = build_model(args, device, inferred_num_patches=inferred_num_patches)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch, resumed_ckpt_path = load_checkpoint_if_needed(model, optimizer, args, device, rank)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 为本次运行创建独立的时间戳目录及指标文件路径（仅 rank 0 实际创建）
    if rank == 0:
        if resumed_ckpt_path:
            # 继续训练时沿用原 checkpoint 所在目录，保持日志与模型连续
            run_dir = os.path.dirname(resumed_ckpt_path)
        else:
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.dataset}"
            run_dir = os.path.join(args.output_dir, run_timestamp)
        os.makedirs(run_dir, exist_ok=True)
        metrics_path = os.path.join(run_dir, "epoch_metrics.csv")
    else:
        run_dir = None
        metrics_path = None

    # 记录运行开始时间（用于日志中的已运行总时长）
    start_time = time.time()

    last_ckpt_path = resumed_ckpt_path

    if start_epoch > args.epochs:
        if rank == 0:
            print(
                f"Checkpoint epoch 已达到 {start_epoch - 1}，不小于目标 epochs={args.epochs}，无需继续训练。"
            )
        if distributed:
            dist.destroy_process_group()
        return

    for epoch in range(start_epoch, args.epochs + 1):
        # 分布式时需要设置 epoch，以确保各 rank 的 shuffle 一致
        if distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        train_loss, train_contrast, train_mae = train_one_epoch(
            model, train_loader, optimizer, device, num_patches, args, epoch, rank, world_size, start_time
        )
        if rank == 0:
            elapsed = time.time() - start_time
            elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))
            print(
                f"[{elapsed_str}] [Train] Epoch {epoch} | Loss: {train_loss:.4f} | "
                f"Contrast: {train_contrast:.4f} | MAE: {train_mae:.4f}"
            )

        val_loss, val_contrast, val_mae = evaluate(
            model, val_loader, device, num_patches, args, epoch, rank, world_size, start_time
        )

        # 记录每个 epoch 的损失到 CSV（仅 rank 0）
        if rank == 0:
            file_exists = os.path.exists(metrics_path)
            with open(metrics_path, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        "epoch",
                        "train_loss",
                        "train_contrast_loss",
                        "train_mae_loss",
                        "val_loss",
                        "val_contrast_loss",
                        "val_mae_loss",
                    ])
                writer.writerow([
                    epoch,
                    float(train_loss),
                    float(train_contrast),
                    float(train_mae),
                    float(val_loss),
                    float(val_contrast),
                    float(val_mae),
                ])

        # 保存当前 epoch 模型，并删除上一个 epoch 的模型（仅 rank 0 实际执行）
        last_ckpt_path = save_checkpoint(model, optimizer, args, epoch, rank, run_dir, last_ckpt_path)


if __name__ == "__main__":
    main()
