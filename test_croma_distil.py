"""
Distillation trained model evaluation for WHU/BigEarthNet/Houston2013.

- WHU and BigEarthNet: per-class IoU + mIoU
- Houston2013: per-class accuracy + mean accuracy
"""

import argparse
import csv
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import WHUOptSarPatchDataset, BigEarthNetDataset, Houston2013PatchDataset, CLASS_NAMES
from pretrain_croma import CROMA


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate distillation checkpoints on test splits")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to distillation checkpoint (.pt)")
    parser.add_argument("--dataset", type=str, choices=["whu", "bigearthnet", "houston2013"], required=True)
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root path")
    parser.add_argument("--split", type=str, default=None, help="Dataset split override")
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--vit_patch_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--encoder_dim", type=int, default=None)
    parser.add_argument("--encoder_layers", type=int, default=None)
    parser.add_argument("--attention_heads", type=int, default=None)
    parser.add_argument("--decoder_dim", type=int, default=None)
    parser.add_argument("--decoder_layers", type=int, default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_csv", type=str, default="./", help="Save metrics to CSV path")
    return parser.parse_args()


def _pick(value: Optional[int], ckpt_args: dict, key: str, fallback: int) -> int:
    if value is not None:
        return value
    if ckpt_args is not None and key in ckpt_args:
        return int(ckpt_args[key])
    return int(fallback)


def _select_split(dataset: str, override: Optional[str]) -> str:
    if override:
        return override
    if dataset == "bigearthnet":
        return "test"
    return "val"


def create_loader(args) -> Tuple[DataLoader, Optional[int]]:
    split = _select_split(args.dataset, args.split)
    if args.dataset == "whu":
        ds = WHUOptSarPatchDataset(
            root_dir=args.data_root,
            split=split if split in {"train", "val"} else "val",
            patch_size=args.image_size,
            stride_ratio=0.9,
            num_ratio=1.0,
        )
    elif args.dataset == "bigearthnet":
        ds = BigEarthNetDataset(
            root=args.data_root,
            split=split if split in {"train", "validation", "test"} else "test",
            ratio=1.0,
        )
    else:
        ds = Houston2013PatchDataset(
            root_dir=args.data_root,
            split=split if split in {"train", "val"} else "val",
            patch_size=args.image_size,
            stride=args.image_size,
        )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    inferred_num_patches = None
    try:
        sample = ds[0]
        optical_sample = sample[0]
        _, h, w = optical_sample.shape
        inferred_num_patches = (h // args.vit_patch_size) * (w // args.vit_patch_size)
    except Exception:
        inferred_num_patches = None

    return loader, inferred_num_patches


def build_components(args, device, num_patches: int):
    if args.dataset == "whu":
        opt_ch, radar_ch = 4, 1
    elif args.dataset == "bigearthnet":
        opt_ch, radar_ch = 10, 2
    else:
        opt_ch, radar_ch = 144, 1

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

    from train_croma_whu_distil import OpticalSatelliteClient, RadarSatelliteClient, GroundServer

    optical_client = OpticalSatelliteClient(
        optical_encoder=croma.optical_encoder,
        attn_bias=croma.attn_bias,
        encoder_dim=args.encoder_dim,
        num_patches=num_patches,
        num_classes=args.num_classes,
    ).to(device)

    radar_client = RadarSatelliteClient(
        radar_encoder=croma.radar_encoder,
        attn_bias=croma.attn_bias,
        encoder_dim=args.encoder_dim,
        num_patches=num_patches,
        num_classes=args.num_classes,
    ).to(device)

    ground_server = GroundServer(
        cross_encoder=croma.cross_encoder,
        encoder_dim=args.encoder_dim,
        num_patches=num_patches,
        num_classes=args.num_classes,
    ).to(device)

    return optical_client, radar_client, ground_server


def load_checkpoint(ckpt_path: str, optical_client, radar_client, ground_server, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    optical_client.load_state_dict(ckpt["optical_client_state_dict"], strict=True)
    radar_client.load_state_dict(ckpt["radar_client_state_dict"], strict=True)
    ground_server.load_state_dict(ckpt["ground_server_state_dict"], strict=True)
    return ckpt


@torch.no_grad()
def evaluate(loader, optical_client, radar_client, ground_server, device, num_classes: int):
    conf = torch.zeros((num_classes, num_classes), device=device, dtype=torch.int64)

    optical_client.eval()
    radar_client.eval()
    ground_server.eval()

    for optical, sar, labels in loader:
        optical = optical.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        h, w = optical.shape[-2:]
        attn_bias = optical_client.attn_bias.to(device)

        optical_enc = optical_client(optical)
        radar_enc = radar_client(sar)
        logits = ground_server(
            radar_encodings=radar_enc,
            optical_encodings=optical_enc,
            attn_bias=attn_bias,
            output_size=(h, w),
        )
        preds = torch.argmax(logits, dim=1)

        valid = (labels >= 0) & (labels < num_classes)
        if valid.any():
            y_true = labels[valid].long()
            y_pred = preds[valid].long()
            idx = y_true * num_classes + y_pred
            conf += torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

    return conf


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})

    args.vit_patch_size = _pick(args.vit_patch_size, ckpt_args, "vit_patch_size", 8)
    args.image_size = _pick(args.image_size, ckpt_args, "image_size", 256)
    args.encoder_dim = _pick(args.encoder_dim, ckpt_args, "encoder_dim", 768)
    args.encoder_layers = _pick(args.encoder_layers, ckpt_args, "encoder_layers", 6)
    args.attention_heads = _pick(args.attention_heads, ckpt_args, "attention_heads", 16)
    args.decoder_dim = _pick(args.decoder_dim, ckpt_args, "decoder_dim", 512)
    args.decoder_layers = _pick(args.decoder_layers, ckpt_args, "decoder_layers", 1)

    if args.dataset == "bigearthnet":
        args.num_classes = len(CLASS_NAMES)
    elif args.dataset == "houston2013":
        args.num_classes = 15
    else:
        args.num_classes = _pick(args.num_classes, ckpt_args, "num_classes", 8)

    loader, inferred_num_patches = create_loader(args)
    if inferred_num_patches is None:
        if args.image_size % args.vit_patch_size != 0:
            raise ValueError("image_size must be divisible by vit_patch_size")
        num_patches = (args.image_size // args.vit_patch_size) ** 2
    else:
        num_patches = inferred_num_patches

    optical_client, radar_client, ground_server = build_components(args, device, num_patches)
    load_checkpoint(args.checkpoint, optical_client, radar_client, ground_server, device)

    conf = evaluate(loader, optical_client, radar_client, ground_server, device, args.num_classes)

    diag = torch.diag(conf).float()
    row_sum = conf.sum(dim=1).float()
    col_sum = conf.sum(dim=0).float()

    if args.dataset in {"whu", "bigearthnet"}:
        union = row_sum + col_sum - diag
        iou = diag / (union + 1e-6)
        valid = union > 0
        miou = iou[valid].mean().item() if valid.any() else 0.0

        print("Per-class IoU:")
        for i, v in enumerate(iou.tolist()):
            print(f"  class {i}: {v:.4f}")
        print(f"mIoU: {miou:.4f}")

        if args.output_csv:
            os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
            with open(args.output_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["dataset", "split", "checkpoint", "metric", "class", "value"])
                for i, v in enumerate(iou.tolist()):
                    writer.writerow([args.dataset, _select_split(args.dataset, args.split), args.checkpoint, "iou", i, f"{v:.6f}"])
                writer.writerow([args.dataset, _select_split(args.dataset, args.split), args.checkpoint, "miou", "all", f"{miou:.6f}"])
    else:
        acc = diag / (row_sum + 1e-6)
        valid = row_sum > 0
        mean_acc = acc[valid].mean().item() if valid.any() else 0.0

        print("Per-class accuracy:")
        for i, v in enumerate(acc.tolist()):
            print(f"  class {i}: {v:.4f}")
        print(f"Mean accuracy: {mean_acc:.4f}")

        if args.output_csv:
            os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
            with open(args.output_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["dataset", "split", "checkpoint", "metric", "class", "value"])
                for i, v in enumerate(acc.tolist()):
                    writer.writerow([args.dataset, _select_split(args.dataset, args.split), args.checkpoint, "acc", i, f"{v:.6f}"])
                writer.writerow([args.dataset, _select_split(args.dataset, args.split), args.checkpoint, "mean_acc", "all", f"{mean_acc:.6f}"])


if __name__ == "__main__":
    main()
