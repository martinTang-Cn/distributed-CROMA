"""
Distillation trained model evaluation for WHU/BigEarthNet/Houston2013.

- WHU and BigEarthNet: per-class IoU + mIoU
- Houston2013: per-class accuracy + mean accuracy
"""

import argparse
import csv
import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from datasets import WHUOptSarPatchDataset, BigEarthNetDataset, Houston2013PatchDataset, CLASS_NAMES
from pretrain_croma import CROMA


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate distillation checkpoints on test splits")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to distillation checkpoint (.pt)")
    parser.add_argument("--dataset", type=str, choices=["whu", "bigearthnet", "houston2013"], default="whu")
    parser.add_argument("--data_root", type=str, default=None, help="Dataset root path")
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
    parser.add_argument("--plot_confusion", action="store_true", help="Save confusion matrix image")
    parser.add_argument("--confusion_path", type=str, default=None, help="Path to save confusion matrix image")
    parser.add_argument(
        "--confusion_normalize",
        type=str,
        choices=["none", "row", "col"],
        default="row",
        help="Normalize confusion matrix by row (true), column (pred), or none",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Log progress every N batches (set to 1 for every batch)",
    )
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=1.0,
        help="Use a subset of the dataset (0 < ratio <= 1)",
    )
    parser.add_argument("--eval_seed", type=int, default=42, help="Seed for subset sampling")
    parser.add_argument(
        "--confusion_csv",
        type=str,
        default=None,
        help="Path to confusion matrix CSV; skip evaluation and plot directly",
    )
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

    if args.eval_ratio <= 0 or args.eval_ratio > 1.0:
        raise ValueError("eval_ratio must be in (0, 1].")
    if args.eval_ratio < 1.0:
        total = len(ds)
        target = max(1, int(total * args.eval_ratio))
        g = torch.Generator()
        g.manual_seed(args.eval_seed)
        indices = torch.randperm(total, generator=g)[:target].tolist()
        ds = Subset(ds, indices)

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
def evaluate(loader, optical_client, radar_client, ground_server, device, num_classes: int, log_interval: int):
    conf = torch.zeros((num_classes, num_classes), device=device, dtype=torch.int64)

    optical_client.eval()
    radar_client.eval()
    ground_server.eval()

    try:
        total_batches = len(loader)
    except TypeError:
        total_batches = None

    for batch_idx, (optical, sar, labels) in enumerate(loader, start=1):
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

        if log_interval and (batch_idx % log_interval == 0 or (total_batches and batch_idx == total_batches)):
            if total_batches:
                print(f"Processed {batch_idx}/{total_batches} batches")
            else:
                print(f"Processed {batch_idx} batches")

    return conf


def _resolve_confusion_path(args) -> str:
    if args.confusion_path:
        return args.confusion_path
    split_name = _select_split(args.dataset, args.split)
    if args.checkpoint:
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        return os.path.join(".", f"confusion_{args.dataset}_{split_name}_{ckpt_name}.png")
    if args.confusion_csv:
        csv_base = os.path.splitext(os.path.basename(args.confusion_csv))[0]
        return os.path.join(".", f"confusion_{csv_base}.png")
    return os.path.join(".", f"confusion_{args.dataset}_{split_name}.png")


def _resolve_confusion_csv_path(args) -> str:
    image_path = _resolve_confusion_path(args)
    base, _ = os.path.splitext(image_path)
    return f"{base}.csv"


def _normalize_confusion(conf: torch.Tensor, mode: str) -> torch.Tensor:
    conf_float = conf.float()
    if mode == "none":
        return conf_float
    if mode == "col":
        denom = conf_float.sum(dim=0, keepdim=True)
    else:
        denom = conf_float.sum(dim=1, keepdim=True)
    return conf_float / (denom + 1e-6)


def save_confusion_matrix(
    conf: torch.Tensor,
    args,
    class_names: Optional[list] = None,
    pre_normalized: bool = False,
) -> str:
    conf_cpu = conf.detach().to("cpu")
    conf_norm = conf_cpu.float() if pre_normalized else _normalize_confusion(conf_cpu, args.confusion_normalize)

    num_classes = conf_cpu.shape[0]
    if not class_names or len(class_names) != num_classes:
        class_names = [str(i) for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(conf_norm.numpy(), cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    path = _resolve_confusion_path(args)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_confusion_csv(conf: torch.Tensor, args, class_names: Optional[list] = None) -> str:
    conf_cpu = conf.detach().to("cpu")
    conf_norm = _normalize_confusion(conf_cpu, args.confusion_normalize)

    num_classes = conf_cpu.shape[0]
    if not class_names or len(class_names) != num_classes:
        class_names = [str(i) for i in range(num_classes)]

    path = _resolve_confusion_csv_path(args)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["normalize", args.confusion_normalize])
        writer.writerow(["true\\pred"] + class_names)
        if args.confusion_normalize == "none":
            for i, name in enumerate(class_names):
                row = [name] + [str(int(v)) for v in conf_cpu[i].tolist()]
                writer.writerow(row)
        else:
            for i, name in enumerate(class_names):
                row = [name] + [f"{v:.6f}" for v in conf_norm[i].tolist()]
                writer.writerow(row)
    return path


def load_confusion_csv(path: str) -> Tuple[torch.Tensor, list, str]:
    with open(path, "r", newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        raise ValueError("Confusion CSV is empty.")

    normalize = "none"
    start_idx = 0
    if len(rows[0]) >= 2 and rows[0][0].strip().lower() == "normalize":
        normalize = rows[0][1].strip().lower() or "none"
        start_idx = 1

    if len(rows) <= start_idx:
        raise ValueError("Confusion CSV missing header row.")

    header = rows[start_idx]
    if len(header) < 2:
        raise ValueError("Confusion CSV header is invalid.")
    class_names = header[1:]

    data_rows = rows[start_idx + 1 :]
    if not data_rows:
        raise ValueError("Confusion CSV has no matrix rows.")

    values = []
    for row in data_rows:
        if len(row) < 2:
            continue
        values.append([float(x) for x in row[1:]])

    if not values:
        raise ValueError("Confusion CSV contains no numeric values.")

    conf = torch.tensor(values, dtype=torch.float32)
    return conf, class_names, normalize


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    if args.confusion_csv:
        conf, class_names, normalize = load_confusion_csv(args.confusion_csv)
        args.confusion_normalize = normalize
        out_path = save_confusion_matrix(conf, args, class_names, pre_normalized=True)
        print(f"Confusion matrix saved to: {out_path}")
        return

    if not args.checkpoint:
        raise ValueError("--checkpoint is required unless --confusion_csv is provided.")
    if not args.data_root:
        raise ValueError("--data_root is required unless --confusion_csv is provided.")

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

    conf = evaluate(
        loader,
        optical_client,
        radar_client,
        ground_server,
        device,
        args.num_classes,
        args.log_interval,
    )

    if args.plot_confusion:
        if args.dataset == "bigearthnet":
            class_names = None
        elif args.dataset == "houston2013":
            class_names = None
        else:
            class_names = [str(i) for i in range(args.num_classes)]
        out_path = save_confusion_matrix(conf, args, class_names)
        csv_path = save_confusion_csv(conf, args, class_names)
        print(f"Confusion matrix saved to: {out_path}")
        print(f"Confusion matrix CSV saved to: {csv_path}")

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
