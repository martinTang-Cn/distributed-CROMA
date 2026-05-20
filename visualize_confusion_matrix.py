import argparse
import csv
import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets import (
    WHUOptSarPatchDataset,
    BigEarthNetDataset,
    Houston2013PatchDataset,
    CLASS_NAMES,
)
from pretrain_croma import CROMA
from train_croma_whu_distil import (
    OpticalSatelliteClient,
    RadarSatelliteClient,
    GroundServer,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize confusion matrix for distillation checkpoints."
    )
    parser.add_argument("--dataset", type=str, choices=["whu", "bigearthnet", "houston2013"], default="whu")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Distillation checkpoint path (.pt)")
    parser.add_argument("--split", type=str, default="val", help="Dataset split")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--vit_patch_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_batches", type=int, default=0, help="0 means all batches")
    parser.add_argument("--eval_ratio", type=float, default=1.0, help="Use a subset of val set (0,1]")
    parser.add_argument("--output_dir", type=str, default="confusion_outputs")
    return parser.parse_args()


def _strip_ddp_prefix(state_dict):
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _infer_num_patches(optical_tensor: torch.Tensor, patch_size: int) -> int:
    _, h, w = optical_tensor.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(f"Input spatial size must be divisible by {patch_size}, got ({h}, {w})")
    return (h // patch_size) * (w // patch_size)


def _get_dataset(args) -> Tuple[torch.utils.data.Dataset, int, int, int, List[str], int]:
    if args.dataset == "whu":
        split = args.split if args.split in {"train", "val"} else "val"
        ds = WHUOptSarPatchDataset(
            root_dir=args.data_root,
            split=split,
            patch_size=args.image_size,
            stride_ratio=0.9,
            num_ratio=1.0,
        )
        opt_ch, radar_ch = 4, 1
        num_classes = 8
        class_names = [str(i) for i in range(num_classes)]
        ignore_index = None
    elif args.dataset == "bigearthnet":
        split = args.split if args.split in {"train", "validation", "test"} else "validation"
        ds = BigEarthNetDataset(
            root=args.data_root,
            split=split,
            ratio=1.0,
        )
        opt_ch, radar_ch = 10, 2
        num_classes = len(CLASS_NAMES)
        class_names = CLASS_NAMES
        ignore_index = -1
    else:
        split = args.split if args.split in {"train", "val"} else "val"
        ds = Houston2013PatchDataset(
            root_dir=args.data_root,
            split=split,
            patch_size=args.image_size,
            stride=args.image_size,
        )
        opt_ch, radar_ch = 144, 1
        num_classes = 15
        class_names = [str(i) for i in range(num_classes)]
        ignore_index = -1

    return ds, opt_ch, radar_ch, num_classes, class_names, ignore_index


def _build_components(
    ckpt_args: dict,
    opt_ch: int,
    radar_ch: int,
    num_patches: int,
    num_classes: int,
    device: torch.device,
):
    patch_size = int(ckpt_args.get("vit_patch_size", 8))
    encoder_dim = int(ckpt_args.get("encoder_dim", 768))
    encoder_layers = int(ckpt_args.get("encoder_layers", 6))
    attention_heads = int(ckpt_args.get("attention_heads", 16))
    decoder_dim = int(ckpt_args.get("decoder_dim", 512))
    decoder_layers = int(ckpt_args.get("decoder_layers", 1))

    croma = CROMA(
        patch_size=patch_size,
        encoder_dim=encoder_dim,
        encoder_layers=encoder_layers,
        attention_heads=attention_heads,
        decoder_dim=decoder_dim,
        decoder_layers=decoder_layers,
        total_channels=opt_ch + radar_ch,
        num_patches=num_patches,
        opt_channels=opt_ch,
        radar_channels=radar_ch,
    )

    attn_bias = croma.attn_bias

    optical_client = OpticalSatelliteClient(
        optical_encoder=croma.optical_encoder,
        attn_bias=attn_bias,
        encoder_dim=encoder_dim,
        num_patches=num_patches,
        num_classes=num_classes,
    ).to(device)

    radar_client = RadarSatelliteClient(
        radar_encoder=croma.radar_encoder,
        attn_bias=attn_bias,
        encoder_dim=encoder_dim,
        num_patches=num_patches,
        num_classes=num_classes,
    ).to(device)

    ground_server = GroundServer(
        cross_encoder=croma.cross_encoder,
        encoder_dim=encoder_dim,
        num_patches=num_patches,
        num_classes=num_classes,
    ).to(device)

    return optical_client, radar_client, ground_server, attn_bias


def _load_distil_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "optical_client_state_dict" not in ckpt:
        raise ValueError("Checkpoint does not look like a distillation checkpoint.")

    ckpt["optical_client_state_dict"] = _strip_ddp_prefix(ckpt["optical_client_state_dict"])
    ckpt["radar_client_state_dict"] = _strip_ddp_prefix(ckpt["radar_client_state_dict"])
    ckpt["ground_server_state_dict"] = _strip_ddp_prefix(ckpt["ground_server_state_dict"])
    return ckpt


def _update_confusion(conf: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int):
    if y_true.numel() == 0:
        return
    idx = y_true * num_classes + y_pred
    conf += torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def compute_confusion_matrix(
    optical_client: torch.nn.Module,
    radar_client: torch.nn.Module,
    ground_server: torch.nn.Module,
    attn_bias: torch.Tensor,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
    ignore_index: Optional[int],
    max_batches: int,
) -> torch.Tensor:
    optical_client.eval()
    radar_client.eval()
    ground_server.eval()

    conf = torch.zeros((num_classes, num_classes), device=device, dtype=torch.int64)

    with torch.no_grad():
        for batch_idx, (optical, radar, labels) in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break

            optical = optical.to(device, non_blocking=True)
            radar = radar.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            h, w = labels.shape[-2:]
            optical_enc = optical_client(optical)
            radar_enc = radar_client(radar)

            logits = ground_server(
                radar_encodings=radar_enc,
                optical_encodings=optical_enc,
                attn_bias=attn_bias.to(device),
                output_size=(h, w),
            )
            preds = torch.argmax(logits, dim=1)

            if ignore_index is not None:
                valid = labels != ignore_index
                y_true = labels[valid].long()
                y_pred = preds[valid].long()
            else:
                y_true = labels.long().reshape(-1)
                y_pred = preds.long().reshape(-1)

            in_range = (
                (y_true >= 0) & (y_true < num_classes) &
                (y_pred >= 0) & (y_pred < num_classes)
            )
            y_true = y_true[in_range]
            y_pred = y_pred[in_range]
            _update_confusion(conf, y_true, y_pred, num_classes)

    return conf


def _save_csv(conf: np.ndarray, class_names: List[str], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred"] + class_names)
        for i, row in enumerate(conf):
            writer.writerow([class_names[i]] + row.tolist())


def _save_precision_csv(conf: np.ndarray, class_names: List[str], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    col_sum = conf.sum(axis=0)
    row_sum = conf.sum(axis=1)
    diag = np.diag(conf)

    precision = np.zeros_like(col_sum, dtype=np.float64)
    recall = np.zeros_like(row_sum, dtype=np.float64)

    valid_prec = col_sum > 0
    valid_rec = row_sum > 0

    precision[valid_prec] = diag[valid_prec] / col_sum[valid_prec]
    recall[valid_rec] = diag[valid_rec] / row_sum[valid_rec]

    f1 = np.zeros_like(diag, dtype=np.float64)
    valid_f1 = (precision + recall) > 0
    f1[valid_f1] = 2 * precision[valid_f1] * recall[valid_f1] / (precision[valid_f1] + recall[valid_f1])

    mean_precision = precision[valid_prec].mean() if np.any(valid_prec) else 0.0
    mean_recall = recall[valid_rec].mean() if np.any(valid_rec) else 0.0
    mean_f1 = f1[valid_f1].mean() if np.any(valid_f1) else 0.0

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1"])
        for name, p, r, f1_val in zip(class_names, precision.tolist(), recall.tolist(), f1.tolist()):
            writer.writerow([name, f"{p:.6f}", f"{r:.6f}", f"{f1_val:.6f}"])
        writer.writerow(["mean_macro", f"{mean_precision:.6f}", f"{mean_recall:.6f}", f"{mean_f1:.6f}"])


def _plot_confusion(conf: np.ndarray, class_names: List[str], out_path: str, normalize: bool):
    if normalize:
        row_sum = conf.sum(axis=1, keepdims=True)
        denom = np.where(row_sum == 0, 1, row_sum)
        conf_plot = conf / denom
        title = "Confusion Matrix (Normalized)"
    else:
        conf_plot = conf
        title = "Confusion Matrix (Counts)"

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(conf_plot, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    dataset, opt_ch, radar_ch, default_num_classes, class_names, ignore_index = _get_dataset(args)
    if args.eval_ratio <= 0 or args.eval_ratio > 1.0:
        raise ValueError("eval_ratio must be in (0, 1].")
    if args.eval_ratio < 1.0:
        total = len(dataset)
        target = max(1, int(total * args.eval_ratio))
        rng = np.random.RandomState(42)
        indices = rng.choice(total, size=target, replace=False)
        dataset = Subset(dataset, indices.tolist())
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    ckpt = _load_distil_checkpoint(args.checkpoint, device)
    ckpt_args = ckpt.get("args", {})

    num_classes = int(ckpt_args.get("num_classes", default_num_classes))
    if len(class_names) != num_classes:
        class_names = [str(i) for i in range(num_classes)]

    sample = dataset[0]
    optical_sample = sample[0]
    num_patches = _infer_num_patches(optical_sample, int(ckpt_args.get("vit_patch_size", args.vit_patch_size)))

    optical_client, radar_client, ground_server, attn_bias = _build_components(
        ckpt_args,
        opt_ch,
        radar_ch,
        num_patches,
        num_classes,
        device,
    )

    optical_client.load_state_dict(ckpt["optical_client_state_dict"], strict=False)
    radar_client.load_state_dict(ckpt["radar_client_state_dict"], strict=False)
    ground_server.load_state_dict(ckpt["ground_server_state_dict"], strict=False)

    conf = compute_confusion_matrix(
        optical_client=optical_client,
        radar_client=radar_client,
        ground_server=ground_server,
        attn_bias=attn_bias,
        loader=loader,
        num_classes=num_classes,
        device=device,
        ignore_index=ignore_index,
        max_batches=args.max_batches,
    )

    conf_np = conf.cpu().numpy().astype(np.int64)

    prefix = f"{args.dataset}_{args.split}"
    csv_path = os.path.join(args.output_dir, f"{prefix}_confusion.csv")
    png_counts = os.path.join(args.output_dir, f"{prefix}_confusion_counts.png")
    png_norm = os.path.join(args.output_dir, f"{prefix}_confusion_norm.png")
    precision_csv = os.path.join(args.output_dir, f"{prefix}_precision.csv")

    _save_csv(conf_np, class_names, csv_path)
    _save_precision_csv(conf_np, class_names, precision_csv)
    _plot_confusion(conf_np, class_names, png_counts, normalize=False)
    _plot_confusion(conf_np, class_names, png_norm, normalize=True)

    print(f"Saved confusion matrix CSV: {csv_path}")
    print(f"Saved per-class precision CSV: {precision_csv}")
    print(f"Saved confusion matrix (counts): {png_counts}")
    print(f"Saved confusion matrix (normalized): {png_norm}")


if __name__ == "__main__":
    main()
