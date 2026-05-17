import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import pandas as pd
import rasterio
from PIL import Image, ImageDraw, ImageFont

from datasets import CASI_FILE, LIDAR_FILE, Houston2013PatchDataset
from pretrain_croma import CROMA
from train_croma_whu_distil import OpticalSatelliteClient, RadarSatelliteClient, GroundServer


def _strip_ddp_prefix(state_dict):
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _pick_int(value: Optional[int], ckpt_args: Optional[dict], key: str, fallback: int) -> int:
    if value is not None:
        return int(value)
    if ckpt_args is not None and key in ckpt_args:
        return int(ckpt_args[key])
    return int(fallback)


def parse_args():
    parser = argparse.ArgumentParser(description="Infer Houston2013 full image with distillation checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Distillation checkpoint path (.pt)")
    parser.add_argument("--data_root", type=str, required=True, help="Houston2013 root directory")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="Label split to infer")
    parser.add_argument("--image_size", type=int, default=None, help="Patch size used in training")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride for sliding window (default = image_size//2)",
    )
    parser.add_argument("--vit_patch_size", type=int, default=None, help="ViT patch size used in training")
    parser.add_argument("--encoder_dim", type=int, default=None)
    parser.add_argument("--encoder_layers", type=int, default=None)
    parser.add_argument("--attention_heads", type=int, default=None)
    parser.add_argument("--decoder_dim", type=int, default=None)
    parser.add_argument("--decoder_layers", type=int, default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_npy", type=str, default="")
    parser.add_argument("--output_png", type=str, default="")
    parser.add_argument(
        "--legend",
        dest="legend",
        action="store_true",
        help="Add color legend to saved PNG output.",
    )
    parser.add_argument(
        "--no-legend",
        dest="legend",
        action="store_false",
        help="Save PNG output without color legend.",
    )
    parser.set_defaults(legend=True)
    parser.add_argument(
        "--stats_dir",
        type=str,
        default=os.path.join(".", "Statistical_data", "Houston2013"),
        help="Directory containing hsi_stats.csv and lidar_stats.csv",
    )
    parser.add_argument(
        "--save_original_label_space",
        action="store_true",
        help="Add 1 to predictions to match original label space (1..15).",
    )
    return parser.parse_args()


def build_models(args, ckpt_args: Optional[dict], num_patches: int, device: torch.device):
    vit_patch_size = _pick_int(args.vit_patch_size, ckpt_args, "vit_patch_size", 8)
    encoder_dim = _pick_int(args.encoder_dim, ckpt_args, "encoder_dim", 768)
    encoder_layers = _pick_int(args.encoder_layers, ckpt_args, "encoder_layers", 6)
    attention_heads = _pick_int(args.attention_heads, ckpt_args, "attention_heads", 16)
    decoder_dim = _pick_int(args.decoder_dim, ckpt_args, "decoder_dim", 512)
    decoder_layers = _pick_int(args.decoder_layers, ckpt_args, "decoder_layers", 1)
    num_classes = _pick_int(args.num_classes, ckpt_args, "num_classes", 15)

    print("num_classes", num_classes)

    croma = CROMA(
        patch_size=vit_patch_size,
        encoder_dim=encoder_dim,
        encoder_layers=encoder_layers,
        attention_heads=attention_heads,
        decoder_dim=decoder_dim,
        decoder_layers=decoder_layers,
        total_channels=145,
        num_patches=num_patches,
        opt_channels=144,
        radar_channels=1,
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

    return optical_client, radar_client, ground_server, attn_bias, num_classes


def load_checkpoint(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_args = checkpoint.get("args", None)

    optical_state = _strip_ddp_prefix(checkpoint.get("optical_client_state_dict", {}))
    radar_state = _strip_ddp_prefix(checkpoint.get("radar_client_state_dict", {}))
    server_state = _strip_ddp_prefix(checkpoint.get("ground_server_state_dict", {}))

    return checkpoint, ckpt_args, optical_state, radar_state, server_state


def load_stats(stats_dir: str, filename: str) -> Tuple[np.ndarray, np.ndarray]:
    csv_path = os.path.join(stats_dir, filename)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Stats file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "mean" not in df.columns or "std" not in df.columns:
        raise ValueError(f"Stats CSV missing mean/std columns: {csv_path}")
    mean = df["mean"].to_numpy(dtype=np.float32)
    std = df["std"].to_numpy(dtype=np.float32)
    std[std == 0] = 1.0
    return mean, std


def standardize_array(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"Expected (C,H,W) array, got {arr.shape}.")
    mean = mean.reshape(-1, 1, 1)
    std = std.reshape(-1, 1, 1)
    return (arr - mean) / std


def load_full_houston(data_root: str, stats_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    with rasterio.open(os.path.join(data_root, CASI_FILE)) as ds:
        hsi = ds.read().astype(np.float32)
    with rasterio.open(os.path.join(data_root, LIDAR_FILE)) as ds:
        lidar = ds.read().astype(np.float32)

    hsi_mean, hsi_std = load_stats(stats_dir, "hsi_stats.csv")
    lidar_mean, lidar_std = load_stats(stats_dir, "lidar_stats.csv")

    hsi = standardize_array(hsi, hsi_mean, hsi_std)
    lidar = standardize_array(lidar, lidar_mean, lidar_std)

    return hsi, lidar


def build_patch_indices(h: int, w: int, patch_size: int, stride: int) -> List[Tuple[int, int]]:
    def build_starts(length: int) -> List[int]:
        if length <= patch_size:
            return [0]
        starts = list(range(0, length - patch_size + 1, stride))
        last = length - patch_size
        if starts[-1] != last:
            starts.append(last)
        return starts

    top_starts = build_starts(h)
    left_starts = build_starts(w)
    return [(top, left) for top in top_starts for left in left_starts]


def save_prediction_png(pred: np.ndarray, output_path: str, include_legend: bool = True):
    palette = [
        
        31, 119, 180,
        255, 127, 14,
        44, 160, 44,
        214, 39, 40,
        148, 103, 189,
        140, 86, 75,
        227, 119, 194,
        127, 127, 127,
        188, 189, 34,
        23, 190, 207,
        174, 199, 232,
        255, 187, 120,
        152, 223, 138,
        255, 152, 150,
        197, 176, 213,
    ]
    labels = [
        "healthy crass",
        "stressed crass",
        "synthetic crass",
        "tree",
        "soil",
        "water",
        "residential",
        "commercial",
        "road",
        "highway",
        "railway",
        "parking-lot1",
        "parking-lot2",
        "tennis court",
        "running track",
    ]
    if len(palette) < 256 * 3:
        palette = palette + [0] * (256 * 3 - len(palette))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if pred.dtype != np.uint8:
        pred = pred.astype(np.uint8)
    img = Image.fromarray(pred, mode="P")
    img.putpalette(palette)
    if not include_legend:
        img.save(output_path)
        return

    rgb = img.convert("RGB")

    swatch_size = 16
    pad = 6
    text_pad = 6
    legend_item_h = swatch_size + pad
    legend_width = 220
    legend_height = max(rgb.height, legend_item_h * len(labels) + pad)
    canvas = Image.new("RGB", (rgb.width + legend_width, legend_height), (255, 255, 255))
    canvas.paste(rgb, (0, 0))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for i, label in enumerate(labels):
        color = tuple(palette[i * 3 : i * 3 + 3])
        y = pad + i * legend_item_h
        x = rgb.width + pad
        draw.rectangle([x, y, x + swatch_size, y + swatch_size], fill=color, outline=(0, 0, 0))
        draw.text((x + swatch_size + text_pad, y), label, fill=(0, 0, 0), font=font)

    canvas.save(output_path)


def evaluate_val_accuracy(
    pred: np.ndarray,
    data_root: str,
    patch_size: int,
    num_classes: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    dataset = Houston2013PatchDataset(
        root_dir=data_root,
        split="val",
        patch_size=patch_size,
        stride=patch_size,
        drop_empty=False,
        return_coords=True,
        normalize=False,
        norm_type="standard",
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    correct = torch.zeros(num_classes, dtype=torch.int64)
    total = torch.zeros(num_classes, dtype=torch.int64)
    pred_tensor = torch.from_numpy(pred)

    for _, _, labels, coords in loader:
        if isinstance(coords, dict):
            tops = coords["top"]
            lefts = coords["left"]
        else:
            tops = [c["top"] for c in coords]
            lefts = [c["left"] for c in coords]

        for i in range(len(tops)):
            top = int(tops[i])
            left = int(lefts[i])
            label_patch = labels[i]
            pred_patch = pred_tensor[top : top + patch_size, left : left + patch_size]

            valid = (label_patch >= 0) & (label_patch < num_classes)
            if not valid.any():
                continue

            label_valid = label_patch[valid].view(-1).long()
            pred_valid = pred_patch[valid].view(-1).long()

            total += torch.bincount(label_valid, minlength=num_classes)
            correct += torch.bincount(
                label_valid[pred_valid == label_valid], minlength=num_classes
            )

    return correct.numpy(), total.numpy()


@torch.no_grad()
def infer_full_map(
    optical_client: OpticalSatelliteClient,
    radar_client: RadarSatelliteClient,
    ground_server: GroundServer,
    attn_bias: torch.Tensor,
    hsi: np.ndarray,
    lidar: np.ndarray,
    patch_indices: List[Tuple[int, int]],
    patch_size: int,
    num_classes: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    h, w = hsi.shape[1:]
    vote_counts = torch.zeros((num_classes, h, w), dtype=torch.int32)

    optical_client.eval()
    radar_client.eval()
    ground_server.eval()

    num_patches = len(patch_indices)
    for start in range(0, num_patches, batch_size):
        batch_indices = patch_indices[start : start + batch_size]
        optical_batch = []
        lidar_batch = []
        for top, left in batch_indices:
            hsi_patch = hsi[:, top : top + patch_size, left : left + patch_size]
            lidar_patch = lidar[:, top : top + patch_size, left : left + patch_size]
            optical_batch.append(torch.from_numpy(hsi_patch))
            lidar_batch.append(torch.from_numpy(lidar_patch))

        optical_tensor = torch.stack(optical_batch).to(device, non_blocking=True)
        lidar_tensor = torch.stack(lidar_batch).to(device, non_blocking=True)

        logits = ground_server(
            radar_encodings=radar_client(lidar_tensor),
            optical_encodings=optical_client(optical_tensor),
            attn_bias=attn_bias.to(device),
            output_size=(patch_size, patch_size),
        )

        logits_cpu = logits.cpu()
        pred_batch = torch.argmax(logits_cpu, dim=1)
        for i, (top, left) in enumerate(batch_indices):
            patch_pred = pred_batch[i]
            for cls in range(num_classes):
                mask = patch_pred == cls
                if mask.any():
                    vote_counts[cls, top : top + patch_size, left : left + patch_size] += mask.to(torch.int32)

    pred = torch.argmax(vote_counts, dim=0).to(torch.int16).numpy()
    return pred


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    checkpoint, ckpt_args, optical_state, radar_state, server_state = load_checkpoint(args.checkpoint, device)

    image_size = _pick_int(args.image_size, ckpt_args, "image_size", 256)
    stride = args.stride if args.stride is not None else max(1, image_size // 2)

    hsi, lidar = load_full_houston(args.data_root, args.stats_dir)
    patch_indices = build_patch_indices(hsi.shape[1], hsi.shape[2], image_size, stride)

    vit_patch = _pick_int(args.vit_patch_size, ckpt_args, "vit_patch_size", 8)
    num_patches = (image_size // vit_patch) * (image_size // vit_patch)

    optical_client, radar_client, ground_server, attn_bias, num_classes = build_models(
        args, ckpt_args, num_patches, device
    )

    optical_client.load_state_dict(optical_state, strict=True)
    radar_client.load_state_dict(radar_state, strict=True)
    ground_server.load_state_dict(server_state, strict=True)

    pred = infer_full_map(
        optical_client,
        radar_client,
        ground_server,
        attn_bias,
        hsi,
        lidar,
        patch_indices,
        image_size,
        num_classes,
        args.batch_size,
        device,
    )

    if args.save_original_label_space:
        pred = (pred + 1).astype(np.int16)

    if args.output_npy:
        os.makedirs(os.path.dirname(args.output_npy) or ".", exist_ok=True)
        np.save(args.output_npy, pred)

    if args.output_png:
        if pred.max() > 255:
            pred_png = pred.astype(np.uint16)
        else:
            pred_png = pred.astype(np.uint8)
        save_prediction_png(pred_png, args.output_png, include_legend=args.legend)

    correct, total = evaluate_val_accuracy(
        pred,
        args.data_root,
        image_size,
        num_classes,
        args.batch_size,
        args.num_workers,
    )
    print("Per-class accuracy (val):")
    for cls in range(num_classes):
        if total[cls] == 0:
            acc = 0.0
        else:
            acc = float(correct[cls]) / float(total[cls])
        print(f"  class {cls}: {acc:.4f} ({int(correct[cls])}/{int(total[cls])})")

    if args.output_npy:
        print(f"Saved prediction (npy): {args.output_npy}")
    if args.output_png:
        print(f"Saved prediction (png): {args.output_png}")


if __name__ == "__main__":
    main()
