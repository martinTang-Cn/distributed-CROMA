import argparse
import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from datasets import Houston2013PatchDataset, CASI_FILE
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
    parser.add_argument("--stride", type=int, default=None, help="Stride for sliding window (default = image_size)")
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


def save_prediction_png(pred: np.ndarray, output_path: str):
    palette = [
        0, 0, 0,        # class 0
        230, 25, 75,    # class 1
        60, 180, 75,    # class 2
        255, 225, 25,   # class 3
        0, 130, 200,    # class 4
        245, 130, 48,   # class 5
        145, 30, 180,   # class 6
        70, 240, 240,   # class 7
        240, 50, 230,   # class 8
        210, 245, 60,   # class 9
        250, 190, 190,  # class 10
        0, 128, 128,    # class 11
        230, 190, 255,  # class 12
        170, 110, 40,   # class 13
        255, 250, 200,  # class 14
    ]
    if len(palette) < 256 * 3:
        palette = palette + [0] * (256 * 3 - len(palette))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if pred.dtype != np.uint8:
        pred = pred.astype(np.uint8)
    img = Image.fromarray(pred, mode="P")
    img.putpalette(palette)
    img.save(output_path)


@torch.no_grad()
def infer_full_map(
    optical_client: OpticalSatelliteClient,
    radar_client: RadarSatelliteClient,
    ground_server: GroundServer,
    attn_bias: torch.Tensor,
    loader: DataLoader,
    full_size: Tuple[int, int],
    num_classes: int,
    device: torch.device,
) -> np.ndarray:
    h, w = full_size
    sum_logits = torch.zeros((num_classes, h, w), dtype=torch.float32)
    count = torch.zeros((h, w), dtype=torch.int32)

    optical_client.eval()
    radar_client.eval()
    ground_server.eval()

    for optical, lidar, _, coords in loader:
        optical = optical.to(device, non_blocking=True)
        lidar = lidar.to(device, non_blocking=True)
        logits = ground_server(
            radar_encodings=radar_client(lidar),
            optical_encodings=optical_client(optical),
            attn_bias=attn_bias.to(device),
            output_size=optical.shape[-2:],
        )

        logits_cpu = logits.cpu()
        if isinstance(coords, dict):
            tops = coords["top"]
            lefts = coords["left"]
            for i in range(len(tops)):
                top = int(tops[i])
                left = int(lefts[i])
                patch_h = logits_cpu.shape[-2]
                patch_w = logits_cpu.shape[-1]
                sum_logits[:, top : top + patch_h, left : left + patch_w] += logits_cpu[i]
                count[top : top + patch_h, left : left + patch_w] += 1
        else:
            for i, c in enumerate(coords):
                top = int(c["top"])
                left = int(c["left"])
                patch_h = logits_cpu.shape[-2]
                patch_w = logits_cpu.shape[-1]
                sum_logits[:, top : top + patch_h, left : left + patch_w] += logits_cpu[i]
                count[top : top + patch_h, left : left + patch_w] += 1

    count = count.clamp_min(1)
    avg_logits = sum_logits / count.unsqueeze(0)
    pred = torch.argmax(avg_logits, dim=0).to(torch.int16).numpy()
    return pred


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    checkpoint, ckpt_args, optical_state, radar_state, server_state = load_checkpoint(args.checkpoint, device)

    image_size = _pick_int(args.image_size, ckpt_args, "image_size", 256)
    stride = args.stride if args.stride is not None else image_size

    dataset = Houston2013PatchDataset(
        root_dir=args.data_root,
        split=args.split,
        patch_size=image_size,
        stride=stride,
        drop_empty=False,
        return_coords=True,
        normalize=True,
        norm_type="standard",
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    sample = dataset[0][0]
    _, patch_h, patch_w = sample.shape
    num_patches = (patch_h // _pick_int(args.vit_patch_size, ckpt_args, "vit_patch_size", 8)) * (
        patch_w // _pick_int(args.vit_patch_size, ckpt_args, "vit_patch_size", 8)
    )

    optical_client, radar_client, ground_server, attn_bias, num_classes = build_models(
        args, ckpt_args, num_patches, device
    )

    optical_client.load_state_dict(optical_state, strict=True)
    radar_client.load_state_dict(radar_state, strict=True)
    ground_server.load_state_dict(server_state, strict=True)

    full_h, full_w = dataset.label.shape
    pred = infer_full_map(
        optical_client,
        radar_client,
        ground_server,
        attn_bias,
        loader,
        (full_h, full_w),
        num_classes,
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
        save_prediction_png(pred_png, args.output_png)

    if args.output_npy:
        print(f"Saved prediction (npy): {args.output_npy}")
    if args.output_png:
        print(f"Saved prediction (png): {args.output_png}")


if __name__ == "__main__":
    main()
