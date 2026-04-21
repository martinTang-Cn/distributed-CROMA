import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange

from datasets import WHUOptSarPatchDataset, BigEarthNetDataset, Houston2013PatchDataset
from pretrain_croma import CROMA, get_mask, apply_mask_to_alibi


PATCH_SIZE = 8


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize CROMA reconstruction on one optical-radar pair.")
    parser.add_argument("--dataset", type=str, choices=["whu", "bigearthnet", "houston2013"], default="whu")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained checkpoint (.pt)")
    parser.add_argument("--split", type=str, default="val", help="Dataset split")
    parser.add_argument("--idx", type=int, default=-1, help="Sample index. -1 means random")
    parser.add_argument("--image_size", type=int, default=256, help="Patch size used by dataset loader")
    parser.add_argument("--stride_ratio", type=float, default=0.9, help="WHU stride ratio")
    parser.add_argument("--mask_ratio_radar", type=float, default=0.75)
    parser.add_argument("--mask_ratio_optical", type=float, default=0.75)
    parser.add_argument("--rgb_indices", type=str, default="2,1,0", help="Optical RGB channel indices")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="reconstruction_vis.png", help="Output image path")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    return parser.parse_args()


def get_dataset_and_channels(args):
    if args.dataset == "whu":
        split = args.split if args.split in {"train", "val"} else "val"
        ds = WHUOptSarPatchDataset(
            root_dir=args.data_root,
            split=split,
            patch_size=args.image_size,
            stride_ratio=args.stride_ratio,
            num_ratio=1.0,
        )
        opt_ch, radar_ch = 4, 1
    elif args.dataset == "bigearthnet":
        split = args.split if args.split in {"train", "validation", "test"} else "validation"
        ds = BigEarthNetDataset(
            root=args.data_root,
            split=split,
            ratio=1.0,
        )
        opt_ch, radar_ch = 10, 2
    else:
        split = args.split if args.split in {"train", "val"} else "val"
        ds = Houston2013PatchDataset(
            root_dir=args.data_root,
            split=split,
            patch_size=args.image_size,
            stride=args.image_size,
        )
        opt_ch, radar_ch = 10, 1

    return ds, opt_ch, radar_ch


def infer_num_patches(optical_tensor):
    _, h, w = optical_tensor.shape
    if h % PATCH_SIZE != 0 or w % PATCH_SIZE != 0:
        raise ValueError(f"Input spatial size must be divisible by {PATCH_SIZE}, got ({h}, {w})")
    return (h // PATCH_SIZE) * (w // PATCH_SIZE)


def build_model(ckpt_args, opt_ch, radar_ch, num_patches, device):
    encoder_dim = int(ckpt_args.get("encoder_dim", 768))
    encoder_layers = int(ckpt_args.get("encoder_layers", 6))
    attention_heads = int(ckpt_args.get("attention_heads", 16))
    decoder_dim = int(ckpt_args.get("decoder_dim", 512))
    decoder_layers = int(ckpt_args.get("decoder_layers", 1))

    model = CROMA(
        patch_size=PATCH_SIZE,
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
    model.to(device)
    return model


def _strip_ddp_prefix(state_dict):
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = _strip_ddp_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)
    return checkpoint


def reconstruct_once(model, optical, radar, mask_ratio_radar, mask_ratio_optical, device):
    model.eval()
    with torch.no_grad():
        optical = optical.unsqueeze(0).to(device)
        radar = radar.unsqueeze(0).to(device)
        imgs = torch.cat([optical, radar], dim=1)

        bsz = imgs.size(0)
        num_patches = model.num_patches
        radar_mask_info = get_mask(bsz, num_patches, device, mask_ratio_radar)
        optical_mask_info = get_mask(bsz, num_patches, device, mask_ratio_optical)

        radar_masked_attn_bias = apply_mask_to_alibi(
            alibi=model.attn_bias.to(device),
            ids_keep_queries=radar_mask_info["ids_keep"],
            ids_keep_keys=radar_mask_info["ids_keep"],
            batch_size=bsz,
            orig_seq_len=num_patches,
            masked_seq_len=radar_mask_info["len_keep"],
            attention_heads=model.attention_heads,
        )
        optical_masked_attn_bias = apply_mask_to_alibi(
            alibi=model.attn_bias.to(device),
            ids_keep_queries=optical_mask_info["ids_keep"],
            ids_keep_keys=optical_mask_info["ids_keep"],
            batch_size=bsz,
            orig_seq_len=num_patches,
            masked_seq_len=optical_mask_info["len_keep"],
            attention_heads=model.attention_heads,
        )

        radar_encodings = model.radar_encoder(
            imgs=imgs[:, model.opt_channels :, ...],
            attn_bias=radar_masked_attn_bias,
            mask_info=radar_mask_info,
        )
        optical_encodings = model.optical_encoder(
            imgs=imgs[:, : model.opt_channels, ...],
            attn_bias=optical_masked_attn_bias,
            mask_info=optical_mask_info,
        )

        cross_attn_bias = apply_mask_to_alibi(
            alibi=model.attn_bias.to(device),
            ids_keep_queries=radar_mask_info["ids_keep"],
            ids_keep_keys=optical_mask_info["ids_keep"],
            batch_size=bsz,
            orig_seq_len=num_patches,
            masked_seq_len=optical_mask_info["len_keep"],
            attention_heads=model.attention_heads,
        )
        joint_encodings = model.cross_encoder(
            x=radar_encodings,
            context=optical_encodings,
            alibi=cross_attn_bias,
        )

        decoder = model.decoder
        x = decoder.encoder_to_decoder(joint_encodings)
        mask_tokens = decoder.mask_token.repeat(
            bsz,
            radar_mask_info["ids_restore"].shape[1] + 1 - x.shape[1],
            1,
        )
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(
            x,
            dim=1,
            index=radar_mask_info["ids_restore"].unsqueeze(-1).repeat(1, 1, x.shape[2]),
        )

        x = x + decoder.decoder_pos_embed
        x = decoder.linear_output(decoder.decoder(x))

        pred = rearrange(
            x,
            "b (h w) (c i j) -> b c (h i) (w j)",
            c=model.total_channels,
            i=model.patch_size,
            j=model.patch_size,
            h=decoder.h_patches,
            w=decoder.w_patches,
        )

        recon_opt = pred[:, : model.opt_channels, :, :]
        recon_radar = pred[:, model.opt_channels :, :, :]

    return recon_opt[0].cpu(), recon_radar[0].cpu(), optical_mask_info["mask_for_mae"][0].cpu(), radar_mask_info["mask_for_mae"][0].cpu()


def normalize_for_display(arr):
    arr = arr.astype(np.float32)
    p2, p98 = np.percentile(arr, (2, 98))
    if p98 > p2:
        arr = np.clip((arr - p2) / (p98 - p2), 0.0, 1.0)
    else:
        amin, amax = arr.min(), arr.max()
        arr = (arr - amin) / (amax - amin + 1e-6)
    return arr


def optical_to_rgb(opt_tensor, rgb_indices):
    c, h, w = opt_tensor.shape
    idx = [int(x) for x in rgb_indices.split(",")]
    if len(idx) != 3:
        raise ValueError("--rgb_indices must have exactly 3 indices, e.g. 2,1,0")
    if any(i < 0 or i >= c for i in idx):
        raise ValueError(f"rgb index out of range, channels={c}, got {idx}")
    rgb = opt_tensor[idx, :, :].permute(1, 2, 0).numpy()
    return normalize_for_display(rgb)


def radar_to_display(radar_tensor):
    c, _, _ = radar_tensor.shape
    arr = radar_tensor.numpy()
    if c == 1:
        return normalize_for_display(arr[0]), "gray"
    if c >= 2:
        vv = normalize_for_display(arr[0])
        vh = normalize_for_display(arr[1])
        pseudo = np.stack([vv, vh, np.zeros_like(vv)], axis=-1)
        return pseudo, None
    return normalize_for_display(arr[0]), "gray"


def patch_mask_to_image(mask_1d, image_hw, patch_size):
    h, w = image_hw
    hp = h // patch_size
    wp = w // patch_size
    patch_mask = mask_1d.reshape(hp, wp).numpy().astype(np.float32)
    return np.repeat(np.repeat(patch_mask, patch_size, axis=0), patch_size, axis=1)


def blend_visible_original(original, recon, patch_mask_img):
    if original.ndim == 3:
        patch_mask_img = patch_mask_img[None, ...]
    return original * (1.0 - patch_mask_img) + recon * patch_mask_img


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，自动切换到 CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    dataset, opt_ch, radar_ch = get_dataset_and_channels(args)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty.")

    idx = args.idx
    if idx < 0:
        idx = random.randint(0, len(dataset) - 1)
    if idx >= len(dataset):
        raise IndexError(f"idx out of range: {idx} >= {len(dataset)}")

    optical, radar, _ = dataset[idx]
    if optical.shape[0] != opt_ch or radar.shape[0] != radar_ch:
        raise ValueError(
            f"Channel mismatch: expected optical={opt_ch}, radar={radar_ch}; got optical={optical.shape[0]}, radar={radar.shape[0]}"
        )

    num_patches = infer_num_patches(optical)

    # 先用 checkpoint 的 args 构建模型参数；如果不存在则使用默认值
    ckpt_meta = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt_meta.get("args", {}) if isinstance(ckpt_meta, dict) else {}
    model = build_model(ckpt_args, opt_ch, radar_ch, num_patches, device)
    load_checkpoint(model, args.checkpoint, device)

    recon_opt, recon_radar, mask_opt_1d, mask_radar_1d = reconstruct_once(
        model,
        optical,
        radar,
        args.mask_ratio_radar,
        args.mask_ratio_optical,
        device,
    )

    h, w = optical.shape[1], optical.shape[2]
    mask_opt_img = patch_mask_to_image(mask_opt_1d, (h, w), PATCH_SIZE)
    mask_radar_img = patch_mask_to_image(mask_radar_1d, (h, w), PATCH_SIZE)

    optical_np = optical.numpy()
    recon_opt_np = recon_opt.numpy()
    optical_hybrid = blend_visible_original(optical_np, recon_opt_np, mask_opt_img)

    radar_np = radar.numpy()
    recon_radar_np = recon_radar.numpy()
    radar_hybrid = blend_visible_original(radar_np, recon_radar_np, mask_radar_img)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    axes[0, 0].imshow(optical_to_rgb(optical, args.rgb_indices))
    axes[0, 0].set_title("Optical Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(optical_to_rgb(recon_opt, args.rgb_indices))
    axes[0, 1].set_title("Optical Reconstruction")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(optical_to_rgb(torch.from_numpy(optical_hybrid), args.rgb_indices))
    axes[0, 2].set_title("Optical Hybrid (visible=original)")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(mask_opt_img, cmap="magma", vmin=0, vmax=1)
    axes[0, 3].set_title("Optical Mask (1=masked)")
    axes[0, 3].axis("off")

    radar_orig_show, radar_orig_cmap = radar_to_display(radar)
    radar_rec_show, radar_rec_cmap = radar_to_display(recon_radar)
    radar_hybrid_show, radar_hybrid_cmap = radar_to_display(torch.from_numpy(radar_hybrid))

    axes[1, 0].imshow(radar_orig_show, cmap=radar_orig_cmap)
    axes[1, 0].set_title("Radar Original")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(radar_rec_show, cmap=radar_rec_cmap)
    axes[1, 1].set_title("Radar Reconstruction")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(radar_hybrid_show, cmap=radar_hybrid_cmap)
    axes[1, 2].set_title("Radar Hybrid (visible=original)")
    axes[1, 2].axis("off")

    axes[1, 3].imshow(mask_radar_img, cmap="magma", vmin=0, vmax=1)
    axes[1, 3].set_title("Radar Mask (1=masked)")
    axes[1, 3].axis("off")

    fig.suptitle(
        f"Dataset={args.dataset}, idx={idx}, mask_ratio_opt={args.mask_ratio_optical}, mask_ratio_radar={args.mask_ratio_radar}",
        fontsize=12,
    )
    plt.tight_layout()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Sample index: {idx}")
    print(f"Saved figure: {args.output}")


if __name__ == "__main__":
    main()
