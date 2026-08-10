"""
Compute the parameter size of the CROMA components in MB:
- optical_encoder: ViT (BaseTransformer) with encoder_layers layers.
- radar_encoder: ViT (BaseTransformer) with encoder_layers // 2 layers.
- cross_encoder (joint encoder): BaseTransformerCrossAttn with encoder_layers // 2 layers.
- seg_head: 分割头 (Conv2d 堆叠).
"""

from __future__ import annotations

import argparse
from typing import Tuple

import torch

from pretrain_croma import CROMA


def build_components(
    image_size: int,
    vit_patch_size: int,
    encoder_dim: int,
    encoder_layers: int,
    attention_heads: int,
    decoder_dim: int,
    decoder_layers: int,
    radar_channels: int,
    opt_channels: int = 4,
    num_classes: int = 8,
) -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    """构建 CROMA 并返回 (optical_encoder, radar_encoder, cross_encoder, seg_head)。"""
    assert image_size % vit_patch_size == 0, "image_size 必须能被 vit_patch_size 整除"
    num_patches = (image_size // vit_patch_size) ** 2

    croma = CROMA(
        patch_size=vit_patch_size,
        encoder_dim=encoder_dim,
        encoder_layers=encoder_layers,
        attention_heads=attention_heads,
        decoder_dim=decoder_dim,
        decoder_layers=decoder_layers,
        total_channels=opt_channels + radar_channels,
        num_patches=num_patches,
        opt_channels=opt_channels,
        radar_channels=radar_channels,
    )

    # 分割头：与训练脚本中 seg_head 相同的结构
    seg_head = torch.nn.Sequential(
        torch.nn.Conv2d(encoder_dim, encoder_dim, kernel_size=3, padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(encoder_dim, num_classes, kernel_size=1),
    )

    return croma.optical_encoder, croma.radar_encoder, croma.cross_encoder, seg_head


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def format_size(num_params: int, bytes_per_param: int) -> float:
    return num_params * bytes_per_param / (1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="计算 CROMA optical/radar_encoder、cross_encoder (joint) 与 seg_head 的参数大小 (MB)"
    )
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--vit_patch_size", type=int, default=8)
    parser.add_argument("--encoder_dim", type=int, default=192)
    parser.add_argument("--encoder_layers", type=int, default=6)
    parser.add_argument("--attention_heads", type=int, default=16)
    parser.add_argument("--decoder_dim", type=int, default=512)
    parser.add_argument("--decoder_layers", type=int, default=1)
    parser.add_argument("--radar_channels", type=int, default=1)
    parser.add_argument("--opt_channels", type=int, default=144)
    parser.add_argument("--num_classes", type=int, default=15)
    args = parser.parse_args()

    num_patches = (args.image_size // args.vit_patch_size) ** 2
    radar_layers = args.encoder_layers // 2
    joint_layers = args.encoder_layers // 2

    print("=" * 56)
    print("CROMA 组件参数规模计算")
    print("=" * 56)
    print(f"num_patches           : {num_patches}")
    print(f"encoder_dim           : {args.encoder_dim}")
    print(f"optical_encoder layers: {args.encoder_layers}")
    print(f"radar_encoder layers  : {radar_layers} (= encoder_layers // 2)")
    print(f"cross_encoder layers  : {joint_layers} (= encoder_layers // 2)")
    print(f"attention_heads       : {args.attention_heads}")
    print(f"radar_channels        : {args.radar_channels}")
    print(f"opt_channels          : {args.opt_channels}")
    print(f"num_classes           : {args.num_classes}")
    print("-" * 56)

    optical_encoder, radar_encoder, cross_encoder, seg_head = build_components(
        image_size=args.image_size,
        vit_patch_size=args.vit_patch_size,
        encoder_dim=args.encoder_dim,
        encoder_layers=args.encoder_layers,
        attention_heads=args.attention_heads,
        decoder_dim=args.decoder_dim,
        decoder_layers=args.decoder_layers,
        radar_channels=args.radar_channels,
        opt_channels=args.opt_channels,
        num_classes=args.num_classes,
    )

    def report(name: str, module: torch.nn.Module) -> int:
        n = count_params(module)
        print(
            f"{name:18s}: {n:>12,} params | "
            f"fp32 {format_size(n, 4):>9.2f} MB | fp16 {format_size(n, 2):>9.2f} MB"
        )
        return n

    n_opt = report("optical_encoder", optical_encoder)
    n_radar = report("radar_encoder", radar_encoder)
    n_client = n_opt + n_radar
    print("-" * 56)
    print(
        f"{'client 合计':18s}: {n_client:>12,} params | "
        f"fp32 {format_size(n_client, 4):>9.2f} MB | fp16 {format_size(n_client, 2):>9.2f} MB"
    )

    n_joint = report("cross_encoder", cross_encoder)
    n_seg = report("seg_head", seg_head)
    n_server = n_joint + n_seg
    print("-" * 56)
    print(
        f"{'server 合计':18s}: {n_server:>12,} params | "
        f"fp32 {format_size(n_server, 4):>9.2f} MB | fp16 {format_size(n_server, 2):>9.2f} MB"
    )
    print("=" * 56)


if __name__ == "__main__":
    main()
