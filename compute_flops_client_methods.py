"""
Compute the FLOPs of the client-side single-sample forward pass for three training methods:

  方法 1: Fed Split Learning           (train_croma_whu_fed_split_learning.py)
          客户端 = optical_encoder + radar_encoder（连接阶段只编码，发激活给服务器）
  方法 2: Fed Split Learning + Distil  (train_croma_whu_fed_split_distil.py)
          连接阶段客户端前向 = optical_encoder + radar_encoder
          本地训练客户端前向 = optical_encoder->proj_OM->seg_head
                              + radar_encoder->proj_RM->seg_head
  方法 3: Fed Learning / FedAvg        (train_croma_whu_fed_learning.py)
          客户端 = 完整模型 optical_encoder + radar_encoder + cross_encoder + seg_head

基于 fvcore 的 FlopCountAnalysis，逐个组件统计后按方法汇总。

Requires:
    pip install fvcore
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError as exc:  # pragma: no cover - runtime check
    raise SystemExit(
        "Missing dependency: fvcore. Install with: pip install fvcore"
    ) from exc

from pretrain_croma import CROMA
from train_croma_whu_fed_split_distil import Projector


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    opt_channels: int
    radar_channels: int


DATASETS = (
    DatasetConfig("whu-opt-sar", opt_channels=4, radar_channels=1),
    DatasetConfig("bigearthnet", opt_channels=10, radar_channels=2),
    DatasetConfig("houston2013", opt_channels=144, radar_channels=1),
)


def build_components(
    opt_channels: int,
    radar_channels: int,
    image_size: int,
    vit_patch_size: int,
    encoder_dim: int,
    encoder_layers: int,
    attention_heads: int,
    decoder_dim: int,
    decoder_layers: int,
    num_classes: int,
    device: torch.device,
):
    """构建 CROMA 及各组件，返回计算 FLOPs 所需的模块与输入。"""
    assert image_size % vit_patch_size == 0, "image_size 必须能被 vit_patch_size 整除"
    num_patches = (image_size // vit_patch_size) ** 2
    h_patches = int(num_patches ** 0.5)
    w_patches = int(num_patches ** 0.5)

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
    croma.to(device)

    # 分割头（与训练脚本中 seg_head 相同的结构）
    seg_head = nn.Sequential(
        nn.Conv2d(encoder_dim, encoder_dim, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(encoder_dim, num_classes, kernel_size=1),
    ).to(device)

    # 投影模型（方法 2 本地训练使用）
    proj_OM = Projector(dim=encoder_dim).to(device)
    proj_RM = Projector(dim=encoder_dim).to(device)

    attn_bias = croma.attn_bias.to(device)  # [1, heads, N, N]

    # 单样本输入
    optical_input = torch.zeros(1, opt_channels, image_size, image_size, device=device)
    radar_input = torch.zeros(1, radar_channels, image_size, image_size, device=device)
    optical_enc = torch.zeros(1, num_patches, encoder_dim, device=device)
    radar_enc = torch.zeros(1, num_patches, encoder_dim, device=device)
    feat = torch.zeros(1, encoder_dim, h_patches, w_patches, device=device)

    return {
        "optical_encoder": (croma.optical_encoder, (optical_input, attn_bias)),
        "radar_encoder": (croma.radar_encoder, (radar_input, attn_bias)),
        "cross_encoder": (croma.cross_encoder, (radar_enc, optical_enc, attn_bias)),
        "seg_head": (seg_head, (feat,)),
        "proj_OM": (proj_OM, (optical_enc,)),
        "proj_RM": (proj_RM, (radar_enc,)),
    }


def flops_of(model: nn.Module, inputs: Tuple) -> float:
    model.eval()
    with torch.no_grad():
        return float(FlopCountAnalysis(model, inputs).total())


def report(name: str, flops: float) -> None:
    print(f"{name:44s}: {flops:>14,.0f} FLOPs = {flops / 1e9:>8.3f} GFLOPs = {flops / 1e6:>9.3f} MFLOPs")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="计算三种训练方法在客户端上单样本前向传播的 FLOPs"
    )
    parser.add_argument("--dataset", type=str, choices=[d.name for d in DATASETS],
                        default="houston2013")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--vit_patch_size", type=int, default=8)
    parser.add_argument("--encoder_dim", type=int, default=192)
    parser.add_argument("--encoder_layers", type=int, default=6)
    parser.add_argument("--attention_heads", type=int, default=16)
    parser.add_argument("--decoder_dim", type=int, default=512)
    parser.add_argument("--decoder_layers", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=8)
    args = parser.parse_args()

    ds = next(d for d in DATASETS if d.name == args.dataset)
    device = torch.device("cpu")

    comps = build_components(
        opt_channels=ds.opt_channels,
        radar_channels=ds.radar_channels,
        image_size=args.image_size,
        vit_patch_size=args.vit_patch_size,
        encoder_dim=args.encoder_dim,
        encoder_layers=args.encoder_layers,
        attention_heads=args.attention_heads,
        decoder_dim=args.decoder_dim,
        decoder_layers=args.decoder_layers,
        num_classes=args.num_classes,
        device=device,
    )

    # ===== 逐个组件统计 FLOPs =====
    f: Dict[str, float] = {}
    for name, (model, inputs) in comps.items():
        f[name] = flops_of(model, inputs)

    encoders = f["optical_encoder"] + f["radar_encoder"]
    local_OM = f["optical_encoder"] + f["proj_OM"] + f["seg_head"]
    local_RM = f["radar_encoder"] + f["proj_RM"] + f["seg_head"]
    full_model = encoders + f["cross_encoder"] + f["seg_head"]

    methods: Dict[str, Dict[str, float]] = {
        "方法1 Fed Split Learning": {
            "客户端前向 (encoders)": encoders,
        },
        "方法2 Fed Split + Distil": {
            "连接阶段客户端前向 (encoders)": encoders,
            "本地训练客户端前向 (单样本, 两分支)": local_OM + local_RM,
        },
        "方法3 Fed Learning (FedAvg)": {
            "客户端前向 (完整模型)": full_model,
        },
    }

    # ===== 输出 =====
    print("=" * 78)
    print(f"客户端单样本前向 FLOPs 统计 | dataset={args.dataset} | "
          f"image={args.image_size}x{args.image_size} | patch={args.vit_patch_size} | "
          f"dim={args.encoder_dim} | layers={args.encoder_layers}")
    print("=" * 78)
    print("[组件 FLOPs]")
    report("optical_encoder", f["optical_encoder"])
    report("radar_encoder", f["radar_encoder"])
    report("cross_encoder (joint)", f["cross_encoder"])
    report("seg_head", f["seg_head"])
    report("proj_OM", f["proj_OM"])
    report("proj_RM", f["proj_RM"])
    print("-" * 78)
    print("[各方法客户端前向 FLOPs]")
    for method, items in methods.items():
        print(f"  {method}:")
        for item, flops in items.items():
            report(f"    {item}", flops)
    print("=" * 78)


if __name__ == "__main__":
    main()
