"""
Compute FLOPs for optical/radar clients on different datasets.

Requires:
    pip install fvcore
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError as exc:  # pragma: no cover - runtime check
    raise SystemExit(
        "Missing dependency: fvcore. Install with: pip install fvcore"
    ) from exc

from pretrain_croma import CROMA
from train_croma_whu_distil import OpticalSatelliteClient, RadarSatelliteClient


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


def build_clients(
    opt_channels: int,
    radar_channels: int,
    image_size: int = 256,
    vit_patch_size: int = 8,
    encoder_dim: int = 768,
    encoder_layers: int = 6,
    attention_heads: int = 16,
    decoder_dim: int = 512,
    decoder_layers: int = 1,
    num_classes: int = 8,
) -> Tuple[OpticalSatelliteClient, RadarSatelliteClient]:
    if image_size % vit_patch_size != 0:
        raise ValueError("image_size must be divisible by vit_patch_size")
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

    attn_bias = croma.attn_bias

    optical_client = OpticalSatelliteClient(
        optical_encoder=croma.optical_encoder,
        attn_bias=attn_bias,
        encoder_dim=encoder_dim,
        num_patches=num_patches,
        num_classes=num_classes,
    )

    radar_client = RadarSatelliteClient(
        radar_encoder=croma.radar_encoder,
        attn_bias=attn_bias,
        encoder_dim=encoder_dim,
        num_patches=num_patches,
        num_classes=num_classes,
    )

    return optical_client, radar_client


def flops_for_client(model: torch.nn.Module, input_tensor: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        flops = FlopCountAnalysis(model, (input_tensor,)).total()
    return float(flops)


def main() -> None:
    device = torch.device("cpu")
    image_size = 256

    results: Dict[str, Dict[str, float]] = {}

    for ds in DATASETS:
        optical_client, radar_client = build_clients(
            opt_channels=ds.opt_channels,
            radar_channels=ds.radar_channels,
            image_size=image_size,
        )
        optical_client.to(device)
        radar_client.to(device)

        optical_input = torch.zeros(
            1, ds.opt_channels, image_size, image_size, device=device
        )
        radar_input = torch.zeros(
            1, ds.radar_channels, image_size, image_size, device=device
        )

        optical_flops = flops_for_client(optical_client, optical_input)
        radar_flops = flops_for_client(radar_client, radar_input)

        results[ds.name] = {
            "optical_client": optical_flops,
            "radar_client": radar_flops,
        }

    print("FLOPs (single forward, 1x256x256 input):")
    for ds_name, vals in results.items():
        print(
            f"- {ds_name}: optical_client={vals['optical_client']:.3e}, "
            f"radar_client={vals['radar_client']:.3e}"
        )


if __name__ == "__main__":
    main()
