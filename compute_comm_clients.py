"""
Compute communication volume between clients and server for one 256x256 optical+radar pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import argparse
import torch

from pretrain_croma import CROMA
from train_croma_whu_distil import OpticalSatelliteClient, RadarSatelliteClient, GroundServer


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
    image_size: int = 256,
    vit_patch_size: int = 1,
    encoder_dim: int = 192,
    encoder_layers: int = 6,
    attention_heads: int = 16,
    decoder_dim: int = 512,
    decoder_layers: int = 1,
    num_classes: int = 8,
) -> Tuple[OpticalSatelliteClient, RadarSatelliteClient, GroundServer, torch.Tensor]:
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

    ground_server = GroundServer(
        cross_encoder=croma.cross_encoder,
        encoder_dim=encoder_dim,
        num_patches=num_patches,
        num_classes=num_classes,
    )

    return optical_client, radar_client, ground_server, attn_bias


def bytes_of(tensor: Optional[torch.Tensor]) -> int:
    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()


def compute_comm_for_pair(
    optical_client: OpticalSatelliteClient,
    radar_client: RadarSatelliteClient,
    ground_server: GroundServer,
    attn_bias: torch.Tensor,
    optical_input: torch.Tensor,
    radar_input: torch.Tensor,
    enable_distill: bool,
    enable_peer_distill: bool,
) -> Dict[str, int]:
    device = optical_input.device
    H, W = optical_input.shape[-2:]

    with torch.no_grad():
        optical_enc = optical_client(optical_input)
        radar_enc = radar_client(radar_input)

        input_bytes = bytes_of(optical_input) + bytes_of(radar_input)

        optical_student_logits = None
        radar_student_logits = None
        if enable_distill:
            optical_student_logits = optical_client.predict(optical_enc, output_size=(H, W))
            radar_student_logits = radar_client.predict(radar_enc, output_size=(H, W))

        optical_pred_tx = None
        radar_pred_tx = None
        if enable_distill and enable_peer_distill:
            optical_pred_tx = optical_student_logits.detach()
            radar_pred_tx = radar_student_logits.detach()

        optical_act = optical_enc.detach()
        radar_act = radar_enc.detach()

        forward_bytes = bytes_of(optical_act) + bytes_of(radar_act)
        forward_bytes += bytes_of(optical_pred_tx) + bytes_of(radar_pred_tx)

        attn_bias = attn_bias.to(device)
        logits = ground_server(
            radar_encodings=radar_act,
            optical_encodings=optical_act,
            attn_bias=attn_bias,
            output_size=(H, W),
        )
        teacher_logits = logits.detach()

        backward_bytes = 0
        if enable_distill:
            optical_peer_logits_rx = radar_pred_tx if enable_peer_distill else None
            radar_peer_logits_rx = optical_pred_tx if enable_peer_distill else None


            backward_bytes += bytes_of(teacher_logits)
            backward_bytes += bytes_of(teacher_logits)
            backward_bytes += bytes_of(optical_peer_logits_rx)
            backward_bytes += bytes_of(radar_peer_logits_rx)

        return {
            "input_bytes": input_bytes,
            "forward_bytes": forward_bytes,
            "backward_bytes": backward_bytes,
            "total_bytes": forward_bytes + backward_bytes,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute communication volume for one optical+radar pair",
    )
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--enable_distill", action="store_true", help="include teacher logits downlink")
    parser.add_argument(
        "--enable_peer_distill",
        action="store_true",
        help="include peer logits forwarding (requires enable_distill)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")

    results: Dict[str, Dict[str, int]] = {}

    for ds in DATASETS:
        optical_client, radar_client, ground_server, attn_bias = build_components(
            opt_channels=ds.opt_channels,
            radar_channels=ds.radar_channels,
            image_size=args.image_size,
        )
        optical_client.to(device)
        radar_client.to(device)
        ground_server.to(device)

        optical_input = torch.zeros(
            args.batch_size, ds.opt_channels, args.image_size, args.image_size, device=device
        )
        radar_input = torch.zeros(
            args.batch_size, ds.radar_channels, args.image_size, args.image_size, device=device
        )

        comm = compute_comm_for_pair(
            optical_client=optical_client,
            radar_client=radar_client,
            ground_server=ground_server,
            attn_bias=attn_bias,
            optical_input=optical_input,
            radar_input=radar_input,
            enable_distill=args.enable_distill,
            enable_peer_distill=args.enable_peer_distill,
        )
        results[ds.name] = comm

    print("Communication volume (bytes):")
    for ds_name, vals in results.items():
        input_mb = vals["input_bytes"] / (1024 * 1024)
        forward_mb = vals["forward_bytes"] / (1024 * 1024)
        backward_mb = vals["backward_bytes"] / (1024 * 1024)
        total_mb = vals["total_bytes"] / (1024 * 1024)
        print(
            f"- {ds_name}: input={input_mb:.3f} MB, forward={forward_mb:.3f} MB, "
            f"backward={backward_mb:.3f} MB, total={total_mb:.3f} MB"
        )


if __name__ == "__main__":
    main()
