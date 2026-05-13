"""
Count per-class pixel totals for WHU/BigEarthNet/Houston2013.
"""

import argparse
import csv
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader

from datasets import WHUOptSarPatchDataset, BigEarthNetDataset, Houston2013PatchDataset, CLASS_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description="Count per-class pixel totals")
    parser.add_argument("--dataset", type=str, choices=["whu", "bigearthnet", "houston2013"], required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--split", type=str, default=None, help="Split override")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_csv", type=str, default="", help="Save counts to CSV path")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def _select_split(dataset: str, override: Optional[str]) -> str:
    if override:
        return override
    if dataset == "bigearthnet":
        return "test"
    return "val"


def create_loader(args):
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
    return loader


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    if args.dataset == "bigearthnet":
        num_classes = len(CLASS_NAMES)
    elif args.dataset == "houston2013":
        num_classes = 15
    else:
        num_classes = 8

    loader = create_loader(args)
    counts = torch.zeros(num_classes, dtype=torch.int64, device=device)

    for _, _, labels in loader:
        labels = labels.to(device, non_blocking=True)
        valid = (labels >= 0) & (labels < num_classes)
        if valid.any():
            labels = labels[valid].long()
            counts += torch.bincount(labels, minlength=num_classes)

    print("Per-class pixel counts:")
    for i, v in enumerate(counts.tolist()):
        print(f"  class {i}: {int(v)}")
    print(f"Total valid pixels: {int(counts.sum().item())}")

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset", "split", "class", "pixel_count"])
            for i, v in enumerate(counts.tolist()):
                writer.writerow([args.dataset, _select_split(args.dataset, args.split), i, int(v)])
            writer.writerow([args.dataset, _select_split(args.dataset, args.split), "all", int(counts.sum().item())])


if __name__ == "__main__":
    main()
