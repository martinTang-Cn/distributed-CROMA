import os
import argparse
from datetime import datetime
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision.models import resnet50, ResNet50_Weights

from datasets import WHUOptSarPatchDataset, BigEarthNetDataset, Houston2013PatchDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ResNet50 for segmentation with fused optical+radar inputs"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["whu-opt-sar", "bigearthnet", "houston2013"],
        default="whu-opt-sar",
    )
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--stride_ratio", type=float, default=0.9)
    parser.add_argument("--num_ratio", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=0.0)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./resnet50_checkpoints")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--save_best_only", action="store_true")
    parser.add_argument("--ignore_index", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_channels_and_classes(dataset: str) -> Tuple[int, int]:
    if dataset == "whu-opt-sar":
        return 4 + 1, 8
    if dataset == "bigearthnet":
        return 10 + 2, 19
    return 144 + 1, 15


def create_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    if args.dataset == "whu-opt-sar":
        train_set = WHUOptSarPatchDataset(
            root_dir=args.data_root,
            split="train",
            patch_size=args.image_size,
            stride_ratio=args.stride_ratio,
            num_ratio=args.num_ratio,
        )
        val_set = WHUOptSarPatchDataset(
            root_dir=args.data_root,
            split="val",
            patch_size=args.image_size,
            stride_ratio=args.stride_ratio,
            num_ratio=1.0,
        )
    elif args.dataset == "bigearthnet":
        train_set = BigEarthNetDataset(
            root=args.data_root,
            split="train",
            ratio=args.num_ratio,
        )
        val_set = BigEarthNetDataset(
            root=args.data_root,
            split="validation",
            ratio=1.0,
        )
    else:
        train_set = Houston2013PatchDataset(
            root_dir=args.data_root,
            split="train",
            patch_size=args.image_size,
            stride=args.image_size,
        )
        val_set = Houston2013PatchDataset(
            root_dir=args.data_root,
            split="val",
            patch_size=args.image_size,
            stride=args.image_size,
        )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def adapt_resnet_conv1(model: nn.Module, in_channels: int) -> None:
    if model.conv1.in_channels == in_channels:
        return

    old_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )

    with torch.no_grad():
        if in_channels == 3:
            new_conv.weight.copy_(old_conv.weight)
        elif in_channels > 3:
            repeat = (in_channels + 2) // 3
            weight = old_conv.weight.repeat(1, repeat, 1, 1)[:, :in_channels, :, :]
            weight = weight * (3.0 / in_channels)
            new_conv.weight.copy_(weight)
        else:
            weight = old_conv.weight[:, :in_channels, :, :]
            if in_channels == 1:
                weight = weight.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(weight)

    model.conv1 = new_conv


def build_model(in_channels: int, num_classes: int, pretrained: bool) -> nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    backbone = resnet50(weights=weights)
    adapt_resnet_conv1(backbone, in_channels)

    class ResNetSeg(nn.Module):
        def __init__(self, base: nn.Module, num_classes: int):
            super().__init__()
            self.stem = nn.Sequential(
                base.conv1,
                base.bn1,
                base.relu,
                base.maxpool,
            )
            self.layer1 = base.layer1
            self.layer2 = base.layer2
            self.layer3 = base.layer3
            self.layer4 = base.layer4
            self.classifier = nn.Conv2d(2048, num_classes, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h, w = x.shape[-2:]
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.classifier(x)
            return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

    return ResNetSeg(backbone, num_classes)


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int,
) -> Tuple[float, float]:
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)
        if ignore_index >= 0:
            mask = targets != ignore_index
            if mask.sum() == 0:
                return 0.0, 0.0
            correct = (preds[mask] == targets[mask]).sum().item()
            total = mask.sum().item()
        else:
            correct = (preds == targets).sum().item()
            total = targets.numel()

        acc = correct / max(total, 1)
    return acc, float(total)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: nn.Module,
    ignore_index: int,
    max_grad_norm: float,
    log_interval: int,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_count = 0.0

    for step, (optical, sar, target) in enumerate(loader):
        optical = optical.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        imgs = torch.cat([optical, sar], dim=1)
        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = loss_fn(logits, target)
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        acc, count = compute_metrics(logits, target, ignore_index)
        total_loss += loss.item()
        total_acc += acc * count
        total_count += count

        if log_interval > 0 and (step + 1) % log_interval == 0:
            avg_loss = total_loss / (step + 1)
            avg_acc = total_acc / max(total_count, 1.0)
            print(f"step {step+1}: loss={avg_loss:.4f}, acc={avg_acc:.4f}")

    avg_loss = total_loss / max(len(loader), 1)
    avg_acc = total_acc / max(total_count, 1.0)
    return avg_loss, avg_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    ignore_index: int,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_count = 0.0

    with torch.no_grad():
        for optical, sar, target in loader:
            optical = optical.to(device, non_blocking=True)
            sar = sar.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            imgs = torch.cat([optical, sar], dim=1)
            logits = model(imgs)
            loss = loss_fn(logits, target)

            acc, count = compute_metrics(logits, target, ignore_index)
            total_loss += loss.item()
            total_acc += acc * count
            total_count += count

    avg_loss = total_loss / max(len(loader), 1)
    avg_acc = total_acc / max(total_count, 1.0)
    return avg_loss, avg_acc


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = create_loaders(args)
    in_channels, num_classes = infer_channels_and_classes(args.dataset)

    model = build_model(in_channels, num_classes, args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=args.ignore_index)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_fn,
            args.ignore_index,
            args.max_grad_norm,
            args.log_interval,
        )
        print(f"train loss={train_loss:.4f}, acc={train_acc:.4f}")

        if epoch % args.val_interval == 0:
            val_loss, val_acc = evaluate(
                model, val_loader, device, loss_fn, args.ignore_index
            )
            print(f"val loss={val_loss:.4f}, acc={val_acc:.4f}")

            should_save = True
            if args.save_best_only:
                if val_loss < best_val:
                    best_val = val_loss
                else:
                    should_save = False

            if should_save:
                ckpt_path = os.path.join(
                    args.output_dir, f"resnet50_{args.dataset}_{timestamp}_e{epoch}.pt"
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "args": vars(args),
                    },
                    ckpt_path,
                )
                print(f"saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
