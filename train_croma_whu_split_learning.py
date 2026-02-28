"""
Split Learning 分割训练脚本

场景描述:
- 光学卫星: 部署 optical_encoder，处理光学图像
- 雷达卫星: 部署 radar_encoder，处理雷达图像
- 地面服务器: 部署 cross_encoder + seg_head，完成融合和分割

训练流程:
1. 卫星端前向传播，生成中间激活值
2. 激活值传输到地面服务器（模拟通信）
3. 服务器完成前向传播和损失计算
4. 服务器反向传播，计算激活值梯度
5. 梯度传回卫星，卫星完成编码器反向传播

注意: Split Learning 仅用于分割训练，预训练阶段不使用。
"""

import os
import argparse
from datetime import datetime
import time
import csv
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import WHUOptSarPatchDataset
from pretrain_croma import CROMA, get_alibi

OPT_CHANNELS = 4
SAR_CHANNELS = 1


@dataclass
class SplitLearningStats:
    """记录 Split Learning 通信统计"""
    # 前向传播: 激活值传输 (卫星 -> 服务器)
    forward_optical_bytes: int = 0
    forward_radar_bytes: int = 0
    # 反向传播: 梯度传输 (服务器 -> 卫星)
    backward_optical_bytes: int = 0
    backward_radar_bytes: int = 0

    def add_forward(self, optical_act: torch.Tensor, radar_act: torch.Tensor):
        """记录前向传播的激活值大小"""
        self.forward_optical_bytes += optical_act.numel() * optical_act.element_size()
        self.forward_radar_bytes += radar_act.numel() * radar_act.element_size()

    def add_backward(self, optical_grad: torch.Tensor, radar_grad: torch.Tensor):
        """记录反向传播的梯度大小"""
        self.backward_optical_bytes += optical_grad.numel() * optical_grad.element_size()
        self.backward_radar_bytes += radar_grad.numel() * radar_grad.element_size()

    def total_bytes(self) -> int:
        return (self.forward_optical_bytes + self.forward_radar_bytes +
                self.backward_optical_bytes + self.backward_radar_bytes)

    def reset(self):
        self.forward_optical_bytes = 0
        self.forward_radar_bytes = 0
        self.backward_optical_bytes = 0
        self.backward_radar_bytes = 0


class OpticalSatelliteClient(nn.Module):
    """
    光学卫星客户端: 部署 optical_encoder
    
    职责:
    - 本地持有光学图像
    - 前向传播生成光学编码
    - 接收梯度并更新本地模型
    """

    def __init__(self, optical_encoder: nn.Module, attn_bias: torch.Tensor):
        super().__init__()
        self.encoder = optical_encoder
        self.register_buffer('attn_bias', attn_bias)

    def forward(self, optical_imgs: torch.Tensor) -> torch.Tensor:
        """
        前向传播: 光学图像 -> 光学编码
        
        Args:
            optical_imgs: [B, C_opt, H, W] 光学图像
            
        Returns:
            optical_encodings: [B, N, D] 光学编码（将传输到服务器）
        """
        attn_bias = self.attn_bias.to(optical_imgs.device)
        optical_encodings = self.encoder(
            imgs=optical_imgs, attn_bias=attn_bias, mask_info=None
        )
        return optical_encodings

    def local_step(self, optical_imgs: torch.Tensor, grad_from_server: torch.Tensor,
                   optimizer: torch.optim.Optimizer, max_grad_norm: float = 0.0):
        """
        完成本地反向传播和参数更新
        
        Args:
            optical_imgs: 原始光学图像
            grad_from_server: 服务器返回的激活值梯度
            optimizer: 本地优化器
            max_grad_norm: 梯度裁剪阈值
        """
        optimizer.zero_grad(set_to_none=True)
        # 重新前向传播以建立计算图
        optical_encodings = self.forward(optical_imgs)
        # 使用服务器传来的梯度进行反向传播
        optical_encodings.backward(grad_from_server)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_grad_norm)
        optimizer.step()


class RadarSatelliteClient(nn.Module):
    """
    雷达卫星客户端: 部署 radar_encoder
    
    职责:
    - 本地持有雷达图像
    - 前向传播生成雷达编码
    - 接收梯度并更新本地模型
    """

    def __init__(self, radar_encoder: nn.Module, attn_bias: torch.Tensor):
        super().__init__()
        self.encoder = radar_encoder
        self.register_buffer('attn_bias', attn_bias)

    def forward(self, radar_imgs: torch.Tensor) -> torch.Tensor:
        """
        前向传播: 雷达图像 -> 雷达编码
        
        Args:
            radar_imgs: [B, C_sar, H, W] 雷达图像
            
        Returns:
            radar_encodings: [B, N, D] 雷达编码（将传输到服务器）
        """
        attn_bias = self.attn_bias.to(radar_imgs.device)
        radar_encodings = self.encoder(
            imgs=radar_imgs, attn_bias=attn_bias, mask_info=None
        )
        return radar_encodings

    def local_step(self, radar_imgs: torch.Tensor, grad_from_server: torch.Tensor,
                   optimizer: torch.optim.Optimizer, max_grad_norm: float = 0.0):
        """
        完成本地反向传播和参数更新
        
        Args:
            radar_imgs: 原始雷达图像
            grad_from_server: 服务器返回的激活值梯度
            optimizer: 本地优化器
            max_grad_norm: 梯度裁剪阈值
        """
        optimizer.zero_grad(set_to_none=True)
        radar_encodings = self.forward(radar_imgs)
        radar_encodings.backward(grad_from_server)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_grad_norm)
        optimizer.step()


class GroundServer(nn.Module):
    """
    地面服务器: 部署 cross_encoder + seg_head
    
    职责:
    - 接收卫星传来的编码
    - 通过 cross_encoder 融合多模态信息
    - 完成分割预测
    - 计算损失并生成梯度传回卫星
    """

    def __init__(self, cross_encoder: nn.Module, encoder_dim: int, num_patches: int, num_classes: int):
        super().__init__()
        self.cross_encoder = cross_encoder
        self.encoder_dim = encoder_dim
        self.num_patches = num_patches

        # 假设 patch 在空间上是正方形网格
        self.h_patches = int(num_patches ** 0.5)
        self.w_patches = int(num_patches ** 0.5)

        self.seg_head = nn.Sequential(
            nn.Conv2d(encoder_dim, encoder_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(encoder_dim, num_classes, kernel_size=1),
        )

    def forward(self, radar_encodings: torch.Tensor, optical_encodings: torch.Tensor,
                attn_bias: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        """
        前向传播: 融合编码并生成分割预测
        
        Args:
            radar_encodings: [B, N, D] 雷达编码（从雷达卫星接收）
            optical_encodings: [B, N, D] 光学编码（从光学卫星接收）
            attn_bias: 注意力偏置
            output_size: (H, W) 输出分割图的尺寸
            
        Returns:
            logits: [B, num_classes, H, W] 分割预测
        """
        joint_encodings = self.cross_encoder(
            x=radar_encodings, context=optical_encodings, alibi=attn_bias
        )

        b, n, d = joint_encodings.shape
        assert n == self.h_patches * self.w_patches

        # [B, N, D] -> [B, D, H_p, W_p]
        feat = (
            joint_encodings.view(b, self.h_patches, self.w_patches, d)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        logits_low = self.seg_head(feat)

        H, W = output_size
        logits = F.interpolate(logits_low, size=(H, W), mode="bilinear", align_corners=False)
        return logits


class SplitLearningTrainer:
    """
    Split Learning 训练协调器
    
    模拟卫星与地面服务器之间的 Split Learning 训练过程
    """

    def __init__(
        self,
        optical_client: OpticalSatelliteClient,
        radar_client: RadarSatelliteClient,
        ground_server: GroundServer,
        attn_bias: torch.Tensor,
        device: torch.device,
    ):
        self.optical_client = optical_client
        self.radar_client = radar_client
        self.ground_server = ground_server
        self.attn_bias = attn_bias
        self.device = device
        self.stats = SplitLearningStats()

    def train_step(
        self,
        optical_imgs: torch.Tensor,
        radar_imgs: torch.Tensor,
        labels: torch.Tensor,
        optical_optimizer: torch.optim.Optimizer,
        radar_optimizer: torch.optim.Optimizer,
        server_optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        max_grad_norm: float = 0.0,
    ) -> float:
        """
        执行一次 Split Learning 训练步骤
        
        流程:
        1. 卫星端前向传播
        2. 激活值传输到服务器（模拟）
        3. 服务器前向传播 + 损失计算
        4. 服务器反向传播，获取激活值梯度
        5. 梯度传回卫星（模拟）
        6. 卫星端反向传播 + 参数更新
        
        Returns:
            loss: 当前步骤的损失值
        """
        optical_imgs = optical_imgs.to(self.device, non_blocking=True)
        radar_imgs = radar_imgs.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        H, W = optical_imgs.shape[-2:]

        # ========== 阶段 1: 卫星端前向传播 ==========
        # 光学卫星
        self.optical_client.train()
        optical_encodings = self.optical_client(optical_imgs)
        
        # 雷达卫星
        self.radar_client.train()
        radar_encodings = self.radar_client(radar_imgs)

        # ========== 阶段 2: 激活值传输到服务器（模拟通信） ==========
        # 在实际场景中，这里会有网络传输
        # 我们使用 .detach() 模拟传输切断，同时保留 requires_grad 以便计算梯度
        optical_act = optical_encodings.detach().requires_grad_(True)
        radar_act = radar_encodings.detach().requires_grad_(True)
        
        # 记录通信量
        self.stats.add_forward(optical_act, radar_act)

        # ========== 阶段 3: 服务器前向传播 + 损失计算 ==========
        self.ground_server.train()
        server_optimizer.zero_grad(set_to_none=True)
        
        attn_bias = self.attn_bias.to(self.device)
        logits = self.ground_server(
            radar_encodings=radar_act,
            optical_encodings=optical_act,
            attn_bias=attn_bias,
            output_size=(H, W)
        )
        loss = criterion(logits, labels)

        # ========== 阶段 4: 服务器反向传播 ==========
        loss.backward()
        
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.ground_server.parameters(), max_grad_norm)
        server_optimizer.step()

        # 获取激活值的梯度（将传回卫星）
        optical_grad = optical_act.grad.detach()
        radar_grad = radar_act.grad.detach()

        # ========== 阶段 5: 梯度传回卫星（模拟通信） ==========
        self.stats.add_backward(optical_grad, radar_grad)

        # ========== 阶段 6: 卫星端反向传播 + 参数更新 ==========
        # 光学卫星本地更新
        self.optical_client.local_step(optical_imgs, optical_grad, optical_optimizer, max_grad_norm)
        
        # 雷达卫星本地更新
        self.radar_client.local_step(radar_imgs, radar_grad, radar_optimizer, max_grad_norm)

        return loss.item()

    @torch.no_grad()
    def evaluate_step(
        self,
        optical_imgs: torch.Tensor,
        radar_imgs: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module,
    ) -> Tuple[float, torch.Tensor]:
        """
        执行一次评估步骤
        
        Returns:
            loss: 损失值
            preds: 预测结果
        """
        optical_imgs = optical_imgs.to(self.device, non_blocking=True)
        radar_imgs = radar_imgs.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        H, W = optical_imgs.shape[-2:]

        self.optical_client.eval()
        self.radar_client.eval()
        self.ground_server.eval()

        optical_encodings = self.optical_client(optical_imgs)
        radar_encodings = self.radar_client(radar_imgs)

        attn_bias = self.attn_bias.to(self.device)
        logits = self.ground_server(
            radar_encodings=radar_encodings,
            optical_encodings=optical_encodings,
            attn_bias=attn_bias,
            output_size=(H, W)
        )
        loss = criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)

        return loss.item(), preds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split Learning 分割训练 - 卫星与地面服务器协同"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="../whu-opt-sar",
        help="WHU 光学-SAR 数据根目录",
    )
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # 分开设置不同组件的学习率
    parser.add_argument("--lr_optical", type=float, default=1e-4, help="光学编码器学习率")
    parser.add_argument("--lr_radar", type=float, default=1e-4, help="雷达编码器学习率")
    parser.add_argument("--lr_server", type=float, default=1e-4, help="服务器学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    
    parser.add_argument("--encoder_dim", type=int, default=768)
    parser.add_argument("--encoder_layers", type=int, default=12)
    parser.add_argument("--attention_heads", type=int, default=16)
    parser.add_argument("--decoder_dim", type=int, default=512)
    parser.add_argument("--decoder_layers", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        required=True,
        help="预训练 CROMA checkpoint 路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../CROMA_checkpoint/croma_whu_split_learning_checkpoints",
    )
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--num_ratio", type=float, default=1.0)
    parser.add_argument("--stride_ratio", type=float, default=0.9)
    parser.add_argument("--max_grad_norm", type=float, default=0.0)
    return parser.parse_args()


def create_loaders(args):
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


def build_split_learning_components(args, device):
    """
    构建 Split Learning 各组件:
    - 光学卫星客户端
    - 雷达卫星客户端
    - 地面服务器
    """
    assert args.image_size % 8 == 0
    num_patches = (args.image_size // 8) ** 2

    # 构建完整 CROMA 以加载预训练权重
    croma = CROMA(
        patch_size=8,
        encoder_dim=args.encoder_dim,
        encoder_layers=args.encoder_layers,
        attention_heads=args.attention_heads,
        decoder_dim=args.decoder_dim,
        decoder_layers=args.decoder_layers,
        total_channels=OPT_CHANNELS + SAR_CHANNELS,
        num_patches=num_patches,
        opt_channels=OPT_CHANNELS,
        radar_channels=SAR_CHANNELS,
    )

    # 加载预训练权重
    ckpt = torch.load(args.pretrained_ckpt, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = croma.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Warning] Missing keys: {missing}")
    if unexpected:
        print(f"[Warning] Unexpected keys: {unexpected}")

    # 获取 attn_bias
    attn_bias = croma.attn_bias

    # 构建光学卫星客户端
    optical_client = OpticalSatelliteClient(
        optical_encoder=croma.optical_encoder,
        attn_bias=attn_bias,
    ).to(device)

    # 构建雷达卫星客户端
    radar_client = RadarSatelliteClient(
        radar_encoder=croma.radar_encoder,
        attn_bias=attn_bias,
    ).to(device)

    # 构建地面服务器
    ground_server = GroundServer(
        cross_encoder=croma.cross_encoder,
        encoder_dim=args.encoder_dim,
        num_patches=num_patches,
        num_classes=args.num_classes,
    ).to(device)

    return optical_client, radar_client, ground_server, attn_bias, num_patches


def train_one_epoch(
    trainer: SplitLearningTrainer,
    train_loader: DataLoader,
    optical_optimizer: torch.optim.Optimizer,
    radar_optimizer: torch.optim.Optimizer,
    server_optimizer: torch.optim.Optimizer,
    args,
    epoch: int,
    start_time: float,
):
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    num_pixels = 0
    trainer.stats.reset()

    for step, (optical, sar, labels) in enumerate(train_loader):
        loss = trainer.train_step(
            optical_imgs=optical,
            radar_imgs=sar,
            labels=labels,
            optical_optimizer=optical_optimizer,
            radar_optimizer=radar_optimizer,
            server_optimizer=server_optimizer,
            criterion=criterion,
            max_grad_norm=args.max_grad_norm,
        )

        batch_pixels = labels.numel()
        total_loss += loss * batch_pixels
        num_pixels += batch_pixels

        if (step + 1) % args.log_interval == 0:
            elapsed = time.time() - start_time
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            comm_mb = trainer.stats.total_bytes() / (1024 * 1024)
            print(
                f"[{elapsed_str}] Epoch {epoch} | Step {step+1}/{len(train_loader)} | "
                f"Loss: {loss:.4f} | Comm: {comm_mb:.2f} MB"
            )

    avg_loss = total_loss / max(1, num_pixels)
    return avg_loss, trainer.stats.total_bytes()


def evaluate(
    trainer: SplitLearningTrainer,
    val_loader: DataLoader,
    args,
    epoch: int,
    start_time: float,
):
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    num_pixels = 0
    num_classes = args.num_classes
    intersection = torch.zeros(num_classes, device=trainer.device)
    union = torch.zeros(num_classes, device=trainer.device)

    for optical, sar, labels in val_loader:
        loss, preds = trainer.evaluate_step(optical, sar, labels, criterion)

        labels = labels.to(trainer.device)
        for c in range(num_classes):
            pred_c = preds == c
            label_c = labels == c
            intersection[c] += (pred_c & label_c).sum()
            union[c] += (pred_c | label_c).sum()

        batch_pixels = labels.numel()
        total_loss += loss * batch_pixels
        num_pixels += batch_pixels

    avg_loss = total_loss / max(1, num_pixels)

    iou_per_class = intersection / (union + 1e-6)
    valid = union > 0
    miou = iou_per_class[valid].mean().item() if valid.any() else 0.0

    elapsed = time.time() - start_time
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    print(f"[{elapsed_str}] [Val] Epoch {epoch} | Loss: {avg_loss:.4f} | mIoU: {miou:.4f}")

    return avg_loss, miou


def save_checkpoint(
    optical_client: OpticalSatelliteClient,
    radar_client: RadarSatelliteClient,
    ground_server: GroundServer,
    optical_optimizer: torch.optim.Optimizer,
    radar_optimizer: torch.optim.Optimizer,
    server_optimizer: torch.optim.Optimizer,
    args,
    epoch: int,
    run_dir: str,
    last_ckpt_path: Optional[str] = None,
) -> str:
    """保存 Split Learning 各组件的 checkpoint"""
    os.makedirs(run_dir, exist_ok=True)

    if last_ckpt_path is not None and os.path.exists(last_ckpt_path):
        try:
            os.remove(last_ckpt_path)
        except OSError:
            pass

    ckpt_path = os.path.join(run_dir, f"split_learning_checkpoint_epoch_{epoch}.pt")
    torch.save(
        {
            "epoch": epoch,
            "optical_client_state_dict": optical_client.state_dict(),
            "radar_client_state_dict": radar_client.state_dict(),
            "ground_server_state_dict": ground_server.state_dict(),
            "optical_optimizer_state_dict": optical_optimizer.state_dict(),
            "radar_optimizer_state_dict": radar_optimizer.state_dict(),
            "server_optimizer_state_dict": server_optimizer.state_dict(),
            "args": vars(args),
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")
    return ckpt_path


def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print("=" * 60)
    print("Split Learning 配置:")
    print("  - 光学卫星: optical_encoder")
    print("  - 雷达卫星: radar_encoder")
    print("  - 地面服务器: cross_encoder + seg_head")
    print("=" * 60)

    train_loader, val_loader = create_loaders(args)
    print(f"Train patches: {len(train_loader.dataset)}, Val patches: {len(val_loader.dataset)}")

    # 构建 Split Learning 组件
    optical_client, radar_client, ground_server, attn_bias, num_patches = \
        build_split_learning_components(args, device)

    # 为每个组件创建独立优化器
    optical_optimizer = torch.optim.AdamW(
        optical_client.parameters(), lr=args.lr_optical, weight_decay=args.weight_decay
    )
    radar_optimizer = torch.optim.AdamW(
        radar_client.parameters(), lr=args.lr_radar, weight_decay=args.weight_decay
    )
    server_optimizer = torch.optim.AdamW(
        ground_server.parameters(), lr=args.lr_server, weight_decay=args.weight_decay
    )

    # 创建训练协调器
    trainer = SplitLearningTrainer(
        optical_client=optical_client,
        radar_client=radar_client,
        ground_server=ground_server,
        attn_bias=attn_bias,
        device=device,
    )

    # 运行目录和指标文件
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, run_timestamp)
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "epoch_metrics.csv")

    start_time = time.time()
    last_ckpt_path = None
    total_comm_bytes = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, epoch_comm_bytes = train_one_epoch(
            trainer, train_loader,
            optical_optimizer, radar_optimizer, server_optimizer,
            args, epoch, start_time
        )
        total_comm_bytes += epoch_comm_bytes

        val_loss, val_miou = evaluate(trainer, val_loader, args, epoch, start_time)

        # 记录指标
        file_exists = os.path.exists(metrics_path)
        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "val_mIoU",
                    "epoch_comm_MB",
                    "total_comm_MB",
                ])
            writer.writerow([
                epoch,
                float(train_loss),
                float(val_loss),
                float(val_miou),
                epoch_comm_bytes / (1024 * 1024),
                total_comm_bytes / (1024 * 1024),
            ])

        last_ckpt_path = save_checkpoint(
            optical_client, radar_client, ground_server,
            optical_optimizer, radar_optimizer, server_optimizer,
            args, epoch, run_dir, last_ckpt_path
        )

    print("=" * 60)
    print(f"训练完成! 总通信量: {total_comm_bytes / (1024 * 1024 * 1024):.2f} GB")
    print("=" * 60)


if __name__ == "__main__":
    main()
