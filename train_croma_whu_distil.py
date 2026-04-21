"""
Distillation 分割训练脚本（卫星-地面协同）

场景描述:
- 光学卫星: 部署 optical_encoder + 本地 seg_head
- 雷达卫星: 部署 radar_encoder + 本地 seg_head
- 地面服务器: 部署 cross_encoder + seg_head，完成融合和监督训练

训练流程:
1. 卫星端前向传播，生成中间激活值
2. 激活值传输到地面服务器（模拟通信）
3. 卫星端预测 logits 上行到地面服务器（模拟通信）
4. 服务器完成前向传播、监督损失计算并更新
5. 服务器将自己的预测 logits 下行回传给两个卫星客户端
6. 服务器将“另一端卫星的预测 logits”转发给对端卫星（模拟通信）
7. 两端卫星客户端同时用“服务器预测”和“对端卫星预测”做 KL 蒸馏更新

注意: 该脚本用于分割训练，预训练阶段不使用。
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

from datasets import WHUOptSarPatchDataset, BigEarthNetDataset, Houston2013PatchDataset, CLASS_NAMES
from pretrain_croma import CROMA, get_alibi

OPT_CHANNELS = 4
SAR_CHANNELS = 1


@dataclass
class SplitLearningStats:
    """记录训练通信统计"""
    # 上行: 激活值/预测传输 (卫星 -> 服务器)
    forward_optical_bytes: int = 0
    forward_radar_bytes: int = 0
    # 下行: 服务器/转发预测传输 (服务器 -> 卫星)
    backward_optical_bytes: int = 0
    backward_radar_bytes: int = 0

    def add_forward(
        self,
        optical_act: torch.Tensor,
        radar_act: torch.Tensor,
        optical_pred: Optional[torch.Tensor] = None,
        radar_pred: Optional[torch.Tensor] = None,
    ):
        """记录上行通信量（激活值 + 可选的卫星端预测）。"""
        self.forward_optical_bytes += optical_act.numel() * optical_act.element_size()
        self.forward_radar_bytes += radar_act.numel() * radar_act.element_size()
        if optical_pred is not None:
            self.forward_optical_bytes += optical_pred.numel() * optical_pred.element_size()
        if radar_pred is not None:
            self.forward_radar_bytes += radar_pred.numel() * radar_pred.element_size()

    def add_backward(
        self,
        optical_teacher_pred: torch.Tensor,
        radar_teacher_pred: torch.Tensor,
        optical_peer_pred: Optional[torch.Tensor] = None,
        radar_peer_pred: Optional[torch.Tensor] = None,
    ):
        """记录下行通信量（服务器教师预测 + 可选的转发另一卫星预测）。"""
        self.backward_optical_bytes += optical_teacher_pred.numel() * optical_teacher_pred.element_size()
        self.backward_radar_bytes += radar_teacher_pred.numel() * radar_teacher_pred.element_size()
        if optical_peer_pred is not None:
            self.backward_optical_bytes += optical_peer_pred.numel() * optical_peer_pred.element_size()
        if radar_peer_pred is not None:
            self.backward_radar_bytes += radar_peer_pred.numel() * radar_peer_pred.element_size()

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

    def __init__(
        self,
        optical_encoder: nn.Module,
        attn_bias: torch.Tensor,
        encoder_dim: int,
        num_patches: int,
        num_classes: int,
    ):
        super().__init__()
        self.encoder = optical_encoder
        self.register_buffer('attn_bias', attn_bias)
        self.h_patches = int(num_patches ** 0.5)
        self.w_patches = int(num_patches ** 0.5)
        self.seg_head = nn.Sequential(
            nn.Conv2d(encoder_dim, encoder_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(encoder_dim, num_classes, kernel_size=1),
        )

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

    def predict(self, optical_encodings: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        b, n, d = optical_encodings.shape
        assert n == self.h_patches * self.w_patches
        feat = (
            optical_encodings.view(b, self.h_patches, self.w_patches, d)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        logits_low = self.seg_head(feat)
        return F.interpolate(logits_low, size=output_size, mode="bilinear", align_corners=False)

    def local_step(
        self,
        optical_imgs: torch.Tensor,
        teacher_logits: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        temperature: float = 1.0,
        max_grad_norm: float = 0.0,
    ) -> float:
        """
        完成本地反向传播和参数更新
        
        Args:
            optical_imgs: 原始光学图像
            teacher_logits: 服务器返回的教师预测
            optimizer: 本地优化器
            temperature: 蒸馏温度
            max_grad_norm: 梯度裁剪阈值
        """
        optimizer.zero_grad(set_to_none=True)
        optical_encodings = self.forward(optical_imgs)
        student_logits = self.predict(optical_encodings, output_size=optical_imgs.shape[-2:])
        teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
        kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
        kd_loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
        optimizer.step()
        return kd_loss.item()


class RadarSatelliteClient(nn.Module):
    """
    雷达卫星客户端: 部署 radar_encoder
    
    职责:
    - 本地持有雷达图像
    - 前向传播生成雷达编码
    - 接收梯度并更新本地模型
    """

    def __init__(
        self,
        radar_encoder: nn.Module,
        attn_bias: torch.Tensor,
        encoder_dim: int,
        num_patches: int,
        num_classes: int,
    ):
        super().__init__()
        self.encoder = radar_encoder
        self.register_buffer('attn_bias', attn_bias)
        self.h_patches = int(num_patches ** 0.5)
        self.w_patches = int(num_patches ** 0.5)
        self.seg_head = nn.Sequential(
            nn.Conv2d(encoder_dim, encoder_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(encoder_dim, num_classes, kernel_size=1),
        )

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

    def predict(self, radar_encodings: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        b, n, d = radar_encodings.shape
        assert n == self.h_patches * self.w_patches
        feat = (
            radar_encodings.view(b, self.h_patches, self.w_patches, d)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        logits_low = self.seg_head(feat)
        return F.interpolate(logits_low, size=output_size, mode="bilinear", align_corners=False)

    
    def local_step(
        self,
        radar_imgs: torch.Tensor,
        teacher_logits: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        temperature: float = 1.0,
        max_grad_norm: float = 0.0,
    ) -> float:
        """
        完成本地反向传播和参数更新
        
        Args:
            radar_imgs: 原始雷达图像
            teacher_logits: 服务器返回的教师预测
            optimizer: 本地优化器
            temperature: 蒸馏温度
            max_grad_norm: 梯度裁剪阈值
        """
        optimizer.zero_grad(set_to_none=True)
        radar_encodings = self.forward(radar_imgs)
        student_logits = self.predict(radar_encodings, output_size=radar_imgs.shape[-2:])
        teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
        kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
        kd_loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
        optimizer.step()
        return kd_loss.item()


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
        enable_distill: bool = False,
        enable_peer_distill: bool = False,
        distill_temperature: float = 1.0,
        max_grad_norm: float = 0.0,
    ) -> Tuple[float, float, float, float]:
        """
        执行一次 Split Learning 训练步骤
        
        流程:
        1. 卫星端前向传播
        2. 激活值传输到服务器（模拟）
        3. 服务器前向传播 + 损失计算
        """
        optical_imgs = optical_imgs.to(self.device, non_blocking=True)
        radar_imgs = radar_imgs.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        H, W = optical_imgs.shape[-2:]

        # ========== 阶段 1: 卫星端前向传播（只做一次前向） ==========
        self.optical_client.train()
        optical_encodings = self.optical_client(optical_imgs)

        self.radar_client.train()
        radar_encodings = self.radar_client(radar_imgs)

        optical_student_logits = None
        radar_student_logits = None
        if enable_distill:
            optical_student_logits = self.optical_client.predict(optical_encodings, output_size=(H, W))
            radar_student_logits = self.radar_client.predict(radar_encodings, output_size=(H, W))

        # ========== 阶段 1.5: 卫星预测上行到服务器（模拟通信） ==========
        # 使用 detach() 模拟跨设备传输（切断梯度）
        optical_pred_tx = None
        radar_pred_tx = None
        if enable_distill and enable_peer_distill:
            optical_pred_tx = optical_student_logits.detach()
            radar_pred_tx = radar_student_logits.detach()

        # ========== 阶段 2: 激活值传输到服务器（模拟通信） ==========
        optical_act = optical_encodings.detach()
        radar_act = radar_encodings.detach()
        self.stats.add_forward(optical_act, radar_act, optical_pred=optical_pred_tx, radar_pred=radar_pred_tx)

        # ========== 阶段 3: 服务器前向传播 + 损失计算 ==========
        self.ground_server.train()
        server_optimizer.zero_grad(set_to_none=True)
        attn_bias = self.attn_bias.to(self.device)
        logits = self.ground_server(
            radar_encodings=radar_act,
            optical_encodings=optical_act,
            attn_bias=attn_bias,
            output_size=(H, W),
        )
        server_ce_loss = criterion(logits, labels)

        # ========== 阶段 4: 服务器反向传播并更新参数 ==========
        server_ce_loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.ground_server.parameters(), max_grad_norm)
        server_optimizer.step()

        teacher_logits = logits.detach()

        optical_kd_loss = 0.0
        radar_kd_loss = 0.0
        if enable_distill:
            # ========== 阶段 5: 服务器预测传回卫星（模拟通信） ==========
            # 服务器把“自己的预测”以及“另一颗卫星的预测（转发）”下行给各自的卫星
            optical_peer_logits_rx = radar_pred_tx if enable_peer_distill else None
            radar_peer_logits_rx = optical_pred_tx if enable_peer_distill else None
            self.stats.add_backward(
                optical_teacher_pred=teacher_logits,
                radar_teacher_pred=teacher_logits,
                optical_peer_pred=optical_peer_logits_rx,
                radar_peer_pred=radar_peer_logits_rx,
            )

            # ========== 阶段 6: 卫星端 KL 蒸馏更新（使用缓存的 student logits） ==========
            optical_kd_loss = self._distill_and_update(
                model=self.optical_client,
                student_logits=optical_student_logits,
                teacher_logits=teacher_logits,
                peer_logits=optical_peer_logits_rx,
                optimizer=optical_optimizer,
                temperature=distill_temperature,
                max_grad_norm=max_grad_norm,
            )

            radar_kd_loss = self._distill_and_update(
                model=self.radar_client,
                student_logits=radar_student_logits,
                teacher_logits=teacher_logits,
                peer_logits=radar_peer_logits_rx,
                optimizer=radar_optimizer,
                temperature=distill_temperature,
                max_grad_norm=max_grad_norm,
            )

        server_ce_loss_value = server_ce_loss.item()
        total_loss = server_ce_loss_value + optical_kd_loss + radar_kd_loss
        return total_loss, server_ce_loss_value, optical_kd_loss, radar_kd_loss

    def _distill_and_update(
        self,
        model: nn.Module,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        peer_logits: Optional[torch.Tensor],
        optimizer: torch.optim.Optimizer,
        temperature: float = 1.0,
        max_grad_norm: float = 0.0,
    ) -> float:
        """使用 student logits 与 teacher/peer logits 计算 KL 蒸馏并更新 model 的参数，返回标量 loss 值。"""
        optimizer.zero_grad(set_to_none=True)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=1)

        teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
        kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

        if peer_logits is not None:
            peer_probs = F.softmax(peer_logits / temperature, dim=1)
            kd_loss = kd_loss + F.kl_div(student_log_probs, peer_probs, reduction="batchmean")

        kd_loss = kd_loss * (temperature ** 2)
        kd_loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        return kd_loss.item()

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
        description="Distillation 分割训练 - 卫星与地面服务器协同"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/featurize/data",
        help="WHU 光学-SAR 数据根目录",
    )
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # 分开设置不同组件的学习率
    parser.add_argument("--lr_optical", type=float, default=1e-4, help="光学编码器学习率")
    parser.add_argument("--lr_radar", type=float, default=1e-4, help="雷达编码器学习率")
    parser.add_argument("--lr_server", type=float, default=1e-4, help="服务器学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    
    parser.add_argument("--encoder_dim", type=int, default=768)
    parser.add_argument("--encoder_layers", type=int, default=6)
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
        default="/home/featurize/work/CROMA_checkpoint/croma_whu_distil_checkpoints",
    )
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--num_ratio", type=float, default=1.0)
    parser.add_argument("--stride_ratio", type=float, default=0.9)
    parser.add_argument("--max_grad_norm", type=float, default=0.0)
    parser.add_argument("--distill_temperature", type=float, default=1.0)
    parser.add_argument(
        "--disable_distill",
        action="store_false",
        dest="enable_distill",
        help="关闭蒸馏（默认开启）",
    )
    parser.set_defaults(enable_distill=True)
    parser.add_argument(
        "--enable_peer_distill",
        action="store_true",
        help="开启卫星间知识蒸馏（默认关闭）",
    )
    parser.add_argument("--dataset", type=str, choices=["whu", "bigearthnet", "houston2013"], default="whu",
                        help="选择使用的数据集：'whu' 使用 WHUOptSarPatchDataset，'bigearthnet' 使用 BigEarthNetDataset，'houston2013' 使用 Houston2013PatchDataset")
    return parser.parse_args()


def create_loaders(args):
    # 支持 WHU、BigEarthNet 和 Houston2013 三种数据集
    if args.dataset == "whu":
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
    else:  # houston2013
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

    # 尝试从数据集中推断实际的 patch 数，以避免与 args.image_size 不一致
    inferred_num_patches = None
    try:
        sample = train_set[0]
        # sample 返回 (optical, sar, ...) 其中 optical shape = (C, H, W)
        optical_sample = sample[0]
        _, H, W = optical_sample.shape
        inferred_num_patches = (H // 8) * (W // 8)
    except Exception:
        inferred_num_patches = None

    return train_loader, val_loader, inferred_num_patches


def build_split_learning_components(args, device, inferred_num_patches=None):
    """
    构建 Split Learning 各组件:
    - 光学卫星客户端
    - 雷达卫星客户端
    - 地面服务器
    """
    # Prefer inferred_num_patches (from actual dataset samples) to avoid mismatch
    if inferred_num_patches is None:
        assert args.image_size % 8 == 0, "image_size 必须能被 8 整除，以适配 CROMA 的 patch_size=8"
        num_patches = (args.image_size // 8) ** 2
    else:
        num_patches = inferred_num_patches

    if args.dataset == "whu":
        opt_ch = 4
        radar_ch = 1
    elif args.dataset == "bigearthnet":
        opt_ch = 10
        radar_ch = 2
    else:  # houston2013
        opt_ch = 10
        radar_ch = 1

    # 构建完整 CROMA 以加载预训练权重
    croma = CROMA(
        patch_size=8,
        encoder_dim=args.encoder_dim,
        encoder_layers=args.encoder_layers,
        attention_heads=args.attention_heads,
        decoder_dim=args.decoder_dim,
        decoder_layers=args.decoder_layers,
        total_channels=opt_ch + radar_ch,
        num_patches=num_patches,
        opt_channels=opt_ch,
        radar_channels=radar_ch,
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
        encoder_dim=args.encoder_dim,
        num_patches=num_patches,
        num_classes=args.num_classes,
    ).to(device)

    # 构建雷达卫星客户端
    radar_client = RadarSatelliteClient(
        radar_encoder=croma.radar_encoder,
        attn_bias=attn_bias,
        encoder_dim=args.encoder_dim,
        num_patches=num_patches,
        num_classes=args.num_classes,
    ).to(device)

    # 构建地面服务器
    ground_server = GroundServer(
        cross_encoder=croma.cross_encoder,
        encoder_dim=args.encoder_dim,
        num_patches=num_patches,
        num_classes=args.num_classes,
    ).to(device)

    optical_client_params = sum(p.numel() for p in optical_client.parameters())
    radar_client_params = sum(p.numel() for p in radar_client.parameters())
    ground_server_params = sum(p.numel() for p in ground_server.parameters())
    print(f"optical_client_params: {optical_client_params/1e6:.2f}M")
    print(f"radar_client_params: {radar_client_params/1e6:.2f}M")
    print(f"ground_server_params: {ground_server_params/1e6:.2f}M")

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
    # BigEarthNet 与 Houston2013 都使用 ignore_index=-1。
    if args.dataset == "bigearthnet":
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
    elif args.dataset == "houston2013":
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
    else:
        criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_server_ce_loss = 0.0
    total_optical_kd_loss = 0.0
    total_radar_kd_loss = 0.0
    num_pixels = 0
    trainer.stats.reset()

    for step, (optical, sar, labels) in enumerate(train_loader):
        loss, server_ce_loss, optical_kd_loss, radar_kd_loss = trainer.train_step(
            optical_imgs=optical,
            radar_imgs=sar,
            labels=labels,
            optical_optimizer=optical_optimizer,
            radar_optimizer=radar_optimizer,
            server_optimizer=server_optimizer,
            criterion=criterion,
            enable_distill=args.enable_distill,
            enable_peer_distill=args.enable_peer_distill,
            distill_temperature=args.distill_temperature,
            max_grad_norm=args.max_grad_norm,
        )

        batch_pixels = labels.numel()
        total_loss += loss * batch_pixels
        total_server_ce_loss += server_ce_loss * batch_pixels
        total_optical_kd_loss += optical_kd_loss * batch_pixels
        total_radar_kd_loss += radar_kd_loss * batch_pixels
        num_pixels += batch_pixels

        if (step + 1) % args.log_interval == 0:
            elapsed = time.time() - start_time
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            comm_mb = trainer.stats.total_bytes() / (1024 * 1024)
            print(
                f"[{elapsed_str}] Epoch {epoch} | Step {step+1}/{len(train_loader)} | "
                f"Loss: {loss:.4f} | "
                f"server_ce_loss: {server_ce_loss:.4f} | "
                f"optical_kd_loss: {optical_kd_loss:.4f} | "
                f"radar_kd_loss: {radar_kd_loss:.4f} | "
                f"Comm: {comm_mb:.2f} MB"
            )

    avg_loss = total_loss / max(1, num_pixels)
    avg_server_ce_loss = total_server_ce_loss / max(1, num_pixels)
    avg_optical_kd_loss = total_optical_kd_loss / max(1, num_pixels)
    avg_radar_kd_loss = total_radar_kd_loss / max(1, num_pixels)

    elapsed = time.time() - start_time
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    print(
        f"[{elapsed_str}] [Train] Epoch {epoch} | "
        f"avg_loss: {avg_loss:.4f} | "
        f"avg_server_ce_loss: {avg_server_ce_loss:.4f} | "
        f"avg_optical_kd_loss: {avg_optical_kd_loss:.4f} | "
        f"avg_radar_kd_loss: {avg_radar_kd_loss:.4f}"
    )

    return avg_loss, trainer.stats.total_bytes()


def evaluate(
    trainer: SplitLearningTrainer,
    val_loader: DataLoader,
    args,
    epoch: int,
    start_time: float,
):
    if args.dataset == "bigearthnet":
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
    elif args.dataset == "houston2013":
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
    else:
        criterion = nn.CrossEntropyLoss()

    if args.dataset == "houston2013":
        total_loss = 0.0
        num_pixels = 0
        num_eval_classes = args.num_classes  # 评估 0..14，-1 为未标注
        conf = torch.zeros((num_eval_classes, num_eval_classes), device=trainer.device)

        for optical, sar, labels in val_loader:
            loss, preds = trainer.evaluate_step(optical, sar, labels, criterion)
            labels = labels.to(trainer.device)

            valid = labels != -1
            valid_count = int(valid.sum().item())
            if valid_count > 0:
                y_true = labels[valid].long()
                y_pred = preds[valid].long()

                in_range = (
                    (y_true >= 0) & (y_true < num_eval_classes) &
                    (y_pred >= 0) & (y_pred < num_eval_classes)
                )
                y_true = y_true[in_range]
                y_pred = y_pred[in_range]

                if y_true.numel() > 0:
                    idx = y_true * num_eval_classes + y_pred
                    conf += torch.bincount(
                        idx,
                        minlength=num_eval_classes * num_eval_classes,
                    ).reshape(num_eval_classes, num_eval_classes)

            batch_pixels = labels.numel()
            total_loss += loss * batch_pixels
            num_pixels += batch_pixels

        avg_loss = total_loss / max(1, num_pixels)

        total = conf.sum()
        if total > 0:
            diag = torch.diag(conf)
            oa = (diag.sum() / total).item()

            per_class_total = conf.sum(dim=1)
            valid_cls = per_class_total > 0
            class_acc = torch.zeros_like(per_class_total)
            class_acc[valid_cls] = diag[valid_cls] / per_class_total[valid_cls]
            aa = class_acc[valid_cls].mean().item() if valid_cls.any() else 0.0

            row_marginal = conf.sum(dim=1)
            col_marginal = conf.sum(dim=0)
            pe = (row_marginal * col_marginal).sum() / (total * total)
            po = diag.sum() / total
            kappa = ((po - pe) / (1 - pe)).item() if float(1 - pe) > 1e-12 else 0.0
        else:
            oa, aa, kappa = 0.0, 0.0, 0.0

        elapsed = time.time() - start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        print(f"[{elapsed_str}] [Val] Epoch {epoch} | Loss: {avg_loss:.4f} | OA: {oa:.4f} | AA: {aa:.4f} | Kappa: {kappa:.4f}")
        return avg_loss, {"oa": oa, "aa": aa, "kappa": kappa}

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

    return avg_loss, {"miou": miou}


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

    ckpt_path = os.path.join(run_dir, f"distil_checkpoint_epoch_{epoch}.pt")
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
    print("Distillation 配置:")
    print("  - 光学卫星: optical_encoder + local seg_head")
    print("  - 雷达卫星: radar_encoder + local seg_head")
    print("  - 地面服务器: cross_encoder + seg_head (teacher)")
    print("=" * 60)

    # BigEarthNet 使用 19 个有效类别（参见 datasets.CLASS_NAMES 和 label_mapping）
    # 若未显式指定，则自动覆盖为 19，避免标签越界
    if args.dataset == "bigearthnet":
        num_be_classes = len(CLASS_NAMES)
        if args.num_classes != num_be_classes:
            print(f"[Info] BigEarthNet: 覆盖 num_classes 从 {args.num_classes} 到 {num_be_classes}")
            args.num_classes = num_be_classes
    elif args.dataset == "houston2013":
        # Houston2013 在数据集中已平移为 -1..14，其中 -1 是未标注（ignore）。
        num_houston_classes = 15
        if args.num_classes != num_houston_classes:
            print(f"[Info] Houston2013: 覆盖 num_classes 从 {args.num_classes} 到 {num_houston_classes}")
            args.num_classes = num_houston_classes

    train_loader, val_loader, inferred_num_patches = create_loaders(args)
    print(f"Train patches: {len(train_loader.dataset)}, Val patches: {len(val_loader.dataset)}")

    # 构建 Split Learning 组件
    optical_client, radar_client, ground_server, attn_bias, num_patches = \
        build_split_learning_components(args, device, inferred_num_patches=inferred_num_patches)

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
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.dataset}"
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

        val_loss, val_metrics = evaluate(trainer, val_loader, args, epoch, start_time)

        # 记录指标
        file_exists = os.path.exists(metrics_path)
        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            if args.dataset == "houston2013":
                if not file_exists:
                    writer.writerow([
                        "epoch",
                        "train_loss",
                        "val_loss",
                        "val_OA",
                        "val_AA",
                        "val_kappa",
                        "epoch_comm_MB",
                        "total_comm_MB",
                    ])
                writer.writerow([
                    epoch,
                    float(train_loss),
                    float(val_loss),
                    float(val_metrics["oa"]),
                    float(val_metrics["aa"]),
                    float(val_metrics["kappa"]),
                    epoch_comm_bytes / (1024 * 1024),
                    total_comm_bytes / (1024 * 1024),
                ])
            else:
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
                    float(val_metrics["miou"]),
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
