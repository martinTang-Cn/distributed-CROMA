import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import argparse
import torch

from datasets import WHUOptSarPatchDataset
from pretrain_croma import CROMA
from train_croma_whu_segmentation import CROMASegmentation, OPT_CHANNELS, SAR_CHANNELS

# 8类名称（根据数据集实际类别替换）
CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7']

# 从 tab20 中取 8 种颜色作为默认配色（可自定义）
_base_cmap = plt.get_cmap('tab20')
CMAP_COLORS = [_base_cmap(i) for i in range(8)]


def load_segmentation_model(seg_ckpt_path: str, device: torch.device):
    """从分割 checkpoint 加载完整的 CROMA 分割模型，用于可视化推理。"""
    ckpt = torch.load(seg_ckpt_path, map_location=device)
    args_dict = ckpt.get("args", {})

    image_size = args_dict.get("image_size", 256)
    encoder_dim = args_dict.get("encoder_dim", 768)
    encoder_layers = args_dict.get("encoder_layers", 12)
    attention_heads = args_dict.get("attention_heads", 16)
    decoder_dim = args_dict.get("decoder_dim", 512)
    decoder_layers = args_dict.get("decoder_layers", 1)
    num_classes = args_dict.get("num_classes", 8)

    assert image_size % 8 == 0, "image_size 必须能被 8 整除，以适配 CROMA 的 patch_size=8"
    num_patches = (image_size // 8) ** 2

    # 构建与训练时相同配置的 CROMA + 分割头
    croma = CROMA(
        patch_size=8,
        encoder_dim=encoder_dim,
        encoder_layers=encoder_layers,
        attention_heads=attention_heads,
        decoder_dim=decoder_dim,
        decoder_layers=decoder_layers,
        total_channels=OPT_CHANNELS + SAR_CHANNELS,
        num_patches=num_patches,
        opt_channels=OPT_CHANNELS,
        radar_channels=SAR_CHANNELS,
    )

    seg_model = CROMASegmentation(croma_model=croma, num_classes=num_classes)
    state_dict = ckpt.get("model_state_dict", ckpt)
    seg_model.load_state_dict(state_dict, strict=False)

    seg_model.to(device)
    seg_model.eval()
    return seg_model, num_classes


def visualize_sample(dataset, idx=None, save_path=None, model=None, device=None, num_classes: int = None):
    """
    可视化数据集中的一个样本
    
    Args:
        dataset: OpticalSARDataset实例
        idx: 样本索引，如果为None则随机选择
    """
    # 随机选择一个样本（使用 SystemRandom 以避免被全局 seed 影响）
    if idx is None:
        idx = random.SystemRandom().randint(0, len(dataset) - 1)
    
    # 获取样本
    optical, sar, label = dataset[idx]
    
    print(f"样本索引: {idx}")
    print(f"光学图像shape: {optical.shape}")
    print(f"SAR图像shape: {sar.shape}")
    print(f"标签shape: {label.shape}")
    
    # 转换为numpy数组以便可视化
    optical_np = optical.numpy()  # (C, H, W)
    sar_np = sar.numpy()  # (C, H, W)
    label_np = label.numpy()  # (H, W) 或 (C, H, W)
    
    # 光学图像：选取前三个通道作为RGB
    optical_rgb = optical_np[:3, :, :].transpose(1, 2, 0)  # (H, W, 3)
    optical_rgb = optical_rgb[:, :, ::-1]
    
    # 归一化到0-1范围用于显示
    optical_rgb = normalize_for_display(optical_rgb)
    
    # SAR图像：如果是单通道，直接显示；如果是多通道，取第一个通道或者前3个通道
    if sar_np.shape[0] == 1:
        sar_display = sar_np[0]  # (H, W)
        sar_display = normalize_for_display(sar_display)
        sar_cmap = 'gray'
    elif sar_np.shape[0] >= 3:
        sar_display = sar_np[:3, :, :].transpose(1, 2, 0)  # (H, W, 3)
        sar_display = normalize_for_display(sar_display)
        sar_cmap = None
    else:
        sar_display = sar_np[0]  # 取第一个通道
        sar_display = normalize_for_display(sar_display)
        sar_cmap = 'gray'
    
    # 标签：如果是多通道，取第一个通道
    if label_np.ndim == 3:
        label_display = label_np[0]
    else:
        label_display = label_np
    
    # 是否需要显示预测结果
    show_pred = model is not None

    # 创建可视化
    ncols = 4 if show_pred else 3
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
    
    # 显示光学图像
    axes[0].imshow(optical_rgb)
    axes[0].set_title(f'Optical Image (RGB)\nShape: {optical.shape}', fontsize=12)
    axes[0].axis('off')
    
    # 显示SAR图像
    if sar_cmap:
        axes[1].imshow(sar_display, cmap=sar_cmap)
    else:
        axes[1].imshow(sar_display)
    axes[1].set_title(f'SAR Image\nShape: {sar.shape}', fontsize=12)
    axes[1].axis('off')
    
    # 显示分割标签（离散颜色映射，适用于多类分割）
    if num_classes is None:
        num_classes = len(CLASS_NAMES)
    cmap = ListedColormap(CMAP_COLORS[:num_classes])
    norm = BoundaryNorm(np.arange(num_classes + 1) - 0.5, ncolors=num_classes)

    label_img = axes[2].imshow(label_display, cmap=cmap, norm=norm)
    axes[2].set_title(f'Label\nShape: {label.shape}', fontsize=12)
    axes[2].axis('off')

    # 添加离散颜色条显示标签值（0..num_classes-1）
    cbar = plt.colorbar(label_img, ax=axes[2], fraction=0.046, pad=0.04, ticks=np.arange(num_classes))
    cbar.ax.set_yticklabels([str(i) for i in range(num_classes)])

    # 添加图例：使用颜色块映射到类别名称
    patches = [mpatches.Patch(color=CMAP_COLORS[i], label=CLASS_NAMES[i]) for i in range(num_classes)]
    axes[2].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

    # 推理并可视化预测结果
    if show_pred:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with torch.no_grad():
            imgs = torch.cat(
                [optical.unsqueeze(0), sar.unsqueeze(0)], dim=1
            ).to(device)
            logits = model(imgs)
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy()

        pred_img = axes[3].imshow(pred, cmap=cmap, norm=norm)
        axes[3].set_title('Prediction', fontsize=12)
        axes[3].axis('off')

        # 也为预测结果添加一个颜色条（可选）
        plt.colorbar(pred_img, ax=axes[3], fraction=0.046, pad=0.04, ticks=np.arange(num_classes))
    
    # 显示唯一的标签值
    unique_labels = np.unique(label_display)
    print(f"标签中的唯一值: {unique_labels}")
    print(f"标签类别数: {len(unique_labels)}")
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def normalize_for_display(image):
    """
    将图像归一化到0-1范围用于显示
    
    Args:
        image: numpy数组
        
    Returns:
        归一化后的图像
    """
    image = image.astype(np.float32)
    
    # 使用百分位数进行拉伸，避免极值影响
    p2, p98 = np.percentile(image, (2, 98))
    
    if p98 - p2 > 0:
        image = np.clip((image - p2) / (p98 - p2), 0, 1)
    else:
        # 如果所有值都相同，直接归一化
        img_min = image.min()
        img_max = image.max()
        if img_max - img_min > 0:
            image = (image - img_min) / (img_max - img_min)
        else:
            image = np.zeros_like(image)
    
    return image


def visualize_multiple_samples(dataset, num_samples=3):
    """
    可视化多个随机样本
    
    Args:
        dataset: OpticalSARDataset实例
        num_samples: 要可视化的样本数量
    """
    # 随机选择多个不重复的索引（使用 SystemRandom 以避免被全局 seed 影响）
    indices = random.SystemRandom().sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    for idx in indices:
        visualize_sample(dataset, idx)
        print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize WHU Optical-SAR dataset samples and segmentation outputs')
    parser.add_argument('--idx', type=int, default=None, help='样本索引，省略则随机')
    parser.add_argument('--save', type=str, default=None, help='保存图像路径（可选）')
    parser.add_argument('--num', type=int, default=1, help='可视化多个样本数量（若>1则随机选择）')
    parser.add_argument('--split', type=str, default='test', help='WHUOptSarPatchDataset 的 split')
    parser.add_argument('--root', type=str, default='../whu-opt-sar', help='WHU 数据集根目录')
    parser.add_argument('--seg_ckpt', type=str, default=None, help='分割模型 checkpoint 路径 (.pt)，若提供则可视化预测结果')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = WHUOptSarPatchDataset(root_dir=args.root, split=args.split)

    print(f"数据集总样本数: {len(dataset)}\n")

    model = None
    num_classes = len(CLASS_NAMES)
    if args.seg_ckpt is not None:
        print(f"加载分割模型权重: {args.seg_ckpt}")
        model, num_classes = load_segmentation_model(args.seg_ckpt, device)

    if args.num > 1:
        indices = random.sample(range(len(dataset)), min(args.num, len(dataset)))
        for i, idx in enumerate(indices):
            save_path = None
            if args.save:
                base, ext = os.path.splitext(args.save)
                save_path = f"{base}_{i}{ext}"
            visualize_sample(dataset, idx=idx, save_path=save_path, model=model, device=device, num_classes=num_classes)
            print("\n" + "="*80 + "\n")
    else:
        visualize_sample(dataset, idx=args.idx, save_path=args.save, model=model, device=device, num_classes=num_classes)
