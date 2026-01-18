import os
from typing import Optional, Callable, Tuple, List, Literal
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import rasterio
import random

class WHUOptSarPatchDataset(Dataset):
    """
    光学-SAR多模态遥感图像分割数据集，使用滑动窗口裁切扩充样本
    
    Args:
        root_dir: 数据集根目录
        split: 数据集划分，'train' 或 'val'
        train_ratio: 训练集比例，默认0.85
        patch_size: 裁切窗口大小，默认256
        stride_ratio: 步长相对于窗口大小的比例，默认0.9（即重叠10%）
        optical_dir: 光学图像子目录名
        sar_dir: SAR图像子目录名
        label_dir: 标签子目录名
        transform: 可选的数据增强变换
        random_seed: 随机种子，用于数据集划分
    """
    
    def __init__(
        self,
        root_dir: str,
        split: Literal['train', 'val'] = 'train',
        train_ratio: float = 0.8,
        patch_size: int = 256,
        stride_ratio: float = 0.9,
        optical_dir: str = 'optical',
        sar_dir: str = 'sar',
        label_dir: str = 'lbl',
        transform: Optional[Callable] = None,
        random_seed: int = 42,
        num_ratio: float = 1.0
    ):
        self.root_dir = root_dir
        self.optical_dir = os.path.join(root_dir, optical_dir)
        self.sar_dir = os.path.join(root_dir, sar_dir)
        self.label_dir = os.path.join(root_dir, label_dir)
        self.split = split
        self.patch_size = patch_size
        self.stride = int(patch_size * stride_ratio)
        self.transform = transform
        self.num_ratio = num_ratio #用一部分数据集

        
        # 原数据集的标签是0,10,20,...,70，将其映射到0,1,2,...,7
        self.label_mapping = {0: 0, 10: 1, 20: 2, 30: 3, 40: 4, 50: 5, 60: 6, 70: 7}
        
        # 获取所有文件
        all_files = self._get_file_list()
        
        # 划分训练集和验证集
        random.seed(random_seed)
        shuffled_files = all_files.copy()
        random.shuffle(shuffled_files)
        
        train_size = int(len(shuffled_files) * train_ratio)
        if split == 'train':
            self.image_files = shuffled_files[:train_size]
        else:
            self.image_files = shuffled_files[train_size:]
        
        # 生成所有patch的索引
        self.patches = self._generate_patches()
        
    def _get_file_list(self) -> List[str]:
        """获取所有图像文件名"""
        optical_files = set([f for f in os.listdir(self.optical_dir) if f.endswith('.tif')])
        sar_files = set([f for f in os.listdir(self.sar_dir) if f.endswith('.tif')])
        label_files = set([f for f in os.listdir(self.label_dir) if f.endswith('.tif')])
        common_files = optical_files & sar_files & label_files
        return sorted(list(common_files))
    
    def _generate_patches(self) -> List[Tuple[int, int, int]]:
        """
        生成所有patch的索引
        返回列表，每个元素为 (file_idx, row_start, col_start)
        """
        patches = []
        
        for file_idx, filename in enumerate(self.image_files):
            # 读取一个标签文件获取图像尺寸
            label_path = os.path.join(self.label_dir, filename)
            with rasterio.open(label_path) as src:
                height, width = src.height, src.width
            
            # 计算滑动窗口的起始位置
            row_starts = list(range(0, height - self.patch_size + 1, self.stride))
            col_starts = list(range(0, width - self.patch_size + 1, self.stride))
            
            # 确保覆盖到边界
            if row_starts[-1] + self.patch_size < height:
                row_starts.append(height - self.patch_size)
            if col_starts[-1] + self.patch_size < width:
                col_starts.append(width - self.patch_size)
            
            # 生成所有窗口位置
            for row in row_starts:
                for col in col_starts:
                    patches.append((file_idx, row, col))
            
        # 随机打乱
        shuffled_patches = patches.copy()
        random.shuffle(shuffled_patches)

        n = len(shuffled_patches)
        target = int(n * self.num_ratio)

        if target <= 0:
            return []

        # 当 num_ratio <= 1 时，直接截取子集（降采样或等于原始数量）
        if self.num_ratio <= 1.0:
            return shuffled_patches[:target]

        # 当 num_ratio > 1 时，需要扩增到目标数量。
        # 如果目标不超过原始数量，直接截取；否则使用有放回采样补足数量。
        if target <= n:
            return shuffled_patches[:target]

        # target > n: 使用有放回采样扩增到目标数量
        # 直接从打乱后的补丁列表中有放回采样 target 个
        expanded = random.choices(shuffled_patches, k=target)
        return expanded
    
    def __len__(self) -> int:
        return len(self.patches)
    
    def _read_tif(self, file_path: str) -> np.ndarray:
        """使用rasterio读取tif文件"""
        with rasterio.open(file_path) as src:
            image = src.read()
        return image
    
    def _read_patch(self, file_path: str, row: int, col: int) -> np.ndarray:
        """读取指定位置的patch"""
        with rasterio.open(file_path) as src:
            # 读取窗口：(col_off, row_off, width, height)
            window = ((row, row + self.patch_size), (col, col + self.patch_size))
            patch = src.read(window=window)
        return patch
    
    def _map_labels(self, label: np.ndarray) -> np.ndarray:
        """
        映射标签值：0,10,20,...,70 -> 0,1,2,...,7
        
        Args:
            label: 原始标签数组
            
        Returns:
            映射后的标签数组
        """
        mapped_label = np.zeros_like(label, dtype=np.int64)
        for old_val, new_val in self.label_mapping.items():
            mapped_label[label == old_val] = new_val
        return mapped_label
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取一个patch样本
        
        Returns:
            (optical_patch, sar_patch, label_patch):
            - optical_patch: 光学图像patch，shape (C, patch_size, patch_size)
            - sar_patch: SAR图像patch，shape (C, patch_size, patch_size)
            - label_patch: 标签patch，shape (patch_size, patch_size)
        """
        file_idx, row, col = self.patches[idx]
        filename = self.image_files[file_idx]
        
        # 构建文件路径
        optical_path = os.path.join(self.optical_dir, filename)
        sar_path = os.path.join(self.sar_dir, filename)
        label_path = os.path.join(self.label_dir, filename)
        
        # 读取patch
        optical_patch = self._read_patch(optical_path, row, col)
        sar_patch = self._read_patch(sar_path, row, col)
        label_patch = self._read_patch(label_path, row, col)
        
        # 如果标签只有一个通道，去掉通道维度
        if label_patch.shape[0] == 1:
            label_patch = label_patch[0]
        
        # 映射标签值：0,10,20,...,70 -> 0,1,2,...,7
        label_patch = self._map_labels(label_patch)
        
        # 转换为torch张量
        optical_patch = torch.from_numpy(optical_patch).float()
        sar_patch = torch.from_numpy(sar_patch).float()
        label_patch = torch.from_numpy(label_patch).long() if label_patch.ndim == 2 else torch.from_numpy(label_patch).float()
        
        # 应用数据增强
        if self.transform is not None:
            optical_patch, sar_patch, label_patch = self.transform(optical_patch, sar_patch, label_patch)
        
        return optical_patch, sar_patch, label_patch
    
