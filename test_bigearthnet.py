#!/usr/bin/env python3
"""
测试 BigEarthNetDataset：随机抽取一条样本，打印形状并保存 s2 的 B/G/R 通道为图片。

使用方法（在仓库根目录或任意位置都可执行）：
    python3 CROMA/test_bigearthnet.py

数据根目录已写死为 `/home/featurize/data`，若需修改请编辑本文件。
"""
import os
import sys
import random
import numpy as np
from PIL import Image
from typing import Tuple

# 确保能导入同目录的 datasets.py
sys.path.insert(0, os.path.dirname(__file__))
from datasets import BigEarthNetDataset


def normalize_channel(ch: np.ndarray) -> np.ndarray:
    # 使用 2-98 百分位做线性拉伸并映射到 0-255
    p2, p98 = np.percentile(ch, (2, 98))
    ch = (ch - p2) / (p98 - p2 + 1e-8)
    ch = np.clip(ch, 0.0, 1.0)
    return (ch * 255.0).astype(np.uint8)


def ensure_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.array(x)


PALETTE_19 = [
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    (0, 128, 128),
    (192, 192, 192),
    (128, 128, 128),
    (255, 0, 0),
    (0, 255, 0),
    (255, 255, 0),
    (0, 0, 255),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 255),
    (255, 0, 128),
]


def target_to_label_map(target: np.ndarray) -> np.ndarray:
    t = ensure_numpy(target)
    # Squeeze singleton dims
    t = np.squeeze(t)
    if t.ndim == 2:
        return t.astype(np.int32)
    if t.ndim == 3:
        # possible shapes: (C, H, W) or (H, W, C)
        c0, c1, c2 = t.shape
        if c0 == 19:
            # (C,H,W)
            return np.argmax(t, axis=0).astype(np.int32)
        if c2 == 19:
            # (H,W,C)
            return np.argmax(t, axis=2).astype(np.int32)
        # fallback: if channels small, take argmax along axis 0
        return np.argmax(t, axis=0).astype(np.int32)
    raise ValueError('Unsupported target shape: ' + str(t.shape))


def label_map_to_rgb(label_map: np.ndarray, palette: Tuple[Tuple[int, int, int]] = None) -> np.ndarray:
    if palette is None:
        palette = PALETTE_19
    h, w = label_map.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for i, col in enumerate(palette):
        mask = label_map == i
        if not mask.any():
            continue
        out[mask] = col
    return out


def to_pil(img_arr: np.ndarray) -> Image.Image:
    return Image.fromarray(img_arr)


def main():
    root = 'F:\数据集\BigEarthNet'

    print('Initializing dataset from', root)
    ds = BigEarthNetDataset(root=root, split='train', ratio=1.0, seed=42)

    n = len(ds)
    print('Dataset length:', n)
    if n == 0:
        print('数据集为空，检查 /home/featurize/data 是否存在并包含 metadata.parquet')
        return

    # 使用 metadata 的第一行（固定为 0）以便复现
    idx = 3652
    print('Using first index (0) from metadata')

    sample = ds[idx]
    # 返回 (s2, s1, target)
    s2, s1, target = sample

    print('s2.shape =', getattr(s2, 'shape', None))
    print('s1.shape =', getattr(s1, 'shape', None))
    print('target.shape =', getattr(target, 'shape', None))

    # 若为 torch.Tensor -> 转为 numpy
    try:
        import torch
        if isinstance(s2, torch.Tensor):
            s2 = s2.numpy()
    except Exception:
        pass

    # s2 expected shape: (C, H, W)
    if s2.ndim != 3 or s2.shape[0] < 3:
        print('s2 形状不符合预期，跳过保存图片')
        return

    # 提取 B(0), G(1), R(2)
    b = normalize_channel(s2[0])
    g = normalize_channel(s2[1])
    r = normalize_channel(s2[2])

    rgb = np.stack([r, g, b], axis=2)

    # 处理 s1：取第一个通道并归一化到 0-255
    try:
        if isinstance(s1, np.ndarray):
            s1_arr = s1
        else:
            s1_arr = ensure_numpy(s1)
    except Exception:
        s1_arr = ensure_numpy(s1)

    if s1_arr.ndim == 3:
        # (C,H,W) -> first channel
        s1_ch = s1_arr[0]
    elif s1_arr.ndim == 2:
        s1_ch = s1_arr
    else:
        s1_ch = np.squeeze(s1_arr)

    s1_gray = normalize_channel(s1_ch.astype(np.float32))

    # 处理 target -> 标签映射 -> 彩色图
    try:
        label_map = target_to_label_map(target)
    except Exception:
        label_map = target_to_label_map(ensure_numpy(target))

    target_rgb = label_map_to_rgb(label_map)

    # 转为 PIL 图像并保证大小一致（使用 s2 的尺寸）
    pil_s2 = to_pil(rgb)
    pil_s1 = to_pil(s1_gray)
    pil_target = to_pil(target_rgb)

    # 统一尺寸
    w, h = pil_s2.size
    pil_s1 = pil_s1.resize((w, h), resample=Image.BILINEAR)
    pil_target = pil_target.resize((w, h), resample=Image.NEAREST)

    # 合并为一张横向对比图
    spacing = 4
    out_img = Image.new('RGB', (w * 3 + spacing * 2, h), (255, 255, 255))
    out_img.paste(pil_s2, (0, 0))
    out_img.paste(pil_s1.convert('RGB'), (w + spacing, 0))
    out_img.paste(pil_target, ((w + spacing) * 2 - spacing, 0))

    out_path = os.path.join(os.path.dirname(__file__), 'image', 's2_s1_target_comparison.png')
    out_path_s2 = os.path.join(os.path.dirname(__file__), 'image', 's2.png')
    out_path_s1 = os.path.join(os.path.dirname(__file__), 'image', 's1.png')
    out_img.save(out_path)
    pil_s2.save(out_path_s2)
    pil_s1.save(out_path_s1)
    print('Saved comparison image to', out_path)


if __name__ == '__main__':
    main()
