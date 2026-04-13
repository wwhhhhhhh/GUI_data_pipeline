"""
可写字区域：主体禁止层 + 图像复杂度惩罚 + 二值化。
仅依赖 numpy；若有 scipy 则用更快的形态学。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage as _ndi
from scipy.signal import convolve2d


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)


def _sobel_mag(gray: np.ndarray) -> np.ndarray:
    g = gray
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = convolve2d(g, kx, mode="same", boundary="symm")
    gy = convolve2d(g, ky, mode="same", boundary="symm")
    return np.sqrt(gx * gx + gy * gy)


def _box_blur(x: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return x
    k = 2 * r + 1
    kernel = np.ones((k, k), dtype=np.float32) / (k * k)
    return convolve2d(x, kernel, mode="same", boundary="symm")


def _lap_energy(gray: np.ndarray, win: int = 5) -> np.ndarray:
    """小窗口拉普拉斯绝对值平滑，作为纹理强度代理。"""
    lap_k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    lap = np.abs(convolve2d(gray, lap_k, mode="same", boundary="symm"))
    return _box_blur(lap, win // 2)


def _normalize01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # 用 0~99th percentile：保留绝对大小关系
    # 天空/草地（边缘弱）→ 低值；树枝（边缘强）→ 高值
    lo = float(x.min())
    hi = float(np.percentile(x, 99))
    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def union_subject_masks(
    masks: List[dict],
    *,
    h: int,
    w: int,
    forbid_labels: Optional[set] = None,
    label_key: str = "label",
) -> np.ndarray:
    """
    masks: [{"mask": bool (H,W), "label": str, ...}, ...]

    - forbid_labels 为 None：合并列表中**全部** mask（适用于列表里只有“主体”实例）。
    - forbid_labels 非空：仅合并 label 属于该集合的 mask（如 person/animal）。
    """
    acc = np.zeros((h, w), dtype=bool)
    for m in masks:
        mk = m.get("mask")
        if mk is None:
            continue
        mk = np.asarray(mk, dtype=bool)
        if mk.shape != (h, w):
            raise ValueError(f"mask shape {mk.shape} != {(h, w)}")
        if forbid_labels is not None:
            lb = m.get(label_key)
            if lb not in forbid_labels:
                continue
        acc |= mk
    return acc


def dilate_binary(mask: np.ndarray, iterations: int = 3) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if iterations <= 0:
        return m
    struct = np.ones((3, 3), dtype=bool)
    return _ndi.binary_dilation(m, structure=struct, iterations=iterations)


def complexity_map(
    rgb: np.ndarray,
    *,
    smooth_sigma: float = 0.0,   # 0 = 自动（图像短边的 1/8）
    lap_win: int = 21,            # 拉普拉斯窗口，比之前 5 大，捕捉更宽纹理
) -> np.ndarray:
    """
    计算全图区域级复杂度，值域 [0, 1]。

    核心思路：
      1. Sobel 梯度幅值 + 大半径高斯平滑（sigma ≈ 图像短边 /8，约 40~80px）
         → 将边缘能量扩散到区域尺度
      2. Laplacian 能量 + 同样的大平滑
         → 捕捉纹理密度
      3. 加权融合 → 归一化

    效果：天空/地板等大片低纹理区 → 接近 0（适合写字）
          工具台/树枝等密集纹理区 → 接近 1（不适合写字）
    """
    h, w = rgb.shape[:2]
    # 平滑半径自适应图像尺寸：短边 1/8，最小 20px，最大 80px
    sigma = smooth_sigma if smooth_sigma > 0 else float(
        max(20, min(80, min(h, w) // 8))
    )

    gray = rgb_to_gray(rgb)

    # Sobel 梯度 → 大高斯平滑
    gmag = _sobel_mag(gray)
    gmag_s = _ndi.gaussian_filter(gmag, sigma=sigma)

    # Laplacian 能量 → 大高斯平滑（lap_win 比之前大，覆盖更多纹理）
    lap = _lap_energy(gray, win=lap_win)
    lap_s = _ndi.gaussian_filter(lap, sigma=sigma)

    c = _normalize01(gmag_s) * 0.6 + _normalize01(lap_s) * 0.4
    return np.clip(c, 0.0, 1.0).astype(np.float32)


def build_writable(
    rgb: np.ndarray,
    masks: List[dict],
    *,
    h: int,
    w: int,
    forbid_labels: Optional[set] = None,
    label_key: str = "label",
    dilate_iter: int = 14,
    complexity_thresh: float = 0.50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    一步计算最终可写字掩码及复杂度图。

    流程：
      1. 合并主体 mask → 形态学膨胀 → 主体禁区 forb
      2. 计算区域级复杂度图 comp（大高斯平滑，sigma≈短边/8）
      3. 可写区域 = (~forb) & (comp <= complexity_thresh)
         若 masks 为空，forb 全为 False，可写区域仅由复杂度决定。

    Returns:
      writable  : bool HxW，True = 可写字
      comp      : float32 HxW [0,1]，0=低复杂/适合写字，1=高复杂/不适合
      subj_mask : bool HxW，True = 主体区域（膨胀前）
    """
    subj = union_subject_masks(masks, h=h, w=w,
                               forbid_labels=forbid_labels, label_key=label_key)
    forb = dilate_binary(subj, iterations=dilate_iter)
    comp = complexity_map(rgb)
    writable = (~forb) & (comp <= complexity_thresh)
    return writable, comp, subj
