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
    blur_r: int = 8,
    lap_win: int = 5,
) -> np.ndarray:
    gray = rgb_to_gray(rgb)
    gmag = _sobel_mag(gray)
    gmag = _box_blur(gmag, blur_r)
    lap = _lap_energy(gray, win=lap_win)
    c = _normalize01(gmag) * 0.6 + _normalize01(lap) * 0.4
    return np.clip(c, 0.0, 1.0)


def build_writable_score(
    rgb: np.ndarray,
    subject_union: np.ndarray,
    *,
    dilate_iter: int = 12,
    complexity_gamma: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      score: float32 HxW in [0,1],越大越适合写字
      complexity: float32 HxW in [0,1]

    complexity_gamma=0 表示完全不考虑图像复杂度，score = 1(自由区) / 0(主体禁区)。
    dilate_iter 越大，主体周围留白越多，文字越不会贴边。
    """
    forb = dilate_binary(subject_union, iterations=dilate_iter)
    comp = complexity_map(rgb)
    free = (~forb).astype(np.float32)
    if complexity_gamma == 0.0:
        score = free
    else:
        score = free * (1.0 - comp) ** complexity_gamma
    return score.astype(np.float32), comp.astype(np.float32)


def threshold_writable(score: np.ndarray, quantile: float = 0.65) -> np.ndarray:
    thr = float(np.quantile(score.reshape(-1), quantile))
    return score >= thr
