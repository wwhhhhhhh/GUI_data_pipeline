"""
可写字区域：主体禁止层 + 图像复杂度惩罚 + 二值化。
使用 OpenCV 加速卷积与形态学（2K 图像下比 scipy 快 5–10×）。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)


def _sobel_mag(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)
    return cv2.magnitude(gx, gy)


def _box_blur(x: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return x
    k = 2 * r + 1
    return cv2.blur(x, (k, k), borderType=cv2.BORDER_REFLECT)


def _lap_energy(gray: np.ndarray, win: int = 5) -> np.ndarray:
    """小窗口拉普拉斯绝对值平滑，作为纹理强度代理。"""
    lap_k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    lap = np.abs(cv2.filter2D(gray, cv2.CV_32F, lap_k, borderType=cv2.BORDER_REFLECT))
    return _box_blur(lap, win // 2)


def _gaussian(x: np.ndarray, sigma: float, truncate: float = 3.0) -> np.ndarray:
    ksize = int(2 * (sigma * truncate) + 1) | 1
    return cv2.GaussianBlur(x, (ksize, ksize), sigma, borderType=cv2.BORDER_REFLECT)


def _normalize01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
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

    - forbid_labels 为 None：合并列表中**全部** mask。
    - forbid_labels 非空：仅合并 label 属于该集合的 mask。
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


_STRUCT_3x3 = np.ones((3, 3), dtype=np.uint8)


def dilate_binary(mask: np.ndarray, iterations: int = 3) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if iterations <= 0:
        return m
    out = cv2.dilate(m.astype(np.uint8), _STRUCT_3x3, iterations=iterations)
    return out.astype(bool)


def complexity_map(
    rgb: np.ndarray,
    *,
    smooth_sigma: float = 0.0,
    lap_win: int = 9,
) -> np.ndarray:
    """
    计算全图区域级复杂度，值域 [0, 1]。

    Sobel 梯度 + Laplacian 能量，各自大高斯平滑后加权融合归一化。
    """
    h, w = rgb.shape[:2]
    sigma = smooth_sigma if smooth_sigma > 0 else float(
        max(12, min(40, min(h, w) // 16))
    )

    gray = rgb_to_gray(rgb)

    gmag = _sobel_mag(gray)
    gmag_s = _gaussian(gmag, sigma)

    lap = _lap_energy(gray, win=lap_win)
    lap_s = _gaussian(lap, sigma)

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
    comp_dilate_iter: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    一步计算最终可写字掩码及复杂度图。

    Returns：
      writable  : bool HxW
      comp      : float32 HxW [0,1]
      subj_mask : bool HxW
      forb_mask : bool HxW
    """
    subj = union_subject_masks(masks, h=h, w=w,
                               forbid_labels=forbid_labels, label_key=label_key)
    forb = dilate_binary(subj, iterations=dilate_iter)
    comp = complexity_map(rgb)
    comp_low = (comp <= complexity_thresh).astype(np.uint8)
    if comp_dilate_iter > 0:
        comp_low = cv2.morphologyEx(
            comp_low, cv2.MORPH_CLOSE, _STRUCT_3x3,
            iterations=comp_dilate_iter,
        )
    writable = (~forb) & comp_low.astype(bool)
    return writable, comp, subj, forb
