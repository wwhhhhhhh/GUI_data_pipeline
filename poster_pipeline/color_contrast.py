"""区域内中位色 + 基于色卡的反差字色选择。"""
from __future__ import annotations

import random
from typing import Dict, Optional, Tuple

import numpy as np

from .color_palette import pick_contrast_color


def rgb_median(rgb: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float]:
    m = np.asarray(mask, dtype=bool)
    px = rgb[m]
    if len(px) == 0:
        return 128.0, 128.0, 128.0
    return tuple(float(np.median(px[:, i])) for i in range(3))


def relative_luminance(r: float, g: float, b: float) -> float:
    """sRGB 近似亮度0..1"""
    def _lin(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    R, G, B = _lin(r), _lin(g), _lin(b)
    return 0.2126 * R + 0.7152 * G + 0.0722 * B


def contrast_text_rgb(rgb: np.ndarray, mask: np.ndarray) -> Tuple[int, int, int, int, int, int]:
    """
    Returns (fg_r,fg_g,fg_b, bg_r,bg_g,bg_b)  —— bg 用中位色代表；fg 为黑或白中对比更大者。
    """
    br, bg, bb = rgb_median(rgb, mask)
    y = relative_luminance(br, bg, bb)
    if y > 0.45:
        fg = (20, 24, 28)
    else:
        fg = (248, 250, 252)
    return (*fg, int(br), int(bg), int(bb))


def contrast_from_palette(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    rng: Optional[random.Random] = None,
    anchor_id: Optional[int] = None,
    min_ratio: float = 4.5,
    top_k: int = 8,
) -> Dict:
    """
    基于色卡的对比色挑选。

    Args:
      rgb       — 源图 HxWx3 uint8
      mask      — 区域 mask（bool HxW），用来采样区域内中位色作为"背景色"
      rng       — 随机源（保证可复现）
      anchor_id — 若给定则优先复用该 id（保证一张图内颜色一致），
                  只有当该 id 与当前区域背景对比度仍 ≥ min_ratio 时才沿用；
                  否则重新挑一个并返回新 id。
      min_ratio — 最低 WCAG 对比度（默认 4.5，常规正文阅读门槛）
      top_k     — 从对比度最高的 top_k 个里随机挑

    Returns: dict {id, hex, rgb, name, name_zh, name_en, contrast, bg_rgb}
    """
    rng = rng or random.Random()
    br, bg_v, bb = rgb_median(rgb, mask)
    bg_rgb = (br, bg_v, bb)

    if anchor_id is not None:
        from .color_palette import PALETTE, contrast_ratio as _cr
        anchor = next((p for p in PALETTE if p["id"] == anchor_id), None)
        if anchor is not None:
            ratio = _cr(anchor["rgb"], bg_rgb)
            if ratio >= min_ratio:
                from .color_palette import _annotate
                out = _annotate(anchor, rng, ratio)
                out["bg_rgb"] = (int(br), int(bg_v), int(bb))
                return out

    chosen = pick_contrast_color(bg_rgb, rng, min_ratio=min_ratio, top_k=top_k)
    chosen["bg_rgb"] = (int(br), int(bg_v), int(bb))
    return chosen
