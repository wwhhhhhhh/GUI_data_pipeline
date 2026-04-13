"""区域内中位色 + 简单反差字色（亮底深字 / 暗底亮字）。"""
from __future__ import annotations

from typing import Tuple

import numpy as np


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
