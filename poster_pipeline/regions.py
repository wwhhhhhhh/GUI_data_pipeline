"""连通域：从二值可写掩码提取多个区域，按面积与平均分数排序。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy import ndimage as ndi


@dataclass
class WritableRegion:
    label_id: int
    mask: np.ndarray  # bool HxW, True only this component
    bbox: Tuple[int, int, int, int]  # y0,y1,x0,x1 inclusive-exclusive? use y0,y1,x0,x1 as slice
    area: int
    mean_score: float


def label_components(binary_mask: np.ndarray) -> Tuple[np.ndarray, int]:
    m = np.asarray(binary_mask, dtype=bool)
    lbl, n = ndi.label(m)
    return lbl, int(n)


def extract_regions(score: np.ndarray, binary_mask: np.ndarray, *, top_k: int = 8) -> List[WritableRegion]:
    lbl, n = label_components(binary_mask)
    if n == 0:
        return []
    regions: List[WritableRegion] = []
    for lid in range(1, n + 1):
        comp = lbl == lid
        area = int(comp.sum())
        if area < 64:  # 太小跳过
            continue
        ys, xs = np.where(comp)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        ms = float(score[comp].mean()) if score.size else 0.0
        mask = comp
        regions.append(WritableRegion(lid, mask, (y0, y1, x0, x1), area, ms))
    regions.sort(key=lambda r: r.area * r.mean_score, reverse=True)
    return regions[:top_k]
