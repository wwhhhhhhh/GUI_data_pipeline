"""
自适应海报布局：基于连通域分析的多区域自动排版。

设计思路（参考 PosterLayout CVPR 2023 及图形设计三分法则）：

  1. 对可写字掩码做连通域标记（scipy.ndimage.label，8 连通）
  2. 对每个连通域计算几何特征：
       - 质心位置（相对坐标）→ 区域位置分类（top/bottom/left/right/center）
       - 外接矩形宽高比         → 文字方向（横排 h / 竖排 v）
       - 区域面积               → 字号层级（主标题/副标题/装饰）
       - 复杂度均值             → 质量分（越低频越适合写字）
  3. 综合评分排序，取 top-K 区域
  4. 按位置和方向确定对齐方式与扫描风格（传给 plan_layout）
  5. 字号按层级缩放：第 1 区 1.0，第 2 区 0.72，第 3 区 0.55

区域位置判断（三分法则，以质心相对坐标为准）：
  cy < 0.38            → top    （上部横排，居中，适合主标题）
  cy > 0.62            → bottom （下部横排，居中，适合品牌/副标）
  cx < 0.38            → left   （左侧竖排，左对齐）
  cx > 0.62            → right  （右侧竖排，右对齐）
  其余                  → center （中部横排，居中，权重较低）

文字方向判断（宽高比 aspect = bbox_w / bbox_h）：
  aspect > 1.5         → 横排 h（宽幅区）
  aspect < 0.75        → 竖排 v（细长区）
  0.75 ~ 1.5           → 看位置：left/right → v，其余 → h

评分公式：score = area_ratio × pos_weight × (1 - avg_complexity)
  pos_weight: top=1.3, bottom=1.1, left/right=1.0, center=0.6
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy import ndimage as ndi


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class TextZone:
    """一个可用于排版的图像区域。"""
    mask:       np.ndarray              # bool HxW，该区域内真正可写的像素
    direction:  str                     # "h" 横排 / "v" 竖排
    align:      str                     # "left" / "center" / "right"
    position:   str                     # "top"/"bottom"/"left"/"right"/"center"
    scan_style: str                     # 传给 plan_layout 的 style 参数
    score:      float                   # 综合质量分（越高越优先）
    font_scale: float = 1.0             # 字号缩放系数（由层级决定）
    bbox:       Tuple[int,int,int,int] = field(default_factory=lambda: (0,0,0,0))
    # (y0, y1, x0, x1)


# ---------------------------------------------------------------------------
# 主接口
# ---------------------------------------------------------------------------

_POS_WEIGHT = {"top": 1.3, "bottom": 1.1, "left": 1.0, "right": 1.0, "center": 0.6}
_POS_ALIGN  = {"top": "center", "bottom": "center", "left": "left", "right": "right", "center": "center"}
# 字号层级缩放：第 1~3 优先区
_FONT_SCALES = [1.0, 0.72, 0.55]


def find_text_zones(
    writable:       np.ndarray,
    complexity:     np.ndarray,
    *,
    min_area_ratio: float = 0.03,   # 区域至少占全图面积的比例
    max_zones:      int   = 3,      # 最多取几个区域
) -> List[TextZone]:
    """
    从可写字掩码中自动识别适合排版的区域列表，按综合质量分排序。

    参数：
      writable      — bool HxW，可写字区域（由 build_writable 得到）
      complexity    — float32 HxW [0,1]，复杂度图
      min_area_ratio— 区域最小面积（占全图比例）；过小的碎片忽略
      max_zones     — 最多使用的区域数（字号层级最多 3 级）

    返回：
      按 score 降序排列的 TextZone 列表，已设置 font_scale。
    """
    h, w = writable.shape

    # 8 连通标记（斜向相邻视为同一区域，减少碎片化）
    struct8 = np.ones((3, 3), dtype=bool)
    labeled, n_comp = ndi.label(writable, structure=struct8)

    if n_comp == 0:
        return []

    min_pixels = int(min_area_ratio * h * w)
    zones: List[TextZone] = []

    for cid in range(1, n_comp + 1):
        comp_mask = labeled == cid       # bool HxW，该连通域像素

        # ── 面积过滤 ──────────────────────────────────────────────────────
        area = int(comp_mask.sum())
        if area < min_pixels:
            continue

        area_ratio = area / (h * w)

        # ── 几何特征 ──────────────────────────────────────────────────────
        ys, xs = np.where(comp_mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1

        cy_rel = (y0 + y1) / 2.0 / h   # 质心相对 y（0=顶, 1=底）
        cx_rel = (x0 + x1) / 2.0 / w   # 质心相对 x（0=左, 1=右）
        bbox_h = y1 - y0
        bbox_w = x1 - x0
        aspect = bbox_w / max(bbox_h, 1)  # > 1=宽, < 1=高

        # ── 位置分类（三分法则） ─────────────────────────────────────────
        if cy_rel < 0.38:
            position = "top"
        elif cy_rel > 0.62:
            position = "bottom"
        elif cx_rel < 0.38:
            position = "left"
        elif cx_rel > 0.62:
            position = "right"
        else:
            position = "center"

        # ── 文字方向（宽高比 + 位置综合判断） ───────────────────────────
        if aspect > 1.5:
            direction = "h"
        elif aspect < 0.75:
            direction = "v"
        else:
            # 中间段：left/right 区域倾向竖排，其余横排
            direction = "v" if position in ("left", "right") else "h"

        # ── plan_layout 扫描风格 ─────────────────────────────────────────
        if direction == "v":
            scan_style = "v_left" if position == "left" else "v_right"
        elif position == "bottom":
            scan_style = "h_bottom"
        elif position == "center":
            scan_style = "h_center"
        else:
            scan_style = "h_top"

        # ── 对齐方式 ─────────────────────────────────────────────────────
        align = _POS_ALIGN[position]

        # ── 综合质量分 ───────────────────────────────────────────────────
        avg_complexity = float(complexity[comp_mask].mean())
        score = area_ratio * _POS_WEIGHT[position] * (1.0 - avg_complexity)

        zones.append(TextZone(
            mask=comp_mask,
            direction=direction,
            align=align,
            position=position,
            scan_style=scan_style,
            score=score,
            bbox=(y0, y1, x0, x1),
        ))

    # ── 按评分排序，取 top-K ─────────────────────────────────────────────
    zones.sort(key=lambda z: z.score, reverse=True)
    zones = zones[:max_zones]

    # ── 字号层级：主标题 1.0 → 副标题 0.72 → 装饰 0.55 ─────────────────
    for i, z in enumerate(zones):
        z.font_scale = _FONT_SCALES[min(i, len(_FONT_SCALES) - 1)]

    return zones
