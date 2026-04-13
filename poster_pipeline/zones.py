"""
基于主体位置的海报分区（Zone）。

设计思路：
  1. 从禁区 mask（主体 + dilation）得到主体 bounding box
  2. 以主体 bbox 为中心，划分上/下/左/右等候选区（原子 zone）
  3. 支持将相邻原子 zone 合并为"复合 zone"（如 top_wide = top_left ∪ top ∪ top_right）
  4. 用复杂度图给每个区打质量分（低频 → 高分 → 天空/草地 > 树枝）
  5. 按 style 从候选区里选出目标区，返回 (name, mask, direction, quality)
  6. 每个 zone 独立分配完整语料，非连通区域各自写字

zone direction:
  "h" → 横排（适合宽区：top/bottom）
  "v" → 竖排（适合窄区：left/right）
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 原子 zone 的 8 方位（以主体 bbox 为参照）
# ---------------------------------------------------------------------------
#
#  ┌──────────┬──────────┬──────────┐
#  │ top_left │   top    │ top_right│
#  ├──────────┼──────────┼──────────┤
#  │  left    │ [主体]   │  right   │
#  ├──────────┼──────────┼──────────┤
#  │ bot_left │ bottom   │ bot_right│
#  └──────────┴──────────┴──────────┘
#
# ---------------------------------------------------------------------------

# 复合 zone：多个原子 zone 的 mask 取并集，形成更大的连续区域
COMPOSITE_ZONES: Dict[str, List[str]] = {
    "top_wide":   ["top_left", "top", "top_right"],    # 顶部全幅（横跨主体上方）
    "bot_wide":   ["bot_left", "bottom", "bot_right"], # 底部全幅
    "left_tall":  ["top_left", "left", "bot_left"],    # 左侧全高
    "right_tall": ["top_right", "right", "bot_right"], # 右侧全高
    "top_half_l": ["top_left", "top"],                 # 顶部左半（偏左上）
    "top_half_r": ["top", "top_right"],                # 顶部右半（偏右上）
    "bot_half_l": ["bot_left", "bottom"],              # 底部左半
    "bot_half_r": ["bottom", "bot_right"],             # 底部右半
}

# ---------------------------------------------------------------------------
# 风格 → 期望分区顺序（每项: zone名, 文字方向）
# ---------------------------------------------------------------------------
POSTER_STYLES: Dict[str, List[Tuple[str, str]]] = {
    # ── 经典原子 zone 样式 ───────────────────────────────────────────────────
    "top_title":    [("top",        "h"), ("bottom",    "h")],
    "sidebar_r":    [("top",        "h"), ("right",     "v"), ("bottom",    "h")],
    "sidebar_l":    [("top",        "h"), ("left",      "v"), ("bottom",    "h")],
    "split":        [("left",       "v"), ("right",     "v")],
    "bottom_cap":   [("bottom",     "h")],
    "frame":        [("top",        "h"), ("left",      "v"),
                     ("right",      "v"), ("bottom",    "h")],
    "diagonal":     [("top_left",   "h"), ("bot_right", "h")],
    "diagonal_alt": [("top_right",  "h"), ("bot_left",  "h")],

    # ── 复合 zone 样式（多原子合并，区域更大，视觉更丰富）────────────────
    "top_banner":   [("top_wide",   "h")],                      # 顶部全幅大标题
    "bot_banner":   [("bot_wide",   "h")],                      # 底部全幅横排
    "h_banner":     [("top_wide",   "h"), ("bot_wide",   "h")], # 上下全幅横排
    "v_columns":    [("left_tall",  "v"), ("right_tall", "v")], # 左右全高竖排
    "l_shape_l":    [("top_wide",   "h"), ("left",       "v")], # 宽顶 + 左竖（L 形）
    "l_shape_r":    [("top_wide",   "h"), ("right",      "v")], # 宽顶 + 右竖（Γ 形）
    "frame_wide":   [("top_wide",   "h"), ("left_tall",  "v"),
                     ("right_tall", "v"), ("bot_wide",   "h")], # 全边框宽幅
    "diag_wide":    [("top_half_l", "h"), ("bot_half_r", "h")], # 左上+右下宽对角
    "diag_wide_r":  [("top_half_r", "h"), ("bot_half_l", "h")], # 右上+左下宽对角
    "sidebar_wide": [("top_wide",   "h"), ("right_tall", "v")], # 全幅顶部 + 右全高
}

STYLE_NAMES = list(POSTER_STYLES.keys())


# ---------------------------------------------------------------------------
# 主体定位
# ---------------------------------------------------------------------------

def locate_subject(forb_mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """返回禁区（主体 + dilation）的 bounding box (y0, y1, x0, x1)。"""
    ys, xs = np.where(forb_mask)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


# ---------------------------------------------------------------------------
# Zone 矩形定义（原子 zone）
# ---------------------------------------------------------------------------

def _zone_rect(
    name: str, h: int, w: int,
    sy0: int, sy1: int, sx0: int, sx1: int,
) -> Optional[Tuple[int, int, int, int]]:
    """
    根据主体 bbox 返回指定原子 zone 的矩形 (y0, y1, x0, x1)。
    """
    table = {
        "top":       (0,   sy0, 0,   w  ),
        "bottom":    (sy1, h,   0,   w  ),
        "left":      (0,   h,   0,   sx0),
        "right":     (0,   h,   sx1, w  ),
        "top_left":  (0,   sy0, 0,   sx0),
        "top_right": (0,   sy0, sx1, w  ),
        "bot_left":  (sy1, h,   0,   sx0),
        "bot_right": (sy1, h,   sx1, w  ),
    }
    r = table.get(name)
    if r is None:
        return None
    y0, y1, x0, x1 = r
    return (y0, y1, x0, x1) if y1 > y0 and x1 > x0 else None


# ---------------------------------------------------------------------------
# 复合 zone mask 构建（原子 zone ∪ 复合 zone 统一入口）
# ---------------------------------------------------------------------------

def _build_zone_mask(
    zone_name: str, h: int, w: int,
    sy0: int, sy1: int, sx0: int, sx1: int,
    forb_mask: np.ndarray,
) -> np.ndarray:
    """
    构建 zone 的可写像素 mask（True = 可写）。
    支持原子 zone（top/bottom/left/right/四角）和复合 zone（COMPOSITE_ZONES）。
    复合 zone 取各原子 zone 的非禁区像素的并集。
    """
    zm = np.zeros((h, w), dtype=bool)
    atomics = COMPOSITE_ZONES.get(zone_name, [zone_name])
    for atomic in atomics:
        rect = _zone_rect(atomic, h, w, sy0, sy1, sx0, sx1)
        if rect is None:
            continue
        y0, y1, x0, x1 = rect
        zm[y0:y1, x0:x1] |= ~forb_mask[y0:y1, x0:x1]
    return zm


# ---------------------------------------------------------------------------
# 复杂度品质打分
# ---------------------------------------------------------------------------

def _zone_quality(complexity: np.ndarray, zone_mask: np.ndarray) -> float:
    """
    zone 在复杂度图上的质量分：mean(1 - complexity) over zone free pixels。
    分越高表示该区越"低频"，越适合写字。
    """
    if not zone_mask.any():
        return 0.0
    return float((1.0 - complexity)[zone_mask].mean())


# ---------------------------------------------------------------------------
# 主接口
# ---------------------------------------------------------------------------

def build_zones(
    h: int,
    w: int,
    forb_mask: np.ndarray,
    complexity: np.ndarray,
    style: str,
    *,
    raw_subj_mask: Optional[np.ndarray] = None,
    min_area_ratio: float = 0.03,   # zone 有效像素至少占全图比例
    min_quality: float = 0.30,      # zone 质量下限（低于此 = 太复杂，跳过）
) -> List[Tuple[str, np.ndarray, str, float]]:
    """
    按 style 返回可用 zone 列表，每项：
      (zone_name, zone_mask_bool HxW, direction "h"/"v", quality_score)

    zone_mask：该 zone 矩形内的非禁区像素（True = 可写）。
    已按 quality 过滤低频不足的区域。

    支持 COMPOSITE_ZONES 中定义的复合 zone（多原子 zone 合并）。

    raw_subj_mask: 未膨胀的主体 mask，用于计算 bbox（zone 边界贴近主体轮廓）。
                   像素可写性仍然由 forb_mask（已膨胀）决定。
                   不传则用 forb_mask 本身计算 bbox。
    """
    # bbox 用 raw mask 定义（zone 边界紧贴主体原始轮廓，不受 dilation 压缩）
    bbox = locate_subject(raw_subj_mask if raw_subj_mask is not None else forb_mask)
    style_def = POSTER_STYLES.get(style, POSTER_STYLES["top_title"])
    result: List[Tuple[str, np.ndarray, str, float]] = []

    if bbox is None:
        # 没有主体：以图像中心 1/6 边距处为虚拟主体 bbox，让 style 分区正常工作
        # 虚拟 bbox 使各方向 zone 各占约 1/6~1/3 图像，forb_mask 此时全为 False（无禁区）
        cy, cx = h // 2, w // 2
        my, mx = max(1, h // 6), max(1, w // 6)
        bbox = (cy - my, cy + my, cx - mx, cx + mx)

    sy0, sy1, sx0, sx1 = bbox

    # "像素占用"机制：按 style_def 顺序处理 zone，每个 zone 自动去掉
    # 已被前置 zone 占用的像素，从根本上消除跨 zone 文字重叠。
    # 注意：只有通过质量检测、真正被使用的 zone 才占用像素；
    # 质量不达标的 zone 不占用像素，留给后续 zone 或 fallback 使用。
    claimed = np.zeros((h, w), dtype=bool)

    for zone_name, direction in style_def:
        zm = _build_zone_mask(zone_name, h, w, sy0, sy1, sx0, sx1, forb_mask)
        zm = zm & ~claimed        # 去除已被前序（已采用）zone 的像素

        if not zm.any():
            continue

        area_ratio = zm.sum() / (h * w)
        if area_ratio < min_area_ratio:
            continue   # zone 太小，不占用

        quality = _zone_quality(complexity, zm)
        if quality < min_quality:
            continue   # zone 太复杂，不占用，留给后续 zone

        claimed |= zm             # 确认采用，才标记为已占用
        result.append((zone_name, zm, direction, quality))

    return result


# ---------------------------------------------------------------------------
# Fallback：当 style 的 zone 全部失效时，尝试备用 style 序列
# ---------------------------------------------------------------------------

_FALLBACK_STYLES = [
    "top_title", "sidebar_r", "sidebar_l",
    "bottom_cap", "split", "top_banner", "bot_banner",
]


def build_zones_with_fallback(
    h: int,
    w: int,
    forb_mask: np.ndarray,
    complexity: np.ndarray,
    style: str,
    *,
    raw_subj_mask: Optional[np.ndarray] = None,
    min_area_ratio: float = 0.03,
    min_quality: float = 0.30,
) -> List[Tuple[str, np.ndarray, str, float]]:
    """
    先用 style 请求的 zone，若一个都找不到则依次尝试备用 style，
    最终兜底：降质量门槛（0.15），取质量最好的 zone。
    """
    zones = build_zones(
        h, w, forb_mask, complexity, style,
        raw_subj_mask=raw_subj_mask,
        min_area_ratio=min_area_ratio,
        min_quality=min_quality,
    )
    if zones:
        return zones

    # 备用 style 序列
    for fb_style in _FALLBACK_STYLES:
        if fb_style == style:
            continue
        zones = build_zones(
            h, w, forb_mask, complexity, fb_style,
            raw_subj_mask=raw_subj_mask,
            min_area_ratio=min_area_ratio,
            min_quality=min_quality,
        )
        if zones:
            return zones

    # 最终兜底：降低质量门槛，全图 zone
    zm = ~forb_mask
    if zm.any():
        q = _zone_quality(complexity, zm)
        return [("full", zm, "h", q)]

    return []


# ---------------------------------------------------------------------------
# 语料分配（辅助工具，在 pipeline 外部使用时仍可调用）
# ---------------------------------------------------------------------------

def split_corpus(corpus: str, n: int) -> List[str]:
    """
    把语料按空格分词后，尽量均匀分成 n 份。
    若词数不足 n，后续份为空串。
    """
    if n <= 0:
        return []
    words = corpus.split()
    if not words:
        return [""] * n
    size = max(1, (len(words) + n - 1) // n)
    parts = []
    for i in range(n):
        chunk = words[i * size: (i + 1) * size]
        parts.append(" ".join(chunk))
    while len(parts) < n:
        parts.append("")
    return parts[:n]
