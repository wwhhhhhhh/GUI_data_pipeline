"""
自适应海报布局：全图分析 → 策略打分 → 选最优布局。

核心思路（区别于连通域方案）：
  1. 预定义 11 种候选排版策略（上下、左右、中心、全边框、L 形等），
     每种策略由 1~4 个矩形区域组成。
  2. 对每种策略，计算其各区域内可写字像素的"密度 × 绝对面积"综合得分。
  3. 选得分最高的策略 → 确定最终布局。
  4. 在策略区域内与 writable mask 取交，得到实际可用像素，
     交给 plan_layout 做扫描线排版。
  5. 按可用面积降序给各区分配字号层级（1.0 → 0.72 → 0.55 → 0.45）。

策略选择原理（举例）：
  · 主体在图像正中（人像、画作）→ 上下各有大片留白 → top_bottom 得高分
  · 画框四周花边复杂，中心清晰 → center 得高分
  · 横幅图像主体偏中，两侧细长留白 → left_right 得高分
  · 主体偏下，顶部大片天空 → top_only 得高分

评分公式（单区）：
  zone_score = density × abs_frac
    density  = writable_px_in_zone / zone_area       ∈ [0,1]
    abs_frac = writable_px_in_zone / (H × W)         ∈ [0,1]
  strategy_score = mean(zone_score_i)

  density 体现"该区有多纯净/低噪"，abs_frac 体现"绝对可写面积大不大"，
  两者相乘后：大片干净区域 >> 小片干净区域 >> 大片复杂区域。

参考：PosterLayout CVPR 2023（DSF 设计序列）、三分法则
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class TextZone:
    """一个已确定方向与对齐方式的排版区域。"""
    mask:       np.ndarray                           # bool HxW，可写像素
    direction:  str                                  # "h" 横排 / "v" 竖排
    align:      str                                  # "left"/"center"/"right"
    position:   str                                  # 区域标签（top/bottom/left/…）
    scan_style: str                                  # 传给 plan_layout 的 style
    score:      float                                # 质量得分（越高越优先）
    font_scale: float = 1.0                          # 字号缩放（由层级决定）
    bbox: Tuple[int, int, int, int] = field(
        default_factory=lambda: (0, 0, 0, 0)
    )                                                # (y0, y1, x0, x1)


# ---------------------------------------------------------------------------
# 策略定义
# ---------------------------------------------------------------------------
# 每个策略 = 若干区域定义（相对坐标，区域之间设计为互不重叠）。
# y0r/y1r/x0r/x1r ∈ [0,1]，最终乘以图像 H/W 得到像素坐标。
#
# 区域方向由 scan 字段隐含：v_left / v_right → 竖排；h_* → 横排。

_ZD = Dict[str, object]   # zone definition dict type alias

STRATEGIES: Dict[str, List[_ZD]] = {

    # ── 上下横排：主体居中（人像、画作、产品正面）────────────────────────
    "top_bottom": [
        {"label": "top",    "y0r": 0.00, "y1r": 0.35,
         "x0r": 0.00, "x1r": 1.00, "scan": "h_top",    "align": "center"},
        {"label": "bottom", "y0r": 0.65, "y1r": 1.00,
         "x0r": 0.00, "x1r": 1.00, "scan": "h_bottom", "align": "center"},
    ],

    # ── 左右竖排：横幅图像主体宽，两侧细长留白 ───────────────────────────
    "left_right": [
        {"label": "left",  "y0r": 0.00, "y1r": 1.00,
         "x0r": 0.00, "x1r": 0.28, "scan": "v_left",  "align": "left"},
        {"label": "right", "y0r": 0.00, "y1r": 1.00,
         "x0r": 0.72, "x1r": 1.00, "scan": "v_right", "align": "right"},
    ],

    # ── 全边框：主体完全居中，四周均有留白（正方形构图多见）────────────
    "frame": [
        {"label": "top",    "y0r": 0.00, "y1r": 0.30,
         "x0r": 0.00, "x1r": 1.00, "scan": "h_top",    "align": "center"},
        {"label": "bottom", "y0r": 0.70, "y1r": 1.00,
         "x0r": 0.00, "x1r": 1.00, "scan": "h_bottom", "align": "center"},
        {"label": "left",   "y0r": 0.30, "y1r": 0.70,
         "x0r": 0.00, "x1r": 0.25, "scan": "v_left",   "align": "left"},
        {"label": "right",  "y0r": 0.30, "y1r": 0.70,
         "x0r": 0.75, "x1r": 1.00, "scan": "v_right",  "align": "right"},
    ],

    # ── 中心横带：边缘纹理复杂（画框花边、装饰边），中心留白 ─────────────
    "center": [
        {"label": "center", "y0r": 0.22, "y1r": 0.78,
         "x0r": 0.15, "x1r": 0.85, "scan": "h_center", "align": "center"},
    ],

    # ── 仅顶部：主体占下半图，上方天空/空白大 ───────────────────────────
    "top_only": [
        {"label": "top", "y0r": 0.00, "y1r": 0.42,
         "x0r": 0.00, "x1r": 1.00, "scan": "h_top", "align": "center"},
    ],

    # ── 仅底部：主体占上半图，下方有地面/空白 ───────────────────────────
    "bottom_only": [
        {"label": "bottom", "y0r": 0.58, "y1r": 1.00,
         "x0r": 0.00, "x1r": 1.00, "scan": "h_bottom", "align": "center"},
    ],

    # ── 顶部横排 + 右侧竖列（Γ 形）──────────────────────────────────────
    "top_right": [
        {"label": "top",   "y0r": 0.00, "y1r": 0.36,
         "x0r": 0.00, "x1r": 1.00, "scan": "h_top",   "align": "center"},
        {"label": "right", "y0r": 0.36, "y1r": 1.00,
         "x0r": 0.72, "x1r": 1.00, "scan": "v_right", "align": "right"},
    ],

    # ── 顶部横排 + 左侧竖列（L 形）──────────────────────────────────────
    "top_left": [
        {"label": "top",  "y0r": 0.00, "y1r": 0.36,
         "x0r": 0.00, "x1r": 1.00, "scan": "h_top",  "align": "center"},
        {"label": "left", "y0r": 0.36, "y1r": 1.00,
         "x0r": 0.00, "x1r": 0.28, "scan": "v_left", "align": "left"},
    ],

    # ── 右侧竖列 + 底部横排（反 L 形）───────────────────────────────────
    "bottom_right": [
        {"label": "right",  "y0r": 0.00, "y1r": 0.64,
         "x0r": 0.72, "x1r": 1.00, "scan": "v_right",  "align": "right"},
        {"label": "bottom", "y0r": 0.64, "y1r": 1.00,
         "x0r": 0.00, "x1r": 1.00, "scan": "h_bottom", "align": "center"},
    ],

    # ── 左侧竖列 + 底部横排（J 形）──────────────────────────────────────
    "bottom_left": [
        {"label": "left",   "y0r": 0.00, "y1r": 0.64,
         "x0r": 0.00, "x1r": 0.28, "scan": "v_left",  "align": "left"},
        {"label": "bottom", "y0r": 0.64, "y1r": 1.00,
         "x0r": 0.00, "x1r": 1.00, "scan": "h_bottom", "align": "center"},
    ],

    # ── 上下 + 右侧（三边环绕）───────────────────────────────────────────
    "top_bottom_right": [
        {"label": "top",    "y0r": 0.00, "y1r": 0.33,
         "x0r": 0.00, "x1r": 1.00, "scan": "h_top",    "align": "center"},
        {"label": "bottom", "y0r": 0.67, "y1r": 1.00,
         "x0r": 0.00, "x1r": 1.00, "scan": "h_bottom", "align": "center"},
        {"label": "right",  "y0r": 0.33, "y1r": 0.67,
         "x0r": 0.75, "x1r": 1.00, "scan": "v_right",  "align": "right"},
    ],

    # ── 上下 + 左侧（三边环绕）───────────────────────────────────────────
    "top_bottom_left": [
        {"label": "top",    "y0r": 0.00, "y1r": 0.33,
         "x0r": 0.00, "x1r": 1.00, "scan": "h_top",    "align": "center"},
        {"label": "bottom", "y0r": 0.67, "y1r": 1.00,
         "x0r": 0.00, "x1r": 1.00, "scan": "h_bottom", "align": "center"},
        {"label": "left",   "y0r": 0.33, "y1r": 0.67,
         "x0r": 0.00, "x1r": 0.25, "scan": "v_left",   "align": "left"},
    ],
}

# 字号层级：第 1~4 优先区依次缩小
_FONT_SCALES = [1.0, 0.72, 0.55, 0.45]


# ---------------------------------------------------------------------------
# 评分工具
# ---------------------------------------------------------------------------

def _region_mask(h: int, w: int, zd: _ZD) -> np.ndarray:
    """将相对坐标区域定义转为 bool HxW 掩码。"""
    m = np.zeros((h, w), dtype=bool)
    y0, y1 = int(zd["y0r"] * h), int(zd["y1r"] * h)  # type: ignore[arg-type]
    x0, x1 = int(zd["x0r"] * w), int(zd["x1r"] * w)  # type: ignore[arg-type]
    if y1 > y0 and x1 > x0:
        m[y0:y1, x0:x1] = True
    return m


def _zone_score(writable: np.ndarray, h: int, w: int, zd: _ZD) -> float:
    """
    单区评分 = density × abs_frac。
    density  ∈ [0,1]：区域内可写像素占区域面积的比例（越高表示区域越纯净）。
    abs_frac ∈ [0,1]：区域内可写像素占全图面积的比例（越高表示绝对空间越大）。
    两者相乘：偏好"大片干净区域"，同时惩罚"小片干净"和"大片嘈杂"。
    """
    region = _region_mask(h, w, zd)
    rpx = int(region.sum())
    if rpx == 0:
        return 0.0
    wpx = int((writable & region).sum())
    if wpx < 16:          # 像素数过少，视为无效
        return 0.0
    density  = wpx / rpx
    abs_frac = wpx / (h * w)
    return density * abs_frac


def _strategy_score(writable: np.ndarray, h: int, w: int,
                    zone_defs: List[_ZD]) -> float:
    """策略总得分 = 各区得分均值。"""
    scores = [_zone_score(writable, h, w, zd) for zd in zone_defs]
    return sum(scores) / max(len(scores), 1)


# ---------------------------------------------------------------------------
# 主接口
# ---------------------------------------------------------------------------

def find_text_zones(
    writable:       np.ndarray,
    complexity:     np.ndarray,
    *,
    min_area_ratio: float = 0.02,
    max_zones:      int   = 3,
) -> Tuple[List[TextZone], str]:
    """
    分析可写字区域，自动选择最优排版策略，返回有序 TextZone 列表。

    参数：
      writable      — bool HxW，build_writable 输出的可写字掩码
      complexity    — float32 HxW [0,1]，复杂度图
      min_area_ratio— 单个排版区域的最小面积（占全图比例），低于此跳过
      max_zones     — 最多返回几个区域（用户配置）；实际可能少于此数

    返回：
      (zones, strategy_name)
      zones         — 按 score 降序的 TextZone 列表，已设置 font_scale
      strategy_name — 最终选用的策略名（供 debug）
    """
    h, w = writable.shape
    min_px = max(16, int(min_area_ratio * h * w))

    # ── 1. 对所有策略打分，选最高 ────────────────────────────────────────
    ranked = sorted(
        STRATEGIES.items(),
        key=lambda kv: _strategy_score(writable, h, w, kv[1]),
        reverse=True,
    )
    best_name, best_zone_defs = ranked[0]

    # ── 2. 从最优策略的区域定义中提取 TextZone ───────────────────────────
    zones: List[TextZone] = []

    for zd in best_zone_defs:
        region  = _region_mask(h, w, zd)
        zm      = writable & region       # 区域内真正可写的像素
        wpx     = int(zm.sum())
        if wpx < min_px:
            continue                       # 区域可写面积不足，跳过

        ys, xs = np.where(zm)
        y0 = int(ys.min()); y1 = int(ys.max()) + 1
        x0 = int(xs.min()); x1 = int(xs.max()) + 1

        scan      = str(zd["scan"])
        direction = "v" if scan.startswith("v_") else "h"
        avg_comp  = float(complexity[zm].mean())
        # 区域得分：可写面积 × 低复杂度质量
        score = (wpx / (h * w)) * (1.0 - avg_comp)

        zones.append(TextZone(
            mask       = zm,
            direction  = direction,
            align      = str(zd["align"]),
            position   = str(zd["label"]),
            scan_style = scan,
            score      = score,
            bbox       = (y0, y1, x0, x1),
        ))

    # ── 3. 按得分降序排列，取 top max_zones ──────────────────────────────
    zones.sort(key=lambda z: z.score, reverse=True)
    zones = zones[:max_zones]

    # ── 4. 字号层级：主标题 1.0 → 副标题 0.72 → 装饰 0.55 ───────────────
    for i, z in enumerate(zones):
        z.font_scale = _FONT_SCALES[min(i, len(_FONT_SCALES) - 1)]

    return zones, best_name
