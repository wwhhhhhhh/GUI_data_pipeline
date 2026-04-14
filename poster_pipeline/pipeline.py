"""
海报排版 pipeline

流程：
  1. build_writable  → 非主体掩码 ∩ 低复杂度掩码（含两步膨胀）= 可写字区域
  2. find_text_zones → 对 12 种策略打分，选最优策略，确定各区方向/对齐
  3. plan_hierarchical → 按字号层级在每个区域内规划 LineSlot（从大到小依次放置）
  4. fill_slots      → 填入语料（每行独立字号，词指针跨行顺序前进）
  5. render_lines    → 渲染（per-line 颜色与字号，带对比描边）
"""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np

from .auto_layout import find_text_zones
from .color_contrast import contrast_text_rgb
from .layout_scanline import fill_slots, plan_hierarchical, render_lines
from .writable_mask import build_writable


# ---------------------------------------------------------------------------
# 字号层级工具
# ---------------------------------------------------------------------------

def _font_levels(font_px_base: int, font_px_min: int) -> List[int]:
    """
    根据基础字号和全图最小字号，生成降序字号层级列表。

    层级比例：× 2.2 / × 1.7 / × 1.3 / × 1.0，结果按 8px 对齐，
    去除重复后末尾附加 font_px_min（若不在列表中）。

    示例（base=56, min=48）→ [120, 96, 72, 56, 48]
    示例（base=48, min=48）→ [104, 80, 64, 48]
    """
    seen: set = set()
    levels: List[int] = []
    for ratio in [2.2, 1.7, 1.3, 1.0]:
        raw = int(font_px_base * ratio)
        px  = max(font_px_min, (raw + 4) // 8 * 8)   # 8px 对齐，不低于最小值
        if px not in seen:
            seen.add(px)
            levels.append(px)
    if font_px_min not in seen:
        levels.append(font_px_min)
    return levels   # 降序


def _zone_schedule(
    all_levels: List[int],
    font_px_min: int,
    zone_idx: int,
    max_slots: int = 6,
) -> List[int]:
    """
    构造单个区域的字号序列。

    - zone_idx=0（主区）：从最大字号开始，下行逐级缩小，最后补 min。
    - zone_idx=1（副区）：从第二大字号开始，以此类推。
    - 序列长度不超过 max_slots，保证每个槽位都有对应字号。
    """
    start = min(zone_idx, len(all_levels) - 1)
    base  = all_levels[start:]                         # 从对应层级开始
    pad   = [font_px_min] * max(0, max_slots - len(base))
    return (base + pad)[:max_slots]


# ---------------------------------------------------------------------------
# 合并掩码可视化
# ---------------------------------------------------------------------------

def make_combined_mask(
    writable:  np.ndarray,
    forb_mask: np.ndarray,
) -> np.ndarray:
    """
    三层合并可视化图（HxW → HxWx3 uint8）：
      白色  [255,255,255] — 可写字区域
      红色  [210, 55, 55] — 主体禁区（含膨胀安全边距）
      黄色  [220,175,  0] — 复杂度禁区（非主体但纹理过高）
    """
    h, w = writable.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[~forb_mask & ~writable] = [220, 175,   0]
    canvas[forb_mask]               = [210,  55,  55]
    canvas[writable]                = [255, 255, 255]
    return canvas


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def run_poster_pipeline(
    rgb: np.ndarray,
    masks: List[Dict[str, Any]],
    *,
    subject_labels:    Optional[set] = None,
    corpus_text:       str   = "示例标题 用于合成数据海报排版",
    font_path:         Optional[str] = None,
    font_px:           int   = 56,    # 基础字号（用于计算层级）
    font_px_min:       int   = 48,    # 全图最小字号（硬下限）
    # ── 膨胀参数 ──────────────────────────────────────────────────────────
    dilate_iter:       int   = 14,    # 主体禁区膨胀步数
    comp_dilate_iter:  int   = 6,     # 低复杂度区域膨胀步数
    # ── 其他配置 ──────────────────────────────────────────────────────────
    complexity_thresh: float = 0.50,
    min_area_ratio:    float = 0.03,  # 全图可写面积下限
    max_zones:         int   = 3,     # 最多使用几个排版区域
) -> Dict[str, Any]:
    """
    参数：
      rgb               — HxWx3 uint8 图像
      masks             — [{"mask": bool HxW, "label": str}, ...]，可为空列表
      subject_labels    — 需要避开的类别集合；None 表示全部 mask 都避开
      corpus_text       — 空格分隔的词组
      font_px           — 基础字号（像素），用于计算 × 2.2 / 1.7 / 1.3 / 1.0 层级
      font_px_min       — 全图最小字号（像素），所有层级不低于此值
      dilate_iter       — 主体禁区膨胀步数，越大文字离主体越远
      comp_dilate_iter  — 低复杂度区域膨胀步数，越大可写边界越宽松
      complexity_thresh — 复杂度阈值（0~1），低于此才算低复杂可写
      min_area_ratio    — 全图可写区域面积下限，低于此跳过整图
      max_zones         — 最多使用几个排版区域；实际区域数可能更少

    Returns dict：
      preview       — 渲染结果 ndarray (HxWx3 uint8)
      bbox_preview  — 带 bbox 框的渲染结果（调试用）
      lines         — List[TextLine]
      writable      — bool HxW 可写字区域
      complexity    — float32 HxW [0,1] 复杂度图
      subj_mask     — bool HxW 主体区域（膨胀前原始 union）
      forb_mask     — bool HxW 主体禁区（膨胀后，含安全边距）
      combined_mask — HxWx3 uint8 三层合并可视化（白/红/黄）
      debug         — 调试信息字典
    """
    h, w = rgb.shape[:2]

    # ── 1. 可写字区域 ──────────────────────────────────────────────────────
    writable, comp, subj_mask, forb_mask = build_writable(
        rgb, masks,
        h=h, w=w,
        forbid_labels=subject_labels,
        label_key="label",
        dilate_iter=dilate_iter,
        complexity_thresh=complexity_thresh,
        comp_dilate_iter=comp_dilate_iter,
    )

    combined = make_combined_mask(writable, forb_mask)

    writable_ratio = float(writable.sum()) / (h * w)
    debug: Dict[str, Any] = {
        "complexity_thresh":  complexity_thresh,
        "dilate_iter":        dilate_iter,
        "comp_dilate_iter":   comp_dilate_iter,
        "writable_ratio":     round(writable_ratio, 3),
    }

    # ── 2. 全图可写面积过小则跳过 ──────────────────────────────────────────
    if writable_ratio < min_area_ratio:
        debug["skip_reason"] = "writable area too small"
        debug["n_lines"] = 0
        return {"debug": debug, "preview": rgb, "bbox_preview": rgb, "lines": [],
                "writable": writable, "complexity": comp,
                "subj_mask": subj_mask, "forb_mask": forb_mask,
                "combined_mask": combined}

    # ── 3. 字号层级：基础字号受短边 1/8 限制，最小为 font_px_min ────────
    font_px_base = max(font_px_min, min(font_px, min(h, w) // 8))
    all_levels   = _font_levels(font_px_base, font_px_min)

    # ── 4. 策略打分 → 选最优布局 → 提取区域 ─────────────────────────────
    zones, strategy_name = find_text_zones(
        writable, comp,
        min_area_ratio=0.02,
        max_zones=max_zones,
    )
    if not zones:
        debug["skip_reason"] = "no valid zones found"
        debug["n_lines"] = 0
        return {"debug": debug, "preview": rgb, "bbox_preview": rgb, "lines": [],
                "writable": writable, "complexity": comp,
                "subj_mask": subj_mask, "forb_mask": forb_mask,
                "combined_mask": combined}

    # ── 5. 逐区层次排版 ────────────────────────────────────────────────────
    all_lines = []

    for zone_idx, zone in enumerate(zones):
        # 每区背景色采样文字颜色
        fr, fg_v, fb, _br, _bg, _bb = contrast_text_rgb(rgb, zone.mask)
        zone_fg = (fr, fg_v, fb)

        # 构造该区字号序列（主区从最大级别开始，次区从次级开始）
        schedule = _zone_schedule(all_levels, font_px_min, zone_idx, max_slots=7)

        # 竖排 side：从 scan_style 中解析（"v_right" → "right"，"v_left" → "left"）
        v_side = (zone.scan_style.split("_", 1)[1]
                  if zone.direction == "v" and "_" in zone.scan_style
                  else "right")

        slots = plan_hierarchical(
            zone.mask, schedule, zone.direction,
            side=v_side,
            margin=6,
            min_width_chars=2,
            min_height_chars=2,
            max_slots=6,
        )
        if not slots:
            continue

        zone_lines = fill_slots(
            slots, corpus_text,
            font_path=font_path,
            font_px=font_px_min,   # 回退值，实际由 slot.font_px 控制
            align=zone.align,
            fg=zone_fg,
        )
        all_lines.extend(zone_lines)

    # ── 6. 渲染 ────────────────────────────────────────────────────────────
    preview      = render_lines(rgb, all_lines, font_path=font_path, draw_bbox=False)
    bbox_preview = render_lines(rgb, all_lines, font_path=font_path, draw_bbox=True)

    debug["strategy"]      = strategy_name
    debug["n_lines"]       = len(all_lines)
    debug["font_px_base"]  = font_px_base
    debug["font_px_min"]   = font_px_min
    debug["font_levels"]   = all_levels
    debug["zones"]         = [
        {
            "position":   z.position,
            "direction":  z.direction,
            "align":      z.align,
            "scan_style": z.scan_style,
            "score":      round(z.score, 4),
            "bbox":       z.bbox,
        }
        for z in zones
    ]
    debug["lines"] = [asdict(x) for x in all_lines]

    return {
        "debug":         debug,
        "preview":       preview,
        "bbox_preview":  bbox_preview,
        "lines":         all_lines,
        "writable":      writable,
        "complexity":    comp,
        "subj_mask":     subj_mask,
        "forb_mask":     forb_mask,
        "combined_mask": combined,
    }


def save_debug_json(path: str, debug: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(debug, f, ensure_ascii=False, indent=2)
