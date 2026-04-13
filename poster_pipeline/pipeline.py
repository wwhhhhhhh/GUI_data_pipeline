"""
海报排版 pipeline（简化版）

流程：
  1. build_writable  → 非主体掩码 ∩ 低复杂度掩码 = 可写字区域
  2. plan_layout     → 在可写区域内按布局风格找槽位
  3. fill_slots      → 填入语料
  4. render_lines    → 渲染
"""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np

from .color_contrast import contrast_text_rgb
from .layout_scanline import fill_slots, plan_layout, render_lines, STYLES
from .writable_mask import build_writable, complexity_map


# 布局风格 → 文字对齐方式
_ALIGN = {
    "h_top":    "center",
    "h_bottom": "center",
    "h_center": "center",
    "v_right":  "right",
    "v_left":   "left",
    "surround":  "center",
}

LAYOUT_STYLES = STYLES   # 对外暴露，供调用方查询可用风格


def run_poster_pipeline(
    rgb: np.ndarray,
    masks: List[Dict[str, Any]],
    *,
    subject_labels: Optional[set] = None,
    corpus_text: str = "示例标题 用于合成数据海报排版",
    font_path: Optional[str] = None,
    font_px: int = 48,
    layout_style: str = "h_top",
    dilate_iter: int = 14,
    complexity_thresh: float = 0.50,
    min_area_ratio: float = 0.04,
) -> Dict[str, Any]:
    """
    参数：
      rgb               — HxWx3 uint8 图像
      masks             — [{"mask": bool HxW, "label": str}, ...]
      subject_labels    — 需要避开的类别集合；None 表示全部 mask 都避开
      corpus_text       — 空格分隔的词组
      font_px           — 目标字号（像素），会自适应图像尺寸压缩
      layout_style      — 布局风格：h_top / h_bottom / h_center / v_right / v_left / surround
      dilate_iter       — 主体禁区膨胀半径（像素）
      complexity_thresh — 复杂度阈值（0~1），低于此的像素视为可写字
      min_area_ratio    — 可写区域面积下限（占全图比例），低于此直接跳过

    Returns dict:
      preview   — 渲染结果 ndarray
      lines     — List[TextLine]
      writable  — bool HxW 可写字掩码
      complexity— float32 HxW 复杂度图
      debug     — 调试信息
    """
    h, w = rgb.shape[:2]

    # ── 1. 可写字区域 = 非主体 ∩ 低复杂度 ──────────────────────────────
    writable, comp, subj_mask = build_writable(
        rgb, masks,
        h=h, w=w,
        forbid_labels=subject_labels,
        label_key="label",
        dilate_iter=dilate_iter,
        complexity_thresh=complexity_thresh,
    )

    writable_ratio = float(writable.sum()) / (h * w)
    debug: Dict[str, Any] = {
        "layout_style":     layout_style,
        "complexity_thresh": complexity_thresh,
        "writable_ratio":   round(writable_ratio, 3),
    }

    # ── 2. 面积过小则跳过 ────────────────────────────────────────────────
    if writable_ratio < min_area_ratio:
        debug["skip_reason"] = "writable area too small"
        debug["n_lines"] = 0
        return {"debug": debug, "preview": rgb, "lines": [],
                "writable": writable, "complexity": comp, "subj_mask": subj_mask}

    # ── 3. 自适应字号：不超过图像短边的 1/8，最小 24px ─────────────────
    font_px_used = max(24, min(font_px, min(h, w) // 8))

    # ── 4. 排版规划：在可写掩码内按风格找槽位 ───────────────────────────
    slots = plan_layout(writable, font_px_used, style=layout_style)
    if not slots:
        debug["skip_reason"] = "no valid slots"
        debug["n_lines"] = 0
        return {"debug": debug, "preview": rgb, "lines": [],
                "writable": writable, "complexity": comp, "subj_mask": subj_mask}

    # ── 5. 文字颜色（从可写区域采样背景色） ─────────────────────────────
    fr, fg_v, fb, br, bg_v, bb = contrast_text_rgb(rgb, writable)
    zone_fg = (fr, fg_v, fb)
    align = _ALIGN.get(layout_style, "center")

    # ── 6. 文本填充 ──────────────────────────────────────────────────────
    lines = fill_slots(
        slots, corpus_text,
        font_path=font_path, font_px=font_px_used,
        align=align, fg=zone_fg,
    )

    # ── 7. 渲染 ──────────────────────────────────────────────────────────
    preview = render_lines(rgb, lines, font_path=font_path)

    debug["n_lines"]        = len(lines)
    debug["font_px_used"]   = font_px_used
    debug["fg_rgb"]         = list(zone_fg)
    debug["bg_median_rgb"]  = [br, bg_v, bb]
    debug["lines"]          = [asdict(x) for x in lines]

    return {
        "debug":      debug,
        "preview":    preview,
        "lines":      lines,
        "writable":   writable,
        "complexity": comp,
        "subj_mask":  subj_mask,
    }


def save_debug_json(path: str, debug: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(debug, f, ensure_ascii=False, indent=2)
