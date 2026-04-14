"""
海报排版 pipeline

流程：
  1. build_writable  → 非主体掩码 ∩ 低复杂度掩码（含小膨胀）= 可写字区域
  2. find_text_zones → 连通域分析，按位置/宽高比自动确定各区方向、对齐、字号层级
  3. plan_layout     → 在每个区域内扫描槽位
  4. fill_slots      → 填入语料（每区独立获得完整语料）
  5. render_lines    → 渲染
"""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np

from .auto_layout import find_text_zones
from .color_contrast import contrast_text_rgb
from .layout_scanline import fill_slots, plan_layout, render_lines
from .writable_mask import build_writable


def run_poster_pipeline(
    rgb: np.ndarray,
    masks: List[Dict[str, Any]],
    *,
    subject_labels: Optional[set] = None,
    corpus_text: str = "示例标题 用于合成数据海报排版",
    font_path: Optional[str] = None,
    font_px: int = 48,
    dilate_iter: int = 14,
    comp_dilate_iter: int = 3,
    complexity_thresh: float = 0.50,
    min_area_ratio: float = 0.04,
    max_zones: int = 3,
) -> Dict[str, Any]:
    """
    参数：
      rgb               — HxWx3 uint8 图像
      masks             — [{"mask": bool HxW, "label": str}, ...]，可为空列表
      subject_labels    — 需要避开的类别集合；None 表示全部 mask 都避开
      corpus_text       — 空格分隔的词组
      font_px           — 目标字号（像素），会自适应图像尺寸压缩
      dilate_iter       — 主体禁区膨胀半径（像素）
      comp_dilate_iter  — 低复杂度区域额外膨胀半径，扩充可写边界
      complexity_thresh — 复杂度阈值（0~1），低于此视为可写字
      min_area_ratio    — 可写区域面积下限（占全图比例），低于此直接跳过
      max_zones         — 最多使用几个排版区域（字号层级最多 3 级）

    Returns dict：
      preview    — 渲染结果 ndarray
      lines      — List[TextLine]
      writable   — bool HxW 可写字掩码
      complexity — float32 HxW 复杂度图
      subj_mask  — bool HxW 主体区域（膨胀前）
      debug      — 调试信息
    """
    h, w = rgb.shape[:2]

    # ── 1. 可写字区域 ──────────────────────────────────────────────────────
    writable, comp, subj_mask = build_writable(
        rgb, masks,
        h=h, w=w,
        forbid_labels=subject_labels,
        label_key="label",
        dilate_iter=dilate_iter,
        complexity_thresh=complexity_thresh,
        comp_dilate_iter=comp_dilate_iter,
    )

    writable_ratio = float(writable.sum()) / (h * w)
    debug: Dict[str, Any] = {
        "complexity_thresh":  complexity_thresh,
        "writable_ratio":     round(writable_ratio, 3),
    }

    # ── 2. 面积过小则跳过 ──────────────────────────────────────────────────
    if writable_ratio < min_area_ratio:
        debug["skip_reason"] = "writable area too small"
        debug["n_lines"] = 0
        return {"debug": debug, "preview": rgb, "lines": [],
                "writable": writable, "complexity": comp, "subj_mask": subj_mask}

    # ── 3. 自适应字号：不超过图像短边的 1/8，最小 24px ──────────────────
    font_px_base = max(24, min(font_px, min(h, w) // 8))

    # ── 4. 自动识别排版区域 ────────────────────────────────────────────────
    zones = find_text_zones(
        writable, comp,
        min_area_ratio=0.03,
        max_zones=max_zones,
    )
    if not zones:
        debug["skip_reason"] = "no valid zones"
        debug["n_lines"] = 0
        return {"debug": debug, "preview": rgb, "lines": [],
                "writable": writable, "complexity": comp, "subj_mask": subj_mask}

    # ── 5. 逐区排版 ────────────────────────────────────────────────────────
    all_lines = []

    for zone in zones:
        zone_font_px = max(24, int(font_px_base * zone.font_scale))

        # 每区独立从背景色采样文字颜色
        fr, fg_v, fb, _br, _bg, _bb = contrast_text_rgb(rgb, zone.mask)
        zone_fg = (fr, fg_v, fb)

        slots = plan_layout(zone.mask, zone_font_px, style=zone.scan_style)
        if not slots:
            continue

        zone_lines = fill_slots(
            slots, corpus_text,
            font_path=font_path, font_px=zone_font_px,
            align=zone.align, fg=zone_fg,
        )
        all_lines.extend(zone_lines)

    # ── 6. 渲染 ────────────────────────────────────────────────────────────
    preview = render_lines(rgb, all_lines, font_path=font_path)

    debug["n_lines"]       = len(all_lines)
    debug["font_px_base"]  = font_px_base
    debug["zones"]         = [
        {
            "position":   z.position,
            "direction":  z.direction,
            "align":      z.align,
            "scan_style": z.scan_style,
            "score":      round(z.score, 3),
            "font_scale": z.font_scale,
            "bbox":       z.bbox,
        }
        for z in zones
    ]
    debug["lines"] = [asdict(x) for x in all_lines]

    return {
        "debug":      debug,
        "preview":    preview,
        "lines":      all_lines,
        "writable":   writable,
        "complexity": comp,
        "subj_mask":  subj_mask,
    }


def save_debug_json(path: str, debug: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(debug, f, ensure_ascii=False, indent=2)
