"""
海报排版 pipeline

流程：
  1. build_writable  → 非主体掩码 ∩ 低复杂度掩码（含两步膨胀）= 可写字区域
  2. find_text_zones → 对 11 种策略打分，选最优策略，确定各区方向/对齐/字号层级
  3. plan_layout     → 在每个区域内扫描槽位（band.all 保证不跨噪点）
  4. fill_slots      → 填入语料（每区独立获得完整语料）
  5. render_lines    → 渲染（per-line 颜色与字号，带对比描边）
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


def make_combined_mask(
    writable:  np.ndarray,
    forb_mask: np.ndarray,
) -> np.ndarray:
    """
    三层合并可视化图（HxW → HxWx3 uint8）：
      白色  [255,255,255] — 可写字区域
      红色  [210, 55, 55] — 主体禁区（含膨胀安全边距）
      黄色  [220,175,  0] — 复杂度禁区（非主体但纹理过高）

    三个类别互斥，覆盖全图所有像素：
      writable ⊆ ~forb_mask
      复杂度禁区 = ~forb_mask & ~writable
    """
    h, w = writable.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    # 先填复杂度禁区（非主体、非可写）
    canvas[~forb_mask & ~writable] = [220, 175, 0]
    # 再覆盖主体禁区（含膨胀缓冲）
    canvas[forb_mask]               = [210,  55, 55]
    # 最后覆盖可写区域（白色，优先级最高）
    canvas[writable]                = [255, 255, 255]
    return canvas


def run_poster_pipeline(
    rgb: np.ndarray,
    masks: List[Dict[str, Any]],
    *,
    subject_labels:   Optional[set] = None,
    corpus_text:      str   = "示例标题 用于合成数据海报排版",
    font_path:        Optional[str] = None,
    font_px:          int   = 48,
    # ── 两个膨胀参数，均可由调用方配置 ──
    dilate_iter:      int   = 14,    # 主体禁区膨胀半径（像素步数）
    comp_dilate_iter: int   = 6,     # 低复杂度区域膨胀半径（扩充可写边界）
    # ── 其他配置 ──
    complexity_thresh: float = 0.50,
    min_area_ratio:   float = 0.03,  # 全图可写面积下限
    max_zones:        int   = 3,     # 最多使用几个排版区域
) -> Dict[str, Any]:
    """
    参数：
      rgb               — HxWx3 uint8 图像
      masks             — [{"mask": bool HxW, "label": str}, ...]，可为空列表
      subject_labels    — 需要避开的类别集合；None 表示全部 mask 都避开
      corpus_text       — 空格分隔的词组
      font_px           — 基础字号（像素），自动限制不超过短边 1/8，最小 24
      dilate_iter       — 主体禁区膨胀步数，越大文字离主体越远
      comp_dilate_iter  — 低复杂度区域膨胀步数，越大可写边界越宽松
      complexity_thresh — 复杂度阈值（0~1），低于此才算低复杂可写
      min_area_ratio    — 全图可写区域面积下限，低于此跳过整图
      max_zones         — 最多使用几个排版区域；实际区域数可能更少

    Returns dict：
      preview       — 渲染结果 ndarray (HxWx3 uint8)
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
        return {"debug": debug, "preview": rgb, "lines": [],
                "writable": writable, "complexity": comp,
                "subj_mask": subj_mask, "forb_mask": forb_mask,
                "combined_mask": combined}

    # ── 3. 自适应字号：不超过短边 1/8，最小 24px ─────────────────────────
    font_px_base = max(24, min(font_px, min(h, w) // 8))

    # ── 4. 策略打分 → 选最优布局 → 提取区域 ─────────────────────────────
    zones, strategy_name = find_text_zones(
        writable, comp,
        min_area_ratio=0.02,
        max_zones=max_zones,
    )
    if not zones:
        debug["skip_reason"] = "no valid zones found"
        debug["n_lines"] = 0
        return {"debug": debug, "preview": rgb, "lines": [],
                "writable": writable, "complexity": comp,
                "subj_mask": subj_mask, "forb_mask": forb_mask,
                "combined_mask": combined}

    # ── 5. 逐区排版 ────────────────────────────────────────────────────────
    all_lines = []

    for zone in zones:
        zone_font_px = max(24, int(font_px_base * zone.font_scale))

        # 每区独立从该区背景色采样文字颜色
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

    debug["strategy"]      = strategy_name
    debug["n_lines"]       = len(all_lines)
    debug["font_px_base"]  = font_px_base
    debug["zones"]         = [
        {
            "position":   z.position,
            "direction":  z.direction,
            "align":      z.align,
            "scan_style": z.scan_style,
            "score":      round(z.score, 4),
            "font_scale": z.font_scale,
            "bbox":       z.bbox,
        }
        for z in zones
    ]
    debug["lines"] = [asdict(x) for x in all_lines]

    return {
        "debug":         debug,
        "preview":       preview,
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
