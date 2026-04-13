"""端到端骨架：图 + 分割 masks → Zone 分区 → 排版 → 预览图。

流程：
  1. union_subject_masks  → 主体合并
  2. dilate_binary        → 主体禁区扩展
  3. complexity_map       → 全图复杂度（0=低频/天空, 1=高频/树枝）
  4. build_zones          → 按 style + 复杂度选出可用分区（支持复合 zone）
  5. plan_layout          → 每个 zone 独立排版槽位
  6. fill_slots           → 每个 zone 独立填入完整语料（非连通区域各自写字）
  7. render_lines         → 统一渲染
"""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np

from .color_contrast import contrast_text_rgb
from .layout_scanline import draw_lines, fill_slots, plan_layout, render_lines, scanline_wrap
from .writable_mask import complexity_map, dilate_binary, union_subject_masks
from .zones import STYLE_NAMES, build_zones_with_fallback


def run_poster_pipeline(
    rgb: np.ndarray,
    masks: List[Dict[str, Any]],
    *,
    subject_labels: Optional[set] = None,
    corpus_text: str = "示例标题 用于合成数据海报排版",
    font_path: Optional[str] = None,
    font_px: int = 48,
    layout_style: str = "top_title",
    dilate_iter: int = 14,
    min_zone_quality: float = 0.30,
) -> Dict[str, Any]:
    """
    参数：
      rgb            — HxWx3 uint8 图像
      masks          — [{"mask": bool HxW, "label": str}, ...]
      subject_labels — 需要避开的 label 集合；None 表示全部 mask 都避开
      corpus_text    — 空格分隔的词组，将被分配到各 zone
      font_px        — 字号（像素）
      layout_style   — 排版风格，见 zones.STYLE_NAMES
      dilate_iter    — 主体禁区膨胀半径（像素），越大留白越多
      min_zone_quality — zone 质量下限，低于此跳过（0~1，0=不过滤）
    """
    h, w = rgb.shape[:2]

    # ── 1. 主体禁区 ──────────────────────────────────────────────────────────
    subj = union_subject_masks(
        masks, h=h, w=w,
        forbid_labels=subject_labels,
        label_key="label",
    )
    forb = dilate_binary(subj, iterations=dilate_iter)

    # ── 2. 复杂度图（用于 zone 质量评分） ──────────────────────────────────
    comp = complexity_map(rgb)

    # ── 3. Zone 分区 ─────────────────────────────────────────────────────────
    # bbox 用未膨胀的 subj 定义（zone 边界贴近主体原始轮廓），
    # 像素可写性仍由 forb（已膨胀）决定
    zones = build_zones_with_fallback(
        h, w, forb, comp, layout_style,
        raw_subj_mask=subj,
        min_quality=min_zone_quality,
    )

    debug: Dict[str, Any] = {
        "style": layout_style,
        "zones": [
            {"name": name, "direction": direction, "quality": round(quality, 3)}
            for name, _, direction, quality in zones
        ],
    }

    if not zones:
        return {"debug": debug, "preview": rgb, "lines": [], "writable_binary": ~forb}

    # ── 4. 每 zone 独立排版 ──────────────────────────────────────────────────
    # · 各 zone 独立获得完整语料、独立颜色
    # · 按 quality 排名决定字号层次（最优 zone 全字号，其余 70%）
    # · max_slots 随 zone 数量缩减，避免多 zone 时文字过密
    # · 字号下限：max(24px, 图像短边 4%)，上限：图像高度 1/8

    ranked = sorted(range(len(zones)), key=lambda i: -zones[i][3])
    n_zones = len(zones)
    min_font = max(24, int(min(rgb.shape[:2]) * 0.04))
    font_px_primary   = max(min_font, min(font_px, rgb.shape[0] // 8))
    font_px_secondary = max(min_font, int(font_px_primary * 0.70))
    # zone 越多，每个 zone 的槽位越少，避免文字铺满全图
    max_slots_per_zone = max(2, 6 // max(1, n_zones))

    all_lines = []

    for i, (zone_name, zone_mask, direction, quality) in enumerate(zones):
        zone_font = font_px_primary if (ranked.index(i) == 0) else font_px_secondary

        # ── 4a. 每 zone 独立文字颜色 ──────────────────────────────────────
        zfr, zfg, zfb, *_ = contrast_text_rgb(rgb, zone_mask)
        zone_fg = (zfr, zfg, zfb)

        # ── 4b. 水平对齐：侧边竖列左/右对齐，顶底区居中 ──────────────────
        if direction == "v":
            align = "left"   # 竖排时 align 无水平效果
        elif zone_name in {"right", "right_tall", "top_right", "bot_right"}:
            align = "right"
        elif zone_name in {"left", "left_tall", "top_left", "bot_left"}:
            align = "left"
        else:
            align = "center"  # top/bottom/wide/half/full → 居中

        # ── 4c. 横排子风格（决定从顶还是从底开始放槽位） ─────────────────
        if direction == "v":
            plan_style = "v_right" if "right" in zone_name else "v_left"
        else:
            plan_style = "h_bottom" if ("bot" in zone_name or "bottom" in zone_name) else "h_top"

        # 竖排每个 zone 只取 1 列（多列会重复相同文字，视觉混乱）
        effective_max_slots = 1 if direction == "v" else max_slots_per_zone
        slots = plan_layout(
            zone_mask, zone_font,
            style=plan_style,
            max_slots=effective_max_slots,
        )
        lines = fill_slots(
            slots, corpus_text,
            font_path=font_path, font_px=zone_font,
            align=align, fg=zone_fg,
        )
        all_lines.extend(lines)

    # ── 5. 渲染（各行颜色/字号已存在 TextLine 内，直接渲染） ────────────────
    preview = render_lines(rgb, all_lines, font_path=font_path)

    # debug 信息：取最优 zone 的颜色作代表
    best_mask = zones[ranked[0]][1]
    fr, fg_c, fb, br, bg_c, bb = contrast_text_rgb(rgb, best_mask)
    debug["n_lines"] = len(all_lines)
    debug["lines"] = [asdict(x) for x in all_lines]
    debug["fg_rgb"] = [fr, fg_c, fb]
    debug["bg_median_rgb"] = [br, bg_c, bb]

    return {
        "debug": debug,
        "preview": preview,
        "lines": all_lines,
        "writable_binary": ~forb,
        "complexity": comp,
    }


def save_debug_json(path: str, debug: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(debug, f, ensure_ascii=False, indent=2)
