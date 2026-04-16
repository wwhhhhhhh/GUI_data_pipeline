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
import random
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np

from .auto_layout import find_text_zones, rank_strategies, zones_for_strategy
from .color_contrast import contrast_from_palette
from .layout_scanline import fill_slots, plan_hierarchical, render_lines
from .writable_mask import build_writable


# ---------------------------------------------------------------------------
# 字号层级工具
# ---------------------------------------------------------------------------

def _font_levels(
    font_px_base: int,
    font_px_min: int,
    *,
    n_tiers: int = 3,
) -> List[int]:
    """
    根据基础字号、全图最小字号和层级数生成降序字号层级列表。

    - n_tiers ∈ {1, 2, 3}：一张图内最多 3 种字号，最少 1 种。
    - 所有层级不低于 font_px_min（硬下限，默认 > 48px）。
    - 结果按 8px 对齐，降序。

    比例按 n_tiers 自动选取：
      1 → [2.0]               （单一字号）
      2 → [2.2, 1.2]          （大/小）
      3 → [2.4, 1.6, 1.0]     （大/中/小）
    """
    if n_tiers <= 1:
        ratios = [1.8]
    elif n_tiers == 2:
        ratios = [2.2, 1.2]
    else:
        ratios = [2.4, 1.6, 1.0]

    seen: set = set()
    levels: List[int] = []
    for ratio in ratios:
        raw = int(font_px_base * ratio)
        px  = max(font_px_min, (raw + 4) // 8 * 8)
        if px not in seen:
            seen.add(px)
            levels.append(px)
    # 保证至少包含最小层，且全部 ≥ font_px_min
    levels = [max(px, font_px_min) for px in levels]
    return sorted(set(levels), reverse=True)[:max(1, n_tiers)]


def _zone_schedule(
    all_levels: List[int],
    font_px_min: int,
    zone_idx: int,
    max_slots: int = 6,
) -> List[int]:
    """
    构造单个区域的字号序列。

    - zone_idx=0（主区）：从最大字号开始，按层级降序使用。
    - zone_idx>0（副区）：从次级字号开始（若层级不足则直接复用最小层）。
    - 序列长度按槽位数补齐，末尾用最小字号填充（不低于 font_px_min）。
    """
    if not all_levels:
        return [font_px_min] * max_slots
    start = min(zone_idx, len(all_levels) - 1)
    base  = all_levels[start:]
    tail  = all_levels[-1]                              # 最小层级
    pad   = [tail] * max(0, max_slots - len(base))
    seq   = (base + pad)[:max_slots]
    return [max(px, font_px_min) for px in seq]


# ---------------------------------------------------------------------------
# meta 构造
# ---------------------------------------------------------------------------

def _text_meta(ln) -> Dict[str, Any]:
    """
    单行文字的 meta 信息：
      bbox       — [x0, y0, x1, y1]（y1 随方向计算：横排= y+font_px，
                   竖排= y + len(text)*font_px）
      text       — 文字内容
      font_px    — 字号（像素值）
      direction  — "h" 横排 / "v" 竖排
      font_name  — 实际加载到的字体文件名
      color      — {"rgb": [r,g,b], "name": "红"}（名称为色卡随机中/英名之一）
    """
    if ln.direction == "v":
        y1 = ln.y + max(1, len(ln.text)) * ln.font_px
    else:
        y1 = ln.y + ln.font_px
    return {
        "bbox":      [int(ln.x0), int(ln.y), int(ln.x1), int(y1)],
        "text":      ln.text,
        "font_px":   int(ln.font_px),
        "direction": ln.direction,
        "font_name": ln.font_name,
        "color": {
            "rgb":  [int(ln.fg[0]), int(ln.fg[1]), int(ln.fg[2])],
            "name": ln.color_name,
        },
    }


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
    # ── 字号层级 & 配色 ───────────────────────────────────────────────────
    n_tiers:           Optional[int] = None,   # 1~3；None 表示每张图随机
    seed:              Optional[int] = None,   # 随机种子（控制层级数、颜色选择）
    color_min_contrast: float = 4.5,
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
    rng = random.Random(seed)
    tiers = n_tiers if n_tiers is not None else rng.randint(1, 3)
    tiers = max(1, min(3, tiers))
    font_px_base = max(font_px_min, min(font_px, min(h, w) // 8))
    all_levels   = _font_levels(font_px_base, font_px_min, n_tiers=tiers)
    debug["n_tiers"] = tiers

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
    anchor_id: Optional[int] = None   # 一张图内尽量沿用同一色卡项

    for zone_idx, zone in enumerate(zones):
        # 色卡采样：优先沿用 anchor（对比足够时），否则按对比度重新挑一个
        color = contrast_from_palette(
            rgb, zone.mask,
            rng=rng,
            anchor_id=anchor_id,
            min_ratio=color_min_contrast,
        )
        if anchor_id is None:
            anchor_id = color["id"]
        zone_fg    = tuple(int(v) for v in color["rgb"])
        color_name = color["name"]

        # 构造该区字号序列（主区从最大级别开始，次区从次级开始）
        schedule = _zone_schedule(all_levels, font_px_min, zone_idx, max_slots=7)

        # 竖排 side：从 scan_style 中解析（"v_right" → "right"，"v_left" → "left"）
        v_side = (zone.scan_style.split("_", 1)[1]
                  if zone.direction == "v" and "_" in zone.scan_style
                  else "right")

        slots = plan_hierarchical(
            zone.mask, schedule, zone.direction,
            side=v_side,
            align=zone.align,
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
            color_name=color_name,
        )
        all_lines.extend(zone_lines)

    # ── 6. 渲染 ────────────────────────────────────────────────────────────
    preview      = render_lines(rgb, all_lines, font_path=font_path, draw_bbox=True)
    bbox_preview = preview   # 保持兼容，指向同一结果

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
    debug["texts"] = [_text_meta(ln) for ln in all_lines]

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


# ---------------------------------------------------------------------------
# 多 layout 生成（bbox-only，不渲染不填文字）
# ---------------------------------------------------------------------------

def _slots_to_scaled_boxes(
    slots,
    *,
    img_h: int,
    target_h: int = 1024,
) -> Dict[str, List[int]]:
    """将 LineSlot 列表转为 scaled_boxes_1024h 格式。"""
    scale = target_h / img_h
    boxes: Dict[str, List[int]] = {}
    for i, s in enumerate(slots):
        if s.direction == "v":
            y1 = s.y + s.height
        else:
            y1 = s.y + s.font_px
        boxes[str(i)] = [
            int(round(s.x0 * scale)),
            int(round(s.y * scale)),
            int(round(s.x1 * scale)),
            int(round(y1 * scale)),
        ]
    return boxes


def _slots_complexity(
    slots,
    comp: np.ndarray,
) -> Dict[str, float]:
    """计算每个 slot 区域内的平均复杂度（× 255 映射到 0–255 区间）。"""
    result: Dict[str, float] = {}
    for i, s in enumerate(slots):
        if s.direction == "v":
            y1 = s.y + s.height
        else:
            y1 = s.y + s.font_px
        y0c = max(0, s.y); y1c = min(comp.shape[0], y1)
        x0c = max(0, s.x0); x1c = min(comp.shape[1], s.x1)
        if y1c > y0c and x1c > x0c:
            val = float(comp[y0c:y1c, x0c:x1c].mean()) * 255.0
        else:
            val = 0.0
        result[str(i)] = round(val, 2)
    return result


def run_multi_layouts(
    rgb: np.ndarray,
    masks: List[Dict[str, Any]],
    *,
    subject_labels:    Optional[set] = None,
    font_px:           int   = 56,
    font_px_min:       int   = 48,
    dilate_iter:       int   = 14,
    comp_dilate_iter:  int   = 6,
    complexity_thresh: float = 0.50,
    min_area_ratio:    float = 0.03,
    max_zones:         int   = 3,
    max_layouts:       int   = 10,
    seed:              Optional[int] = None,
) -> Dict[str, Any]:
    """
    生成多种 layout 方案（bbox-only，不渲染不填文字）。

    为每种有效策略 × 每种层级数(1/2/3) 组合生成一个 layout，
    取前 max_layouts 个。结果格式兼容目标 JSON 的 matched_layouts。

    Returns dict:
      matched_layouts — List[Dict]，每项 {"layoutN": {"low_level_complexity": {...},
                        "scaled_boxes_1024h": {...}}}
      writable        — bool HxW
      complexity      — float32 HxW
      subj_mask       — bool HxW
      forb_mask       — bool HxW
      debug           — 调试信息
    """
    h, w = rgb.shape[:2]
    rng = random.Random(seed)

    # ── 1. 共享的 writable + complexity ────────────────────────────────────
    writable, comp, subj_mask, forb_mask = build_writable(
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
        "writable_ratio": round(writable_ratio, 3),
        "img_size":       [w, h],
    }

    if writable_ratio < min_area_ratio:
        debug["skip_reason"] = "writable area too small"
        return {"matched_layouts": [], "writable": writable, "complexity": comp,
                "subj_mask": subj_mask, "forb_mask": forb_mask, "debug": debug}

    # ── 2. 排列候选策略 ────────────────────────────────────────────────────
    ranked = rank_strategies(writable, comp)
    font_px_base = max(font_px_min, min(font_px, min(h, w) // 8))

    # ── 3. 策略 × 层级数 组合生成 layout ──────────────────────────────────
    layouts: List[Dict] = []
    seen_signatures: set = set()     # 去重：避免不同参数生成相同 bbox

    for strat_name, strat_score in ranked:
        if strat_score <= 0:
            continue
        zones = zones_for_strategy(
            writable, comp, strat_name,
            min_area_ratio=0.02, max_zones=max_zones,
        )
        if not zones:
            continue

        for n_tiers in [1, 2, 3]:
            all_levels = _font_levels(font_px_base, font_px_min, n_tiers=n_tiers)
            all_slots = []

            for zone_idx, zone in enumerate(zones):
                schedule = _zone_schedule(all_levels, font_px_min, zone_idx, max_slots=7)
                v_side = (zone.scan_style.split("_", 1)[1]
                          if zone.direction == "v" and "_" in zone.scan_style
                          else "right")
                slots = plan_hierarchical(
                    zone.mask, schedule, zone.direction,
                    side=v_side,
                    align=zone.align,
                    margin=6,
                    min_width_chars=2,
                    min_height_chars=2,
                    max_slots=6,
                )
                all_slots.extend(slots)

            if not all_slots:
                continue

            # 去重
            sig = tuple((s.x0, s.y, s.x1, s.y + (s.height if s.direction == "v" else s.font_px))
                        for s in all_slots)
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)

            raw_boxes = []
            for s in all_slots:
                if s.direction == "v":
                    y1 = s.y + s.height
                else:
                    y1 = s.y + s.font_px
                raw_boxes.append([int(s.x0), int(s.y), int(s.x1), int(y1)])

            idx = len(layouts)
            layouts.append({
                f"layout{idx}": {
                    "low_level_complexity": _slots_complexity(all_slots, comp),
                    "scaled_boxes_1024h":   _slots_to_scaled_boxes(
                        all_slots, img_h=h, target_h=1024,
                    ),
                    "raw_boxes": raw_boxes,
                }
            })
            if len(layouts) >= max_layouts:
                break
        if len(layouts) >= max_layouts:
            break

    debug["n_layouts"]  = len(layouts)
    debug["strategies"] = [(n, round(s, 4)) for n, s in ranked[:5]]

    return {
        "matched_layouts": layouts,
        "writable":        writable,
        "complexity":      comp,
        "subj_mask":       subj_mask,
        "forb_mask":       forb_mask,
        "debug":           debug,
    }
