"""
海报排版：三层分离 + 层次字号支持

  1. plan_hierarchical(mask, size_schedule, direction, ...)
       纯几何，按字号从大到小依次扫描 mask，产生带字号标注的 LineSlot 列表。
       横排：从上到下，每行字号由 size_schedule 决定。
       竖排：从外侧到内侧，每列字号由 size_schedule 决定。

  2. fill_slots(slots, corpus, font_path, ...)
       纯文本分配：每个槽位按自己的 slot.font_px 计算可容纳文字量。

  3. render_lines(rgb, lines, font_path, ...)
       纯渲染：per-line 字号与颜色，可选绘制 bbox 轮廓。

  ── 旧接口兼容 ────────────────────────────────────────────────────────────
  plan_layout / scanline_wrap / draw_lines 保留，供外部代码平滑过渡。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = ImageDraw = ImageFont = None


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class LineSlot:
    """一行/列文字的几何槽位（纯布局，没有文字内容）。"""
    y: int          # 顶部 y
    x0: int         # 左边界
    x1: int         # 右边界
    height: int     # 槽位高度（横排 = font_px；竖排 = 可用字符数 × font_px）
    direction: str = "h"    # "h" 横排  "v" 竖排
    font_px: int   = 0      # 该槽位字号（0 表示使用调用方默认值）


@dataclass
class TextLine:
    """已填充文字的槽位，用于渲染。"""
    y: int
    x0: int
    x1: int
    text: str
    direction: str = "h"               # "h" 横排  "v" 竖排
    align: str     = "left"            # "left" / "center" / "right"（横排有效）
    fg: Tuple[int, int, int] = (248, 250, 252)   # 文字颜色（per-line）
    font_px: int   = 48                # 该行字号（per-line，支持层次大小）


# ---------------------------------------------------------------------------
# 字体工具
# ---------------------------------------------------------------------------

def _load_font(font_path: Optional[str], font_px: int):
    if ImageFont is None:
        return None
    candidates = list(filter(None, [font_path])) + [
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, font_px)
        except Exception:
            continue
    return ImageFont.load_default()


def _text_width(text: str, font) -> int:
    if font is None or Image is None:
        return 0
    img = Image.new("L", (1, 1))
    dr = ImageDraw.Draw(img)
    try:
        return int(dr.textlength(text, font=font))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# 几何工具
# ---------------------------------------------------------------------------

def _widest_run(arr: np.ndarray) -> Optional[Tuple[int, int]]:
    """返回一维 bool 数组中最长 True 连续段的 (start, end_exclusive)，无则 None。"""
    if not arr.any():
        return None
    xs = np.flatnonzero(arr)
    breaks = np.where(np.diff(xs) > 1)[0] + 1
    segs = np.split(xs, breaks)
    best = max(segs, key=len)
    return int(best[0]), int(best[-1]) + 1


# ---------------------------------------------------------------------------
# 层次字号扫描（核心新逻辑）
# ---------------------------------------------------------------------------

def _h_slots_hierarchical(
    mask: np.ndarray,
    size_schedule: List[int],   # 字号列表，降序，如 [120, 96, 72, 56, 48, 48]
    margin: int,
    min_width_chars: int,
    line_gap_ratio: float,      # 行间距 = font_px × ratio
    max_slots: int,
) -> List[LineSlot]:
    """
    横向层次扫描：按 size_schedule 从大到小依次寻找可放置的行。

    策略：
      - 对每个字号 target_font，从当前 y 向下扫描，步长 target_font//4
      - 找到宽度 ≥ min_width_chars × target_font 的最宽连续可写带时放置槽位
      - 该字号找不到时跳过（继续尝试更小的字号，但 y 不前进）
      - 成功放置后 y 前进到槽位底部 + 行间距
    """
    h, w = mask.shape
    slots: List[LineSlot] = []
    y = 0

    for target_font in size_schedule:
        if len(slots) >= max_slots or y >= h:
            break
        min_w = min_width_chars * target_font
        step = max(1, target_font // 4)

        # 从 y 向下扫描
        found_y: Optional[int] = None
        best_run: Optional[Tuple[int, int]] = None
        scan_y = y
        while scan_y + target_font <= h:
            band = mask[scan_y: scan_y + target_font, :]
            col_clear = band.all(axis=0)
            run = _widest_run(col_clear)
            if run is not None and (run[1] - run[0] - 2 * margin) >= min_w:
                found_y = scan_y
                best_run = run
                break
            scan_y += step

        if found_y is None:
            continue   # 此字号放不下，跳过（y 不变，尝试更小字号）

        x0 = best_run[0] + margin  # type: ignore[index]
        x1 = best_run[1] - margin  # type: ignore[index]
        slots.append(LineSlot(
            y=found_y, x0=x0, x1=x1,
            height=target_font, direction="h",
            font_px=target_font,
        ))
        gap = max(4, int(target_font * line_gap_ratio))
        y = found_y + target_font + gap

    return slots


def _v_slots_hierarchical(
    mask: np.ndarray,
    size_schedule: List[int],   # 字号列表，降序（每列一个字号）
    margin: int,
    min_height_chars: int,
    col_gap_ratio: float,       # 列间距 = font_px × ratio
    max_slots: int,
    side: str,                  # "right" | "left"
) -> List[LineSlot]:
    """
    竖向层次扫描：从外侧向内依次放置列，每列字号由 size_schedule 决定。

    side="right"：从右侧开始，越靠外字号越大。
    side="left" ：从左侧开始，越靠外字号越大。
    """
    h, w = mask.shape
    slots: List[LineSlot] = []

    x = (w if side == "right" else 0)

    for target_font in size_schedule:
        if len(slots) >= max_slots:
            break
        gap = max(4, int(target_font * col_gap_ratio))

        if side == "right":
            col_x = x - target_font
            if col_x < 0:
                break
        else:
            col_x = x
            if col_x + target_font > w:
                break

        strip = mask[:, col_x: col_x + target_font]
        row_clear = strip.all(axis=1)
        run = _widest_run(row_clear)
        if run is not None:
            avail_h = run[1] - run[0] - 2 * margin
            min_h = min_height_chars * target_font
            if avail_h >= min_h:
                slots.append(LineSlot(
                    y=run[0] + margin, x0=col_x, x1=col_x + target_font,
                    height=avail_h, direction="v",
                    font_px=target_font,
                ))

        if side == "right":
            x = col_x - gap
        else:
            x = col_x + target_font + gap

    return slots


def plan_hierarchical(
    mask: np.ndarray,
    size_schedule: List[int],
    direction: str,
    *,
    side: str = "right",            # 竖排有效："right" | "left"
    margin: int = 6,
    min_width_chars: int = 2,
    min_height_chars: int = 2,
    line_gap_ratio: float = 0.18,   # 横排行间距比例
    col_gap_ratio: float = 0.25,    # 竖排列间距比例
    max_slots: int = 6,
) -> List[LineSlot]:
    """
    层次字号扫描：根据 size_schedule 在 mask 中规划 LineSlot 列表。

    参数：
      mask          — bool HxW，可写区域
      size_schedule — 字号列表（降序），如 [120, 96, 72, 56, 48, 48]
                      每个元素对应一行/列的目标字号
      direction     — "h" 横排 | "v" 竖排
      side          — 竖排时从哪侧开始（"right" | "left"）
      margin        — 槽位与可写边界之间的内缩像素
      min_width_chars  — 横排最少容纳字符数（× font_px 得到最小宽度）
      min_height_chars — 竖排最少容纳字符数（× font_px 得到最小高度）
      line_gap_ratio   — 横排行间距相对字号的比例
      col_gap_ratio    — 竖排列间距相对字号的比例
      max_slots     — 最多返回几个槽位
    """
    if direction == "h":
        return _h_slots_hierarchical(
            mask, size_schedule, margin, min_width_chars,
            line_gap_ratio, max_slots,
        )
    else:
        return _v_slots_hierarchical(
            mask, size_schedule, margin, min_height_chars,
            col_gap_ratio, max_slots, side,
        )


# ---------------------------------------------------------------------------
# 第二层：fill_slots — 纯文本分配（支持 per-slot 字号）
# ---------------------------------------------------------------------------

def _join_words(words: List[str]) -> str:
    if not words:
        return ""
    result = words[0]
    for w in words[1:]:
        sep = " " if (result[-1].isascii() or w[0].isascii()) else ""
        result += sep + w
    return result


def fill_slots(
    slots: List[LineSlot],
    corpus: str,
    font_path: Optional[str] = None,
    font_px: int = 48,             # 字号回退值（slot.font_px == 0 时使用）
    align: str = "left",
    fg: Tuple[int, int, int] = (248, 250, 252),
) -> List[TextLine]:
    """
    将 corpus 分配到槽位，返回 TextLine 列表。

    横排槽（direction='h'）：按空格切词，贪心填充每行。
      - 语料词指针跨槽顺序前进；耗尽后重头循环。
    竖排槽（direction='v'）：按字符填充，每槽独立从语料头取字。
      - capacity = slot.height // slot.font_px

    每个槽位使用 slot.font_px（优先）或 font_px（回退）。
    align / fg 为 per-zone 全局值，统一写入 TextLine。
    """
    if not slots or not corpus.strip():
        return []

    # 按字号缓存字体
    _font_cache: dict = {}

    def _get_font(px: int):
        if px not in _font_cache:
            _font_cache[px] = _load_font(font_path, px)
        return _font_cache[px]

    lines: List[TextLine] = []
    words = corpus.split()
    chars = [c for c in corpus if c.strip()]

    wi = 0   # 横排词指针（跨 h 槽连续前进）

    for slot in slots:
        slot_fpx = slot.font_px if slot.font_px > 0 else font_px
        font = _get_font(slot_fpx)

        if slot.direction == "h":
            # 词指针耗尽时从头循环（确保每个槽都有内容）
            if wi >= len(words):
                wi = 0
            avail_w = slot.x1 - slot.x0
            line_words: List[str] = []
            used_w = 0
            while wi < len(words):
                word = words[wi]
                word_w = _text_width(word, font)
                sep_w = (_text_width(" ", font)
                         if (line_words and (line_words[-1][-1].isascii()
                                             or word[0].isascii()))
                         else 0)
                if used_w + sep_w + word_w <= avail_w:
                    line_words.append(word)
                    used_w += sep_w + word_w
                    wi += 1
                else:
                    break
            if line_words:
                lines.append(TextLine(
                    y=slot.y, x0=slot.x0, x1=slot.x1,
                    text=_join_words(line_words),
                    direction="h", align=align, fg=fg,
                    font_px=slot_fpx,
                ))

        else:  # direction == "v"
            # 竖排：每列独立从语料头截取
            capacity = max(1, slot.height // slot_fpx)
            chunk = "".join(chars[:capacity])
            if chunk:
                lines.append(TextLine(
                    y=slot.y, x0=slot.x0, x1=slot.x1,
                    text=chunk,
                    direction="v", align=align, fg=fg,
                    font_px=slot_fpx,
                ))

    return lines


# ---------------------------------------------------------------------------
# 第三层：render_lines — 纯渲染
# ---------------------------------------------------------------------------

def render_lines(
    rgb: np.ndarray,
    lines: List[TextLine],
    fg: Optional[Tuple[int, int, int]] = None,  # 旧接口兼容，优先使用 ln.fg
    font_path: Optional[str] = None,
    font_px: int = 48,
    stroke_width: int = 2,
    draw_bbox: bool = False,   # True 时在每行文字区域绘制轮廓框（调试用）
) -> np.ndarray:
    """
    横排：按 ln.align 定位后绘制整行文本。
    竖排：逐字在 (x0, y + i*ln.font_px) 绘制，字符居中于列宽。
    每行独立使用 ln.fg 颜色和 ln.font_px 字号；带对比描边提升可读性。

    draw_bbox=True 时，在文字区域周围绘制细线矩形（颜色为描边色，便于
    查看排版网格与实际像素的对应关系）。
    """
    if Image is None:
        raise RuntimeError("需要 pillow：pip install pillow")

    img = Image.fromarray(rgb.copy())
    dr = ImageDraw.Draw(img)

    # 按字号缓存字体对象，避免重复加载
    _font_cache: dict = {}

    def _get_font(px: int):
        if px not in _font_cache:
            _font_cache[px] = _load_font(font_path, px)
        return _font_cache[px]

    for ln in lines:
        line_fg  = ln.fg if ln.fg is not None else (fg or (248, 250, 252))
        line_fpx = ln.font_px if ln.font_px else font_px
        font     = _get_font(line_fpx)

        lum = 0.299 * line_fg[0] + 0.587 * line_fg[1] + 0.114 * line_fg[2]
        stroke_fill = (0, 0, 0) if lum > 128 else (255, 255, 255)

        if ln.direction == "h":
            tw = _text_width(ln.text, font)
            slot_w = ln.x1 - ln.x0
            if ln.align == "center":
                x = ln.x0 + max(0, (slot_w - tw) // 2)
            elif ln.align == "right":
                x = max(ln.x0, ln.x1 - tw)
            else:
                x = ln.x0

            if draw_bbox:
                dr.rectangle(
                    [ln.x0, ln.y, ln.x1, ln.y + line_fpx],
                    outline=stroke_fill, width=1,
                )

            dr.text(
                (x, ln.y), ln.text,
                fill=line_fg, font=font,
                stroke_width=stroke_width, stroke_fill=stroke_fill,
            )

        else:  # vertical
            col_w   = ln.x1 - ln.x0
            v_end   = ln.y + len(ln.text) * line_fpx

            if draw_bbox:
                dr.rectangle(
                    [ln.x0, ln.y, ln.x1, v_end],
                    outline=stroke_fill, width=1,
                )

            for i, char in enumerate(ln.text):
                char_w = _text_width(char, font)
                x_off  = max(0, (col_w - char_w) // 2)
                dr.text(
                    (ln.x0 + x_off, ln.y + i * line_fpx),
                    char,
                    fill=line_fg, font=font,
                    stroke_width=stroke_width, stroke_fill=stroke_fill,
                )

    return np.asarray(img)


# ---------------------------------------------------------------------------
# 旧接口兼容（plan_layout / scanline_wrap / draw_lines）
# ---------------------------------------------------------------------------

STYLES = ("h_top", "h_bottom", "h_center", "v_right", "v_left", "surround")


def _h_slots_uniform(
    mask: np.ndarray,
    font_px: int,
    margin: int,
    min_w: int,
    line_spacing: float,
) -> List[LineSlot]:
    h, _ = mask.shape
    step  = int(font_px * (1.0 + line_spacing))
    slots: List[LineSlot] = []
    y = 0
    while y + font_px <= h:
        band = mask[y: y + font_px, :]
        col_clear = band.all(axis=0)
        run = _widest_run(col_clear)
        if run is not None and (run[1] - run[0] - 2 * margin) >= min_w:
            slots.append(LineSlot(
                y=y, x0=run[0] + margin, x1=run[1] - margin,
                height=font_px, direction="h", font_px=font_px,
            ))
        y += step
    return slots


def _v_slots_uniform(
    mask: np.ndarray,
    font_px: int,
    margin: int,
    min_h: int,
    col_spacing: float,
    side: str,
) -> List[LineSlot]:
    h, w = mask.shape
    col_step = int(font_px * (1.0 + col_spacing))
    slots: List[LineSlot] = []
    xs_iter = (
        range(w - font_px, -1, -col_step)
        if side == "right"
        else range(0, w - font_px + 1, col_step)
    )
    for x in xs_iter:
        if x < 0 or x + font_px > w:
            continue
        strip = mask[:, x: x + font_px]
        row_clear = strip.all(axis=1)
        run = _widest_run(row_clear)
        if run is None:
            continue
        avail_h = run[1] - run[0] - 2 * margin
        if avail_h >= min_h:
            slots.append(LineSlot(
                y=run[0] + margin, x0=x, x1=x + font_px,
                height=avail_h, direction="v", font_px=font_px,
            ))
    return slots


def plan_layout(
    mask: np.ndarray,
    font_px: int,
    *,
    style: str = "h_top",
    line_spacing: float = 0.25,
    col_spacing: float = 0.3,
    margin: int = 6,
    min_width_chars: int = 2,
    min_height_chars: int = 2,
    max_slots: int = 6,
) -> List[LineSlot]:
    """旧接口：单一字号扫描，保留供兼容。新代码请用 plan_hierarchical。"""
    h, w = mask.shape[:2]
    min_w = min_width_chars * font_px
    min_h = min_height_chars * font_px

    if style in ("h_top", "h_bottom", "h_center"):
        all_slots = _h_slots_uniform(mask, font_px, margin, min_w, line_spacing)
        n = len(all_slots)
        if style == "h_top":
            return all_slots[:max_slots]
        elif style == "h_bottom":
            return all_slots[max(0, n - max_slots):]
        else:
            mid = n // 2; half = max_slots // 2
            return all_slots[max(0, mid - half): mid + half]

    elif style in ("v_right", "v_left"):
        side  = style[2:]
        slots = _v_slots_uniform(mask, font_px, margin, min_h, col_spacing, side)
        return slots[:max_slots]

    elif style == "surround":
        top_mask = mask.copy(); top_mask[h // 2:, :] = False
        top_sl   = _h_slots_uniform(top_mask, font_px, margin, min_w, line_spacing)[:max_slots // 2]
        bot_mask = mask.copy(); bot_mask[:h // 2, :] = False
        bot_all  = _h_slots_uniform(bot_mask, font_px, margin, min_w, line_spacing)
        bot_sl   = bot_all[max(0, len(bot_all) - max_slots // 2):]
        return top_sl + bot_sl

    raise ValueError(f"未知 style: {style!r}，可选: {STYLES}")


def scanline_wrap(
    mask: np.ndarray,
    text: str,
    *,
    font_px: int = 48,
    font_path: Optional[str] = None,
    style: str = "h_top",
    y_step: Optional[int] = None,
) -> List[TextLine]:
    """plan_layout + fill_slots 组合入口（旧接口兼容）。"""
    slots = plan_layout(mask, font_px, style=style)
    return fill_slots(slots, text, font_path=font_path, font_px=font_px)


def draw_lines(
    rgb: np.ndarray,
    lines: List[TextLine],
    fg: Tuple[int, int, int],
    *,
    font_px: int = 48,
    font_path: Optional[str] = None,
    stroke_width: int = 2,
) -> np.ndarray:
    """render_lines 别名（旧接口兼容）。"""
    return render_lines(rgb, lines, fg, font_path=font_path,
                        font_px=font_px, stroke_width=stroke_width)
