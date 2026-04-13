"""
海报排版：三层分离 + 多风格支持

  1. plan_layout(mask, font_px, style=...)
       纯几何，扫描 mask，按 style 输出 LineSlot 列表
       支持: h_top / h_bottom / h_center / v_right / v_left / surround

  2. fill_slots(slots, corpus, font_path, font_px)
       纯文本分配：横排按词贪心填充，竖排按字符填充

  3. render_lines(rgb, lines, fg, font_path, font_px)
       纯渲染：横排左对齐，竖排逐字绘制
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = ImageDraw = ImageFont = None

# 支持的布局风格
STYLES = ("h_top", "h_bottom", "h_center", "v_right", "v_left", "surround")


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class LineSlot:
    """一行/列文字的几何槽位（纯布局，没有文字内容）。"""
    y: int          # 顶部 y
    x0: int         # 左边界
    x1: int         # 右边界
    height: int     # 槽位高度（横排 ≈ font_px；竖排 = N*font_px）
    direction: str = "h"   # "h" 横排  "v" 竖排


@dataclass
class TextLine:
    """已填充文字的槽位，用于渲染。"""
    y: int
    x0: int
    x1: int
    text: str
    direction: str = "h"              # "h" 横排  "v" 竖排
    align: str = "left"               # "left" / "center" / "right"（横排有效）
    fg: Tuple[int, int, int] = (248, 250, 252)   # 文字颜色（per-line）
    font_px: int = 48                 # 该行字号（per-line，支持层次大小）


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
# 第一层：plan_layout — 纯几何
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


def _h_slots(
    mask: np.ndarray,
    font_px: int,
    margin: int,
    min_w: int,
    line_spacing: float,
) -> List[LineSlot]:
    """扫描 mask 得到所有有效横排槽位（从上到下）。"""
    h, w = mask.shape[:2]
    step = int(font_px * (1.0 + line_spacing))
    slots: List[LineSlot] = []
    y = 0
    while y + font_px <= h:
        band = mask[y: y + font_px, :]
        col_clear = band.all(axis=0)
        run = _widest_run(col_clear)
        if run is not None:
            avail_w = run[1] - run[0] - 2 * margin
            if avail_w >= min_w:
                slots.append(LineSlot(
                    y=y,
                    x0=run[0] + margin,
                    x1=run[1] - margin,
                    height=font_px,
                    direction="h",
                ))
        y += step
    return slots


def _v_slots(
    mask: np.ndarray,
    font_px: int,
    margin: int,
    min_h_chars: int,
    col_spacing: float,
    side: str,
) -> List[LineSlot]:
    """扫描 mask 得到竖排槽位。side='right' 从右往左，'left' 从左往右。"""
    h, w = mask.shape[:2]
    col_step = int(font_px * (1.0 + col_spacing))
    min_h = min_h_chars * font_px
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
        row_clear = strip.all(axis=1)           # 每行在这个列带里是否全 True
        run = _widest_run(row_clear)
        if run is None:
            continue
        avail_h = run[1] - run[0] - 2 * margin
        if avail_h >= min_h:
            slots.append(LineSlot(
                y=run[0] + margin,
                x0=x,
                x1=x + font_px,
                height=avail_h,
                direction="v",
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
    """
    纯几何：按 style 扫描 mask，返回槽位列表。

    style 可选:
      h_top      横排，优先顶部
      h_bottom   横排，优先底部
      h_center   横排，居中区域
      v_right    竖排，从右侧开始（列从右往左）
      v_left     竖排，从左侧开始
      surround   环绕：上半区横排 + 下半区横排
    """
    h, w = mask.shape[:2]
    min_w = min_width_chars * font_px

    if style in ("h_top", "h_bottom", "h_center"):
        all_slots = _h_slots(mask, font_px, margin, min_w, line_spacing)
        n = len(all_slots)
        if style == "h_top":
            return all_slots[:max_slots]
        elif style == "h_bottom":
            return all_slots[max(0, n - max_slots):]
        else:  # h_center
            mid = n // 2
            half = max_slots // 2
            return all_slots[max(0, mid - half): mid + half]

    elif style in ("v_right", "v_left"):
        side = style[2:]   # "right" or "left"
        slots = _v_slots(mask, font_px, margin, min_height_chars, col_spacing, side)
        return slots[:max_slots]

    elif style == "surround":
        # 上半区：h_top
        top_mask = mask.copy()
        top_mask[h // 2 :, :] = False
        top_slots = _h_slots(top_mask, font_px, margin, min_w, line_spacing)[:max_slots // 2]

        # 下半区：h_bottom（取最后几行）
        bot_mask = mask.copy()
        bot_mask[: h // 2, :] = False
        bot_all = _h_slots(bot_mask, font_px, margin, min_w, line_spacing)
        bot_slots = bot_all[max(0, len(bot_all) - max_slots // 2):]

        return top_slots + bot_slots

    raise ValueError(f"未知 style: {style!r}，可选: {STYLES}")


# ---------------------------------------------------------------------------
# 第二层：fill_slots — 纯文本分配
# ---------------------------------------------------------------------------

def fill_slots(
    slots: List[LineSlot],
    corpus: str,
    font_path: Optional[str] = None,
    font_px: int = 48,
    align: str = "left",
    fg: Tuple[int, int, int] = (248, 250, 252),
) -> List[TextLine]:
    """
    将 corpus 分配到槽位，返回 TextLine 列表。

    横排槽（direction='h'）：按空格切词，贪心填充每行。
    竖排槽（direction='v'）：按字符填充，每槽最多放 height//font_px 个字符。

    align: "left" / "center" / "right"，横排时控制文字水平位置。
    fg: 文字颜色，per-line 存储在 TextLine 中。
    """
    if not slots or not corpus.strip():
        return []

    font = _load_font(font_path, font_px)
    lines: List[TextLine] = []

    words = corpus.split()
    chars = [c for c in corpus if c.strip()]

    wi = 0   # 横排词指针（跨槽顺序填词）

    for slot in slots:
        if slot.direction == "h":
            if wi >= len(words):
                continue
            avail_w = slot.x1 - slot.x0
            line_words: List[str] = []
            used_w = 0
            while wi < len(words):
                word = words[wi]
                word_w = _text_width(word, font)
                # 中文词间无分隔；英文词间留一个空格宽度
                sep_w = _text_width(" ", font) if (line_words and (
                    line_words[-1][-1].isascii() or word[0].isascii())) else 0
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
                    direction="h",
                    align=align,
                    fg=fg,
                    font_px=font_px,
                ))

        else:  # direction == "v"
            # 竖排：每列槽位独立从语料头开始，避免多列之间字符接续导致顺序混乱
            capacity = slot.height // font_px
            chunk = "".join(chars[:capacity])
            if chunk:
                lines.append(TextLine(
                    y=slot.y, x0=slot.x0, x1=slot.x1,
                    text=chunk,
                    direction="v",
                    align=align,
                    fg=fg,
                    font_px=font_px,
                ))

    return lines


def _join_words(words: List[str]) -> str:
    if not words:
        return ""
    result = words[0]
    for w in words[1:]:
        sep = " " if (result[-1].isascii() or w[0].isascii()) else ""
        result += sep + w
    return result


# ---------------------------------------------------------------------------
# 第三层：render_lines — 纯渲染
# ---------------------------------------------------------------------------

def render_lines(
    rgb: np.ndarray,
    lines: List[TextLine],
    fg: Optional[Tuple[int, int, int]] = None,   # 保留兼容旧接口，优先使用 ln.fg
    font_path: Optional[str] = None,
    font_px: int = 48,
    stroke_width: int = 2,
) -> np.ndarray:
    """
    横排：按 ln.align 定位后绘制整行文本。
    竖排：逐字在 (x0, y + i*ln.font_px) 绘制，字符居中于列宽。
    每行独立使用 ln.fg 颜色和 ln.font_px 字号；带对比描边提升可读性。
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
        line_fg = ln.fg if ln.fg is not None else (fg or (248, 250, 252))
        line_fpx = ln.font_px if ln.font_px else font_px
        font = _get_font(line_fpx)

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
            dr.text(
                (x, ln.y),
                ln.text,
                fill=line_fg,
                font=font,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill,
            )
        else:  # vertical
            col_w = ln.x1 - ln.x0
            for i, char in enumerate(ln.text):
                char_w = _text_width(char, font)
                x_off = max(0, (col_w - char_w) // 2)
                dr.text(
                    (ln.x0 + x_off, ln.y + i * line_fpx),
                    char,
                    fill=line_fg,
                    font=font,
                    stroke_width=stroke_width,
                    stroke_fill=stroke_fill,
                )

    return np.asarray(img)


# ---------------------------------------------------------------------------
# 兼容旧接口（pipeline.py 调用）
# ---------------------------------------------------------------------------

def scanline_wrap(
    mask: np.ndarray,
    text: str,
    *,
    font_px: int = 48,
    font_path: Optional[str] = None,
    style: str = "h_top",
    y_step: Optional[int] = None,   # 旧参数，忽略
) -> List[TextLine]:
    """plan_layout + fill_slots 组合入口。"""
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
    """render_lines 别名。"""
    return render_lines(rgb, lines, fg, font_path=font_path, font_px=font_px, stroke_width=stroke_width)
