#!/usr/bin/env python3
"""
从 JSONL 文件批量跑 poster_pipeline。

JSONL 格式（每行一个 JSON 对象）：
  {
    "img_path": "s3://.../<folder>/<stem>",   # 用末段文件名在本地查找图片
    "masks":    [{"size": [H, W], "counts": "..."}, ...],   # RLE 格式
    "bboxes":   [[x0,y0,x1,y1], ...],
    "cat_names": ["boat", ...],
    "img_id":   "...",
    "confidence_scores": [...]
  }

masks 为空列表时，视为全图无主体，仅用复杂度图决定可写区域。

用法：
  python poster_pipeline/run_jsonl.py \\
      --jsonl  /path/to/data.jsonl \\
      --img_dir /path/to/images \\
      --out_dir /path/to/output  \\
      --n 10
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import cycle
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    from pycocotools import mask as mask_utils
except ImportError:
    print("需要 pycocotools：pip install pycocotools", file=sys.stderr)
    sys.exit(1)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from poster_pipeline.pipeline import run_poster_pipeline, save_debug_json  # noqa: E402

# 布局风格循环（对应 layout_scanline.STYLES）
STYLE_CYCLE = [
    "h_top",
    "h_bottom",
    "h_center",
    "v_right",
    "v_left",
    "surround",
]

# 常见中文字体路径（跨平台候选）
FONT_CANDIDATES = [
    # macOS
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    # Windows
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    # Linux
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]

FONT_PX = 56
CORPUS = "华为 智慧生活 影像旗舰 极致性能"

# 需要作为主体避开的类别（与 COCO 类名一致）
SUBJECT_LABELS = {
    "person",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _find_font() -> str | None:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return p
    return None


def _find_image(img_dir: Path, stem: str) -> Path | None:
    """在 img_dir 中查找 stem（无扩展名），支持常见图片格式。"""
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        p = img_dir / (stem + ext)
        if p.exists():
            return p
        # 有些文件系统大小写敏感
        p2 = img_dir / (stem + ext.upper())
        if p2.exists():
            return p2
    # 模糊匹配：stem 作为前缀
    matches = sorted(img_dir.glob(stem + "*"))
    for m in matches:
        if m.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            return m
    return None


def decode_rle_masks(raw_masks: list, img_h: int, img_w: int) -> list[dict]:
    """
    将 JSONL 中的 RLE mask 列表解码为 pipeline 所需格式：
      [{"mask": bool HxW, "label": str}, ...]

    cat_names 信息通过 raw_masks 与调用方传入的 cat_names 对应。
    """
    result = []
    for item in raw_masks:
        rle = {
            "size": item["size"],
            "counts": (
                item["counts"].encode("utf-8")
                if isinstance(item["counts"], str)
                else item["counts"]
            ),
        }
        m_dec = mask_utils.decode(rle)   # HxW, 0/1
        result.append(m_dec.astype(bool))
    return result


def process_entry(
    entry: dict,
    img_dir: Path,
    out_dir: Path,
    style: str,
    font_path: str | None,
) -> bool:
    """处理单条 JSONL 记录，返回是否成功。"""
    # ── 找本地图片 ──────────────────────────────────────────────────────────
    img_path_s3 = entry.get("img_path", "")
    stem = img_path_s3.rstrip("/").rsplit("/", 1)[-1]   # 取 S3 路径最后一段
    local_path = _find_image(img_dir, stem)
    if local_path is None:
        print(f"  [跳过] 找不到图片: {stem!r}  (在 {img_dir})")
        return False

    bgr = cv2.imread(str(local_path))
    if bgr is None:
        print(f"  [跳过] 读取失败: {local_path}")
        return False
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = rgb.shape[:2]

    # ── 解码 masks ───────────────────────────────────────────────────────────
    raw_masks = entry.get("masks", [])
    cat_names = entry.get("cat_names", [])

    masks: list[dict] = []
    if raw_masks:
        bool_masks = decode_rle_masks(raw_masks, img_h, img_w)
        for i, bm in enumerate(bool_masks):
            label = cat_names[i] if i < len(cat_names) else "object"
            masks.append({"mask": bm, "label": label})

    has_subject = len(masks) > 0
    forbidden = set(cat_names) & SUBJECT_LABELS if has_subject else set()
    subject_labels = SUBJECT_LABELS if has_subject else None

    print(f"  图片: {local_path.name}  ({img_w}x{img_h})  "
          f"masks={len(masks)}  禁区={forbidden or '(无主体，全图复杂度过滤)'}  风格={style}")

    # ── 运行 pipeline ────────────────────────────────────────────────────────
    try:
        result = run_poster_pipeline(
            rgb,
            masks,
            subject_labels=subject_labels,
            corpus_text=CORPUS,
            font_path=font_path,
            font_px=FONT_PX,
            layout_style=style,
            dilate_iter=14,
            complexity_thresh=0.50,
            min_area_ratio=0.03,
        )
    except Exception as e:
        import traceback
        print(f"  [错误] pipeline 出错: {e}")
        traceback.print_exc()
        return False

    # ── 输出目录 ─────────────────────────────────────────────────────────────
    img_id = entry.get("img_id", stem)
    out_sub = out_dir / img_id
    out_sub.mkdir(parents=True, exist_ok=True)

    # 原图
    Image.fromarray(rgb).save(out_sub / "image.png")

    # 渲染结果
    Image.fromarray(result["preview"]).save(out_sub / "preview.png")

    # 主体 mask（白=主体区域，黑=背景）
    subj = result.get("subj_mask")
    if subj is not None:
        Image.fromarray((subj.astype(np.uint8) * 255)).save(out_sub / "subject_mask.png")

    # 复杂度图连续值（白=低复杂/适合写字，黑=高复杂）
    comp = result.get("complexity")
    if comp is not None:
        c_vis = ((1.0 - comp) * 255).astype(np.uint8)
        Image.fromarray(c_vis).save(out_sub / "complexity_map.png")

        # 复杂度二值化（白=低复杂可写，黑=高复杂）
        thresh = result["debug"].get("complexity_thresh", 0.50)
        comp_binary = (comp <= thresh).astype(np.uint8) * 255
        Image.fromarray(comp_binary).save(out_sub / "complexity_binary.png")

    # 可写区域（白=可写，黑=禁区）
    writable = result.get("writable")
    if writable is not None:
        Image.fromarray((writable.astype(np.uint8) * 255)).save(out_sub / "writable_mask.png")

    # debug JSON
    save_debug_json(str(out_sub / "debug.json"), result["debug"])

    n_lines = result["debug"].get("n_lines", 0)
    skip = result["debug"].get("skip_reason", "")
    writable_ratio = result["debug"].get("writable_ratio", 0)
    print(f"  writable={writable_ratio:.1%}  lines={n_lines}  "
          f"{('skip: ' + skip) if skip else ''}-> {out_sub}")
    return True


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="从 JSONL 批量跑 poster_pipeline")
    parser.add_argument("--jsonl",   required=True,  help="JSONL 文件路径")
    parser.add_argument("--img_dir", required=True,  help="图片目录（本地）")
    parser.add_argument("--out_dir", default=None,   help="输出目录（默认在 JSONL 同级 _out）")
    parser.add_argument("--n",       type=int, default=0, help="最多处理 N 条（0=全部）")
    parser.add_argument("--font",    default=None,   help="中文字体路径（不填自动检测）")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    img_dir    = Path(args.img_dir)
    out_dir    = Path(args.out_dir) if args.out_dir else jsonl_path.parent / (jsonl_path.stem + "_out")
    font_path  = args.font or _find_font()

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"JSONL:   {jsonl_path}")
    print(f"图片目录: {img_dir}")
    print(f"输出目录: {out_dir}")
    print(f"字体:    {font_path or '(Pillow 内置)'}")

    style_iter = cycle(STYLE_CYCLE)
    ok = err = 0

    with open(jsonl_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[行{lineno}] JSON 解析失败: {e}")
                continue

            style = next(style_iter)
            print(f"\n[{lineno}]", end=" ")
            success = process_entry(entry, img_dir, out_dir, style, font_path)
            if success:
                ok += 1
            else:
                err += 1

            if args.n > 0 and (ok + err) >= args.n:
                break

    print(f"\n完成：成功 {ok} 张，跳过/失败 {err} 张。输出: {out_dir}")


if __name__ == "__main__":
    main()
