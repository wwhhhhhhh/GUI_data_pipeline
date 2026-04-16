#!/usr/bin/env python3
"""
从 JSON 文件批量跑 poster_pipeline，输出带 layout bbox 信息的新 JSON。

输入：标准 JSON 数组（每条含 img_path, masks, bboxes, cat_names, img_id 等）。
输出：
  1. 新 JSON（_layout.json）：在每条记录上追加 matched_layouts 字段。
  2. 可选：逐张保存渲染 preview / debug 中间图像。

用法示例：
  python poster_pipeline/run_json.py \\
      --json            data.json \\
      --img_dir         /images \\
      --out_dir         /output \\
      --n               20 \\
      --max_layouts     10 \\
      --save_images \\
      --save_debug_images
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from pycocotools import mask as mask_utils
except ImportError:
    print("需要 pycocotools：pip install pycocotools", file=sys.stderr)
    sys.exit(1)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from poster_pipeline.pipeline import (  # noqa: E402
    make_combined_mask,
    run_multi_layouts,
    run_poster_pipeline,
    save_debug_json,
)

SUBJECT_LABELS = {
    "person",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
}

_FONT_CANDIDATES = [
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]

CORPUS = "华为 智慧生活 影像旗舰 极致性能"


def _find_font() -> str | None:
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            return p
    return None


def _find_image(img_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp",
                ".JPG", ".JPEG", ".PNG", ".WEBP", ".BMP"):
        p = img_dir / (stem + ext)
        if p.exists():
            return p
    for p in sorted(img_dir.glob(stem + "*")):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            return p
    return None


def _decode_rle_masks(raw_masks: list, cat_names: list) -> list[dict]:
    result = []
    for i, item in enumerate(raw_masks):
        rle = {
            "size": item["size"],
            "counts": (
                item["counts"].encode("utf-8")
                if isinstance(item["counts"], str)
                else item["counts"]
            ),
        }
        m_dec = mask_utils.decode(rle)
        label = cat_names[i] if i < len(cat_names) else "object"
        result.append({"mask": m_dec.astype(bool), "label": label})
    return result


def _save_img(path: Path, arr: np.ndarray, *, is_color: bool = True) -> None:
    if is_color and arr.ndim == 3:
        cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(str(path), arr)


# ---------------------------------------------------------------------------
# 单条处理
# ---------------------------------------------------------------------------

def process_entry(
    entry:             dict,
    img_dir:           Path,
    out_dir:           Path | None,
    font_path:         str | None,
    idx:               int,
    *,
    dilate_iter:       int,
    comp_dilate_iter:  int,
    complexity_thresh: float,
    max_zones:         int,
    font_px:           int,
    font_px_min:       int,
    max_layouts:       int,
    save_images:       bool,
    save_debug_images: bool,
) -> dict | None:
    """处理单条记录，返回追加了 matched_layouts 的 entry（副本），或 None。"""

    img_path_s3 = entry.get("img_path", "")
    stem = img_path_s3.rstrip("/").rsplit("/", 1)[-1]
    local_path = _find_image(img_dir, stem)
    if local_path is None:
        print(f"  [跳过] 找不到图片: {stem!r}")
        return None

    bgr = cv2.imread(str(local_path))
    if bgr is None:
        print(f"  [跳过] 读取失败: {local_path}")
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = rgb.shape[:2]

    # ── 解码 masks ─────────────────────────────────────────────────────
    raw_masks = entry.get("masks", [])
    cat_names = entry.get("cat_names", [])
    masks = _decode_rle_masks(raw_masks, cat_names) if raw_masks else []
    subject_labels = SUBJECT_LABELS if masks else None
    forbidden = set(cat_names) & SUBJECT_LABELS if masks else set()

    print(f"  {local_path.name}  ({img_w}×{img_h})  masks={len(masks)}  "
          f"禁区={forbidden or '(无主体)'}", end="")

    # ── 多 layout 生成 ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    ml_result = run_multi_layouts(
        rgb, masks,
        subject_labels    = subject_labels,
        font_px           = font_px,
        font_px_min       = font_px_min,
        dilate_iter       = dilate_iter,
        comp_dilate_iter  = comp_dilate_iter,
        complexity_thresh = complexity_thresh,
        min_area_ratio    = 0.03,
        max_zones         = max_zones,
        max_layouts       = max_layouts,
        seed              = idx,
    )
    t1 = time.perf_counter()

    n_layouts = len(ml_result["matched_layouts"])
    print(f"  layouts={n_layouts}  {t1-t0:.2f}s", end="")

    # ── 构造输出 entry（原始数据 + matched_layouts）────────────────────
    out_entry = {}
    for k, v in entry.items():
        out_entry[k] = v
    out_entry["matched_layouts"] = ml_result["matched_layouts"]

    # ── 可选：保存图像 ────────────────────────────────────────────────
    if (save_images or save_debug_images) and out_dir is not None:
        img_id  = entry.get("img_id", stem) or stem
        safe_id = img_id.replace("/", "_").replace("\\", "_")
        out_sub = out_dir / f"{idx:06d}_{safe_id}"
        out_sub.mkdir(parents=True, exist_ok=True)

        if save_images:
            # 跑一次完整 pipeline 获取渲染结果
            full_result = run_poster_pipeline(
                rgb, masks,
                subject_labels    = subject_labels,
                corpus_text       = CORPUS,
                font_path         = font_path,
                font_px           = font_px,
                font_px_min       = font_px_min,
                dilate_iter       = dilate_iter,
                comp_dilate_iter  = comp_dilate_iter,
                complexity_thresh = complexity_thresh,
                min_area_ratio    = 0.03,
                max_zones         = max_zones,
            )
            _save_img(out_sub / "image.png", rgb)
            _save_img(out_sub / "preview.png", full_result["preview"])
            save_debug_json(str(out_sub / "debug.json"), full_result["debug"])

            if save_debug_images:
                combined = full_result.get("combined_mask")
                if combined is not None:
                    _save_img(out_sub / "combined_mask.png", combined)
                forb = full_result.get("forb_mask")
                if forb is not None:
                    _save_img(out_sub / "subject_mask.png",
                              forb.astype(np.uint8) * 255, is_color=False)
                comp = full_result.get("complexity")
                if comp is not None:
                    _save_img(out_sub / "complexity_map.png",
                              ((1.0 - comp) * 255).astype(np.uint8), is_color=False)
                writable = full_result.get("writable")
                if writable is not None:
                    _save_img(out_sub / "writable_mask.png",
                              writable.astype(np.uint8) * 255, is_color=False)

        print(f"  -> {out_sub.name}", end="")

    print()
    return out_entry


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="从 JSON 批量跑 poster_pipeline，输出带 layout 的新 JSON")
    ap.add_argument("--json",    required=True, help="输入 JSON 文件路径（顶层为数组）")
    ap.add_argument("--img_dir", required=True, help="图片目录（本地）")
    ap.add_argument("--out_dir", default=None,  help="图像输出目录（默认 JSON 同级 _out/）")
    ap.add_argument("--out_json",default=None,  help="输出 JSON 路径（默认 <input>_layout.json）")
    ap.add_argument("--n",       type=int, default=0, help="最多处理 N 条（0=全部）")
    ap.add_argument("--font",    default=None,  help="中文字体路径")
    # ── pipeline 参数 ─────────────────────────────────────────────────────
    ap.add_argument("--max_zones",        type=int,   default=3,    help="最多排版区域数")
    ap.add_argument("--dilate_iter",      type=int,   default=14,   help="主体禁区膨胀步数")
    ap.add_argument("--comp_dilate",      type=int,   default=6,    help="复杂度区域膨胀步数")
    ap.add_argument("--complexity_thresh",type=float, default=0.50, help="复杂度阈值 0~1")
    ap.add_argument("--font_px",          type=int,   default=56,   help="基础字号像素")
    ap.add_argument("--font_px_min",      type=int,   default=48,   help="全图最小字号像素")
    # ── 多 layout 参数 ────────────────────────────────────────────────────
    ap.add_argument("--max_layouts",      type=int,   default=10,   help="每张图最多生成几种 layout")
    # ── 输出控制 ──────────────────────────────────────────────────────────
    ap.add_argument("--save_images",      action="store_true", help="保存 image/preview/debug.json")
    ap.add_argument("--save_debug_images",action="store_true", help="保存 mask/complexity 等调试图（隐含 --save_images）")
    args = ap.parse_args()

    if args.save_debug_images:
        args.save_images = True

    json_path = Path(args.json)
    img_dir   = Path(args.img_dir)
    out_dir   = (Path(args.out_dir)
                 if args.out_dir
                 else json_path.parent / (json_path.stem + "_out"))
    out_json  = (Path(args.out_json)
                 if args.out_json
                 else json_path.parent / (json_path.stem + "_layout.json"))
    font_path = args.font or _find_font()

    if args.save_images:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"输入 JSON:         {json_path}")
    print(f"图片目录:          {img_dir}")
    print(f"输出 JSON:         {out_json}")
    if args.save_images:
        print(f"图像输出:          {out_dir}")
    print(f"max_layouts:       {args.max_layouts}")
    print(f"save_images:       {args.save_images}")
    print(f"save_debug_images: {args.save_debug_images}")
    print(f"字体:              {font_path or '(Pillow 内置)'}")
    print()

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        for key in ("data", "items", "annotations", "images"):
            if key in data and isinstance(data[key], list):
                entries = data[key]
                break
        else:
            print("无法识别 JSON 结构")
            sys.exit(1)
    else:
        print(f"无法识别 JSON 类型: {type(data)}")
        sys.exit(1)

    total = len(entries)
    print(f"共 {total} 条记录\n")

    output_entries: list = []
    ok = err = 0

    for idx, entry in enumerate(entries):
        if args.n > 0 and (ok + err) >= args.n:
            break
        print(f"[{idx+1}/{total}]", end=" ")
        result = process_entry(
            entry, img_dir,
            out_dir if args.save_images else None,
            font_path, idx,
            dilate_iter       = args.dilate_iter,
            comp_dilate_iter  = args.comp_dilate,
            complexity_thresh = args.complexity_thresh,
            max_zones         = args.max_zones,
            font_px           = args.font_px,
            font_px_min       = args.font_px_min,
            max_layouts       = args.max_layouts,
            save_images       = args.save_images,
            save_debug_images = args.save_debug_images,
        )
        if result is not None:
            output_entries.append(result)
            ok += 1
        else:
            err += 1

    # ── 写出带 layout 的新 JSON ─────────────────────────────────────────
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_entries, f, ensure_ascii=False, indent=2)

    print(f"\n完成：成功 {ok} 张，跳过/失败 {err} 张。")
    print(f"输出 JSON: {out_json}")
    if args.save_images:
        print(f"图像输出:  {out_dir}")


if __name__ == "__main__":
    main()
