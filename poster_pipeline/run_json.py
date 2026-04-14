#!/usr/bin/env python3
"""
从 JSON 文件批量跑 poster_pipeline。

JSON 格式（文件顶层为数组，每个元素一条记录）：
  [
    {
      "img_path": "s3://.../<folder>/<stem>",
      "masks":    [{"size": [H, W], "counts": "RLE字符串"}, ...],
      "bboxes":   [[x0,y0,x1,y1], ...],
      "cat_names": ["boat", ...],
      "img_id":   "...",
      "confidence_scores": [...]
    },
    ...
  ]

masks 为空列表时，视为全图无主体，仅用复杂度决定可写区域。

用法示例：
  python poster_pipeline/run_json.py \\
      --json            /path/to/data.json \\
      --img_dir         /path/to/images \\
      --out_dir         /path/to/output \\
      --n               20 \\
      --max_zones       3 \\
      --dilate_iter     14 \\
      --comp_dilate     8 \\
      --complexity_thresh 0.50 \\
      --font_px         56
"""
from __future__ import annotations

import argparse
import json
import sys
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

from poster_pipeline.pipeline import make_combined_mask, run_poster_pipeline, save_debug_json  # noqa: E402

# 需要作为主体避开的类别（与 COCO 类名一致）
SUBJECT_LABELS = {
    "person",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
}

# 常见中文字体路径（跨平台候选，自动选第一个存在的）
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
    """在 img_dir 中查找 stem（无扩展名）对应的图片文件。"""
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
    """将 JSON 中的 RLE mask 解码为 pipeline 格式 [{"mask": bool HxW, "label": str}]。"""
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


def _save_outputs(out_sub: Path, rgb: np.ndarray, result: dict,
                  complexity_thresh: float) -> None:
    """统一存储所有输出图像与 JSON。"""
    # 原图
    Image.fromarray(rgb).save(out_sub / "image.png")

    # 排版渲染结果
    Image.fromarray(result["preview"]).save(out_sub / "preview.png")

    # 三层合并可视化（白=可写，红=主体禁区，黄=复杂度禁区）
    combined = result.get("combined_mask")
    if combined is not None:
        Image.fromarray(combined).save(out_sub / "combined_mask.png")

    # 主体 mask（白=主体含安全边距，黑=其余）
    forb = result.get("forb_mask")
    if forb is not None:
        Image.fromarray((forb.astype(np.uint8) * 255)).save(out_sub / "subject_mask.png")

    # 复杂度连续图（白=低复杂/适合写字）
    comp = result.get("complexity")
    if comp is not None:
        c_vis = ((1.0 - comp) * 255).astype(np.uint8)
        Image.fromarray(c_vis).save(out_sub / "complexity_map.png")

        # 复杂度二值化（白=低复杂可写，黑=高复杂）
        comp_bin = (comp <= complexity_thresh).astype(np.uint8) * 255
        Image.fromarray(comp_bin).save(out_sub / "complexity_binary.png")

    # 最终可写区域
    writable = result.get("writable")
    if writable is not None:
        Image.fromarray((writable.astype(np.uint8) * 255)).save(out_sub / "writable_mask.png")

    # debug JSON
    save_debug_json(str(out_sub / "debug.json"), result["debug"])


def process_entry(
    entry:             dict,
    img_dir:           Path,
    out_dir:           Path,
    font_path:         str | None,
    idx:               int,
    *,
    dilate_iter:       int,
    comp_dilate_iter:  int,
    complexity_thresh: float,
    max_zones:         int,
    font_px:           int,
) -> bool:
    """处理单条记录，返回是否成功。"""
    # ── 找本地图片 ──────────────────────────────────────────────────────────
    img_path_s3 = entry.get("img_path", "")
    stem = img_path_s3.rstrip("/").rsplit("/", 1)[-1]
    local_path = _find_image(img_dir, stem)
    if local_path is None:
        print(f"  [跳过] 找不到图片: {stem!r}")
        return False

    bgr = cv2.imread(str(local_path))
    if bgr is None:
        print(f"  [跳过] 读取失败: {local_path}")
        return False
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ── 解码 masks ───────────────────────────────────────────────────────────
    raw_masks = entry.get("masks", [])
    cat_names = entry.get("cat_names", [])
    masks = _decode_rle_masks(raw_masks, cat_names) if raw_masks else []

    subject_labels = SUBJECT_LABELS if masks else None
    forbidden = set(cat_names) & SUBJECT_LABELS if masks else set()

    img_h, img_w = rgb.shape[:2]
    print(f"  {local_path.name}  ({img_w}×{img_h})  masks={len(masks)}  "
          f"禁区={forbidden or '(无主体，全图复杂度过滤)'}")

    # ── 运行 pipeline ────────────────────────────────────────────────────────
    try:
        result = run_poster_pipeline(
            rgb, masks,
            subject_labels    = subject_labels,
            corpus_text       = CORPUS,
            font_path         = font_path,
            font_px           = font_px,
            dilate_iter       = dilate_iter,
            comp_dilate_iter  = comp_dilate_iter,
            complexity_thresh = complexity_thresh,
            min_area_ratio    = 0.03,
            max_zones         = max_zones,
        )
    except Exception as e:
        import traceback
        print(f"  [错误] {e}")
        traceback.print_exc()
        return False

    # ── 输出目录 ─────────────────────────────────────────────────────────────
    img_id   = entry.get("img_id", stem) or stem
    safe_id  = img_id.replace("/", "_").replace("\\", "_")
    out_sub  = out_dir / f"{idx:06d}_{safe_id}"
    out_sub.mkdir(parents=True, exist_ok=True)

    _save_outputs(out_sub, rgb, result, complexity_thresh)

    # ── 终端摘要 ─────────────────────────────────────────────────────────────
    n_lines    = result["debug"].get("n_lines", 0)
    skip       = result["debug"].get("skip_reason", "")
    wr         = result["debug"].get("writable_ratio", 0)
    strategy   = result["debug"].get("strategy", "?")
    zones_info = result["debug"].get("zones", [])
    zone_sum   = [(z["position"], z["direction"]) for z in zones_info]
    print(f"  strategy={strategy}  writable={wr:.1%}  zones={zone_sum}  "
          f"lines={n_lines}  {('skip: ' + skip) if skip else ''}-> {out_sub.name}")
    return True


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="从 JSON 数组批量跑 poster_pipeline")
    ap.add_argument("--json",    required=True, help="JSON 文件路径（顶层为数组）")
    ap.add_argument("--img_dir", required=True, help="图片目录（本地）")
    ap.add_argument("--out_dir", default=None,  help="输出目录（默认 JSON 同级 _out/）")
    ap.add_argument("--n",       type=int, default=0,    help="最多处理 N 条（0=全部）")
    ap.add_argument("--font",    default=None,            help="中文字体路径")
    # ── 可配置参数 ──────────────────────────────────────────────────────────
    ap.add_argument("--max_zones",        type=int,   default=3,    help="最多排版区域数（默认 3）")
    ap.add_argument("--dilate_iter",      type=int,   default=14,   help="主体禁区膨胀步数（默认 14）")
    ap.add_argument("--comp_dilate",      type=int,   default=6,    help="复杂度区域膨胀步数（默认 6）")
    ap.add_argument("--complexity_thresh",type=float, default=0.50, help="复杂度阈值 0~1（默认 0.50）")
    ap.add_argument("--font_px",          type=int,   default=56,   help="基础字号像素（默认 56）")
    args = ap.parse_args()

    json_path = Path(args.json)
    img_dir   = Path(args.img_dir)
    out_dir   = (Path(args.out_dir)
                 if args.out_dir
                 else json_path.parent / (json_path.stem + "_out"))
    font_path = args.font or _find_font()

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"JSON:             {json_path}")
    print(f"图片目录:          {img_dir}")
    print(f"输出目录:          {out_dir}")
    print(f"字体:             {font_path or '(Pillow 内置，不支持中文)'}")
    print(f"主体膨胀:          {args.dilate_iter} 步")
    print(f"复杂度膨胀:        {args.comp_dilate} 步")
    print(f"复杂度阈值:        {args.complexity_thresh}")
    print(f"最大区域数:        {args.max_zones}")
    print(f"基础字号:          {args.font_px}px")

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
            print("无法识别 JSON 结构，期望顶层为数组或含 'data'/'items' 键的对象")
            sys.exit(1)
    else:
        print(f"无法识别 JSON 类型: {type(data)}")
        sys.exit(1)

    total = len(entries)
    print(f"共 {total} 条记录\n")

    ok = err = 0
    for idx, entry in enumerate(entries):
        if args.n > 0 and (ok + err) >= args.n:
            break
        print(f"[{idx+1}/{total}]", end=" ")
        success = process_entry(
            entry, img_dir, out_dir, font_path, idx,
            dilate_iter       = args.dilate_iter,
            comp_dilate_iter  = args.comp_dilate,
            complexity_thresh = args.complexity_thresh,
            max_zones         = args.max_zones,
            font_px           = args.font_px,
        )
        ok += success
        err += not success

    print(f"\n完成：成功 {ok} 张，跳过/失败 {err} 张。输出: {out_dir}")


if __name__ == "__main__":
    main()
