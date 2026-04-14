#!/usr/bin/env python3
"""
从 JSON 文件批量跑 poster_pipeline。

JSON 格式（文件内容为 JSON 数组，每个元素一条记录）：
  [
    {
      "img_path": "s3://.../<folder>/<stem>",   # 用末段文件名在本地查找图片
      "masks":    [{"size": [H, W], "counts": "RLE字符串"}, ...],
      "bboxes":   [[x0,y0,x1,y1], ...],
      "cat_names": ["boat", ...],
      "img_id":   "...",
      "confidence_scores": [...]
    },
    ...
  ]

masks 为空列表时，视为全图无主体，仅用复杂度图决定可写区域。

用法：
  python poster_pipeline/run_json.py \\
      --json    /path/to/data.json \\
      --img_dir /path/to/images \\
      --out_dir /path/to/output  \\
      --n 10
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

from poster_pipeline.pipeline import run_poster_pipeline, save_debug_json  # noqa: E402

# 需要作为主体避开的类别（与 COCO 类名一致）
SUBJECT_LABELS = {
    "person",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
}

FONT_PX = 56
CORPUS = "华为 智慧生活 影像旗舰 极致性能"

# 常见中文字体路径（跨平台候选）
_FONT_CANDIDATES = [
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def _find_font() -> str | None:
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            return p
    return None


def _find_image(img_dir: Path, stem: str) -> Path | None:
    """在 img_dir 中查找 stem（无扩展名），支持常见图片格式。"""
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp",
                ".JPG", ".JPEG", ".PNG", ".WEBP", ".BMP"):
        p = img_dir / (stem + ext)
        if p.exists():
            return p
    # 模糊匹配：以 stem 为前缀
    for p in sorted(img_dir.glob(stem + "*")):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            return p
    return None


def _decode_rle_masks(raw_masks: list, cat_names: list) -> list[dict]:
    """将 RLE mask 列表解码为 pipeline 所需格式。"""
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
        m_dec = mask_utils.decode(rle)          # HxW, 0/1 uint8
        label = cat_names[i] if i < len(cat_names) else "object"
        result.append({"mask": m_dec.astype(bool), "label": label})
    return result


def process_entry(
    entry:      dict,
    img_dir:    Path,
    out_dir:    Path,
    font_path:  str | None,
    idx:        int,
) -> bool:
    """处理单条记录，返回是否成功。"""
    # ── 找本地图片 ──────────────────────────────────────────────────────────
    img_path_s3 = entry.get("img_path", "")
    stem = img_path_s3.rstrip("/").rsplit("/", 1)[-1]
    local_path = _find_image(img_dir, stem)
    if local_path is None:
        print(f"  [跳过] 找不到图片: {stem!r}  (在 {img_dir})")
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

    has_subject = bool(masks)
    subject_labels = SUBJECT_LABELS if has_subject else None
    forbidden = set(cat_names) & SUBJECT_LABELS if has_subject else set()

    img_h, img_w = rgb.shape[:2]
    print(f"  {local_path.name}  ({img_w}×{img_h})  "
          f"masks={len(masks)}  禁区={forbidden or '(无主体，全图复杂度过滤)'}")

    # ── 运行 pipeline ────────────────────────────────────────────────────────
    try:
        result = run_poster_pipeline(
            rgb,
            masks,
            subject_labels=subject_labels,
            corpus_text=CORPUS,
            font_path=font_path,
            font_px=FONT_PX,
            dilate_iter=14,
            comp_dilate_iter=3,
            complexity_thresh=0.50,
            min_area_ratio=0.03,
            max_zones=3,
        )
    except Exception as e:
        import traceback
        print(f"  [错误] {e}")
        traceback.print_exc()
        return False

    # ── 输出目录 ─────────────────────────────────────────────────────────────
    img_id = entry.get("img_id", stem) or stem
    # 防止 img_id 中含路径分隔符等不合法字符
    safe_id = img_id.replace("/", "_").replace("\\", "_")
    out_sub = out_dir / f"{idx:06d}_{safe_id}"
    out_sub.mkdir(parents=True, exist_ok=True)

    # 原图
    Image.fromarray(rgb).save(out_sub / "image.png")

    # 渲染结果
    Image.fromarray(result["preview"]).save(out_sub / "preview.png")

    # 主体 mask（白=主体，黑=背景；masks 为空则全黑）
    subj = result.get("subj_mask")
    if subj is not None:
        Image.fromarray((subj.astype(np.uint8) * 255)).save(out_sub / "subject_mask.png")

    # 复杂度图（白=低复杂/适合写字，黑=高复杂）
    comp = result.get("complexity")
    thresh = result["debug"].get("complexity_thresh", 0.50)
    if comp is not None:
        c_vis = ((1.0 - comp) * 255).astype(np.uint8)
        Image.fromarray(c_vis).save(out_sub / "complexity_map.png")

        # 复杂度二值化（白=低复杂可写，黑=高复杂）
        comp_bin = (comp <= thresh).astype(np.uint8) * 255
        Image.fromarray(comp_bin).save(out_sub / "complexity_binary.png")

    # 最终可写区域（白=可写，黑=禁区）
    writable = result.get("writable")
    if writable is not None:
        Image.fromarray((writable.astype(np.uint8) * 255)).save(out_sub / "writable_mask.png")

    # debug JSON
    save_debug_json(str(out_sub / "debug.json"), result["debug"])

    n_lines      = result["debug"].get("n_lines", 0)
    skip         = result["debug"].get("skip_reason", "")
    wr           = result["debug"].get("writable_ratio", 0)
    zones_info   = result["debug"].get("zones", [])
    zone_summary = [(z["position"], z["direction"]) for z in zones_info]
    print(f"  writable={wr:.1%}  zones={zone_summary}  lines={n_lines}  "
          f"{('skip: ' + skip) if skip else ''}-> {out_sub.name}")
    return True


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="从 JSON 数组批量跑 poster_pipeline")
    parser.add_argument("--json",    required=True,  help="JSON 文件路径（内容为数组）")
    parser.add_argument("--img_dir", required=True,  help="图片目录（本地）")
    parser.add_argument("--out_dir", default=None,   help="输出目录（默认 JSON 同级 _out/）")
    parser.add_argument("--n",       type=int, default=0, help="最多处理 N 条（0=全部）")
    parser.add_argument("--font",    default=None,   help="中文字体路径（不填自动检测）")
    args = parser.parse_args()

    json_path = Path(args.json)
    img_dir   = Path(args.img_dir)
    out_dir   = Path(args.out_dir) if args.out_dir else json_path.parent / (json_path.stem + "_out")
    font_path = args.font or _find_font()

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"JSON:    {json_path}")
    print(f"图片目录: {img_dir}")
    print(f"输出目录: {out_dir}")
    print(f"字体:    {font_path or '(Pillow 内置，不支持中文)'}")

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # 兼容：数组 or {"data": [...]} / {"items": [...]}
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

    print(f"共 {len(entries)} 条记录")

    ok = err = 0
    for idx, entry in enumerate(entries):
        if args.n > 0 and (ok + err) >= args.n:
            break
        print(f"\n[{idx+1}/{len(entries)}]", end=" ")
        success = process_entry(entry, img_dir, out_dir, font_path, idx)
        if success:
            ok += 1
        else:
            err += 1

    print(f"\n完成：成功 {ok} 张，跳过/失败 {err} 张。输出: {out_dir}")


if __name__ == "__main__":
    main()
