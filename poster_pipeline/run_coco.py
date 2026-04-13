#!/usr/bin/env python3
"""用 COCO val2017 图片 + 实例分割标注跑 poster_pipeline，取前 10 张有标注的图。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from poster_pipeline.pipeline import run_poster_pipeline, save_debug_json  # noqa: E402

IMG_DIR = Path("/Users/wuwenhao/Downloads/val2017")
ANN_FILE = Path("/Users/wuwenhao/Downloads/annotations/instances_val2017.json")
OUT_DIR = _ROOT / "_coco_demo_out"

# 只避开人物和动物
SUBJECT_LABELS = {
    "person",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
}

FONT_PATH = "/System/Library/Fonts/STHeiti Medium.ttc"
FONT_PX = 56

# 10 张图循环使用多种海报风格（含复合 zone 样式）
STYLE_CYCLE = [
    "top_title",    # 上标题 + 下副标
    "sidebar_r",    # 顶部横排 + 右侧竖列
    "top_banner",   # 顶部全幅大标题（top_left+top+top_right 合并）
    "h_banner",     # 上下全幅横排
    "v_columns",    # 左右全高竖排（left_tall+right_tall）
    "l_shape_r",    # 宽顶 + 右竖（Γ 形）
    "frame_wide",   # 全边框宽幅
    "diag_wide",    # 左上+右下宽对角
    "sidebar_l",    # 顶部横排 + 左侧竖列
    "bot_banner",   # 底部全幅横排
]


def coco_poly_to_mask(segmentation, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in segmentation:
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask.astype(bool)


def load_coco_data(ann_file: Path, img_dir: Path, n: int = 10):
    with open(ann_file) as f:
        data = json.load(f)
    cat_map = {c["id"]: c["name"] for c in data["categories"]}
    ann_by_img = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        seg = ann.get("segmentation", [])
        if not isinstance(seg, list) or len(seg) == 0:
            continue
        ann_by_img[ann["image_id"]].append(ann)

    selected = []
    for img_info in data["images"]:
        img_path = img_dir / img_info["file_name"]
        if not img_path.exists():
            continue
        anns = ann_by_img.get(img_info["id"], [])
        if not anns:
            continue
        selected.append((img_info, anns))
        if len(selected) >= n:
            break
    return selected, cat_map


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"加载标注: {ANN_FILE}")
    items, cat_map = load_coco_data(ANN_FILE, IMG_DIR, n=10)
    print(f"找到 {len(items)} 张可用图片")

    for idx, (img_info, anns) in enumerate(items):
        img_path = IMG_DIR / img_info["file_name"]
        h, w = img_info["height"], img_info["width"]
        stem = img_path.stem
        style = STYLE_CYCLE[idx]

        print(f"\n[{idx+1}/10] {img_path.name}  ({w}x{h})  风格={style}")

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print("  读取失败，跳过")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        masks = []
        for ann in anns:
            m = coco_poly_to_mask(ann["segmentation"], h, w)
            label = cat_map.get(ann["category_id"], "object")
            masks.append({"mask": m, "label": label, "score": 1.0})

        labels_in_img = set(m["label"] for m in masks)
        forbidden = labels_in_img & SUBJECT_LABELS
        print(f"  禁区: {forbidden or '(无人物/动物)'}")

        try:
            result = run_poster_pipeline(
                rgb,
                masks,
                subject_labels=SUBJECT_LABELS,
                corpus_text="华为 智慧生活 影像旗舰 极致性能",
                font_path=FONT_PATH,
                font_px=FONT_PX,
                layout_style=style,
                dilate_iter=14,
                min_zone_quality=0.30,
            )
        except Exception as e:
            import traceback
            print(f"  pipeline 出错: {e}")
            traceback.print_exc()
            continue

        out_sub = OUT_DIR / stem
        out_sub.mkdir(exist_ok=True)

        Image.fromarray(rgb).save(out_sub / "image.png")
        Image.fromarray(result["preview"]).save(out_sub / "preview.png")

        wb = result.get("writable_binary")
        if wb is not None:
            Image.fromarray((wb.astype(np.uint8) * 255)).save(out_sub / "writable_mask.png")

        comp = result.get("complexity")
        if comp is not None:
            c_vis = ((1 - comp) * 255).astype(np.uint8)  # 白=低复杂度=好
            Image.fromarray(c_vis).save(out_sub / "quality_map.png")

        save_debug_json(out_sub / "debug.json", result["debug"])

        zones_info = result["debug"].get("zones", [])
        n_lines = result["debug"].get("n_lines", 0)
        print(f"  zones={[z['name'] for z in zones_info]}  lines={n_lines}  -> {out_sub}")

    print(f"\n完成，输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
