#!/usr/bin/env python3
"""合成一张「天空背景 + 椭圆主体」与伪 SAM3 mask，跑通管线并落盘。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from poster_pipeline.pipeline import run_poster_pipeline, save_debug_json  # noqa: E402


def synthetic_scene(h: int = 360, w: int = 640) -> tuple[np.ndarray, list]:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sky = np.stack(
        [
            120 + 40 * np.sin(xx / 80) + yy * 0.05,
            170 + 30 * np.cos(xx / 90) + yy * 0.08,
            230 - yy * 0.15,
        ],
        axis=-1,
    )
    noise = np.random.randn(h, w, 3).astype(np.float32) * 18
    rgb = np.clip(sky + noise, 0, 255).astype(np.uint8)
    # 纹理块（高复杂度）
    rgb[200:320, 100:260] = np.clip(rgb[200:320, 100:260].astype(np.int16) + np.random.randint(-40, 40, (120, 160, 3)), 0, 255).astype(
        np.uint8
    )

    cy, cx = h // 2, w // 2
    ellipse = ((xx - cx) / (w * 0.22)) ** 2 + ((yy - cy) / (h * 0.35)) ** 2 <= 1.0
    masks = [{"mask": ellipse.astype(bool), "label": "subject", "score": 1.0}]
    return rgb, masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(_ROOT / "_poster_demo_out"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rgb, masks = synthetic_scene()
    np.save(out / "image.npy", rgb)
    # 伪 SAM3：主体 label 需要避开
    result = run_poster_pipeline(
        rgb,
        masks,
        subject_labels={"subject"},
        corpus_text="合成海报数据扫描线折行 不规则区域 反差色 无网环境可解析",
        quantile=0.58,
    )

    from PIL import Image

    Image.fromarray(rgb).save(out / "image.png")
    Image.fromarray(masks[0]["mask"].astype(np.uint8) * 255).save(out / "subject_mask.png")
    wb = result.get("writable_binary")
    if wb is not None:
        Image.fromarray((wb.astype(np.uint8) * 255)).save(out / "writable_mask.png")
    Image.fromarray(result["preview"]).save(out / "preview.png")
    save_debug_json(out / "layout_debug.json", result["debug"])

    meta = {
        "image": str(out / "image.png"),
        "masks": [{"label": "subject", "png": str(out / "subject_mask.png")}],
        "note": "真实 SAM3 请替换 masks 为多实例 JSON/npy，并在 pipeline 中映射 subject_labels",
    }
    with open(out / "synthetic_manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
