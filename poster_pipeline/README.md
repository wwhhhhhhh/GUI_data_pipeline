# poster_pipeline — 海报自动排版工程

基于实例分割 mask 的海报自动排版工程。输入图像与主体分割结果，自动识别适合写字的区域，对 12 种布局策略打分选最优，按字号层级（从大到小）在每个区域内规划文字 bbox，支持一张图输出多种 layout 方案。

---

## 整体思路

```
输入图像 + 实例分割 masks（可为空列表）
        │
        ▼
  ① 主体禁区构建          writable_mask.py（OpenCV 加速）
     union 所有主体 mask → cv2.dilate（dilate_iter 步）→ 禁区 forb
     masks 为空时 forb 全为 False（全图参与候选）
        │
        ▼
  ② 复杂度图              writable_mask.py
     cv2.Sobel 梯度 + Laplacian 能量 → cv2.GaussianBlur 平滑
     （sigma ≈ 短边/16，范围 12~40px，更贴合实际纹理边界）
     → 区域级复杂度，归一化 [0,1]
        │
        ▼
  ③ 可写字区域             writable_mask.py
     comp_low = (complexity ≤ threshold)
     comp_low = cv2.morphologyEx(CLOSE, comp_dilate_iter 步)  ← 形态学闭运算
     writable = (~forb) & comp_low
        │
        ▼
  ④ 策略打分 → 选最优布局  auto_layout.py
     预定义 12 种候选策略 → density × abs_frac 打分 → 选最高
     支持 rank_strategies() 返回全部策略排名（用于多 layout 生成）
        │
        ▼
  ⑤ 字号层级（1–3 层随机） pipeline.py
     n_tiers ∈ {1,2,3}，随机或指定：
       1 层 → [base×1.8]            （单一字号）
       2 层 → [base×2.2, base×1.2]  （大/小）
       3 层 → [base×2.4, base×1.6, base×1.0]（大/中/小）
     所有层级 ≥ font_px_min（默认 48px），8px 对齐
        │
        ▼
  ⑥ 层次 bbox 规划 + 对齐  layout_scanline.py → plan_hierarchical
     按字号从大到小扫描放置 LineSlot → 两级对齐：
       同层级：统一 x0/x1（交集对齐）
       跨层级：按 align 方向吸附（左/右/居中）
        │
        ▼
  ⑦ 文本分配              layout_scanline.py → fill_slots
     横排：贪心填词，语料词指针跨槽顺序前进
     竖排：逐字填入，capacity = height // font_px
        │
        ▼
  ⑧ 色卡对比色选取         color_palette.py + color_contrast.py
     90 色色卡 → 按 WCAG 对比度排序 → top_k 中随机选一个
     同一张图内通过 anchor_id 尽量沿用同一色卡项（颜色一致性）
     随机选取中/英颜色名
        │
        ▼
  ⑨ 文字渲染              layout_scanline.py → render_lines
     per-line 颜色与字号，带对比描边，可选 draw_bbox
        │
        ▼
     输出：preview / matched_layouts / debug.json / meta（含 bbox/字号/颜色名/字体名）
```

---

## 模块说明

### `pipeline.py` — 端到端入口

对外暴露 `run_poster_pipeline()`、`run_multi_layouts()`、`make_combined_mask()`。

#### `run_poster_pipeline` 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `rgb` | — | HxWx3 uint8 图像 |
| `masks` | — | `[{"mask": bool HxW, "label": str}, ...]`，可为空列表 |
| `subject_labels` | `None` | 需要避开的类别集合；`None` = 全部 mask 都避开 |
| `corpus_text` | `"示例标题..."` | 空格分隔的词组 |
| `font_path` | `None` | 字体路径，自动回退系统中文字体 |
| `font_px` | `56` | 基础字号（像素），用于计算层级 |
| `font_px_min` | `48` | 全图最小字号（像素），硬下限 |
| `dilate_iter` | `14` | 主体禁区膨胀步数 |
| `comp_dilate_iter` | `6` | 低复杂度区域闭运算步数 |
| `complexity_thresh` | `0.50` | 复杂度阈值 |
| `min_area_ratio` | `0.03` | 全图可写面积下限 |
| `max_zones` | `3` | 最多排版区域数 |
| `n_tiers` | `None` | 字号层级数 1–3；`None` 随机 |
| `seed` | `None` | 随机种子（层级数、颜色选择） |
| `color_min_contrast` | `4.5` | 最低 WCAG 对比度 |

#### `run_multi_layouts` — 多 layout 生成（bbox-only）

为每种有效策略 × 每种层级数(1/2/3) 组合生成一个 layout，不渲染不填文字，仅输出 bbox 几何信息。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `max_layouts` | `10` | 每张图最多生成几种 layout |
| 其余参数 | 同上 | 与 `run_poster_pipeline` 共享 |

返回：
```python
{
    "matched_layouts": [
        {"layout0": {
            "low_level_complexity": {"0": 72.64, "1": 211.38},
            "scaled_boxes_1024h":   {"0": [208, 888, 837, 952], "1": [405, 968, 641, 1006]}
        }},
        {"layout1": {...}},
        ...
    ],
    "writable": ..., "complexity": ..., "debug": ...
}
```

- `scaled_boxes_1024h`：bbox 坐标统一缩放到高度 1024px
- `low_level_complexity`：框内平均复杂度 × 255（0–255 区间）
- `build_writable` 只算一次，layout 规划可复用

#### 返回字典（`run_poster_pipeline`）

| 键 | 说明 |
|---|---|
| `preview` | 渲染结果 ndarray (HxWx3 uint8) |
| `lines` | `List[TextLine]`，每行含 `font_px`、`fg`、`text`、`direction`、`font_name`、`color_name` |
| `writable` | bool HxW 可写字区域 |
| `complexity` | float32 HxW [0,1] 复杂度图 |
| `subj_mask` / `forb_mask` | 主体原始 / 膨胀后禁区 |
| `combined_mask` | HxWx3 三层合并可视化（白/红/黄） |
| `debug` | 含 `strategy`、`zones`、`n_lines`、`font_levels`、`texts`（enriched meta） |

`debug["texts"]` 每条包含：
```json
{
  "bbox": [x0, y0, x1, y1],
  "text": "...",
  "font_px": 96,
  "direction": "h",
  "font_name": "STHeiti Medium.ttc",
  "color": {"rgb": [0, 0, 128], "name": "藏青"}
}
```

---

### `auto_layout.py` — 策略打分自动分区

预定义 12 种布局策略，每种由 1–4 个矩形区域组成，评分公式 `density × abs_frac` 选最优。

新增接口：
- `rank_strategies(writable, complexity)` → 全部策略按得分降序
- `zones_for_strategy(writable, complexity, name)` → 指定策略提取 TextZone

| 策略名 | 形状 | 典型场景 |
|---|---|---|
| `top_bottom` | 上下横条 | 主体居中（人像、产品正面） |
| `left_right` | 左右竖条 | 横幅图像两侧留白 |
| `frame` | 全边框（四边） | 主体完全居中 |
| `center` | 中心横带 | 边缘复杂，中心清晰 |
| `top_only` / `bottom_only` | 仅顶 / 仅底 | 主体偏上或偏下 |
| `top_right` / `top_left` | Γ 形 / L 形 | 主体偏左或偏右 |
| `bottom_right` / `bottom_left` | 反 L / J 形 | 主体偏左下或偏右下 |
| `top_bottom_right` / `top_bottom_left` | 三边环绕 | 主体居左或居右 |

---

### `writable_mask.py` — 主体禁区 + 复杂度图（OpenCV 加速）

全部卷积和形态学操作使用 OpenCV（`cv2.Sobel`、`cv2.GaussianBlur`、`cv2.blur`、`cv2.dilate`、`cv2.morphologyEx`），2K 图像下比 scipy 快 **5–10×**。

**复杂度图**：`sigma ≈ 短边/16`（旧实现 1/8），`lap_win=9`（旧 21），减少高纹理能量向低复杂区域的扩散偏移。

**可写字区域**：低复杂度掩码使用 `binary_closing`（闭运算 = 膨胀 + 腐蚀）取代单向膨胀，填小孔平滑边界但不整体外扩。

---

### `layout_scanline.py` — 三层排版引擎

#### `plan_hierarchical` — 层次字号扫描 + 两级对齐

**两级对齐机制**（核心改进）：

1. **同字号组内**：取所有 [x0, x1] 的交集，统一两侧边缘
2. **跨字号组**：按 align 方向吸附
   - `left`：所有层级左边缘对齐
   - `right`：所有层级右边缘对齐
   - `center`：以最窄层级中心为锚点，宽层级向其居中

安全钳位：所有对齐结果不超出 mask 可写范围。

#### `fill_slots` — 文本分配

每个 TextLine 携带 `font_name`（实际加载到的字体文件名）和 `color_name`（色卡名）。

---

### `color_palette.py` — 90 色色卡 + 对比色挑选

| 功能 | 说明 |
|---|---|
| `PALETTE` | 90 种颜色（hex/rgb/中英文名列表） |
| `pick_contrast_color(bg_rgb, rng)` | 按 WCAG 对比度排序，从 top_k 中随机挑一个 |
| `contrast_ratio(fg, bg)` | WCAG 相对亮度对比度 |

### `color_contrast.py` — 区域配色

| 函数 | 说明 |
|---|---|
| `contrast_from_palette(rgb, mask, *, anchor_id, ...)` | 色卡对比色挑选，支持 anchor_id 沿用同色 |
| `contrast_text_rgb(rgb, mask)` | 旧接口（简单黑/白二选一） |

---

## 入口脚本

### `run_json.py` — JSON 批量处理（主入口）

**核心功能**：读入 JSON → 逐张生成多种 layout → 输出带 `matched_layouts` 的新 JSON。

```bash
# 仅输出 JSON（最快，不存图）
python poster_pipeline/run_json.py \
    --json              data.json \
    --img_dir           /images \
    --max_layouts       10

# 同时保存渲染图
python poster_pipeline/run_json.py \
    --json              data.json \
    --img_dir           /images \
    --save_images

# 保存全部调试图（mask/complexity 等）
python poster_pipeline/run_json.py \
    --json              data.json \
    --img_dir           /images \
    --save_images \
    --save_debug_images
```

#### 全部参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--json` | — | 输入 JSON 文件路径（顶层为数组） |
| `--img_dir` | — | 图片目录（本地） |
| `--out_dir` | 同级 `_out/` | 图像输出目录 |
| `--out_json` | `<input>_layout.json` | 输出 JSON 路径 |
| `--n` | `0`（全部） | 最多处理 N 条 |
| `--font` | 自动检测 | 中文字体路径 |
| `--max_zones` | `3` | 最多排版区域数 |
| `--dilate_iter` | `14` | 主体禁区膨胀步数 |
| `--comp_dilate` | `6` | 复杂度闭运算步数 |
| `--complexity_thresh` | `0.50` | 复杂度阈值 |
| `--font_px` | `56` | 基础字号 |
| `--font_px_min` | `48` | 全图最小字号 |
| `--max_layouts` | `10` | 每张图最多生成几种 layout |
| `--save_images` | `False` | 保存 image/preview/debug.json |
| `--save_debug_images` | `False` | 保存 mask/complexity 等调试图（隐含 `--save_images`） |

#### 输入 JSON 格式

```json
[
  {
    "img_path": "s3://.../<stem>",
    "masks": [{"size": [H, W], "counts": "RLE字符串"}, ...],
    "cat_names": ["boat", "person"],
    "img_id": "...",
    "bboxes": [[x0,y0,x1,y1], ...],
    "confidence_scores": [...]
  }
]
```

#### 输出 JSON 格式

在原始记录上追加 `matched_layouts` 字段：

```json
{
  "img_path": "...",
  "masks": [...],
  "matched_layouts": [
    {
      "layout0": {
        "low_level_complexity": {"0": 72.64, "1": 211.38},
        "scaled_boxes_1024h": {"0": [208, 888, 837, 952], "1": [405, 968, 641, 1006]}
      }
    },
    {
      "layout1": {
        "low_level_complexity": {"0": 117.31},
        "scaled_boxes_1024h": {"0": [67, 910, 979, 958]}
      }
    }
  ]
}
```

- bbox 坐标缩放到 **高度 1024px**
- bbox 高度 = 字号大小（像素值，已缩放）
- `low_level_complexity` = 框内平均复杂度 × 255

#### 可选图像输出（`--save_images`）

| 文件 | 说明 |
|---|---|
| `image.png` | 原图 |
| `preview.png` | 排版渲染结果 |
| `debug.json` | 调试信息 |

#### 额外调试图（`--save_debug_images`）

| 文件 | 说明 |
|---|---|
| `combined_mask.png` | 三层合并图（白=可写，红=主体禁区，黄=复杂度禁区） |
| `subject_mask.png` | 主体禁区 |
| `complexity_map.png` | 复杂度连续图 |
| `writable_mask.png` | 最终可写区域 |

### `run_coco.py` — COCO 演示

```bash
python poster_pipeline/run_coco.py
```

---

## Python API

```python
import cv2
import numpy as np
from poster_pipeline.pipeline import run_poster_pipeline, run_multi_layouts

rgb = cv2.cvtColor(cv2.imread("photo.jpg"), cv2.COLOR_BGR2RGB)

# ── 单 layout（含渲染）────────────────────────────────────────────────
result = run_poster_pipeline(
    rgb, masks,
    subject_labels = {"person"},
    corpus_text    = "华为 智慧生活 影像旗舰",
    font_px        = 56,
    font_px_min    = 48,
    n_tiers        = 3,          # 3 层字号
    seed           = 42,         # 可复现
)

# 查看 enriched meta
for t in result["debug"]["texts"]:
    print(f"[{t['font_px']}px] {t['direction']} color={t['color']['name']} '{t['text']}'")

# ── 多 layout（bbox-only，不渲染）─────────────────────────────────────
ml = run_multi_layouts(
    rgb, masks,
    max_layouts = 10,
    font_px     = 56,
    font_px_min = 48,
)

for layout in ml["matched_layouts"]:
    for k, v in layout.items():
        print(f"{k}: boxes={v['scaled_boxes_1024h']}")

# ── 无主体（纯背景图）────────────────────────────────────────────────
result = run_poster_pipeline(rgb, [], corpus_text="华为 智慧生活")
```

---

## 性能

| 图像尺寸 | 单 layout（含渲染） | 10 layout（bbox-only） |
|---|---|---|
| 640×480 (COCO) | ~50ms | ~80ms |
| 2048×1365 (2K) | ~350ms | ~660ms |
| 2048×2048 | ~500ms | ~800ms |

核心加速：全部卷积/形态学/高斯平滑使用 OpenCV（比 scipy 快 5–10×），图像保存使用 `cv2.imwrite`（比 PIL 快 3–6×）。

---

## 依赖

```
numpy
Pillow
opencv-python
pycocotools        # RLE mask 解码（run_json.py）
```

> 注：不再依赖 `scipy`，所有信号处理和形态学操作已迁移至 OpenCV。

字体优先级（自动回退）：
1. `font_path` 参数
2. `/System/Library/Fonts/STHeiti Medium.ttc`（macOS）
3. `C:\Windows\Fonts\msyh.ttc`（Windows 微软雅黑）
4. `/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc`（Linux）
5. Pillow 内置 bitmap（不支持中文，兜底）

---

## 关键设计决策

### 1. 策略打分取代连通域 + 随机风格

12 种预定义策略涵盖海报常见构图，`density × abs_frac` 评分完全数据驱动。多 layout 生成时，按策略得分排序依次使用不同策略，组合 1/2/3 层字号变体。

### 2. 分层字号 + 两级对齐

每张图随机 1–3 种字号层级（可指定），同层级行 bbox 边缘严格对齐（交集），跨层级按 align 方向吸附。bbox 高度即字号像素值，可直接用于下游渲染。

### 3. 色卡对比色 + 图内一致性

90 色色卡按 WCAG 对比度排序挑选，同一张图内通过 `anchor_id` 沿用同色，避免多区域颜色不一致。每个颜色附带随机中/英文名称，写入 meta。

### 4. 复杂度 mask 精准对齐

`sigma ≈ 短边/16`（原 1/8）+ `lap_win=9`（原 21）减少高斯扩散偏移；闭运算（dilate→erode）替代单向膨胀，边界对称不外扩。

### 5. OpenCV 全链路加速

所有 `scipy.signal.convolve2d` → `cv2.Sobel` / `cv2.filter2D` / `cv2.blur`；`scipy.ndimage.gaussian_filter` → `cv2.GaussianBlur`；形态学 → `cv2.dilate` / `cv2.morphologyEx`。2K 图像 pipeline 计算从 ~1.2s 降至 ~0.35s。

### 6. 多 layout 一出多

`run_multi_layouts` 复用同一次 `build_writable` 计算，对 top-K 策略 × 3 种层级数组合生成多达 N 个去重 layout，输出格式兼容下游训练/渲染。
