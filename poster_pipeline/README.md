# poster_pipeline — 海报自动排版工程

基于实例分割 mask 的海报自动排版工程。输入图像与主体分割结果，自动识别适合写字的区域，自适应规划排版，将语料渲染到图上。

---

## 整体思路

```
输入图像 + 实例分割 masks（可为空列表）
        │
        ▼
  ① 主体禁区构建          writable_mask.py
     union 所有主体 mask → 形态学膨胀 → 禁区 forb
     masks 为空时 forb 全为 False（全图参与候选）
        │
        ▼
  ② 复杂度图              writable_mask.py
     Sobel 梯度 + Laplacian 能量 → 大高斯平滑（sigma ≈ 短边/8，20~80px）
     → 区域级复杂度，归一化 [0,1]
     天空/地板 → 低复杂度；树枝/纹理 → 高复杂度
        │
        ▼
  ③ 可写字区域（交集）     writable_mask.py
     comp_low = (complexity ≤ threshold)
     comp_low = dilate(comp_low, 3px)   ← 小膨胀扩充可写边界
     writable = (~forb) & comp_low
        │
        ▼
  ④ 连通域分析 → 自动分区  auto_layout.py
     对 writable 做 8 连通标记，过滤面积过小的碎片
     每个区域计算：质心位置 / 宽高比 / 平均复杂度
     → 位置分类（top/bottom/left/right/center）
     → 文字方向（横排 h / 竖排 v）
     → 综合评分排序，取 top-K（默认 3）
     → 字号层级：1.0 / 0.72 / 0.55
        │
        ▼
  ⑤ 排版规划              layout_scanline.py → plan_layout
     对每个区域的 mask 扫描槽位 LineSlot
     横排：band.all(axis=0) 找最宽连续可写列段
     竖排：strip.all(axis=1) 找最长连续可写行段
        │
        ▼
  ⑥ 文本分配              layout_scanline.py → fill_slots
     每个区域独立获得完整语料，贪心填词（横排）/ 逐字填入（竖排）
        │
        ▼
  ⑦ 颜色选取              color_contrast.py
     每个区域独立采样背景中位色，选对比最强字色（亮底→深字，暗底→白字）
        │
        ▼
  ⑧ 文字渲染              layout_scanline.py → render_lines
     横排/竖排，带对比描边，per-line 颜色与字号
        │
        ▼
     输出：preview / subject_mask / complexity_binary / writable_mask / debug.json
```

---

## 模块说明

### `pipeline.py` — 端到端入口

对外暴露 `run_poster_pipeline()`。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `rgb` | — | HxWx3 uint8 图像 |
| `masks` | — | `[{"mask": bool HxW, "label": str}, ...]`，可为空列表 |
| `subject_labels` | `None` | 需要避开的类别集合；`None` = 全部 mask 都避开 |
| `corpus_text` | `"示例标题..."` | 空格分隔的词组 |
| `font_path` | `None` | 字体路径，自动回退系统中文字体 |
| `font_px` | `48` | 基础字号（像素），自动限制不超过短边 1/8 |
| `dilate_iter` | `14` | 主体禁区膨胀半径（像素） |
| `comp_dilate_iter` | `3` | 低复杂度区域额外膨胀半径（扩充可写边界） |
| `complexity_thresh` | `0.50` | 复杂度阈值，低于此才算可写 |
| `min_area_ratio` | `0.04` | 全图可写区域面积下限，低于此跳过整图 |
| `max_zones` | `3` | 最多使用几个排版区域 |

返回字典：

| 键 | 说明 |
|---|---|
| `preview` | 渲染结果 ndarray (HxWx3) |
| `lines` | `List[TextLine]` |
| `writable` | bool HxW 可写字区域 |
| `complexity` | float32 HxW [0,1] 复杂度图 |
| `subj_mask` | bool HxW 主体区域（膨胀前，masks 为空则全 False） |
| `debug` | 调试信息，含 `zones` 列表（position/direction/score/font_scale/bbox） |

---

### `auto_layout.py` — 连通域自动分区（核心创新）

基于连通域分析的自适应布局，取代预定义风格选择。

#### `TextZone` 数据类

| 字段 | 说明 |
|---|---|
| `mask` | bool HxW，该区域可写像素 |
| `direction` | `"h"` 横排 / `"v"` 竖排 |
| `align` | `"left"` / `"center"` / `"right"` |
| `position` | `"top"` / `"bottom"` / `"left"` / `"right"` / `"center"` |
| `scan_style` | 传给 `plan_layout` 的 style（`h_top` / `h_bottom` / `v_left` / `v_right` / `h_center`） |
| `score` | 综合质量分（area × pos_weight × quality） |
| `font_scale` | 字号缩放系数（1.0 / 0.72 / 0.55） |
| `bbox` | `(y0, y1, x0, x1)` |

#### 位置分类（三分法则）

```
cy_rel < 0.38            → top    权重 1.3（主标题首选）
cy_rel > 0.62            → bottom 权重 1.1（品牌/副标）
cx_rel < 0.38            → left   权重 1.0（竖排）
cx_rel > 0.62            → right  权重 1.0（竖排）
其余                      → center 权重 0.6（视觉中心，权重低）
```

#### 方向判断

```
宽高比 > 1.5   → 横排 h（宽幅横条）
宽高比 < 0.75  → 竖排 v（细长竖条）
中间段：left/right 位置 → 竖排，其余 → 横排
```

#### 综合评分

```
score = (area / total_area) × pos_weight × (1 - avg_complexity)
```

---

### `writable_mask.py` — 主体禁区 + 复杂度图

**`complexity_map(rgb)`**  
大高斯平滑（sigma ≈ 短边/8）扩散边缘能量到区域尺度，有效区分大片低频区（天空）和密集纹理区（树枝）。

**`build_writable(rgb, masks, ..., comp_dilate_iter=3)`**  
返回 `(writable, comp, subj_mask)`：
- 低复杂度区域额外膨胀 `comp_dilate_iter` 步，扩充可写边界，使文字槽位更充裕
- masks 为空时 forb 全 False，可写区域仅由复杂度决定

---

### `layout_scanline.py` — 三层排版引擎

#### `plan_layout(mask, font_px, style)` — 纯几何

在 mask 内扫描槽位。`style` 决定槽位优先顺序：

| style | 说明 |
|---|---|
| `h_top` | 横排，优先顶部 |
| `h_bottom` | 横排，优先底部 |
| `h_center` | 横排，居中 |
| `v_left` | 竖排，从左往右扫列 |
| `v_right` | 竖排，从右往左扫列 |
| `surround` | 上半横排 + 下半横排 |

核心算法（横排）：`band.all(axis=0)` 要求文字高度带内每列都可写，保证文字矩形不跨越主体。

#### `fill_slots` / `render_lines` — 文本填充与渲染

- 横排：PIL `textlength` 实测字宽，贪心填词，不截断
- 竖排：逐字填入，容量 = `height // font_px`
- 渲染：per-line 颜色和字号，带对比描边

---

### `color_contrast.py` — 文字颜色选取

对区域内背景像素取 RGB 中位色，选黑或白中对比度更高者。

---

## 入口脚本

### `run_json.py` — JSON 批量处理（主入口）

从实例分割输出的 JSON 文件（数组格式）批量处理。

```bash
python poster_pipeline/run_json.py \
    --json    /path/to/data.json \
    --img_dir /path/to/images \
    --out_dir /path/to/output \
    --n 20
```

**JSON 格式**（文件顶层为数组）：
```json
[
  {
    "img_path": "s3://.../<stem>",
    "masks": [{"size": [H, W], "counts": "RLE字符串"}, ...],
    "cat_names": ["boat", "person"],
    "img_id": "...",
    "bboxes": [...],
    "confidence_scores": [...]
  }
]
```

- `masks` 为 `[]` 时：无主体，全图用复杂度过滤
- `img_path` 末段文件名在 `--img_dir` 下模糊匹配（`.jpg/.png/.webp` 等）

**输出文件（每张图一子目录）**：

| 文件 | 说明 |
|---|---|
| `image.png` | 原图 |
| `preview.png` | 排版渲染结果 |
| `subject_mask.png` | 主体区域（白=主体） |
| `complexity_map.png` | 复杂度连续图（白=低复杂） |
| `complexity_binary.png` | 复杂度二值（白=低复杂可写） |
| `writable_mask.png` | 最终可写区域 |
| `debug.json` | 调试信息（含 zones 详情） |

### `run_coco.py` — COCO 演示

```bash
python poster_pipeline/run_coco.py
```

（需修改脚本中 `IMG_DIR` / `ANN_FILE` 路径）

---

## Python API

```python
import cv2
import numpy as np
from poster_pipeline.pipeline import run_poster_pipeline
from PIL import Image

rgb = cv2.cvtColor(cv2.imread("photo.jpg"), cv2.COLOR_BGR2RGB)

# ── 有主体 ──────────────────────────────────────────────────────────────────
masks = [{"mask": person_mask_bool, "label": "person"}]
result = run_poster_pipeline(
    rgb, masks,
    subject_labels={"person", "dog"},
    corpus_text="华为 智慧生活 影像旗舰 极致性能",
    font_px=56,
)

# ── 无主体（纯背景图） ────────────────────────────────────────────────────────
result = run_poster_pipeline(rgb, [], corpus_text="华为 智慧生活")

Image.fromarray(result["preview"]).save("poster.png")

# 查看自动识别的排版区域
for z in result["debug"]["zones"]:
    print(z["position"], z["direction"], f"score={z['score']:.3f}")
```

---

## 依赖

```
numpy
scipy
Pillow
opencv-python      # 读图（run_coco.py 还用于 polygon→mask）
pycocotools        # RLE mask 解码（run_json.py）
```

字体优先级（自动回退）：
1. `font_path` 参数
2. `/System/Library/Fonts/STHeiti Medium.ttc`（macOS）
3. `C:\Windows\Fonts\msyh.ttc`（Windows 微软雅黑）
4. `/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc`（Linux）
5. Pillow 内置 bitmap（不支持中文，兜底）

---

## 关键设计决策

### 1. 连通域分析取代预定义风格

旧方案通过指定 `layout_style`（h_top / h_bottom / v_right 等）来控制排版，但预定义风格与实际可写区域形状往往不匹配。新方案：
- 对 writable mask 做 8 连通标记
- 按质心位置 + 宽高比自动决定每个区域的文字方向、对齐方式和扫描顺序
- 按综合评分（面积 × 位置权重 × 低频质量）选取 top-K 区域

### 2. 低复杂度区域小膨胀（comp_dilate_iter=3）

复杂度阈值二值化后，对低复杂度区域额外膨胀 3px，平滑边界、消除碎片、扩充可写空间。这一步在主体禁区过滤之后，不会让文字进入主体区域。

### 3. 字号层级（1.0 → 0.72 → 0.55）

排名第 1 的区域使用基础字号（主标题），第 2 区用 72% 字号（副标题），第 3 区用 55%（装饰性文字）。层级差异产生视觉重心，符合海报排版的设计原则（PosterLayout CVPR 2023 中的 Design Sequence Formation）。

### 4. 三分法则分类位置

质心 `cy < 0.38` → top（上三分之一），`cy > 0.62` → bottom（下三分之一），侧方类似。中心区域（center）权重最低（0.6），避免文字遮挡主体视觉焦点。Top 权重最高（1.3），天空/顶部横幅是海报主标题的最佳位置。

### 5. band.all(axis=0) 保证文字矩形安全

横排槽位要求文字矩形每一列、每一行都在可写区内（不仅最宽区间的首末像素）。这彻底防止文字跨越主体"洞"（复杂纹理孔洞）。

### 6. 无主体时全图复杂度过滤

`masks=[]` 时主体禁区为全零，writable 退化为纯复杂度二值图。适合纯背景产品图、风景图，无需标注主体即可自动找到低纹理可写区。
