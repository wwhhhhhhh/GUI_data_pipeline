# poster_pipeline — 海报自动排版工程

基于实例分割 mask 的海报自动排版工程。输入图像与主体分割结果，自动识别适合写字的区域，对 12 种布局策略打分选最优，按字号层级（从大到小）在每个区域内规划文字 bbox，将语料渲染到图上。

---

## 整体思路

```
输入图像 + 实例分割 masks（可为空列表）
        │
        ▼
  ① 主体禁区构建          writable_mask.py
     union 所有主体 mask → 形态学膨胀（dilate_iter 步）→ 禁区 forb
     masks 为空时 forb 全为 False（全图参与候选）
        │
        ▼
  ② 复杂度图              writable_mask.py
     Sobel 梯度 + Laplacian 能量 → 大高斯平滑（sigma ≈ 短边/8，20~80px）
     → 区域级复杂度，归一化 [0,1]
     天空/地板 → 低复杂度；树枝/纹理 → 高复杂度
        │
        ▼
  ③ 可写字区域（两步交集）  writable_mask.py
     comp_low = (complexity ≤ threshold)
     comp_low = dilate(comp_low, comp_dilate_iter 步)  ← 扩充可写边界
     writable = (~forb) & comp_low
        │
        ▼
  ④ 策略打分 → 选最优布局  auto_layout.py
     预定义 12 种候选策略（上下、左右、全边框、中心、L/Γ/J 形等）
     每种策略由 1~4 个矩形区域组成
     对每个区域计算：density × abs_frac
       density  = 区域内可写像素 / 区域面积
       abs_frac = 区域内可写像素 / 全图面积
     策略得分 = 各区均值 → 选得分最高的策略
        │
        ▼
  ⑤ 字号层级计算            pipeline.py
     base = min(font_px, 短边/8)，不低于 font_px_min
     层级 = base × [2.2, 1.7, 1.3, 1.0]，8px 对齐，末尾附 font_px_min
     示例（base=56, min=48）→ [120, 96, 72, 56, 48]
     zone 0（主区）从最大级开始；zone 1 从次级开始；以此类推
        │
        ▼
  ⑥ 层次 bbox 规划          layout_scanline.py → plan_hierarchical
     对每个区域按字号序列从大到小依次扫描，放置 LineSlot
     横排：band.all(axis=0) 找最宽连续可写列段
     竖排：strip.all(axis=1) 找最长连续可写行段
     每个槽位记录自己的字号（font_px），大字号优先占位
        │
        ▼
  ⑦ 文本分配              layout_scanline.py → fill_slots
     每个槽位按自己的 font_px 计算可容纳字符数
     横排：贪心填词，语料词指针跨槽顺序前进
     竖排：逐字填入，capacity = height // font_px
        │
        ▼
  ⑧ 颜色选取              color_contrast.py
     每个区域独立采样背景中位色，选对比最强字色（亮底→深字，暗底→白字）
        │
        ▼
  ⑨ 文字渲染              layout_scanline.py → render_lines
     横排/竖排，per-line 颜色与字号，带对比描边
     可选 draw_bbox=True 在文字框周围绘制轮廓（调试）
        │
        ▼
     输出：preview / bbox_preview / combined_mask / subject_mask
           complexity_map / complexity_binary / writable_mask / debug.json
```

---

## 模块说明

### `pipeline.py` — 端到端入口

对外暴露 `run_poster_pipeline()` 和 `make_combined_mask()`。

#### `run_poster_pipeline` 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `rgb` | — | HxWx3 uint8 图像 |
| `masks` | — | `[{"mask": bool HxW, "label": str}, ...]`，可为空列表 |
| `subject_labels` | `None` | 需要避开的类别集合；`None` = 全部 mask 都避开 |
| `corpus_text` | `"示例标题..."` | 空格分隔的词组 |
| `font_path` | `None` | 字体路径，自动回退系统中文字体 |
| `font_px` | `56` | 基础字号（像素），用于计算层级（× 2.2/1.7/1.3/1.0） |
| `font_px_min` | `48` | **全图最小字号**（像素），所有层级不低于此值 |
| `dilate_iter` | `14` | **主体禁区膨胀步数**（越大文字离主体越远） |
| `comp_dilate_iter` | `6` | **低复杂度区域膨胀步数**（越大可写边界越宽松） |
| `complexity_thresh` | `0.50` | 复杂度阈值，低于此才算低复杂可写 |
| `min_area_ratio` | `0.03` | 全图可写区域面积下限，低于此跳过整图 |
| `max_zones` | `3` | **最多使用几个排版区域**（实际可能少于此数） |

#### 返回字典

| 键 | 说明 |
|---|---|
| `preview` | 渲染结果 ndarray (HxWx3 uint8) |
| `bbox_preview` | 带槽位轮廓框的渲染结果（调试用） |
| `lines` | `List[TextLine]`，每行含 `font_px`、`fg`、`text`、`direction` |
| `writable` | bool HxW 可写字区域 |
| `complexity` | float32 HxW [0,1] 复杂度图 |
| `subj_mask` | bool HxW 主体区域（膨胀前原始 union） |
| `forb_mask` | bool HxW 主体禁区（膨胀后，含安全边距） |
| `combined_mask` | HxWx3 uint8 三层合并可视化（见下） |
| `debug` | 含 `strategy`、`zones`、`n_lines`、`font_levels` 等 |

#### `make_combined_mask(writable, forb_mask) → HxWx3`

三层合并可视化，一张图看清所有区域：

| 颜色 | 含义 |
|---|---|
| **白色** `[255,255,255]` | 可写字区域（最终可用） |
| **红色** `[210, 55, 55]` | 主体禁区（原始主体 + 膨胀安全边距） |
| **黄色** `[220,175,  0]` | 复杂度禁区（非主体但纹理过高） |

三类互斥，覆盖全图所有像素。

---

### `auto_layout.py` — 策略打分自动分区（核心创新）

#### 设计原理

不依赖连通域分析，而是对**预定义的 12 种布局策略**打分，根据实际可写区域的分布自动选最优策略：

- 主体在图像正中（人像/画作）→ 上下各有大片留白 → **`top_bottom`** 或 **`top_only`** 高分
- 四周花边装饰复杂、中心清晰 → **`center`** 高分
- 横幅图像主体宽、两侧留白 → **`left_right`** 高分
- 主体偏下、顶部大片天空 → **`top_only`** 高分

#### 评分公式

```
单区得分 = density × abs_frac
  density  = 区域内可写像素 / 区域面积         ∈ [0,1]（区域纯净度）
  abs_frac = 区域内可写像素 / (H × W)         ∈ [0,1]（绝对可用面积）

策略得分 = 各区得分均值
```

`density × abs_frac` 的几何意义：大片干净区域 >> 小片干净区域 >> 大片复杂区域，同时惩罚极小的可写区。

#### 12 种候选策略

| 策略名 | 形状 | 典型场景 |
|---|---|---|
| `top_bottom` | 上下横条 | 主体居中（人像、产品正面） |
| `left_right` | 左右竖条 | 横幅图像两侧留白 |
| `frame` | 全边框（四边） | 主体完全居中、四周均有留白 |
| `center` | 中心横带 | 边缘复杂（画框花边），中心清晰 |
| `top_only` | 仅顶部 | 主体占下半图，顶部大片天空 |
| `bottom_only` | 仅底部 | 主体占上半图，底部地面/空白 |
| `top_right` | 顶横 + 右竖（Γ形） | 主体偏左 |
| `top_left` | 顶横 + 左竖（L形） | 主体偏右 |
| `bottom_right` | 右竖 + 底横（反L） | 主体偏左下 |
| `bottom_left` | 左竖 + 底横（J形） | 主体偏右下 |
| `top_bottom_right` | 上下横 + 右竖 | 主体居左 |
| `top_bottom_left` | 上下横 + 左竖 | 主体居右 |

---

### `writable_mask.py` — 主体禁区 + 复杂度图

**`complexity_map(rgb)`**
大高斯平滑（sigma ≈ 短边/8，范围 20~80px）把 Sobel 梯度和 Laplacian 能量扩散到区域尺度，有效区分大片低频区（天空）和密集纹理区（树枝）。

**`build_writable(rgb, masks, ..., dilate_iter=14, comp_dilate_iter=6)`**
返回 `(writable, comp, subj_mask, forb_mask)`：

1. 主体 mask union → 膨胀 `dilate_iter` 步 → `forb_mask`（安全禁区）
2. 复杂度图 → 二值化 → 膨胀 `comp_dilate_iter` 步（扩充低复杂区边界）
3. `writable = (~forb_mask) & comp_low_dilated`

两个膨胀参数均可配置：`dilate_iter` 控制文字与主体的距离，`comp_dilate_iter` 控制可写区域边界的宽松程度。

---

### `layout_scanline.py` — 三层排版引擎

#### `plan_hierarchical(mask, size_schedule, direction, ...)` — 层次字号扫描（主接口）

按 `size_schedule`（降序字号列表）在 mask 内依次规划槽位，每个槽位记录自己的字号：

**横排（direction="h"）**：
- 从上到下，对每个目标字号 `target_font`，从当前 y 向下扫描
- `band.all(axis=0)` 找最宽连续可写列段，宽度 ≥ `min_width_chars × target_font` 时放置槽位
- 成功后 y 前进到槽位底部 + 行间距；找不到时跳过该字号（y 不变，尝试更小的下一级）

**竖排（direction="v"）**：
- 从外侧（左或右）向内依次放置列，每列取下一个字号
- `strip.all(axis=1)` 找最长连续可写行段，高度 ≥ `min_height_chars × target_font` 时放置槽位

```
示例字号序列（zone 0，base=56, min=48）：[120, 96, 72, 56, 48, 48]
  → 放置 120px 大标题 → 96px 副标题 → 72px 小标题 → 56px 正文 → ...
```

主要参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `size_schedule` | — | 字号列表，降序 |
| `direction` | — | `"h"` 横排 / `"v"` 竖排 |
| `side` | `"right"` | 竖排起始侧（`"left"` / `"right"`） |
| `margin` | `6` | 槽位内缩边距（像素） |
| `min_width_chars` | `2` | 横排最少容纳字符数 |
| `line_gap_ratio` | `0.18` | 行间距 = font_px × ratio |
| `col_gap_ratio` | `0.25` | 列间距 = font_px × ratio |
| `max_slots` | `6` | 最多返回槽位数 |

#### `fill_slots(slots, corpus, ...)` — 文本分配

每个槽位使用 `slot.font_px` 计算可容纳量：
- 横排：PIL `textlength` 实测字宽，贪心填词，语料耗尽时循环
- 竖排：`capacity = slot.height // slot.font_px`，独立从语料头截取

#### `render_lines(rgb, lines, ..., draw_bbox=False)` — 渲染

- per-line 颜色（`ln.fg`）和字号（`ln.font_px`），带对比描边
- `draw_bbox=True`：在每行文字区域绘制细线矩形轮廓（调试用，保存为 `bbox_preview.png`）

---

### `color_contrast.py` — 文字颜色选取

对区域内背景像素取 RGB 中位色，选黑或白中对比度更高者作为前景色。

---

## 入口脚本

### `run_json.py` — JSON 批量处理（主入口）

```bash
python poster_pipeline/run_json.py \
    --json              /path/to/data.json \
    --img_dir           /path/to/images \
    --out_dir           /path/to/output \
    --n                 20 \
    --max_zones         3 \
    --dilate_iter       14 \
    --comp_dilate       6 \
    --complexity_thresh 0.50 \
    --font_px           56 \
    --font_px_min       48
```

所有参数说明：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--json` | — | JSON 文件路径（顶层为数组） |
| `--img_dir` | — | 图片目录（本地） |
| `--out_dir` | 同级 `_out/` | 输出目录 |
| `--n` | `0`（全部） | 最多处理 N 条 |
| `--font` | 自动检测 | 中文字体路径 |
| `--max_zones` | `3` | 最多排版区域数 |
| `--dilate_iter` | `14` | 主体禁区膨胀步数 |
| `--comp_dilate` | `6` | 复杂度区域膨胀步数 |
| `--complexity_thresh` | `0.50` | 复杂度阈值 |
| `--font_px` | `56` | 基础字号（用于计算层级） |
| `--font_px_min` | `48` | 全图最小字号（硬下限） |

**JSON 格式**（顶层为数组）：
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

`masks` 为 `[]` 时：无主体，全图仅用复杂度过滤。

**输出文件（每张图一子目录）**：

| 文件 | 说明 |
|---|---|
| `image.png` | 原图 |
| `preview.png` | 排版渲染结果（纯文字） |
| `bbox_preview.png` | 带槽位轮廓框的渲染结果（调试，可见排版网格） |
| `combined_mask.png` | **三层合并图**（白=可写，红=主体禁区，黄=复杂度禁区） |
| `subject_mask.png` | 主体禁区（含膨胀安全边距） |
| `complexity_map.png` | 复杂度连续图（白=低复杂） |
| `complexity_binary.png` | 复杂度二值图（白=低复杂可写） |
| `writable_mask.png` | 最终可写区域 |
| `debug.json` | 含 `strategy`、`zones`、`n_lines`、`font_levels` 等 |

### `run_coco.py` — COCO 演示

```bash
python poster_pipeline/run_coco.py
```

（需修改脚本中 `IMG_DIR` / `ANN_FILE` 路径，输出与 `run_json.py` 相同的文件集）

---

## Python API

```python
import cv2
import numpy as np
from poster_pipeline.pipeline import run_poster_pipeline, make_combined_mask
from PIL import Image

rgb = cv2.cvtColor(cv2.imread("photo.jpg"), cv2.COLOR_BGR2RGB)

# ── 有主体 ──────────────────────────────────────────────────────────────────
masks = [{"mask": person_mask_bool, "label": "person"}]
result = run_poster_pipeline(
    rgb, masks,
    subject_labels    = {"person", "dog"},
    corpus_text       = "华为 智慧生活 影像旗舰 极致性能",
    font_px           = 56,        # 基础字号，用于计算层级
    font_px_min       = 48,        # 全图最小字号（硬下限）
    dilate_iter       = 14,        # 主体禁区膨胀步数
    comp_dilate_iter  = 6,         # 复杂度区域膨胀步数
    max_zones         = 3,         # 最多 3 个排版区域
)

# ── 无主体（纯背景图）────────────────────────────────────────────────────────
result = run_poster_pipeline(rgb, [], corpus_text="华为 智慧生活")

Image.fromarray(result["preview"]).save("poster.png")
Image.fromarray(result["bbox_preview"]).save("poster_bbox.png")  # 调试用
Image.fromarray(result["combined_mask"]).save("combined.png")

# 查看自动选择的策略与字号层级
print("strategy:", result["debug"]["strategy"])
print("font_levels:", result["debug"]["font_levels"])  # e.g. [120, 96, 72, 56, 48]
for ln in result["debug"]["lines"]:
    print(f"  [{ln['font_px']}px] {ln['direction']} '{ln['text']}'")
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

### 1. 策略打分取代连通域 + 随机风格

旧方案先做连通域分析，再按各域质心位置分类（top/bottom/left/right），容易导致策略选择与实际可写区域分布不匹配（"随机感"）。

新方案：**先全图分析，后决定策略**。12 种预定义策略涵盖海报常见构图，对每种策略计算可写像素在其区域内的密度与绝对面积，取综合最高者。策略选择完全数据驱动：
- 画作/人像（主体正中）→ `top_bottom` 或 `top_only` 得高分
- 画框装饰（边缘复杂）→ `center` 得高分
- 横幅图（主体宽）→ `left_right` 得高分

### 2. 层次字号取代单一字号

旧方案每个区域用同一字号扫描，输出视觉平淡（"上方一行字"）。

新方案：**字号序列从大到小依次分配**，模拟真实海报排版：
- 字号层级由基础字号 `font_px` 按 × 2.2 / 1.7 / 1.3 / 1.0 比例生成，8px 对齐
- 主区（zone 0）从最大级别开始；次区（zone 1）从次级开始，产生跨区字号差异
- 最小字号由 `font_px_min` 控制（硬下限，默认 48px）
- 大字号优先占位，小字号填充剩余空间

```
示例（base=56, min=48）→ 层级 [120, 96, 72, 56, 48]
  zone 0: 120px 大标题 → 96px 副标题 → 72px 正文 → ...
  zone 1: 96px → 72px → 56px → ...
  zone 2: 72px → 56px → 48px → ...
```

### 3. 两步膨胀，两个参数分别可控

- `dilate_iter`（主体膨胀）：控制文字与主体边缘的安全距离
- `comp_dilate_iter`（复杂度膨胀，默认 6）：扩充低复杂度区域的边界，消除碎片、平滑边界、增加可用空间

两者独立配置，互不影响：前者保证文字不压主体，后者保证可写区域充裕。

### 4. 三层合并可视化（combined_mask）

一张图看清所有决策：白=可写，红=主体禁区，黄=复杂度禁区。三类互斥，覆盖全图。便于调参时直观判断膨胀步数是否合适。

### 5. band.all(axis=0) 保证文字矩形安全

横排槽位要求文字高度带内每列都在可写区，彻底防止文字跨越主体洞或复杂纹理缺口。

### 6. 无主体时全图复杂度过滤

`masks=[]` 时 `forb_mask` 全为 False，`writable` 退化为纯复杂度二值图（加膨胀）。适合纯背景产品图、风景图，无需标注主体。
