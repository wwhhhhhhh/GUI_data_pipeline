# poster_pipeline — 海报自动排版工程

基于实例分割 mask 的海报自动排版工程。输入图像与主体分割结果，自动找到适合写字的区域，规划排版，将语料渲染到图上。

---

## 整体思路

```
输入图像 + 实例分割 masks（可为空）
        │
        ▼
  ① 主体禁区构建          writable_mask.py
     union 所有主体 mask → 形态学膨胀 → 禁区 forb
     masks 为空时 forb 全为 False（全图参与候选）
        │
        ▼
  ② 复杂度图              writable_mask.py
     Sobel 梯度 + Laplacian 能量 → 大高斯平滑（sigma≈短边/8）
     → 归一化到 [0,1]
     天空/草地 → 低复杂度（适合写字）
     树枝/纹理 → 高复杂度（不适合写字）
        │
        ▼
  ③ 可写字区域（直接交集） writable_mask.py
     writable = (~forb) & (complexity ≤ threshold)
        │
        ▼
  ④ 排版规划              layout_scanline.py → plan_layout
     对 writable mask 做扫描线找槽位 LineSlot
        │
        ▼
  ⑤ 文本分配              layout_scanline.py → fill_slots
     贪心填词（横排）/ 逐字填入（竖排）
        │
        ▼
  ⑥ 颜色选取              color_contrast.py
     采样可写区背景中位色，选对比最强字色
        │
        ▼
  ⑦ 文字渲染              layout_scanline.py → render_lines
     横排/竖排，带对比描边
        │
        ▼
     输出 preview / subject_mask / complexity_binary / writable_mask / debug.json
```

---

## 模块说明

### `pipeline.py` — 端到端入口

对外暴露 `run_poster_pipeline()`，串联所有步骤。

| 参数 | 说明 |
|---|---|
| `rgb` | HxWx3 uint8 图像 |
| `masks` | `[{"mask": bool HxW, "label": str}, ...]`，可为空列表 |
| `subject_labels` | 需要避开的类别集合（如 `{"person", "dog"}`）；`None` 表示全部 mask 都避开 |
| `corpus_text` | 空格分隔的词组，如 `"华为 智慧生活 影像旗舰 极致性能"` |
| `font_path` | 字体文件路径，默认自动回退到系统中文字体 |
| `font_px` | 字号（像素），建议 48~72 |
| `layout_style` | 排版风格，见下表，默认 `"h_top"` |
| `dilate_iter` | 主体禁区膨胀半径（像素），默认 14 |
| `complexity_thresh` | 复杂度阈值（0~1），低于此才算可写，默认 0.50 |
| `min_area_ratio` | 可写区域面积下限（占全图比例），低于此直接跳过，默认 0.04 |

返回字典：

| 键 | 说明 |
|---|---|
| `preview` | 渲染后图像 ndarray (HxWx3) |
| `lines` | `List[TextLine]` |
| `writable` | bool HxW，可写字区域 |
| `complexity` | float32 HxW [0,1]，复杂度图 |
| `subj_mask` | bool HxW，主体区域（膨胀前，masks 为空则全 False） |
| `debug` | 调试信息字典 |

---

### `writable_mask.py` — 主体禁区 + 复杂度图

**`union_subject_masks(masks, h, w, forbid_labels)`**
将指定类别的实例 mask 合并为一张主体 mask。masks 为空时返回全零掩码。

**`dilate_binary(mask, iterations)`**
对主体 mask 做形态学膨胀，生成安全禁区。

**`complexity_map(rgb, smooth_sigma, lap_win)`**
计算全图复杂度，值域 [0, 1]：
- Sobel 梯度幅值（权重 0.6）+ Laplacian 能量（权重 0.4）
- 各自用大高斯平滑（sigma = short_side/8，范围 20~80 px）扩散到区域尺度
- 归一化采用 0~99th percentile，保留各区域的绝对强弱关系
- 天空、草地等低频区 → 接近 0；树枝、密集纹理 → 接近 1

**`build_writable(rgb, masks, h, w, ...)`**
一步得到 `(writable, comp, subj_mask)`：
- `writable = (~forb) & (comp ≤ complexity_thresh)`
- masks 为空时 forb 全 False，可写区域仅由复杂度决定

---

### `layout_scanline.py` — 三层排版引擎

排版分三个职责完全分离的层：

#### 第一层：`plan_layout(mask, font_px, style)` — 纯几何

在 writable mask 内找文字槽位 `LineSlot`。

核心算法（横排）：
```
对每个候选行带 [y, y+font_px)：
  col_clear = mask[y:y+font_px].all(axis=0)
  → 找最宽的连续可写列段
  → 宽度 ≥ min_width_chars × font_px → 生成槽位
```

竖排：对每个候选列带 `[x, x+font_px)` 做 `mask[:, x:x+font_px].all(axis=1)` 找最长可写行段。

**支持的 6 种布局风格**：

| 风格 | 说明 |
|---|---|
| `h_top` | 横排，优先填顶部槽位 |
| `h_bottom` | 横排，优先填底部槽位 |
| `h_center` | 横排，优先填中部槽位 |
| `v_right` | 竖排，从右侧列往左扫 |
| `v_left` | 竖排，从左侧列往右扫 |
| `surround` | 上半区横排 + 下半区横排 |

#### 第二层：`fill_slots(slots, corpus, font_path, font_px)` — 纯文本分配

- 横排槽：按空格切词，用 PIL `textlength` 实测字宽，贪心填词
- 竖排槽：按字符逐个填入，容量 = `height // font_px`
- 返回 `List[TextLine(y, x0, x1, text, direction, align, fg, font_px)]`

#### 第三层：`render_lines(rgb, lines, font_path)` — 纯渲染

- 横排：按 `ln.align`（left/center/right）定位后绘制
- 竖排：逐字绘制，字符居中于列宽
- 带描边（亮字→黑描边，暗字→白描边）

---

### `color_contrast.py` — 文字颜色选取

对可写区域内的背景像素取 RGB 中位色，计算相对亮度，选择黑色或白色中对比度更高者作为前景色。

---

### `run_jsonl.py` — JSONL 批量处理（主入口）

从实例分割输出的 JSONL 文件批量跑 pipeline，支持 pycocotools RLE mask 格式。

**JSONL 格式**（每行一个 JSON 对象）：
```json
{
  "img_path": "s3://.../<folder>/<stem>",
  "masks": [{"size": [H, W], "counts": "RLE字符串"}, ...],
  "bboxes": [[x0, y0, x1, y1], ...],
  "cat_names": ["boat", "person", ...],
  "img_id": "...",
  "confidence_scores": [...]
}
```

- `masks` 为空列表时：视为无主体，全图仅用复杂度过滤，适合纯背景图
- `img_path` 末段文件名（无扩展名）用于在本地目录查找图片（支持 .jpg/.png/.webp 等）

**输出文件（每张图一个子目录）**：

| 文件 | 说明 |
|---|---|
| `image.png` | 原图 |
| `preview.png` | 排版渲染结果 |
| `subject_mask.png` | 主体区域（白=主体，黑=背景），masks 空则全黑 |
| `complexity_map.png` | 复杂度连续图（白=低复杂/适合写字） |
| `complexity_binary.png` | 复杂度二值化（白=低复杂可写，黑=高复杂禁写） |
| `writable_mask.png` | 最终可写区域（白=可写） |
| `debug.json` | 调试信息 |

---

### `run_coco.py` — COCO 数据集批量演示

从 COCO val2017 取前 N 张含标注的图像，将 polygon 分割转为 bool mask，批量验证 pipeline。

---

## 快速开始

### Python API

```python
import cv2
import numpy as np
from poster_pipeline.pipeline import run_poster_pipeline

rgb = cv2.cvtColor(cv2.imread("photo.jpg"), cv2.COLOR_BGR2RGB)

# 有主体时传入 masks
masks = [{"mask": person_mask_bool, "label": "person"}]
result = run_poster_pipeline(
    rgb,
    masks,
    subject_labels={"person", "dog"},
    corpus_text="华为 智慧生活 影像旗舰 极致性能",
    font_px=56,
    layout_style="h_top",
)

# 无主体时传空列表
result = run_poster_pipeline(rgb, [], corpus_text="华为 智慧生活")

from PIL import Image
Image.fromarray(result["preview"]).save("poster.png")
# 保存辅助 mask
Image.fromarray((result["subj_mask"].astype("uint8") * 255)).save("subject_mask.png")
Image.fromarray(((result["complexity"] <= 0.5).astype("uint8") * 255)).save("complexity_binary.png")
Image.fromarray((result["writable"].astype("uint8") * 255)).save("writable_mask.png")
```

### JSONL 批量处理

```bash
python poster_pipeline/run_jsonl.py \
    --jsonl  /path/to/MTI_split_part00000000.jsonl \
    --img_dir /path/to/MTI_split_part00000000 \
    --out_dir /path/to/output \
    --n 20
```

参数说明：

| 参数 | 说明 |
|---|---|
| `--jsonl` | 输入 JSONL 文件路径 |
| `--img_dir` | 本地图片目录（按 img_path 末段文件名匹配） |
| `--out_dir` | 输出目录（默认 JSONL 同级 `<stem>_out/`） |
| `--n` | 最多处理 N 条（0 = 全部） |
| `--font` | 中文字体路径（不填自动检测系统字体） |

### COCO 演示

```bash
python poster_pipeline/run_coco.py
```

（需提前下载 COCO val2017 图片与标注，修改脚本中的路径）

---

## 依赖

```
numpy
scipy
Pillow
opencv-python      # 读图与 polygon→mask（run_coco.py）
pycocotools        # RLE mask 解码（run_jsonl.py）
```

字体优先级（自动回退）：
1. `font_path` 参数指定路径
2. `/System/Library/Fonts/STHeiti Medium.ttc`（macOS）
3. `C:\Windows\Fonts\msyh.ttc`（Windows 微软雅黑）
4. `/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc`（Linux）
5. Pillow 内置 bitmap 字体（不支持中文，兜底）

---

## 关键设计决策

### 1. 直接交集而非 Zone 分区

旧方案把图像划分为 8 个原子 zone（top/bottom/left/right/四角），在每个 zone 内分别排版。新方案简化为：
```
writable = (~subject_forb) & (complexity ≤ threshold)
```
直接将交集结果传给排版引擎，layout_style 控制槽位的扫描顺序（顶部/底部/竖排等）。优点：逻辑简单、无 zone 质量门槛、可写区域完整保留。

### 2. 大高斯平滑（sigma ≈ 短边/8）

复杂度图先计算 Sobel 梯度和 Laplacian 能量，再用大高斯（sigma 约 40~80 px）平滑，把边缘能量扩散到区域尺度。效果：
- 天空大片低频区 → 整块接近 0（适合写字）
- 密集树枝、纹理区 → 整块接近 1（不适合写字）
- 避免了小平滑半径下天空内微弱差异被过度放大

### 3. 无主体时全图参与复杂度过滤

当 `masks=[]` 时，主体禁区为全零，`writable` 退化为纯复杂度二值图。适合纯背景图（产品摆拍、风景等），直接在低复杂度区域排版，无需人工划定主体。

### 4. `band.all(axis=0)` 保证文字矩形不跨主体

横排槽位要求文字高度带内每列每行都在可写区，彻底避免文字跨越主体"洞"（主体在图像中间时上下两块可写区不会被连通成一个槽位）。

### 5. 归一化用 0~99th percentile

保留各区域的绝对强弱关系：弱边缘的天空对应低复杂度值，强边缘的树枝对应高复杂度值。相对归一化（5th~95th）会把天空内细微差异拉伸到 0~1，导致天空和树枝无法区分。
