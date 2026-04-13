# poster_pipeline — 海报自动排版工程

基于实例分割 mask 的海报自动排版工程。输入图像与主体分割结果，自动找到适合写字的区域，规划排版，将语料渲染到图上。

---

## 整体思路

```
输入图像 + 实例分割 masks
        │
        ▼
  ① 主体禁区构建          writable_mask.py
     union 所有主体 mask，膨胀生成禁区
        │
        ▼
  ② 复杂度图              writable_mask.py
     Sobel + Laplacian 计算每像素边缘强度
     天空/草地 → 低复杂度（适合写字）
     树枝/纹理 → 高复杂度（不适合写字）
        │
        ▼
  ③ Zone 分区             zones.py
     以主体 bbox 为中心划分 8 个原子 zone
     （top / bottom / left / right / 四角）
     相邻原子 zone 可合并为复合 zone（top_wide / left_tall 等）
     按复杂度品质过滤，按风格(style)选取目标 zone
        │
        ▼
  ④ 排版规划              layout_scanline.py → plan_layout
     对每个 zone 的 mask 做 band.all(axis=0) 扫描
     找到文字可完整放入的矩形槽位 LineSlot
        │
        ▼
  ⑤ 文本分配              layout_scanline.py → fill_slots
     每个 zone 独立获得完整语料，各自贪心填入槽位
     非连通区域各自写字，不再均分语料
     横排按词，竖排按字符；用 PIL 实测字宽，保证不溢出槽位
        │
        ▼
  ⑥ 颜色选取              color_contrast.py
     采样最优 zone 的背景中位色
     亮底→深字，暗底→白字
        │
        ▼
  ⑦ 文字渲染              layout_scanline.py → render_lines
     横排：在 (x0, y) 逐行绘制
     竖排：逐字在 (x0, y + i·font_px) 绘制，字符居中于列宽
     带描边（对比色）提升可读性
        │
        ▼
     输出 preview 图 + debug JSON
```

---

## 模块说明

### `pipeline.py` — 端到端入口

对外暴露 `run_poster_pipeline()`，串联所有步骤。

| 参数 | 说明 |
|---|---|
| `rgb` | HxWx3 uint8 图像 |
| `masks` | `[{"mask": bool HxW, "label": str}, ...]` 实例分割结果 |
| `subject_labels` | 需要避开的类别集合（如 `{"person", "dog"}`）；`None` 表示全部 mask 都避开 |
| `corpus_text` | 空格分隔的词组，如 `"华为 智慧生活 影像旗舰 极致性能"` |
| `font_path` | 字体文件路径，默认自动回退到系统中文字体 |
| `font_px` | 字号（像素），海报建议 48~72 |
| `layout_style` | 排版风格，见下表，默认 `"top_title"` |
| `dilate_iter` | 主体禁区膨胀半径（像素），越大文字距主体越远，默认 14 |
| `min_zone_quality` | zone 质量下限（0~1），低于此跳过该区，默认 0.30 |

返回字典包含：`preview`（渲染后图像 ndarray）、`lines`、`writable_binary`、`complexity`、`debug`。

---

### `writable_mask.py` — 主体禁区 + 复杂度图

**`union_subject_masks(masks, h, w, forbid_labels)`**
将指定类别的实例 mask 合并为一张主体 mask。

**`dilate_binary(mask, iterations)`**
对主体 mask 做形态学膨胀，生成安全禁区（文字不能进入的缓冲区）。

**`complexity_map(rgb)`**
计算全图复杂度，值域 [0, 1]：
- 使用 Sobel 梯度幅值（权重 0.6）+ Laplacian 能量（权重 0.4）
- 归一化采用 0~99th percentile，保留各区域的绝对强弱关系
- 天空、草地等低频区 → 接近 0；树枝、纹理等高频区 → 接近 1

---

### `zones.py` — Zone 分区与风格管理

**核心思想**：以主体的 bounding box 为参照，将图像划分为最多 8 个**原子 zone**，同时支持将相邻原子 zone 合并为更大的**复合 zone**：

```
┌─────────────┬───────────┬─────────────┐
│  top_left   │    top    │  top_right  │
├─────────────┼───────────┼─────────────┤
│    left     │  [主体]   │    right    │
├─────────────┼───────────┼─────────────┤
│  bot_left   │  bottom   │  bot_right  │
└─────────────┴───────────┴─────────────┘

复合 zone 示例（取成员并集）：
  top_wide  = top_left ∪ top ∪ top_right   （顶部全幅）
  left_tall = top_left ∪ left ∪ bot_left   （左侧全高）
  bot_wide  = bot_left ∪ bottom ∪ bot_right （底部全幅）
```

每个 zone 的质量分 = 该区非禁像素的均值 `(1 - complexity)`，反映该区有多"低频"（适合写字）。

**`build_zones(h, w, forb_mask, complexity, style, raw_subj_mask)`**
- `raw_subj_mask`：未膨胀的主体 mask，用于计算 bbox（避免 dilation 压缩 zone 空间）
- 按 style 选取指定 zone（原子或复合），过滤面积过小或质量过低的 zone
- 复合 zone 通过 `COMPOSITE_ZONES` 字典定义，自动合并成员原子 zone 的可写像素
- 返回 `List[(name, zone_mask, direction, quality)]`

**`build_zones_with_fallback(...)`**
若请求的 style 没有可用 zone（主体遮挡等极端情况），依次尝试备用 style 序列，最终兜底为全图 zone，保证一定有输出。

**支持的 18 种海报风格**：

| 风格 | 含义 | 使用 zone |
|---|---|---|
| `top_title` | 大标题在上，副标在下 | top + bottom |
| `sidebar_r` | 顶部横排 + 右侧竖列 | top + right + bottom |
| `sidebar_l` | 顶部横排 + 左侧竖列 | top + left + bottom |
| `split` | 主体居中，左右各一竖列 | left + right |
| `bottom_cap` | 底部宽幅横排（电影字幕风） | bottom |
| `frame` | 四边环绕 | top + left + right + bottom |
| `diagonal` | 左上 + 右下对角分布 | top_left + bot_right |
| `diagonal_alt` | 右上 + 左下对角分布 | top_right + bot_left |
| `top_banner` | 顶部全幅大标题（复合） | top_wide（top_left ∪ top ∪ top_right） |
| `bot_banner` | 底部全幅横排（复合） | bot_wide |
| `h_banner` | 上下全幅横排（复合） | top_wide + bot_wide |
| `v_columns` | 左右全高竖排（复合） | left_tall + right_tall |
| `l_shape_l` | 宽顶 + 左竖，L 形（复合） | top_wide + left |
| `l_shape_r` | 宽顶 + 右竖，Γ 形（复合） | top_wide + right |
| `frame_wide` | 全边框宽幅（复合） | top_wide + left_tall + right_tall + bot_wide |
| `diag_wide` | 左上+右下宽对角（复合） | top_half_l + bot_half_r |
| `diag_wide_r` | 右上+左下宽对角（复合） | top_half_r + bot_half_l |
| `sidebar_wide` | 全幅顶部 + 右全高（复合） | top_wide + right_tall |

---

### `layout_scanline.py` — 三层排版引擎

排版分三个职责完全分离的层：

#### 第一层：`plan_layout(mask, font_px, style)` — 纯几何

在 zone mask 内找文字槽位 `LineSlot`。

核心算法（横排）：
```
对每个候选行带 [y, y+font_px)：
  col_clear = mask[y:y+font_px].all(axis=0)
  → 整列在行带内每行都为 True 才算可写
  → 找最宽的连续可写列段
  → 宽度 ≥ min_width_chars × font_px → 生成槽位
```
这保证文字矩形的每一个像素都在可写区内，不会跨越主体的"洞"。

竖排：对每个候选列带 `[x, x+font_px)` 做 `mask[:, x:x+font_px].all(axis=1)` 找最长可写行段。

支持子风格：`h_top` / `h_bottom` / `h_center` / `v_right` / `v_left` / `surround`

#### 第二层：`fill_slots(slots, corpus, font_path, font_px)` — 纯文本分配

- 每个 zone 独立传入**完整语料**，不再跨 zone 均分词语（非连通区域各自写字）
- 横排槽：按空格切词，用 PIL `textlength` 实测字宽，贪心填词；放不下整词直接跳过该槽（不截断）
- 竖排槽：按字符逐个填入，容量 = `height // font_px`
- 返回 `List[TextLine(y, x0, x1, text, direction)]`

#### 第三层：`render_lines(rgb, lines, fg, font_path, font_px)` — 纯渲染

- 横排：`draw.text((x0, y), text, ...)`
- 竖排：逐字 `draw.text((x0 + offset, y + i·font_px), char, ...)`，字符居中于列宽
- 统一加描边（亮字→黑描边，暗字→白描边），提升复杂背景下的可读性

---

### `color_contrast.py` — 文字颜色选取

对最优 zone 区域内的背景像素取 RGB 中位色，计算相对亮度（sRGB 标准），选择黑色或白色中对比度更高者作为前景色。

---

### `regions.py` — 连通域提取（辅助）

从二值 mask 提取连通域，按面积 × 平均分数排序，供调试或旧流程使用。当前主流程已改为 zone 方案，此模块保留备用。

---

### `run_coco.py` — COCO 数据集批量演示

从 COCO val2017 取前 N 张含标注的图像，自动将 polygon 分割转为 bool mask，批量跑 pipeline，输出每张的 `preview.png` / `writable_mask.png` / `quality_map.png` / `debug.json`。

---

## 关键设计决策

### 1. 为什么用 Zone 而不是连通域（connected component）？

连通域会把散落在各处的可写像素归为一组（通过细小路径连通），导致 zone bbox 横跨主体，文字槽位跨越主体"洞"。Zone 基于主体 bbox 直接划分矩形区域，语义清晰，且与海报常见构图（左右留白、上下标题）天然对应。

### 2. 原子 zone + 复合 zone 的组合方式

8 个原子 zone（top/bottom/left/right/四角）是基本单元；`COMPOSITE_ZONES` 字典定义相邻原子 zone 的合并方案（取 mask 并集），生成更大的写字区域：
- `top_wide`（顶部全幅）覆盖主体上方整条横带，比单独的 `top` 更宽，适合大标题；
- `left_tall`（左侧全高）覆盖主体左侧整列，比单独 `left` 更高，适合多行竖排；
- 复合 zone 使文字区域连续、视觉统一，避免分散的短区域导致文字割裂感。

18 种预设 style 覆盖从经典（顶底横排、左右竖排）到复合（L 形、全边框宽幅、宽对角）的常见海报构图。

### 3. 每个 zone 独立填入完整语料

旧方案把语料均分给各 zone，导致 zone 多时每区仅得 1~2 个词、视觉过于稀疏。新方案改为：**每个 zone 独立获得完整语料**，由 `plan_layout` 的 `max_slots` 参数自然限制每区最多放多少行。非连通的区域各自独立排版，相同文字出现在不同区域，形成海报常见的品牌重复感。

### 4. `band.all(axis=0)` 为什么比逐行扫描更好？

旧方法取行内最宽的 True 区间（min~max），当主体形成"洞"时，min~max 会横跨洞，文字画在主体上。`band.all(axis=0)` 要求文字矩形内每一列、每一行都为 True，彻底杜绝这种情况。

### 5. 复杂度归一化为什么用 0~99th percentile？

旧方法用 5th~95th percentile 的相对归一化：若图像大部分是天空，天空内部细微差异也被拉伸到 0~1，使天空与树枝的复杂度差异消失。改用 0~99th percentile 后，绝对边缘强度弱的天空对应低复杂度值，边缘强的树枝对应高复杂度值，两者可以区分。

### 6. bbox 用 raw mask，可写性用 dilated mask

Zone 边界（如 top zone 的高度）由**未膨胀**的主体 bbox 决定，确保 zone 有足够空间放置文字。但 zone 内哪些像素真正可写，由**膨胀后**的禁区决定，确保文字与主体之间有安全间距。

---

## 依赖

```
numpy
scipy
Pillow
opencv-python   # 仅 run_coco.py 用于读图和 polygon→mask
```

字体优先级（自动回退）：
1. `font_path` 参数指定路径
2. `/System/Library/Fonts/STHeiti Medium.ttc`（macOS）
3. `/System/Library/Fonts/PingFang.ttc`（macOS）
4. `/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc`（Linux）
5. `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`（Linux）
6. Pillow 内置 bitmap 字体（不支持中文，兜底）

---

## 快速开始

```python
import cv2, numpy as np
from poster_pipeline.pipeline import run_poster_pipeline

rgb = cv2.cvtColor(cv2.imread("photo.jpg"), cv2.COLOR_BGR2RGB)

# masks 来自实例分割模型（如 SAM / Mask R-CNN）
masks = [{"mask": person_mask, "label": "person"}]

result = run_poster_pipeline(
    rgb,
    masks,
    subject_labels={"person", "dog"},   # 需要避开的类别
    corpus_text="华为 智慧生活 影像旗舰 极致性能",
    font_px=56,
    layout_style="l_shape_r",           # 宽顶 + 右侧竖列（复合 zone）
)

from PIL import Image
Image.fromarray(result["preview"]).save("poster.png")
```

批量跑 COCO 演示：
```bash
python poster_pipeline/run_coco.py
```
输出在 `_coco_demo_out/<image_id>/` 目录下。

---

## 原始背景与参考文献

> 以下为立项时的方法调研，供延伸阅读。

### 算法选型参考

**扫描线折行（已实现）**：对每行 y 求可写区间宽度，按词贪心折行。工程讨论见 StackOverflow *Algorithm for fitting text into an irregular shape*。

**最大内接矩形（MIR）**：在可写 mask 内找最大矩形，适合单行大标题。可用旋转卡壳或 DP 实现。

**学习型排版**：PosterLayout（CVPR 2023）、PosterO（2025）等，对数据和算力要求较高，适合生成式场景。

### 参考链接

- [StackOverflow: fitting text into irregular shape](https://stackoverflow.com/questions/42809043/algorithm-for-fitting-text-into-an-irregular-shape)
- [CVPR 2023 PosterLayout](https://openaccess.thecvf.com/content/CVPR2023/html/Hsu_PosterLayout_A_New_Benchmark_and_Approach_for_Content-Aware_Visual-Textual_Presentation_CVPR_2023_paper.html)
- [PosterO 项目页](https://thekinsley.github.io/PosterO/)

---

## 1. 输入约定（建议）

| 字段 | 说明 |
|------|------|
| `image` | `H×W×3` uint8 RGB |
| `masks` | 列表，每项含 `mask: bool[H,W]`、`label` 或 `class_id`、`score`（可选） |
| `subject_classes` | 需要**避开**的粗类集合，如 `person, animal, vehicle_face, text` 等（由你的 SAM3 后处理或 CLIP 粗分类映射而来） |
| `corpus` / `font` | 你后续提供；本仓库骨架里用占位字符串与 Pillow 渲染 |

SAM3 往往给出**实例掩码**而不直接给语义名：工程上需要一层 **`mask_id → 粗类`**（轻量分类器、CLIP 阈值、或规则：大面积中心连通域 + 形状先验）。没有这一层时，可先用**面积/位置启发式**（最大连通域且靠近图像下半部 → 主体）做 MVP。

---

## 2. 可写字区域：从「禁止层」与「复杂度惩罚」构造

### 2.1 语义/实例禁止层 \(B\)

- 对所有属于 `subject_classes` 的掩码取并集：\(B = \bigcup M_i\)。
- **形态学膨胀** \(B' = \mathrm{dilate}(B)\) 留出笔划与感知边距（避免贴边）。
- 初始可写：\(W_0 = \neg B'\)。

### 2.2 图像复杂度图 \(C\)（纹理/细节不适合排字）

常用、实现成本低且鲁棒的标量：

- **梯度能量**：\(G = \|\nabla I\|\)（灰度上算 Sobel）；局部均值 `box_filter(G)`。
- **拉普拉斯响应方差**（小窗口）：纹理越丰富方差越大。
- **局部熵/直方图散度（可选）**：滑动窗口内统计灰度直方图，纹理越碎则指标越高。

归一化到 \([0,1]\) 后定义惩罚：

\[
W_1 = W_0 \odot (1 - C)^\gamma
\]

\(\gamma\) 控制对复杂纹理的排斥强度。

### 2.3 可选：显著性/「抢注意力」区域（进阶）

若你有轻量显著性（或用人脸框），可并入禁止层，避免字压在视觉焦点上。

### 2.4 二值化与连通域

- `writable = W1 > τ`（τ 可自适应：Otsu 或分位数）。
- **连通域标记** → 多个候选区域；按**面积 × 平均可写分数**排序，取 Top-K 或合并小域。

---

## 3. 颜色与反差字色

在最终排版多边形/掩码内采样像素（建议 **Lab** 或 **Y** 分量）：

- 背景代表色：取 **masked patch的中位数**（对异常点鲁棒）。
- 字色：在 `{近白, 近黑}` 或色轮上选使 \(\Delta E\) 或 \(|L_{\text{fg}} - L_{\text{bg}}|\) 最大的候选；可加**细描边/阴影**（海报常用）提高任意背景下的下限。

---

## 4. 不规则区域排版：已有思路与文献/工程对照

下面按「从经典几何 → 工业排版 → 学习型海报」分层，便于你选型。

### 4.1 水平扫描线 / 「竖条/slab」法（**强推荐作基线**）

**思想**：对单连通（或可拆成若干简单域）区域，对每个纵坐标 \(y\) 求区域与水平线的交集，得到一组区间 \([x_\min(y), x_\max(y)]\)。在每个 \(y\) 上可用宽度随 \(y\) 变化，把**折行**看成「可变行长」的段落排版。

- 工程讨论与实现线索：StackOverflow *Algorithm for fitting text into an irregular shape*（扫描线 + 测量可用宽度）。
- 印刷/PDF 领域：HexaPDF 等将 **Knuth-Plass** 思想扩展到非矩形（分段测量宽度）。

**优点**：实现相对直接、可控、无训练；**缺点**：极端凹多边形会出现「窄腰」导致断行难看，需要后处理（见下）。

### 4.2 最大内接矩形（MIR）/ 旋转卡壳（**块级标题**）

在简单多边形内求（近似）最大轴对齐或旋转矩形，适合**大标题单行/少行**，作首块锚点，再在剩余区域做扫描线。

- 综述/实践：*Maximum Inscribed Rectangle for Irregular Shapes*（旋转卡壳、DP、遗传算法等）。

### 4.3 距离变换 + 中轴（**定「排版脊线」与方向**）

对二值可写区域做 **Euclidean Distance Transform**；脊线（局部极大）近似**中轴**。沿脊线可估计局部**主方向**（曲线切向），用于：

- 决定文字块是**横排**还是**沿弧微旋转**（海报装饰字）；
- 先把区域用 **PCA/最小外接矩形**对齐到主轴，再在轴对齐 bbox 内做矩形排版，最后逆变换（工程上常用 trick）。

### 4.4 学习型 / 数据集驱动（**要「像设计师」时再上**）

- **PosterLayout**（CVPR 2023）：内容感知海报图文排版 benchmark与方法脉络，可参考其「设计序列」与约束思路。
- **PosterO**（2025）：布局树 + LLM，偏生成式海报结构。
- **HDLayout** 等：层次化布局表示，偏「生成视觉文字形状」研究向。

这些对**数据与算力**要求高；你的场景（分割已知、要鲁棒可控）建议：**几何扫描线 + 规则约束**为主，模型为辅。

### 4.5 推荐组合（落地优先级）

1. **MVP**：禁止层 + 复杂度图 → 连通域 → **PCA 对齐 bbox** 内用 **HarfBuzz/Pillow 折行**（矩形近似）。
2. **不规则主路径**：连通域内 **扫描线求行宽** + **贪心断行**（词级宽度用字体度量）。
3. **标题+正文**：MIR/最大空白块找标题带，正文用扫描线在剩余掩码内排版。

---

## 5. 本目录代码骨架

| 文件 | 作用 |
|------|------|
| `writable_mask.py` | 禁止掩码合并、膨胀、复杂度惩罚、二值化 |
| `regions.py` | 连通域、过滤、排序 |
| `color_contrast.py` | 区域内中位色 + 反差字色 |
| `layout_scanline.py` | 扫描线行宽 + 简单词级折行（需 Pillow） |
| `pipeline.py` | 串起一步流程 |
| `demo_synthetic.py` | **无 SAM3 也可跑**：合成图 + 伪掩码，验证连通域与扫描线 |

依赖：`numpy`、`scipy`（卷积/形态学/连通域）；渲染用 `pillow`。可选 `opencv-python` 做加速或替代实现。

---

## 6. 实例数据（图像 + SAM3）

本仓库**不内置**第三方大图版权文件。请你本地准备一组：

1. 任选 3–10 张场景图（天空/墙面/海面/纯色产品底等主体分明的图）。
2. 用 SAM3 导出每图的 `masks`（JSON 或 npy/png）。
3. 将路径填入 `examples/manifest.example.json`（见同目录说明）。

**可立即运行的示例**：

```bash
cd poster_pipeline
python3 demo_synthetic.py --out ../_poster_demo_out
```

将在 `_poster_demo_out/` 生成合成 `image.png`、`subject_mask.png`、`writable_mask.png`、`preview.png` 和 `layout_debug.json`。

---

## 7. 参考文献与链接（便于你深挖）

- Springer：*An Approach to Optimal Text Placement on Images*（构图与注意力相关原则）。
- StackOverflow / Kasyan：不规则形状内排字——**扫描线 + 可用宽度测量**思路。
- CVPR 2023：**PosterLayout**（内容感知海报排版 benchmark）。
- arXiv：**PosterO**（布局树 + LLM）。
- **最大内接矩形**：**Rotating calipers** / DP / 启发式综述（Oreate AI 博文等）。

便于检索的入口链接（外网环境可打开；内网请用标题检索镜像）：

- [StackOverflow: fitting text into irregular shape](https://stackoverflow.com/questions/42809043/algorithm-for-fitting-text-into-an-irregular-shape)
- [Springer: Optimal Text Placement on Images](https://link.springer.com/chapter/10.1007/978-3-642-39360-0_8)
- [CVPR 2023 PosterLayout（OpenAccess索引页）](https://openaccess.thecvf.com/content/CVPR2023/html/Hsu_PosterLayout_A_New_Benchmark_and_Approach_for_Content-Aware_Visual-Textual_Presentation_CVPR_2023_paper.html)
- [PosterO 项目页](https://thekinsley.github.io/PosterO/)
