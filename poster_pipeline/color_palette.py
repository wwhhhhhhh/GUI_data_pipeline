"""
色卡 + 对比色挑选工具。

- PALETTE: 90 种基础颜色（hex/rgb/中英名列表）
- pick_contrast_color: 基于背景色，从色卡中挑选对比色（按 WCAG 对比度排序）
- random_name: 从一项的中英名列表中随机选一个作为可读名称
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 色卡数据
# ---------------------------------------------------------------------------

_RAW_PALETTE = [
    (1, "#000000", (0, 0, 0), ["黑色", "纯黑", "黑", "漆黑"], ["black"]),
    (2, "#000080", (0, 0, 128), ["藏青", "海军蓝", "海军色", "暗蓝", "暗蓝色", "深蓝色"], ["navy blue", "navy", "dark blue"]),
    (3, "#0000ff", (0, 0, 255), ["蓝色", "纯蓝", "蓝"], ["blue"]),
    (4, "#003153", (0, 49, 83), ["普鲁士蓝", "深海军蓝"], ["prussian blue", "deep navy"]),
    (5, "#003399", (0, 51, 153), ["钴蓝", "钴蓝色", "青天蓝", "克莱因蓝", "午夜蓝", "青花蓝"], ["cobalt blue", "cerulean blue", "midnight blue"]),
    (6, "#0033ff", (0, 51, 255), ["群青", "中蓝", "宝蓝色"], ["ultramarine", "medium blue", "sapphire blue"]),
    (7, "#00477d", (0, 71, 125), ["水手蓝", "天青石蓝", "矿蓝"], ["marine blue", "lapis lazuli"]),
    (8, "#006374", (0, 99, 116), ["墨蓝色"], ["ink blue"]),
    (9, "#007ba7", (0, 123, 167), ["蔚蓝色", "钢青色", "钢蓝色"], ["cerulean", "steel blue"]),
    (10, "#008080", (0, 128, 128), ["鸭绿色", "水鸭色", "深青色", "孔雀蓝"], ["teal", "dark cyan", "peacock blue"]),
    (11, "#00bfff", (0, 191, 255), ["深天蓝", "晴空蓝", "天蓝色", "天空蓝"], ["deep sky blue", "sky blue"]),
    (12, "#00ff00", (0, 255, 0), ["莱姆色", "青柠绿", "酸橙色", "绿色"], ["lime", "green"]),
    (13, "#00ff80", (0, 255, 128), ["春绿色", "春绿"], ["spring green"]),
    (14, "#082567", (8, 37, 103), ["蓝宝石色", "青玉色", "暗矿蓝"], ["sapphire blue", "sapphire"]),
    (15, "#0dbf8c", (13, 191, 140), ["青蓝"], ["cyan blue"]),
    (16, "#127436", (18, 116, 54), ["海洋绿", "深绿色", "森林绿", "浓绿色", "铬绿", "海绿"], ["seagreen", "forest green", "viridian"]),
    (17, "#1e90ff", (30, 144, 255), ["道奇蓝", "品蓝", "菊兰色"], ["dodger blue", "azure"]),
    (18, "#20b2aa", (32, 178, 170), ["浅海绿色", "浅海洋绿"], ["light sea green"]),
    (19, "#228b22", (34, 139, 34), ["森林绿", "暗绿色", "深绿色"], ["forest green", "deep green"]),
    (20, "#36bf36", (54, 191, 54), ["柠檬绿", "橙绿色", "酸橙绿"], ["lime green"]),
    (21, "#404040", (64, 64, 64), ["深灰色", "暗岩灰", "深石板灰"], ["dark gray", "dark slate gray"]),
    (22, "#483d8b", (72, 61, 139), ["暗岩蓝", "暗灰蓝色", "深板岩暗蓝灰色"], ["dark slate blue"]),
    (23, "#4d1f00", (77, 31, 0), ["椰褐", "椰壳褐", "深棕色", "暗棕色"], ["coconut brown", "dark brown"]),
    (24, "#4de680", (77, 230, 128), ["中春绿色"], ["medium spring green"]),
    (25, "#50c878", (80, 200, 120), ["祖母绿", "中海绿", "适中的海洋绿"], ["emerald", "medium sea green"]),
    (26, "#5f9ea0", (95, 158, 160), ["军服蓝", "军校蓝", "萨克斯蓝", "暗青色"], ["cadet blue", "saxe blue"]),
    (27, "#66cdaa", (102, 205, 170), ["中碧绿色", "中绿色"], ["medium aquamarine"]),
    (28, "#66ff59", (102, 255, 89), ["亮绿色"], ["bright green"]),
    (29, "#697723", (105, 119, 35), ["暗橄榄绿", "橄榄土褐色", "苔藓绿", "深橄榄绿"], ["dark olive green", "olive drab", "moss green"]),
    (30, "#6a0dad", (106, 13, 173), ["紫色", "三色堇紫", "暗兰紫", "深兰花紫"], ["purple", "pansy", "dark orchid"]),
    (31, "#6a5acd", (106, 90, 205), ["中紫", "适中的紫色", "紫水晶色", "紫罗兰色", "薰衣草紫"], ["medium purple", "amethyst", "violet"]),
    (32, "#704214", (112, 66, 20), ["乌贼墨色", "深褐色", "咖啡色"], ["sepia", "coffee"]),
    (33, "#7fffd4", (127, 255, 212), ["碧绿色", "绿玉", "海蓝宝石色"], ["aquamarine"]),
    (34, "#8000ff", (128, 0, 255), ["堇紫色"], ["violet"]),
    (35, "#808000", (128, 128, 0), ["橄榄色", "橄榄"], ["olive"]),
    (36, "#808080", (128, 128, 128), ["灰色", "昏灰", "岩灰", "石板灰", "铁灰色"], ["grey", "gray", "dim gray", "slate gray", "iron gray"]),
    (37, "#8674a1", (134, 116, 161), ["浅灰紫"], ["grayish purple"]),
    (38, "#89cff0", (137, 207, 240), ["浅蓝", "天空蓝", "亮天蓝色"], ["baby blue", "sky blue"]),
    (39, "#8a2be2", (138, 43, 226), ["蓝紫", "紫罗兰色", "紫罗兰"], ["blue violet"]),
    (40, "#8b008b", (139, 0, 139), ["暗洋红", "深紫红", "深紫红色"], ["dark magenta"]),
    (41, "#8ce600", (140, 230, 0), ["苹果绿", "绿黄", "黄绿色", "嫩绿", "草坪绿", "明亮绿"], ["apple green", "green yellow", "lawn green", "chartreuse"]),
    (42, "#8e4585", (142, 69, 133), ["梅红色"], ["plum"]),
    (43, "#8fbc8f", (143, 188, 143), ["暗海绿", "青瓷绿", "青瓷色"], ["dark sea green", "celadon"]),
    (44, "#90ee90", (144, 238, 144), ["亮绿色", "浅绿色", "淡绿色"], ["light green", "pale green"]),
    (45, "#9acd32", (154, 205, 50), ["草绿", "草绿色", "叶绿"], ["grass green", "foliage green"]),
    (46, "#a0522d", (160, 82, 45), ["赫色", "黄土赭色", "鞍褐", "马鞍棕色", "驼色"], ["sienna", "saddle brown", "camel"]),
    (47, "#a52a2a", (165, 42, 42), ["砖红色", "火砖色", "栗色", "深红色", "勃艮第酒红"], ["dark red"]),
    (48, "#a9a9a9", (169, 169, 169), ["银色", "浅灰色", "亮灰色", "庚斯博罗灰"], ["silver", "light gray", "gainsboro"]),
    (49, "#b784a7", (183, 132, 167), ["紫丁香色"], ["lilac"]),
    (50, "#cd7f32", (184, 115, 51), ["铜色", "秘鲁色", "秘鲁"], ["bronze", "peru"]),
    (51, "#b8ddc8", (184, 221, 200), ["薄荷绿", "水绿色", "青瓷色"], ["pale mint", "aqua green", "celadon"]),
    (52, "#ba55d3", (186, 85, 211), ["中兰紫", "适中的兰花紫"], ["medium orchid"]),
    (53, "#bc8f8f", (188, 143, 143), ["玫瑰褐", "玫瑰棕色", "干枯玫瑰红"], ["rosy brown", "dusty rose"]),
    (54, "#bdb76b", (189, 183, 107), ["暗卡其色"], ["dark khaki"]),
    (55, "#c71585", (199, 21, 133), ["中青紫红", "适中的紫罗兰红色"], ["medium violet red"]),
    (56, "#c9a0dc", (201, 160, 220), ["紫藤色", "木槿紫", "淡紫色", "薰衣草色", "浅紫色"], ["wisteria", "light purple"]),
    (57, "#cccc4d", (204, 204, 77), ["芥末黄", "橄榄黄"], ["mustard", "olive yellow"]),
    (58, "#ccccff", (204, 204, 255), ["薰衣草蓝", "淡紫蓝"], ["lavender blue"]),
    (59, "#ccff00", (204, 255, 0), ["激光柠檬色", "荧光黄绿色"], ["laser lemon", "fluorescent yellow-green"]),
    (60, "#d2691e", (210, 105, 30), ["巧克力色", "燃橙", "燃橙色"], ["chocolate", "burnt orange"]),
    (61, "#d2b48c", (210, 180, 140), ["日晒色", "茶色", "硬木色", "原木色", "浅褐色"], ["tan", "burly wood", "light brown"]),
    (62, "#d94dff", (217, 77, 255), ["锦葵紫"], ["mallow"]),
    (63, "#daa520", (218, 165, 32), ["金菊色", "金麒麟色", "秋麒麟", "暗金黄色"], ["goldenrod", "dark goldenrod"]),
    (64, "#e60000", (230, 0, 0), ["鲜红", "红色", "中国红", "纯红", "腥红", "猩红色", "朱红"], ["red", "china red", "bright red", "scarlet", "vermilion"]),
    (65, "#e6005c", (230, 0, 92), ["胭脂红", "樱桃红", "樱桃色"], ["carmine", "cerise"]),
    (66, "#e68ab8", (230, 138, 184), ["苍紫罗兰色", "苍白的紫罗兰红色"], ["pale violet red"]),
    (67, "#e6c35c", (230, 195, 92), ["茉莉黄"], ["jasmine"]),
    (68, "#e6c3c3", (230, 195, 195), ["雾玫瑰色", "浅玫瑰色", "薄雾玫瑰", "莫兰迪粉"], ["misty rose", "dusty rose"]),
    (69, "#e6d933", (230, 217, 51), ["含羞草黄", "月黄"], ["mimosa", "moon yellow"]),
    (70, "#e9967a", (233, 150, 122), ["暗鲑红", "暗肉色", "深鲜肉色", "深肉色"], ["dark salmon"]),
    (71, "#ee82ee", (238, 130, 238), ["兰紫色", "兰花紫", "紫罗兰色"], ["lavender magenta", "orchid"]),
    (72, "#f08080", (240, 128, 128), ["亮珊瑚色", "淡珊瑚色", "浅珊瑚色"], ["light coral"]),
    (73, "#f0e68c", (240, 230, 140), ["灰金菊色", "苍麒麟色", "浅金菊黄"], ["pale goldenrod"]),
    (74, "#f28500", (242, 133, 0), ["橘色", "暗橙", "深橙色", "万寿菊黄", "橙色"], ["tangerine", "dark orange", "marigold", "orange"]),
    (75, "#f400a1", (244, 0, 161), ["洋玫瑰红"], ["magenta rose"]),
    (76, "#ff007f", (255, 0, 127), ["深粉红", "深粉色", "玫瑰红"], ["rose", "deep pink"]),
    (77, "#ff00ff", (255, 0, 255), ["洋红色", "品红", "品红色", "紫红色", "红紫色", "洋红"], ["magenta", "fuchsia"]),
    (78, "#ff4d40", (255, 77, 64), ["朱红色", "蕃茄红", "番茄红", "番茄色", "柿子橙"], ["tomato", "persimmon", "vermilion"]),
    (79, "#ff6600", (255, 102, 0), ["阳橙", "阳橙色"], ["international orange", "vivid orange", "sun orange"]),
    (80, "#ff66cc", (255, 102, 204), ["浅玫瑰红"], ["rose pink"]),
    (81, "#ff7f50", (255, 127, 80), ["珊瑚红", "珊瑚色", "珊瑚", "鲑肉色", "橙红色"], ["coral", "tropical orange", "salmon color"]),
    (82, "#ffb3e6", (255, 179, 230), ["浅珍珠红"], ["pearl pink"]),
    (83, "#ffbf00", (255, 191, 0), ["琥珀色", "铬黄", "金色", "金黄", "橙黄色"], ["amber", "gold", "tangerine yellow"]),
    (84, "#ffdab9", (255, 218, 185), ["粉扑桃色", "桃色", "陶坯黄", "珍珠桃色"], ["peachpuff", "peach", "pearly peach"]),
    (85, "#ffffff", (255, 255, 255), ["精白", "白色", "纯白", "白", "雪色", "雪白色", "雪白"], ["white", "snow"]),
    (86, "#0abab5", (10, 186, 181), ["蒂芙尼蓝", "青绿色", "湖水绿", "绿松石色", "Tiffany蓝"], ["tiffany blue", "turquoise"]),
    (87, "#00ffff", (0, 255, 255), ["青色", "纯青"], ["cyan", "aqua"]),
    (88, "#ffb7c5", (255, 183, 197), ["樱花粉", "淡粉色", "浅粉色", "糖果粉色", "糖果粉红色"], ["cherry blossom pink", "light pink", "pastel pink", "candy pink"]),
    (89, "#ffff00", (255, 255, 0), ["柠檬黄", "纯黄"], ["lemon yellow", "pure yellow", "yellow"]),
    (90, "#e6e6fa", (230, 230, 250), ["薰衣草紫", "淡紫色"], ["lavender"]),
]


def _to_dict(entry) -> Dict:
    _id, _hex, rgb, zh, en = entry
    return {"id": _id, "hex": _hex, "rgb": rgb, "names_zh": zh, "names_en": en}


PALETTE: List[Dict] = [_to_dict(e) for e in _RAW_PALETTE]


# ---------------------------------------------------------------------------
# 对比度计算（WCAG 相对亮度）
# ---------------------------------------------------------------------------

def _linearize(c: float) -> float:
    c = c / 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(rgb: Tuple[float, float, float]) -> float:
    r, g, b = rgb
    return 0.2126 * _linearize(r) + 0.7152 * _linearize(g) + 0.0722 * _linearize(b)


def contrast_ratio(fg: Tuple[float, float, float], bg: Tuple[float, float, float]) -> float:
    l1 = relative_luminance(fg)
    l2 = relative_luminance(bg)
    hi, lo = (l1, l2) if l1 >= l2 else (l2, l1)
    return (hi + 0.05) / (lo + 0.05)


# ---------------------------------------------------------------------------
# 挑选对比色
# ---------------------------------------------------------------------------

def pick_contrast_color(
    bg_rgb: Tuple[float, float, float],
    rng: Optional[random.Random] = None,
    *,
    min_ratio: float = 4.5,
    top_k: int = 8,
    exclude_ids: Optional[set] = None,
) -> Dict:
    """
    从色卡中挑选一个与背景对比足够的颜色。

    策略：
      1. 计算所有色卡项与背景的 WCAG 对比度
      2. 过滤掉对比度 < min_ratio 的项（若全部不满足则降级）
      3. 在对比度前 top_k 中随机挑一个（避免总是用黑或白）

    Returns: PALETTE 中的一项（dict），带一个随机 name 字段（中英名随机之一）。
    """
    rng = rng or random.Random()
    candidates = []
    exclude = exclude_ids or set()
    for item in PALETTE:
        if item["id"] in exclude:
            continue
        cr = contrast_ratio(item["rgb"], bg_rgb)
        candidates.append((cr, item))
    candidates.sort(key=lambda x: x[0], reverse=True)

    good = [x for x in candidates if x[0] >= min_ratio]
    pool = good if good else candidates
    pool = pool[:top_k] if len(pool) > top_k else pool
    _cr, chosen = rng.choice(pool)
    return _annotate(chosen, rng, _cr)


def _annotate(item: Dict, rng: random.Random, cr: float) -> Dict:
    """从 names_zh + names_en 中随机挑一个作为 name。"""
    pool = list(item["names_zh"]) + list(item["names_en"])
    name = rng.choice(pool) if pool else item["hex"]
    return {
        "id":       item["id"],
        "hex":      item["hex"],
        "rgb":      tuple(item["rgb"]),
        "name":     name,
        "name_zh":  rng.choice(item["names_zh"]) if item["names_zh"] else "",
        "name_en":  rng.choice(item["names_en"]) if item["names_en"] else "",
        "contrast": round(float(cr), 3),
    }
