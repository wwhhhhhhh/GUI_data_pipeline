"""
页面装配引擎 v4.3 — 宽度感知高度预估 + 保守预算
=================================================
关键改进:
  1. 高度预算 = rem_h × 0.65 (留 35% 安全边距)
  2. 组件高度 = BASE_EST × (35 / rem_w) — 越窄文字换行越多,预估越高
  3. 所有容器 h-screen + overflow:hidden, 杜绝溢出
  4. 去掉 split_screen, sidebar 仅在 ≥42rem 启用
"""

import json, random
from components_v4 import (
    COMPONENT_REGISTRY, DAISY_THEMES, HEIGHT_EST,
    set_render_context, get_image_manifest, reset_image_manifest,
)
from layouts import LAYOUT_TEMPLATES, pick_layout
from page_types import PAGE_TYPES, pick_page_type
from resolutions import RESOLUTIONS, pick_resolution, ROOT_FONT_SIZE, SCALE_FACTOR, BREAKPOINTS

# ── 插槽映射 ──
SLOT_MAP = {
    "navbar":["navbar"], "breadcrumb":["breadcrumb"], "sidebar_menu":["sidebar"],
    "sidebar_toc":["sidebar"], "sidebar_contacts":["sidebar"],
    "sidebar_trending":["sidebar"], "stats":["stats","section_0"],
    "hero_banner":["hero","section_0"], "hero_side":["center_content"],
    "auth_form":["center_content"],
    "error_display":["center_content"],
    "table":["table_area","main_content","section_1"],
    "chart_placeholder":["chart_area","main_content","section_1"],
    "card_grid":["section_0","main_content","section_1"],
    "form_group":["main_content","section_1","center_content"],
    "toggle_group":["section_2","extra"],
    "profile_header":["section_0","main_content"],
    "article_body":["main_content","section_1"],
    "chat_bubbles":["chat_area","main_content"],
    "input_area":["input_area","extra"],
    "kanban_columns":["main_content","section_1"],
    "email_list":["main_content","section_1"],
    "email_detail":["section_2","extra"],
    "feed_cards":["main_content","section_1"],
    "file_list":["main_content","section_1"],
    "calendar_grid":["main_content","section_1"],
    "event_list":["section_2","extra"],
    "pricing_table":["section_1","main_content"],
    "feature_grid":["section_1"],
    "testimonial_cards":["section_2"],
    "faq_accordion":["section_3","section_2","extra"],
    "cta_section":["cta","section_3"],
    "footer":["footer"],
    "comment_list":["extra","section_2"],
    "review_list":["extra","section_2"],
    "code_block":["extra","section_2"],
    "timeline":["main_content","section_2"],
    "progress_group":["extra","section_2"],
    "filter_bar":["breadcrumb","section_0"],
    "tabs":["section_0","main_content"],
    "pagination":["extra"],
    "badge_group":["extra"],
    "avatar_group":["extra"],
    "alert":["extra","section_3"],
    "modal_static":["extra"],
    "rating_display":["main_content"],
    "steps":["breadcrumb"],
    "toolbar":["breadcrumb"],
    "story_bar":["section_0"],
    "suggestion_list":["section_2","extra"],
    "product_detail":["main_content","section_1"],
    "related_cards":["extra","section_3"],
    "tag_group":["extra"],
    "date_range_picker":["breadcrumb"],
    "social_login_buttons":["extra","center_content"],
    "divider":[],
}

# ── HTML 模板 ──
HTML_HEAD = '''<!DOCTYPE html>
<html lang="zh" data-theme="{theme}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width={vw}, initial-scale=1.0">
<title>{title}</title>
<link href="https://cdn.jsdelivr.net/npm/daisyui@4.4.19/dist/full.min.css" rel="stylesheet"/>
<script src="https://cdn.tailwindcss.com"></script>
<script>tailwind.config={{theme:{{screens:{{'sm':'{bp_sm}px','md':'{bp_md}px','lg':'{bp_lg}px','xl':'{bp_xl}px'}}}}}}</script>
<style>
html{{font-size:{root_font}px !important;}}
body{{height:{vh}px;max-height:{vh}px;overflow:hidden;-webkit-font-smoothing:antialiased;margin:0;padding:0;}}
.shadow-sm{{box-shadow:0 .0625rem .125rem 0 rgba(0,0,0,.05)!important}}
.shadow,.shadow-md{{box-shadow:0 .25rem .375rem -.0625rem rgba(0,0,0,.1),0 .125rem .25rem -.125rem rgba(0,0,0,.1)!important}}
.shadow-lg,.shadow-xl{{box-shadow:0 .625rem .9375rem -.1875rem rgba(0,0,0,.1),0 .25rem .375rem -.25rem rgba(0,0,0,.1)!important}}
.shadow-2xl{{box-shadow:0 1.25rem 1.5625rem -.3125rem rgba(0,0,0,.1),0 .5rem .625rem -.375rem rgba(0,0,0,.1)!important}}
.border,.border-b,.border-t,.border-l,.border-r{{border-width:.0625rem!important}}
.border-2{{border-width:.125rem!important}}
.divide-y>*+*{{border-top-width:.0625rem!important}}
.img-placeholder{{background:#FF0000!important;display:block}}
::-webkit-scrollbar{{width:.25rem}}
::-webkit-scrollbar-thumb{{background:oklch(var(--bc)/.2);border-radius:.125rem}}
</style>
</head>
<body class="bg-base-200 font-sans">
'''

HTML_TAIL = '''
<script type="application/json" id="image-manifest">{manifest}</script>
<script>
window.addEventListener('load',function(){{try{{var m=JSON.parse(document.getElementById('image-manifest').textContent);m.forEach(function(i){{var e=document.getElementById(i.id);if(e){{var r=e.getBoundingClientRect();i.x=Math.round(r.x);i.y=Math.round(r.y);i.width=Math.round(r.width);i.height=Math.round(r.height)}}}});document.getElementById('image-manifest').textContent=JSON.stringify(m)}}catch(e){{}}}});
</script>
</body></html>'''


def _est_height(comp_name, rem_w, rem_h=35):
    """
    宽度+高度感知的组件高度预估 (rem)
    - 窄视口 → 文字换行多 → 膨胀: × max(1.0, 35/rem_w)
    - 矮视口 → _mi()更少,文字更短 → 收缩: × min(1.0, rem_h/35)
    """
    base = HEIGHT_EST.get(comp_name, 5)
    if base == 0:
        return 0
    width_scale = max(1.0, 35.0 / max(rem_w, 23))
    height_scale = min(1.0, max(rem_h, 23) / 35.0)
    return base * width_scale * height_scale


def assemble_page(corpus_fn, page_type=None, layout_name=None, theme=None, resolution=None):
    reset_image_manifest()

    # ── 先选分辨率, 再根据预算过滤页面类型 ──
    res = pick_resolution(resolution) if isinstance(resolution, str) or resolution is None else resolution
    if theme is None:
        theme = random.choice(DAISY_THEMES)

    rem_w = res["rem_w"]
    rem_h = res["rem_h"]
    h_budget = rem_h * 0.65

    # ── 页面类型选择: 过滤掉 required 组件超预算的类型 ──
    if page_type:
        pt_name, pt_cfg = pick_page_type(page_type)
    else:
        # 计算每种页面类型的 required 高度, 过滤
        valid_types = {}
        for name, cfg in PAGE_TYPES.items():
            req_h = sum(_est_height(c, rem_w, rem_h) for c in cfg["required"])
            if req_h <= h_budget:
                valid_types[name] = cfg
        if not valid_types:
            # 兜底: 用最小的 error_page
            valid_types = {"error_page": PAGE_TYPES["error_page"]}
        pt_name = random.choice(list(valid_types.keys()))
        pt_cfg = valid_types[pt_name]

    # ── Render context: scale with BOTH width and height ──
    area_ratio = (rem_w * rem_h) / (35 * 35)  # 1.0 at 35×35
    ctx = {
        "rem_w": rem_w,
        "rem_h": rem_h,
        "text_s": max(0.25, min(1.0, min(rem_w, rem_h) / 35)),
        "max_cols": res["max_cols"],
        "max_items": max(2, min(5, int(3 * area_ratio + 1))),
    }
    set_render_context(ctx)

    ly_name, ly_cfg = pick_layout(layout_name, pt_cfg.get("layout_prefer"), res)

    # ── 组件选择: required 必含, optional 受预算限制 ──
    required = list(pt_cfg["required"])
    optional = list(pt_cfg["optional"])
    random.shuffle(optional)

    min_c, max_c = pt_cfg["component_range"]
    target = random.randint(min_c, max_c)
    num_opt = max(0, target - len(required))

    # required 组件: 必须包含, 先计算它们的总高度
    used_h = 0.0
    for comp in required:
        used_h += _est_height(comp, rem_w, rem_h)

    # 如果 required 组件本身已经超预算, 就不加任何 optional
    final_components = list(required)
    if used_h < h_budget:
        for comp in optional[:num_opt]:
            est = _est_height(comp, rem_w, rem_h)
            if est == 0:
                final_components.append(comp)
                continue
            if used_h + est <= h_budget:
                final_components.append(comp)
                used_h += est

    # ── 渲染组件 → 填槽 ──
    slots = {s: "" for s in ly_cfg["slots"]}
    rendered = []

    for comp_name in final_components:
        if comp_name not in COMPONENT_REGISTRY:
            continue
        try:
            html = COMPONENT_REGISTRY[comp_name](corpus_fn)
        except Exception as e:
            html = f"<!-- {comp_name} error: {e} -->"
        rendered.append(comp_name)

        preferred = SLOT_MAP.get(comp_name, [])
        placed = False
        for ps in preferred:
            if ps in slots:
                slots[ps] += html + "\n"
                placed = True
                break
        if not placed:
            for fs in ["main_content","extra","section_1","section_2","section_3","section_0"]:
                if fs in slots:
                    slots[fs] += html + "\n"
                    placed = True
                    break
            if not placed:
                avail = [s for s in ly_cfg["slots"] if s not in ("navbar","footer")]
                if avail:
                    slots[random.choice(avail)] += html + "\n"

    title = corpus_fn(max(2, int(random.randint(4, 10) * ctx["text_s"])))
    head = HTML_HEAD.format(
        theme=theme, title=title,
        vw=res["width"], vh=res["height"],
        root_font=ROOT_FONT_SIZE,
        bp_sm=BREAKPOINTS["sm"], bp_md=BREAKPOINTS["md"],
        bp_lg=BREAKPOINTS["lg"], bp_xl=BREAKPOINTS["xl"],
    )
    body = ly_cfg["template"]
    for sn in ly_cfg["slots"]:
        body = body.replace("{" + sn + "}", slots.get(sn, ""))

    manifest = get_image_manifest()
    tail = HTML_TAIL.format(manifest=json.dumps(manifest, ensure_ascii=False))
    full_html = head + body + tail

    meta = {
        "page_type": pt_name, "page_type_label": pt_cfg["label"],
        "layout": ly_name, "layout_label": ly_cfg["label"],
        "theme": theme,
        "resolution": res["label"], "orientation": res["orientation"],
        "width": res["width"], "height": res["height"],
        "rem_w": round(rem_w, 1), "rem_h": round(rem_h, 1),
        "height_budget": round(h_budget, 1), "height_used": round(used_h, 1),
        "root_font_size": ROOT_FONT_SIZE, "scale_factor": SCALE_FACTOR,
        "min_text_px": int(ROOT_FONT_SIZE * 0.75),
        "component_count": len(rendered), "components": rendered,
        "image_count": len(manifest), "images": manifest,
        "render_ctx": ctx,
    }
    return full_html, meta