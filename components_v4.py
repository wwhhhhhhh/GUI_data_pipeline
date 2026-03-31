"""
原子组件库 v4.2 — 视口感知 + 高分辨率截图版
==============================================
每个组件通过线程局部 _ctx 读取:
  _ts()  文本缩放 0.4~1.0
  _mc()  最大列数 1~3
  _mi()  最大条目数 2~8

每个组件注册 HEIGHT_EST: 预估占用 rem
"""

import random
import threading

# ============================================================
# 线程局部上下文
# ============================================================
_local = threading.local()

def set_render_context(ctx):
    _local.ctx = ctx
    _local.images = []

def _ctx():
    return getattr(_local, 'ctx', {})

def _ts():
    return _ctx().get("text_s", 1.0)

def _mc():
    return _ctx().get("max_cols", 2)

def _mi():
    return _ctx().get("max_items", 6)

def get_image_manifest():
    return getattr(_local, 'images', [])

def reset_image_manifest():
    _local.images = []

def _t(lo, hi):
    """按 text_s 缩放后的随机文本长度, 最小返回 2"""
    s = _ts()
    return max(2, int(random.randint(lo, hi) * s))


# ============================================================
# DaisyUI 主题
# ============================================================
DAISY_THEMES = [
    "light", "dark", "cupcake", "bumblebee", "emerald", "corporate",
    "synthwave", "retro", "cyberpunk", "valentine", "halloween",
    "garden", "forest", "aqua", "lofi", "pastel", "fantasy",
    "wireframe", "black", "luxury", "dracula", "cmyk", "autumn",
    "business", "acid", "lemonade", "night", "coffee", "winter",
    "dim", "nord", "sunset",
]

# ============================================================
# 图片关键词
# ============================================================
_KW_AVATAR = ["商务人士头像", "年轻女性头像", "技术工程师头像", "客服人员头像",
              "项目经理头像", "设计师头像", "产品经理头像", "运营人员头像",
              "市场总监头像", "数据分析师头像", "前端开发者头像", "团队领导头像"]
_KW_CARD = ["产品功能展示图", "服务介绍配图", "项目案例图", "数据分析图表",
            "团队协作场景", "办公环境照片", "科技感背景图", "仪表盘截图"]
_KW_HERO = ["科技主题宣传图", "品牌视觉横幅", "产品发布海报", "数据可视化背景",
            "抽象几何背景", "渐变色彩背景", "创意工作坊场景", "技术峰会场景"]
_KW_PRODUCT = ["智能手机正面图", "笔记本电脑展示", "平板设备展示图", "智能手表特写"]
_KW_ARTICLE = ["新闻报道配图", "技术博客插图", "教程说明配图", "行业分析配图"]
_KW_FEED = ["社交动态配图", "生活分享照片", "美食摄影照片", "旅行风景照片"]

def _pk(pool):
    return random.choice(pool)


# ============================================================
# 红色占位图
# ============================================================

def _img_html(keyword, classes="w-full aspect-video"):
    imgs = getattr(_local, 'images', [])
    img_id = f"img_{len(imgs):04d}"
    imgs.append({"id": img_id, "keyword": keyword})
    _local.images = imgs
    return (f'<div id="{img_id}" class="img-placeholder {classes}" '
            f'style="background:#FF0000;" '
            f'data-img-id="{img_id}" data-img-keyword="{keyword}"></div>')

def _avatar_html(keyword, size="w-8 h-8"):
    imgs = getattr(_local, 'images', [])
    img_id = f"img_{len(imgs):04d}"
    imgs.append({"id": img_id, "keyword": keyword})
    _local.images = imgs
    return (f'<div id="{img_id}" class="img-placeholder {size} rounded-full shrink-0" '
            f'style="background:#FF0000;" '
            f'data-img-id="{img_id}" data-img-keyword="{keyword}"></div>')


# ============================================================
# 小工具
# ============================================================

_ic = 0
def _uid():
    global _ic; _ic += 1; return _ic

def _icon():
    icons = [
        '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" class="w-5 h-5 stroke-current"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"/></svg>',
        '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" class="w-5 h-5 stroke-current"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>',
        '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" class="w-5 h-5 stroke-current"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>',
        '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" class="w-5 h-5 stroke-current"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"/></svg>',
        '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" class="w-5 h-5 stroke-current"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/></svg>',
        '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" class="w-5 h-5 stroke-current"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>',
        '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" class="w-5 h-5 stroke-current"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"/></svg>',
        '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" class="w-5 h-5 stroke-current"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"/></svg>',
    ]
    return random.choice(icons)

def _rc():
    return random.choice(["primary","secondary","accent","info","success","warning","error"])

def _num():
    return random.choice([str(random.randint(1,999)), f"{random.randint(1,99)}.{random.randint(0,9)}K",
                          f"{random.randint(1,50)}.{random.randint(0,9)}M", f"{random.randint(10,99)}%",
                          f"¥{random.randint(10,9999):,}", f"${random.randint(10,9999):,}"])

def _trend():
    return f"{random.choice(['↗︎','↘︎','→'])} {random.randint(1,45)}%"

def _btn():
    c = _rc(); o = " btn-outline" if random.random()<0.3 else ""
    return f"btn btn-{c}{o} btn-sm"


# ============================================================
# 组件 (每个带 HEIGHT_EST 预估 rem)
# ============================================================

def render_navbar(cf, **kw):
    brand = cf(_t(3,8))
    items = [cf(_t(2,4)) for _ in range(min(random.randint(3,5), _mi()))]
    links = "".join(f'<li><a>{i}</a></li>' for i in items)
    v = random.choice(["std","search","mini"])
    if v == "search":
        return f'''<div class="navbar bg-base-100 shadow-md"><div class="flex-1"><a class="btn btn-ghost text-base font-bold text-primary">{brand}</a></div><div class="flex-none gap-2"><input type="text" placeholder="搜索..." class="input input-bordered input-sm w-24"/><div class="dropdown dropdown-end"><div tabindex="0" class="btn btn-ghost btn-circle btn-sm avatar"><div class="w-7 rounded-full overflow-hidden">{_avatar_html(_pk(_KW_AVATAR),"w-7 h-7")}</div></div><ul tabindex="0" class="menu menu-sm dropdown-content mt-2 z-[1] p-1 shadow bg-base-100 rounded-box w-40"><li><a>{cf(_t(2,4))}</a></li><li><a>退出</a></li></ul></div></div></div>'''
    elif v == "mini":
        return f'''<div class="navbar bg-base-100 shadow-sm"><div class="flex-1"><a class="text-base font-semibold">{brand}</a></div><div class="flex-none"><ul class="menu menu-horizontal menu-sm px-1">{links}</ul></div></div>'''
    else:
        return f'''<div class="navbar bg-base-100 shadow-md"><div class="flex-1"><a class="btn btn-ghost text-base font-bold">{brand}</a></div><div class="flex-none"><ul class="menu menu-horizontal menu-sm px-1">{links}</ul></div></div>'''

def render_breadcrumb(cf, **kw):
    n = random.randint(2,3)
    items = [cf(_t(2,4)) for _ in range(n)]
    crumbs = "".join(f"<li><a>{i}</a></li>" for i in items[:-1]) + f"<li class='font-semibold'>{items[-1]}</li>"
    return f'<div class="text-xs breadcrumbs"><ul>{crumbs}</ul></div>'

def render_stats(cf, **kw):
    n = min(random.randint(2,4), _mi())
    items = ""
    for _ in range(n):
        c = _rc()
        items += f'<div class="stat p-3"><div class="stat-title text-xs">{cf(_t(2,5))}</div><div class="stat-value text-lg text-{c}">{_num()}</div><div class="stat-desc text-xs">{_trend()}</div></div>'
    return f'<div class="stats shadow w-full bg-base-100 stats-vertical sm:stats-horizontal">{items}</div>'

def render_card_grid(cf, **kw):
    n = min(random.randint(2,6), _mi())
    cols = min(random.choice([1,2,2]), _mc())
    cards = ""
    for _ in range(n):
        title = cf(_t(3,8)); desc = cf(_t(8,25))
        badge = random.choice([None,None,"新","热门","Beta"])
        bh = f' <span class="badge badge-secondary badge-xs">{badge}</span>' if badge else ""
        style = random.choice(["img","compact","text"])
        if style == "img":
            cards += f'<div class="card bg-base-100 shadow-md"><figure class="px-3 pt-3">{_img_html(_pk(_KW_CARD),"w-full aspect-video rounded-lg")}</figure><div class="card-body p-3"><h2 class="card-title text-sm">{title}{bh}</h2><p class="text-xs opacity-70 line-clamp-2">{desc}</p><div class="card-actions justify-end mt-1"><button class="{_btn()}">{cf(_t(2,3))}</button></div></div></div>'
        elif style == "compact":
            cards += f'<div class="card bg-base-100 shadow-sm"><div class="card-body p-3"><div class="flex items-center gap-2"><div class="avatar placeholder"><div class="bg-{_rc()} text-neutral-content rounded w-8 h-8"><span class="text-sm">{title[0]}</span></div></div><div class="min-w-0"><h3 class="font-semibold text-xs truncate">{title}{bh}</h3><p class="text-xs opacity-60 truncate">{desc[:20]}</p></div></div></div></div>'
        else:
            cards += f'<div class="card bg-base-100 shadow-sm"><div class="card-body p-3"><h2 class="card-title text-sm">{title}{bh}</h2><p class="text-xs opacity-70 line-clamp-2">{desc}</p><div class="card-actions justify-end mt-1"><button class="{_btn()}">{cf(_t(2,3))}</button></div></div></div>'
    return f'<div class="grid grid-cols-{cols} gap-3">{cards}</div>'

def render_hero_banner(cf, **kw):
    title = cf(_t(4,10)); sub = cf(_t(10,30))
    v = random.choice(["center","gradient"])
    if v == "gradient":
        return f'<div class="hero min-h-[10rem] rounded-box" style="background:linear-gradient(135deg,oklch(var(--p)) 0%,oklch(var(--s)) 100%);"><div class="hero-content text-center text-primary-content p-4"><div><h1 class="text-xl font-bold">{title}</h1><p class="py-2 text-sm opacity-90">{sub}</p><button class="btn btn-accent btn-sm">{cf(_t(2,4))}</button></div></div></div>'
    return f'<div class="hero min-h-[10rem] bg-base-200 rounded-box"><div class="hero-content text-center p-4"><div><h1 class="text-xl font-bold">{title}</h1><p class="py-2 text-sm opacity-80">{sub}</p><div class="flex gap-2 justify-center"><button class="btn btn-primary btn-sm">{cf(_t(2,4))}</button><button class="btn btn-ghost btn-sm">{cf(_t(2,4))}</button></div></div></div></div>'

def render_table(cf, **kw):
    rows = min(random.randint(3,8), _mi())
    cols_n = min(random.randint(3,5), _mc() + 2)
    headers = [cf(_t(2,4)) for _ in range(cols_n)]
    hh = "".join(f"<th class='text-xs'>{h}</th>" for h in headers)
    rh = ""
    for _ in range(rows):
        cells = ""
        for c in range(cols_n):
            if c == 0 and random.random() < 0.3:
                cells += f'<td><div class="flex items-center gap-1">{_avatar_html(_pk(_KW_AVATAR),"w-6 h-6")}<span class="text-xs">{cf(_t(2,5))}</span></div></td>'
            elif random.random() < 0.2:
                cells += f'<td><span class="badge badge-{_rc()} badge-xs">{cf(_t(2,3))}</span></td>'
            else:
                cells += f"<td class='text-xs'>{cf(_t(2,6))}</td>"
        rh += f"<tr class='hover'>{cells}</tr>"
    return f'<div class="card bg-base-100 shadow-md"><div class="card-body p-0"><div class="overflow-x-auto"><table class="table table-sm"><thead><tr>{hh}</tr></thead><tbody>{rh}</tbody></table></div></div></div>'

def render_sidebar_menu(cf, **kw):
    sections = random.randint(2,3)
    html = ""
    for s in range(sections):
        title = cf(_t(2,4))
        items = ""
        for i in range(random.randint(2,4)):
            active = "active" if s==0 and i==0 else ""
            items += f'<li><a class="{active} text-xs py-1">{_icon()} {cf(_t(2,4))}</a></li>'
        html += f'<li class="menu-title text-xs opacity-50 mt-2">{title}</li>{items}'
    return f'<ul class="menu menu-sm w-full">{html}</ul>'

def render_form_group(cf, **kw):
    n = min(random.randint(3,5), _mi())
    fields = ""
    for _ in range(n):
        ft = random.choice(["text","select","textarea"])
        label = cf(_t(2,4))
        if ft == "select":
            opts = "".join(f'<option>{cf(_t(2,4))}</option>' for _ in range(3))
            fields += f'<div class="form-control w-full"><label class="label py-1"><span class="label-text text-xs">{label}</span></label><select class="select select-bordered select-sm w-full"><option disabled selected>请选择</option>{opts}</select></div>'
        elif ft == "textarea":
            fields += f'<div class="form-control w-full"><label class="label py-1"><span class="label-text text-xs">{label}</span></label><textarea class="textarea textarea-bordered textarea-sm h-16" placeholder="{cf(_t(3,8))}"></textarea></div>'
        else:
            fields += f'<div class="form-control w-full"><label class="label py-1"><span class="label-text text-xs">{label}</span></label><input type="text" placeholder="{cf(_t(3,8))}" class="input input-bordered input-sm w-full"/></div>'
    return f'<div class="card bg-base-100 shadow-md"><div class="card-body p-3"><h2 class="card-title text-sm mb-2">{cf(_t(3,6))}</h2><div class="space-y-1">{fields}</div><div class="card-actions justify-end mt-3"><button class="btn btn-ghost btn-sm">{cf(_t(2,3))}</button><button class="btn btn-primary btn-sm">{cf(_t(2,3))}</button></div></div></div>'

def render_toggle_group(cf, **kw):
    n = min(random.randint(3,5), _mi())
    items = ""
    for _ in range(n):
        checked = "checked" if random.random()<0.4 else ""
        items += f'<div class="flex items-center justify-between py-1"><div><p class="text-xs font-medium">{cf(_t(2,6))}</p><p class="text-xs opacity-50">{cf(_t(4,10))}</p></div><input type="checkbox" class="toggle toggle-{_rc()} toggle-sm" {checked}/></div>'
    return f'<div class="card bg-base-100 shadow-md"><div class="card-body p-3"><h2 class="card-title text-xs mb-1">{cf(_t(2,5))}</h2><div class="divide-y divide-base-200">{items}</div></div></div>'

def render_tabs(cf, **kw):
    n = random.randint(3,5)
    style = random.choice(["tabs-boxed","tabs-bordered","tabs-lifted"])
    tabs = ""
    for i in range(n):
        active = "tab-active" if i==0 else ""
        tabs += f'<a class="tab tab-sm {active}">{cf(_t(2,4))}</a>'
    return f'<div class="tabs {style} w-full">{tabs}</div>'

def render_timeline(cf, **kw):
    n = min(random.randint(2,4), _mi())
    items = ""
    for i in range(n):
        c = _rc(); side = "timeline-start" if i%2==0 else "timeline-end"
        sl = "" if i==0 else "<hr/>"; el = "" if i==n-1 else "<hr/>"
        items += f'<li>{sl}<div class="{side} timeline-box shadow-sm p-2"><p class="font-semibold text-xs">{cf(_t(3,6))}</p><p class="text-xs opacity-60">{cf(_t(5,12))}</p></div><div class="timeline-middle"><div class="w-3 h-3 rounded-full bg-{c}"></div></div>{el}</li>'
    return f'<ul class="timeline timeline-vertical">{items}</ul>'

def render_progress_group(cf, **kw):
    n = min(random.randint(3,5), _mi())
    items = ""
    for _ in range(n):
        c = _rc(); val = random.randint(10,98)
        items += f'<div><div class="flex justify-between mb-0.5"><span class="text-xs font-medium">{cf(_t(2,5))}</span><span class="text-xs opacity-60">{val}%</span></div><progress class="progress progress-{c} w-full" value="{val}" max="100"></progress></div>'
    return f'<div class="card bg-base-100 shadow-md"><div class="card-body p-3"><h2 class="card-title text-xs mb-2">{cf(_t(3,6))}</h2><div class="space-y-2">{items}</div></div></div>'

def render_alert(cf, **kw):
    at = random.choice(["alert-info","alert-success","alert-warning","alert-error"])
    return f'<div class="alert {at} shadow-sm py-2 px-3">{_icon()}<div><h3 class="font-bold text-xs">{cf(_t(2,5))}</h3><div class="text-xs opacity-80">{cf(_t(5,15))}</div></div></div>'

def render_chat_bubbles(cf, **kw):
    n = min(random.randint(3,7), _mi())
    html = ""
    for i in range(n):
        side = "chat-start" if i%2==0 else "chat-end"
        c = random.choice(["chat-bubble-primary","chat-bubble-secondary",""])
        html += f'<div class="chat {side}"><div class="chat-image avatar"><div class="w-8 rounded-full overflow-hidden">{_avatar_html(_pk(_KW_AVATAR),"w-8 h-8")}</div></div><div class="chat-header text-xs opacity-50">{cf(_t(2,3))}</div><div class="chat-bubble {c} text-xs">{cf(_t(5,20))}</div></div>'
    return html

def render_profile_header(cf, **kw):
    return f'''<div class="card bg-base-100 shadow-md"><div class="h-16 bg-gradient-to-r from-primary to-secondary rounded-t-2xl relative overflow-hidden">{_img_html("个人主页封面图","absolute inset-0 w-full h-full opacity-40")}</div><div class="card-body p-3 -mt-8 items-center text-center"><div class="avatar"><div class="w-14 rounded-full ring ring-base-100 ring-offset-base-100 ring-offset-1 overflow-hidden">{_avatar_html(_pk(_KW_AVATAR),"w-14 h-14")}</div></div><h2 class="font-bold text-sm mt-1">{cf(_t(2,6))}</h2><p class="text-xs opacity-60">{cf(_t(5,12))}</p><div class="flex gap-4 mt-1"><div class="text-center"><p class="font-bold text-sm">{_num()}</p><p class="text-xs opacity-50">{cf(_t(2,3))}</p></div><div class="text-center"><p class="font-bold text-sm">{_num()}</p><p class="text-xs opacity-50">{cf(_t(2,3))}</p></div></div><div class="card-actions mt-2"><button class="btn btn-primary btn-sm">{cf(_t(2,3))}</button><button class="btn btn-ghost btn-sm">{cf(_t(2,3))}</button></div></div></div>'''

def render_chart_placeholder(cf, **kw):
    title = cf(_t(3,8))
    ct = random.choice(["bar","area"])
    if ct == "bar":
        n = min(random.randint(4,8), _mi())
        bars = ""
        for _ in range(n):
            h = random.randint(15,95); c = _rc()
            bars += f'<div class="flex-1 flex flex-col items-center justify-end"><div class="w-full bg-{c} rounded-t-sm opacity-80" style="height:{h}%"></div><span class="text-xs opacity-40 mt-0.5">{cf(1)}</span></div>'
        return f'<div class="card bg-base-100 shadow-md"><div class="card-body p-3"><h2 class="card-title text-xs">{title}</h2><div class="h-32 flex items-end gap-0.5 pt-2">{bars}</div></div></div>'
    bg = random.choice(["from-primary/20 to-primary/5","from-secondary/20 to-secondary/5","from-accent/20 to-accent/5"])
    return f'<div class="card bg-base-100 shadow-md"><div class="card-body p-3"><h2 class="card-title text-xs">{title}</h2><div class="h-32 bg-gradient-to-t {bg} rounded-lg flex items-center justify-center"><span class="text-xs opacity-40">{ct.upper()} · {cf(_t(3,6))}</span></div></div></div>'

def render_pagination(cf, **kw):
    total = random.randint(3,10); cur = random.randint(1,total)
    bts = ""
    for p in range(1,min(total+1,6)):
        a = "btn-active" if p==cur else ""
        bts += f'<button class="join-item btn btn-xs {a}">{p}</button>'
    if total > 5: bts += f'<button class="join-item btn btn-xs btn-disabled">…</button><button class="join-item btn btn-xs">{total}</button>'
    return f'<div class="flex justify-center"><div class="join">{bts}</div></div>'

def render_badge_group(cf, **kw):
    n = min(random.randint(3,6), _mi())
    badges = " ".join(f'<span class="badge badge-{_rc()} badge-xs">{cf(_t(2,4))}</span>' for _ in range(n))
    return f'<div class="flex flex-wrap gap-1">{badges}</div>'

def render_avatar_group(cf, **kw):
    n = min(random.randint(3,5), _mi())
    avs = "".join(f'<div class="avatar"><div class="w-8 overflow-hidden">{_avatar_html(_pk(_KW_AVATAR),"w-8 h-8")}</div></div>' for _ in range(n))
    avs += f'<div class="avatar placeholder"><div class="w-8 bg-neutral text-neutral-content"><span class="text-xs">+{random.randint(2,20)}</span></div></div>'
    return f'<div class="avatar-group -space-x-3">{avs}</div>'

def render_footer(cf, **kw):
    return f'<footer class="footer footer-center p-4 bg-base-100 text-base-content mt-auto"><p class="text-xs opacity-60">© 2024 {cf(_t(3,8))} — {cf(_t(4,10))}</p></footer>'

def render_divider(cf, **kw):
    txt = cf(_t(2,4)) if random.random()<0.5 else ""
    return f'<div class="divider text-xs my-1">{txt}</div>'

def render_auth_form(cf, **kw):
    title = cf(_t(3,5)); sub = cf(_t(5,12))
    return f'<div class="card bg-base-100 shadow-xl w-full"><div class="card-body p-4"><h2 class="text-lg font-bold text-center">{title}</h2><p class="text-center text-xs opacity-60 mb-3">{sub}</p><div class="form-control"><label class="label py-0.5"><span class="label-text text-xs">邮箱</span></label><input type="email" placeholder="name@example.com" class="input input-bordered input-sm"/></div><div class="form-control"><label class="label py-0.5"><span class="label-text text-xs">密码</span></label><input type="password" placeholder="••••••••" class="input input-bordered input-sm"/></div><div class="form-control mt-3"><button class="btn btn-primary btn-sm">{cf(_t(2,3))}</button></div><div class="divider text-xs my-1">或</div><button class="btn btn-outline btn-sm w-full">{cf(_t(2,5))}</button></div></div>'

def render_error_display(cf, **kw):
    code = random.choice(["404","500","403","503"])
    return f'<div class="text-center py-8"><h1 class="text-6xl font-bold text-primary opacity-20">{code}</h1><h2 class="text-lg font-bold mt-2">{cf(_t(3,6))}</h2><p class="opacity-60 mt-1 text-sm">{cf(_t(8,20))}</p><div class="mt-4 flex gap-2 justify-center"><button class="btn btn-primary btn-sm">{cf(_t(2,4))}</button><button class="btn btn-ghost btn-sm">{cf(_t(2,4))}</button></div></div>'

def render_article_body(cf, **kw):
    sections = min(random.randint(1,3), max(1, _mi()-1))
    html = f'<h1 class="text-lg font-bold">{cf(_t(5,12))}</h1>'
    html += f'<div class="flex items-center gap-2 mt-2">{_avatar_html(_pk(_KW_AVATAR),"w-6 h-6")}<span class="text-xs">{cf(_t(2,4))}</span><span class="text-xs opacity-50">{cf(_t(3,6))}</span></div><div class="divider my-1"></div>'
    for s in range(sections):
        html += f'<h2 class="text-sm font-semibold mt-3 mb-1">{cf(_t(3,8))}</h2><p class="text-xs leading-relaxed opacity-80 mb-2">{cf(_t(20,50))}</p>'
        if random.random()<0.3:
            html += f'<figure class="my-2">{_img_html(_pk(_KW_ARTICLE),"rounded-lg w-full aspect-video")}</figure>'
    return f'<div class="card bg-base-100 shadow-md"><div class="card-body p-3">{html}</div></div>'

def render_filter_bar(cf, **kw):
    items = ""
    for _ in range(min(random.randint(2,3), _mc()+1)):
        opts = "".join(f'<option>{cf(_t(2,3))}</option>' for _ in range(3))
        items += f'<select class="select select-bordered select-xs"><option disabled selected>{cf(_t(2,3))}</option>{opts}</select>'
    items += f'<button class="btn btn-primary btn-xs">{cf(_t(2,3))}</button>'
    return f'<div class="flex flex-wrap items-center gap-2 p-2 bg-base-100 rounded-box shadow-sm">{items}</div>'

def render_pricing_table(cf, **kw):
    plans = min(random.randint(2,3), _mc())
    html = ""
    for i in range(plans):
        pop = i==1; border = "border-2 border-primary" if pop else ""
        badge = '<div class="badge badge-primary badge-xs absolute -top-2 right-2">推荐</div>' if pop else ""
        feats = "".join(f'<li class="flex items-center gap-1 text-xs"><svg class="w-3 h-3 text-success shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>{cf(_t(3,8))}</li>' for _ in range(random.randint(2,4)))
        html += f'<div class="card bg-base-100 shadow-md {border} relative"><div class="card-body p-3 items-center text-center">{badge}<h3 class="font-bold text-sm">{cf(_t(2,4))}</h3><p class="text-xl font-bold mt-1">¥{random.choice([0,29,99,199])}<span class="text-xs font-normal opacity-50">/月</span></p><ul class="space-y-1 mt-2 text-left w-full">{feats}</ul><button class="btn {"btn-primary" if pop else "btn-outline"} btn-sm w-full mt-2">{cf(_t(2,3))}</button></div></div>'
    return f'<div class="grid grid-cols-{plans} gap-3">{html}</div>'

def render_cta_section(cf, **kw):
    return f'<div class="bg-primary text-primary-content rounded-box p-4 text-center"><h2 class="text-base font-bold">{cf(_t(4,10))}</h2><p class="mt-1 text-xs opacity-80">{cf(_t(8,20))}</p><div class="mt-3 flex gap-2 justify-center"><button class="btn btn-accent btn-sm">{cf(_t(2,4))}</button><button class="btn btn-ghost btn-sm border border-current">{cf(_t(2,4))}</button></div></div>'

def render_feature_grid(cf, **kw):
    n = min(random.randint(2,4), _mi()); cols = min(n, _mc())
    items = ""
    for _ in range(n):
        c = _rc()
        items += f'<div class="card bg-base-100 shadow-sm p-3 text-center"><div class="w-8 h-8 rounded-lg bg-{c}/10 text-{c} flex items-center justify-center mx-auto mb-2">{_icon()}</div><h3 class="font-semibold text-xs">{cf(_t(2,5))}</h3><p class="text-xs opacity-60 mt-1">{cf(_t(5,15))}</p></div>'
    return f'<div class="grid grid-cols-{cols} gap-3">{items}</div>'

def _stars(n=None):
    if n is None: n = random.randint(4,5)
    star = '<span class="text-warning text-sm">★</span>'
    return star * n

def render_testimonial_cards(cf, **kw):
    n = min(random.randint(1,2), _mc())
    items = ""
    for _ in range(n):
        stars = _stars()
        avatar = _avatar_html(_pk(_KW_AVATAR),"w-7 h-7")
        items += f'<div class="card bg-base-100 shadow-sm"><div class="card-body p-3"><div class="flex gap-0.5 mb-1">{stars}</div><p class="text-xs italic opacity-80">{cf(_t(10,25))}</p><div class="flex items-center gap-2 mt-2">{avatar}<div><p class="font-semibold text-xs">{cf(_t(2,4))}</p><p class="text-xs opacity-50">{cf(_t(2,5))}</p></div></div></div></div>'
    return f'<div class="grid grid-cols-{n} gap-3">{items}</div>'

def render_faq_accordion(cf, **kw):
    n = min(random.randint(2,4), _mi())
    name = f"faq_{_uid()}"
    items = ""
    for i in range(n):
        chk = "checked" if i==0 else ""
        items += f'<div class="collapse collapse-arrow bg-base-100 shadow-sm"><input type="radio" name="{name}" {chk}/><div class="collapse-title text-xs font-medium py-2 min-h-0">{cf(_t(4,10))}</div><div class="collapse-content"><p class="text-xs opacity-70">{cf(_t(10,25))}</p></div></div>'
    return f'<div class="space-y-1">{items}</div>'

def render_comment_list(cf, **kw):
    n = min(random.randint(2,4), _mi())
    html = ""
    for _ in range(n):
        html += f'<div class="flex gap-2 p-2">{_avatar_html(_pk(_KW_AVATAR),"w-6 h-6")}<div class="flex-1 min-w-0"><div class="flex items-center gap-1"><span class="font-semibold text-xs">{cf(_t(2,4))}</span><span class="text-xs opacity-40">{random.randint(1,24)}h</span></div><p class="text-xs opacity-80 mt-0.5">{cf(_t(5,20))}</p></div></div>'
    return f'<div class="card bg-base-100 shadow-md"><div class="card-body p-1 divide-y divide-base-200">{html}</div></div>'

def render_kanban_columns(cf, **kw):
    cols = min(random.randint(2,4), _mc())
    statuses = ["待处理","进行中","审核中","已完成"]
    colors = ["warning","info","secondary","success"]
    html = ""
    for c in range(cols):
        cards = ""
        n = min(random.randint(1,3), _mi())
        for _ in range(n):
            cards += f'<div class="card bg-base-100 shadow-sm p-2"><p class="font-medium text-xs">{cf(_t(3,8))}</p><p class="text-xs opacity-60 mt-0.5 line-clamp-1">{cf(_t(5,12))}</p><div class="flex items-center justify-between mt-1"><span class="badge badge-{random.choice(["error","warning","info"])} badge-xs">{cf(_t(1,2))}</span>{_avatar_html(_pk(_KW_AVATAR),"w-4 h-4")}</div></div>'
        html += f'<div class="flex-1 min-w-0"><div class="flex items-center gap-1 mb-2"><div class="w-2 h-2 rounded-full bg-{colors[c%len(colors)]}"></div><h3 class="font-semibold text-xs">{statuses[c%len(statuses)]}</h3><span class="badge badge-ghost badge-xs">{n}</span></div><div class="space-y-1 bg-base-200/50 rounded-box p-1 min-h-[3rem]">{cards}</div></div>'
    return f'<div class="flex gap-2 overflow-x-auto">{html}</div>'

def render_email_list(cf, **kw):
    n = min(random.randint(4,8), _mi())
    rows = ""
    for _ in range(n):
        unread = "font-bold" if random.random()<0.3 else ""
        dot = '<div class="w-1.5 h-1.5 rounded-full bg-primary shrink-0"></div>' if unread else '<div class="w-1.5 h-1.5 shrink-0"></div>'
        rows += f'<div class="flex items-center gap-2 p-2 hover:bg-base-200 cursor-pointer border-b border-base-200 {unread}">{dot}{_avatar_html(_pk(_KW_AVATAR),"w-6 h-6")}<div class="flex-1 min-w-0"><div class="flex justify-between"><span class="text-xs truncate">{cf(_t(2,5))}</span><span class="text-xs opacity-40">{random.randint(1,59)}m</span></div><p class="text-xs truncate">{cf(_t(4,10))}</p></div></div>'
    return f'<div class="card bg-base-100 shadow-md overflow-hidden">{rows}</div>'

def render_feed_cards(cf, **kw):
    n = min(random.randint(2,4), _mi())
    html = ""
    for _ in range(n):
        has_img = random.random()<0.5
        img_h = f'<div class="px-3">{_img_html(_pk(_KW_FEED),"rounded-lg w-full aspect-video")}</div>' if has_img else ""
        html += f'<div class="card bg-base-100 shadow-sm"><div class="card-body p-3 pb-1"><div class="flex items-center gap-2">{_avatar_html(_pk(_KW_AVATAR),"w-7 h-7")}<div><p class="font-semibold text-xs">{cf(_t(2,4))}</p><p class="text-xs opacity-50">{random.randint(1,24)}h</p></div></div><p class="text-xs mt-1">{cf(_t(8,25))}</p></div>{img_h}<div class="card-body p-3 pt-1"><div class="flex gap-4 text-xs opacity-60"><span>❤ {random.randint(0,500)}</span><span>💬 {random.randint(0,50)}</span></div></div></div>'
    return f'<div class="space-y-3">{html}</div>'

def render_calendar_grid(cf, **kw):
    days_h = "".join(f'<th class="text-center text-xs p-1">{d}</th>' for d in ["一","二","三","四","五","六","日"])
    rows = ""; day = 1
    for w in range(5):
        cells = ""
        for d in range(7):
            if (w==0 and d<random.randint(0,2)) or day>31:
                cells += '<td class="p-0.5"></td>'
            else:
                today = "bg-primary text-primary-content rounded" if day==random.randint(10,20) else ""
                cells += f'<td class="p-0.5 text-center"><div class="text-xs {today} p-0.5">{day}</div></td>'
                day += 1
        rows += f'<tr>{cells}</tr>'
    return f'<div class="card bg-base-100 shadow-md"><div class="card-body p-3"><div class="flex justify-between items-center mb-2"><button class="btn btn-ghost btn-xs">◀</button><h2 class="font-bold text-xs">{cf(_t(3,6))}</h2><button class="btn btn-ghost btn-xs">▶</button></div><table class="w-full"><thead><tr>{days_h}</tr></thead><tbody>{rows}</tbody></table></div></div>'

def render_file_list(cf, **kw):
    n = min(random.randint(4,8), _mi())
    exts = [".pdf",".docx",".xlsx",".png",".zip",".csv",".py"]
    rows = ""
    for _ in range(n):
        ext = random.choice(exts); size = f"{random.randint(1,999)} {'KB' if random.random()<0.6 else 'MB'}"
        rows += f'<div class="flex items-center gap-2 p-2 hover:bg-base-200 border-b border-base-200"><div class="w-6 h-6 rounded bg-{_rc()}/10 flex items-center justify-center text-xs font-mono">{ext}</div><div class="flex-1 min-w-0"><p class="text-xs font-medium truncate">{cf(_t(3,8))}{ext}</p><p class="text-xs opacity-50">{size}</p></div></div>'
    return f'<div class="card bg-base-100 shadow-md overflow-hidden">{rows}</div>'

def render_input_area(cf, **kw):
    return f'<div class="flex gap-2 items-end"><button class="btn btn-ghost btn-circle btn-sm">{_icon()}</button><input type="text" placeholder="{cf(_t(3,8))}" class="input input-bordered input-sm flex-1"/><button class="btn btn-primary btn-circle btn-sm">{_icon()}</button></div>'

def render_rating_display(cf, **kw):
    score = round(random.uniform(3.0,5.0),1)
    stars = "".join([f'<input type="radio" name="r_{_uid()}" class="mask mask-star-2 bg-warning"/>' for _ in range(5)])
    return f'<div class="flex items-center gap-3"><p class="text-2xl font-bold">{score}</p><div><div class="rating rating-sm">{stars}</div><p class="text-xs opacity-50">{random.randint(50,5000)} {cf(_t(2,3))}</p></div></div>'

def render_code_block(cf, **kw):
    lang = random.choice(["python","javascript","bash"])
    lines = [cf(_t(6,20)) for _ in range(min(random.randint(3,6), _mi()))]
    code = "\n".join(f"  {l}" for l in lines)
    return f'<div class="card bg-neutral text-neutral-content shadow-md"><div class="card-body p-3"><div class="flex justify-between items-center mb-1"><span class="badge badge-ghost badge-xs">{lang}</span><button class="btn btn-ghost btn-xs">复制</button></div><pre class="text-xs overflow-x-auto leading-relaxed"><code>{code}</code></pre></div></div>'

def render_toolbar(cf, **kw):
    bts = "".join(f'<button class="btn btn-ghost btn-xs gap-0.5">{_icon()}<span class="text-xs">{cf(_t(2,3))}</span></button>' for _ in range(min(random.randint(3,5), _mi())))
    return f'<div class="flex items-center gap-0.5 p-1.5 bg-base-100 rounded-box shadow-sm border border-base-200">{bts}<div class="flex-1"></div><input type="text" placeholder="搜索..." class="input input-bordered input-xs w-20"/></div>'

def render_sidebar_toc(cf, **kw):
    n = min(random.randint(3,6), _mi())
    items = "".join(f'<li><a class="text-xs py-0.5 hover:text-primary {"font-semibold text-primary" if i==0 else ""}">{cf(_t(2,6))}</a></li>' for i in range(n))
    return f'<div class="card bg-base-100 shadow-md"><div class="card-body p-3"><h3 class="font-bold text-xs mb-2">{cf(_t(2,4))}</h3><ul class="space-y-0.5 border-l-2 border-base-200 pl-2">{items}</ul></div></div>'

def render_product_detail(cf, **kw):
    price = random.randint(29,9999)
    return f'<div class="card bg-base-100 shadow-md"><div class="card-body p-3"><div class="space-y-3">{_img_html(_pk(_KW_PRODUCT),"rounded-lg w-full aspect-video")}<h1 class="text-base font-bold">{cf(_t(4,10))}</h1><p class="text-lg font-bold text-primary">¥{price}</p><p class="text-xs opacity-70">{cf(_t(10,30))}</p><div class="flex gap-2 mt-2"><button class="btn btn-primary btn-sm flex-1">{cf(_t(2,3))}</button><button class="btn btn-outline btn-sm">{_icon()}</button></div></div></div></div>'

def render_event_list(cf, **kw):
    n = min(random.randint(2,4), _mi())
    html = ""
    for _ in range(n):
        c = _rc()
        html += f'<div class="flex gap-2 p-1"><div class="w-0.5 rounded-full bg-{c}"></div><div><p class="text-xs font-medium">{cf(_t(3,6))}</p><p class="text-xs opacity-50">{random.randint(0,23):02d}:{random.randint(0,59):02d}</p></div></div>'
    return f'<div class="card bg-base-100 shadow-sm"><div class="card-body p-3"><h3 class="font-bold text-xs mb-1">{cf(_t(2,4))}</h3>{html}</div></div>'

def render_date_range_picker(cf, **kw):
    return f'<div class="flex flex-wrap items-center gap-2 p-2 bg-base-100 rounded-box shadow-sm"><span class="text-xs font-medium">{cf(_t(2,3))}</span><input type="date" class="input input-bordered input-xs"/><span class="text-xs opacity-50">至</span><input type="date" class="input input-bordered input-xs"/><button class="btn btn-primary btn-xs">{cf(_t(2,3))}</button></div>'

def render_hero_side(cf, **kw):
    return f'<div class="space-y-3 max-w-[16rem]"><h1 class="text-lg font-bold">{cf(_t(4,10))}</h1><p class="text-xs opacity-80">{cf(_t(10,25))}</p><div class="flex gap-3"><div class="text-center"><p class="text-base font-bold">{_num()}</p><p class="text-xs opacity-60">{cf(_t(2,3))}</p></div><div class="text-center"><p class="text-base font-bold">{_num()}</p><p class="text-xs opacity-60">{cf(_t(2,3))}</p></div></div></div>'

def render_social_login_buttons(cf, **kw):
    return f'<div class="space-y-1"><button class="btn btn-outline btn-sm w-full gap-1">{_icon()} {cf(_t(2,5))}</button><button class="btn btn-outline btn-sm w-full gap-1">{_icon()} {cf(_t(2,5))}</button></div>'

def render_story_bar(cf, **kw):
    n = min(random.randint(3,6), _mi())
    items = "".join(f'<div class="flex flex-col items-center gap-0.5"><div class="w-10 h-10 rounded-full ring ring-primary ring-offset-1 overflow-hidden">{_avatar_html(_pk(_KW_AVATAR),"w-10 h-10")}</div><span class="text-xs truncate w-10 text-center">{cf(_t(2,3))}</span></div>' for _ in range(n))
    return f'<div class="flex gap-3 overflow-x-auto py-1">{items}</div>'

def render_suggestion_list(cf, **kw):
    n = min(random.randint(2,4), _mi())
    items = "".join(f'<div class="flex items-center gap-2 p-1.5 hover:bg-base-200 rounded">{_avatar_html(_pk(_KW_AVATAR),"w-7 h-7")}<div class="flex-1 min-w-0"><p class="text-xs font-medium">{cf(_t(2,5))}</p><p class="text-xs opacity-50">{cf(_t(3,6))}</p></div><button class="btn btn-primary btn-xs btn-outline">{cf(_t(2,2))}</button></div>' for _ in range(n))
    return f'<div class="card bg-base-100 shadow-sm"><div class="card-body p-3"><h3 class="font-bold text-xs mb-1">{cf(_t(2,4))}</h3>{items}</div></div>'

def _rating_stars():
    star = '<input type="radio" class="mask mask-star-2 bg-warning"/>'
    return star * 5

def render_review_list(cf, **kw):
    n = min(random.randint(2,3), _mi())
    html = ""
    for _ in range(n):
        stars = _rating_stars()
        avatar = _avatar_html(_pk(_KW_AVATAR),"w-6 h-6")
        html += f'<div class="p-2 border-b border-base-200"><div class="flex items-center gap-2 mb-1">{avatar}<span class="text-xs font-medium">{cf(_t(2,3))}</span><div class="rating rating-xs">{stars}</div></div><p class="text-xs opacity-80">{cf(_t(8,25))}</p></div>'
    return f'<div class="card bg-base-100 shadow-md">{html}</div>'

def render_tag_group(cf, **kw):
    return render_badge_group(cf, **kw)

def render_steps(cf, **kw):
    n = min(random.randint(3,5), _mi()); cur = random.randint(1,n)
    items = "".join(f'<li class="step {"step-primary" if i<cur else ""} text-xs">{cf(_t(2,3))}</li>' for i in range(n))
    return f'<ul class="steps steps-horizontal w-full overflow-hidden">{items}</ul>'

def render_modal_static(cf, **kw):
    return f'<div class="card bg-base-100 shadow-xl border border-base-300 max-w-[16rem] mx-auto"><div class="card-body p-3"><h3 class="font-bold text-sm">{cf(_t(3,6))}</h3><p class="text-xs opacity-70 py-1">{cf(_t(8,20))}</p><div class="card-actions justify-end"><button class="btn btn-ghost btn-xs">{cf(_t(2,3))}</button><button class="btn btn-primary btn-xs">{cf(_t(2,3))}</button></div></div></div>'

def render_related_cards(cf, **kw):
    n = min(random.randint(2,3), _mc())
    items = "".join(f'<div class="card card-side bg-base-100 shadow-sm"><figure class="w-16 shrink-0">{_img_html(_pk(_KW_ARTICLE),"h-full w-full")}</figure><div class="card-body p-2"><h3 class="text-xs font-medium line-clamp-2">{cf(_t(4,10))}</h3></div></div>' for _ in range(n))
    return f'<div class="grid grid-cols-1 gap-2">{items}</div>'

def render_email_detail(cf, **kw):
    return f'<div class="card bg-base-100 shadow-md"><div class="card-body p-3"><h2 class="text-sm font-bold">{cf(_t(4,10))}</h2><div class="flex items-center gap-2 mt-1">{_avatar_html(_pk(_KW_AVATAR),"w-6 h-6")}<span class="text-xs">{cf(_t(2,5))}</span><span class="text-xs opacity-40">{random.randint(1,28)}天前</span></div><div class="divider my-1"></div><p class="text-xs opacity-80">{cf(_t(20,50))}</p><div class="flex gap-2 mt-2"><button class="btn btn-primary btn-xs">{cf(_t(2,3))}</button><button class="btn btn-ghost btn-xs">{cf(_t(2,3))}</button></div></div></div>'

def render_sidebar_contacts(cf, **kw):
    n = min(random.randint(4,8), _mi())
    items = ""
    for i in range(n):
        active = "bg-primary/10" if i==0 else ""
        items += f'<div class="flex items-center gap-2 p-2 hover:bg-base-200 rounded {active}">{_avatar_html(_pk(_KW_AVATAR),"w-7 h-7")}<div class="flex-1 min-w-0"><span class="text-xs font-medium truncate block">{cf(_t(2,4))}</span><p class="text-xs opacity-50 truncate">{cf(_t(3,8))}</p></div></div>'
    return f'<div class="space-y-0.5">{items}</div>'

def render_sidebar_trending(cf, **kw):
    n = min(random.randint(3,5), _mi())
    items = "".join(f'<div class="flex items-start gap-1 py-1"><span class="text-xs opacity-40">{i+1}</span><div><p class="text-xs font-medium">{cf(_t(3,8))}</p><p class="text-xs opacity-50">{_num()}</p></div></div>' for i in range(n))
    return f'<div class="card bg-base-100 shadow-sm"><div class="card-body p-3"><h3 class="font-bold text-xs mb-1">{cf(_t(2,4))}</h3><div class="divide-y divide-base-200">{items}</div></div></div>'


# ============================================================
# 组件注册表 + 高度预估(rem)
# ============================================================

COMPONENT_REGISTRY = {}
HEIGHT_EST = {}

def _reg(name, fn, h):
    COMPONENT_REGISTRY[name] = fn
    HEIGHT_EST[name] = h

_reg("navbar", render_navbar, 3.5)
_reg("breadcrumb", render_breadcrumb, 2)
_reg("tabs", render_tabs, 2.5)
_reg("sidebar_menu", render_sidebar_menu, 0)
_reg("pagination", render_pagination, 2.5)
_reg("steps", render_steps, 3)
_reg("card_grid", render_card_grid, 16)
_reg("stats", render_stats, 6)
_reg("hero_banner", render_hero_banner, 12)
_reg("profile_header", render_profile_header, 16)
_reg("badge_group", render_badge_group, 3)
_reg("avatar_group", render_avatar_group, 2.5)
_reg("feature_grid", render_feature_grid, 14)
_reg("testimonial_cards", render_testimonial_cards, 10)
_reg("table", render_table, 14)
_reg("chart_placeholder", render_chart_placeholder, 12)
_reg("progress_group", render_progress_group, 10)
_reg("timeline", render_timeline, 12)
_reg("file_list", render_file_list, 12)
_reg("calendar_grid", render_calendar_grid, 16)
_reg("kanban_columns", render_kanban_columns, 14)
_reg("email_list", render_email_list, 12)
_reg("event_list", render_event_list, 7)
_reg("form_group", render_form_group, 16)
_reg("toggle_group", render_toggle_group, 10)
_reg("filter_bar", render_filter_bar, 3)
_reg("auth_form", render_auth_form, 18)
_reg("input_area", render_input_area, 3)
_reg("date_range_picker", render_date_range_picker, 3)
_reg("alert", render_alert, 4)
_reg("modal_static", render_modal_static, 8)
_reg("error_display", render_error_display, 14)
_reg("rating_display", render_rating_display, 3)
_reg("article_body", render_article_body, 18)
_reg("comment_list", render_comment_list, 10)
_reg("review_list", render_review_list, 8)
_reg("chat_bubbles", render_chat_bubbles, 16)
_reg("code_block", render_code_block, 10)
_reg("faq_accordion", render_faq_accordion, 8)
_reg("feed_cards", render_feed_cards, 16)
_reg("footer", render_footer, 3)
_reg("divider", render_divider, 1.5)
_reg("cta_section", render_cta_section, 8)
_reg("pricing_table", render_pricing_table, 16)
_reg("toolbar", render_toolbar, 3)
_reg("sidebar_toc", render_sidebar_toc, 0)
_reg("sidebar_contacts", render_sidebar_contacts, 0)
_reg("sidebar_trending", render_sidebar_trending, 8)
_reg("story_bar", render_story_bar, 5)
_reg("suggestion_list", render_suggestion_list, 8)
_reg("email_detail", render_email_detail, 10)
_reg("product_detail", render_product_detail, 14)
_reg("related_cards", render_related_cards, 5)
_reg("tag_group", render_tag_group, 2)
_reg("hero_side", render_hero_side, 8)
_reg("social_login_buttons", render_social_login_buttons, 4)