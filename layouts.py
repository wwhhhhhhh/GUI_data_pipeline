"""
布局模板库 — 超高清紧凑视口 (23~44rem)
所有布局都加 overflow:hidden 兜底，杜绝溢出
"""
import random

LAYOUT_TEMPLATES = {
    # ── 1. 纯垂直堆叠 (任何尺寸, 最安全) ──
    "stack": {
        "label": "垂直堆叠",
        "min_rem_w": 0,
        "needs_sidebar": False,
        "slots": ["navbar", "section_0", "section_1", "section_2", "section_3", "footer"],
        "template": """
<div class="h-screen bg-base-200 flex flex-col overflow-hidden">
  {navbar}
  <main class="flex-1 p-3 space-y-3 overflow-hidden">
    {section_0}
    {section_1}
    {section_2}
    {section_3}
  </main>
  {footer}
</div>""",
    },

    # ── 2. 窄内容堆叠 (文章/表单) ──
    "stack_narrow": {
        "label": "窄内容堆叠",
        "min_rem_w": 0,
        "needs_sidebar": False,
        "slots": ["navbar", "breadcrumb", "main_content", "extra", "footer"],
        "template": """
<div class="h-screen bg-base-200 flex flex-col overflow-hidden">
  {navbar}
  <main class="flex-1 p-3 space-y-3 overflow-hidden">
    {breadcrumb}
    {main_content}
    {extra}
  </main>
  {footer}
</div>""",
    },

    # ── 3. Hero + 堆叠 ──
    "hero_stack": {
        "label": "Hero+堆叠",
        "min_rem_w": 0,
        "needs_sidebar": False,
        "slots": ["navbar", "hero", "section_1", "section_2", "section_3", "cta", "footer"],
        "template": """
<div class="h-screen bg-base-200 flex flex-col overflow-hidden">
  {navbar}
  {hero}
  <main class="flex-1 p-3 space-y-3 overflow-hidden">
    {section_1}
    {section_2}
    {section_3}
  </main>
  {cta}
  {footer}
</div>""",
    },

    # ── 4. 左侧边栏 (需要 ≥42rem 宽, 仅横屏最宽时) ──
    "sidebar_left": {
        "label": "左侧边栏+内容",
        "min_rem_w": 42,
        "needs_sidebar": True,
        "slots": ["navbar", "sidebar", "breadcrumb", "section_0", "main_content", "extra", "footer"],
        "template": """
<div class="h-screen bg-base-200 flex flex-col overflow-hidden">
  {navbar}
  <div class="flex flex-1 min-w-0 overflow-hidden">
    <aside class="w-[10rem] shrink-0 bg-base-100 border-r border-base-300 p-2 overflow-hidden">
      {sidebar}
    </aside>
    <main class="flex-1 p-3 space-y-3 min-w-0 overflow-hidden">
      {breadcrumb}
      {section_0}
      {main_content}
      {extra}
    </main>
  </div>
  {footer}
</div>""",
    },

    # ── 5. 居中卡片 (登录/错误) ──
    "center_card": {
        "label": "居中卡片",
        "min_rem_w": 0,
        "needs_sidebar": False,
        "slots": ["center_content", "extra"],
        "template": """
<div class="h-screen bg-base-200 flex items-center justify-center p-3 overflow-hidden">
  <div class="w-full max-w-[18rem] space-y-3 overflow-hidden">
    {center_content}
    {extra}
  </div>
</div>""",
    },

    # ── 6. Dashboard 堆叠 (stats + grid) ──
    "dashboard_stack": {
        "label": "仪表盘堆叠",
        "min_rem_w": 0,
        "needs_sidebar": False,
        "slots": ["navbar", "breadcrumb", "stats", "chart_area", "table_area", "extra", "footer"],
        "template": """
<div class="h-screen bg-base-200 flex flex-col overflow-hidden">
  {navbar}
  <main class="flex-1 p-3 space-y-3 overflow-hidden">
    {breadcrumb}
    {stats}
    {chart_area}
    {table_area}
    {extra}
  </main>
  {footer}
</div>""",
    },

    # ── 7. 聊天布局 ──
    "chat_layout": {
        "label": "聊天布局",
        "min_rem_w": 0,
        "needs_sidebar": False,
        "slots": ["navbar", "chat_area", "input_area"],
        "template": """
<div class="h-screen bg-base-200 flex flex-col overflow-hidden">
  {navbar}
  <main class="flex-1 p-3 space-y-2 overflow-hidden">
    {chat_area}
  </main>
  <div class="border-t border-base-300 bg-base-100 p-2 shrink-0">
    {input_area}
  </div>
</div>""",
    },
}


def pick_layout(name=None, prefer_list=None, res_ctx=None):
    rem_w = res_ctx.get("rem_w", 40) if res_ctx else 40
    allow_sidebar = res_ctx.get("allow_sidebar", False) if res_ctx else False

    valid = {
        k: v for k, v in LAYOUT_TEMPLATES.items()
        if rem_w >= v["min_rem_w"]
        and (allow_sidebar or not v["needs_sidebar"])
    }

    if name and name in valid:
        return name, valid[name]

    if prefer_list:
        pref_valid = [n for n in prefer_list if n in valid]
        if pref_valid and random.random() < 0.7:
            key = random.choice(pref_valid)
            return key, valid[key]

    key = random.choice(list(valid.keys()))
    return key, valid[key]