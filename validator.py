"""
Playwright 后处理验证器
========================
用法:
  pip install playwright && playwright install chromium
  python validator.py -i ./output/html_260327/ -m ./output/meta_260327/ -j 8

功能:
  1. 在 headless Chromium 中以正确 viewport 打开每个 HTML
  2. 检测水平/垂直溢出
  3. 若溢出: 通过 JS 从尾部逐个删除可选组件, 直到不溢出
  4. 保存修正后的 HTML 和更新后的 metadata
  5. 同时更新 image-manifest 中的 boundingBox
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def validate_one(html_path, meta_path, pw_browser, max_retries=10):
    """验证并修复单个 HTML 文件"""

    # 读取 meta
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    width = meta["width"]
    height = meta["height"]
    required_components = []  # 从 page_types 获取
    all_components = meta.get("components", [])

    # 页面必需组件 (不能删除)
    from page_types import PAGE_TYPES
    pt = meta.get("page_type", "")
    if pt in PAGE_TYPES:
        required_components = PAGE_TYPES[pt]["required"]

    # 可删除组件 (从后往前)
    removable = [c for c in reversed(all_components) if c not in required_components]

    context = pw_browser.new_context(viewport={"width": width, "height": height})
    page = context.new_page()

    try:
        page.goto(f"file://{os.path.abspath(html_path)}", wait_until="load", timeout=15000)
        page.wait_for_timeout(500)  # 等待 JS 执行完

        # 检测溢出
        overflow_info = page.evaluate("""() => {
            const b = document.body;
            return {
                scrollW: b.scrollWidth,
                scrollH: b.scrollHeight,
                clientW: document.documentElement.clientWidth,
                clientH: document.documentElement.clientHeight,
                overflowX: b.scrollWidth > document.documentElement.clientWidth + 2,
                overflowY: b.scrollHeight > document.documentElement.clientHeight + 2,
            }
        }""")

        removed = []
        retry = 0

        while (overflow_info["overflowX"] or overflow_info["overflowY"]) and removable and retry < max_retries:
            comp_to_remove = removable.pop(0)
            retry += 1

            # 尝试删除该组件的 DOM (找最后一个同类型的大容器)
            removed_ok = page.evaluate(f"""() => {{
                // 策略: 删除 body 内最后一个 .card / .stats / .hero / .alert / .footer 等
                const selectors = {{
                    'card_grid': '.card',
                    'stats': '.stats',
                    'hero_banner': '.hero',
                    'alert': '.alert',
                    'footer': 'footer',
                    'table': 'table',
                    'timeline': '.timeline',
                    'progress_group': 'progress',
                    'chart_placeholder': '.card',
                    'form_group': '.card',
                    'toggle_group': '.card',
                    'faq_accordion': '.collapse',
                    'comment_list': '.card',
                    'review_list': '.card',
                    'pricing_table': '.card',
                    'cta_section': '.bg-primary',
                    'feature_grid': '.card',
                    'testimonial_cards': '.card',
                    'badge_group': '.badge',
                    'avatar_group': '.avatar-group',
                    'pagination': '.join',
                    'tabs': '.tabs',
                    'feed_cards': '.card',
                    'email_list': '.card',
                    'kanban_columns': '.card',
                    'file_list': '.card',
                    'calendar_grid': '.card',
                    'code_block': '.card',
                    'suggestion_list': '.card',
                    'story_bar': '.avatar',
                }};

                // 通用策略: 删除 main 区域最后一个子元素
                const main = document.querySelector('main') || document.body;
                const children = main.children;
                if (children.length > 1) {{
                    const last = children[children.length - 1];
                    last.remove();
                    return true;
                }}
                return false;
            }}""")

            if removed_ok:
                removed.append(comp_to_remove)

            # 重新检测
            overflow_info = page.evaluate("""() => ({
                scrollW: document.body.scrollWidth,
                scrollH: document.body.scrollHeight,
                clientW: document.documentElement.clientWidth,
                clientH: document.documentElement.clientHeight,
                overflowX: document.body.scrollWidth > document.documentElement.clientWidth + 2,
                overflowY: document.body.scrollHeight > document.documentElement.clientHeight + 2,
            })""")

        # 获取修正后的 HTML
        if removed:
            fixed_html = page.evaluate("() => document.documentElement.outerHTML")
            # 加回 DOCTYPE
            fixed_html = "<!DOCTYPE html>\n" + fixed_html
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(fixed_html)

        # 更新 image manifest (已经由页面内 JS 更新了 boundingBox)
        try:
            manifest_json = page.evaluate("""() => {
                const el = document.getElementById('image-manifest');
                return el ? el.textContent : '[]';
            }""")
            manifest = json.loads(manifest_json)
            meta["images"] = manifest
        except:
            pass

        # 更新 meta
        still_overflow = overflow_info["overflowX"] or overflow_info["overflowY"]
        meta["validation"] = {
            "overflow_x": overflow_info["overflowX"],
            "overflow_y": overflow_info["overflowY"],
            "scroll_w": overflow_info["scrollW"],
            "scroll_h": overflow_info["scrollH"],
            "components_removed": removed,
            "fixed": bool(removed),
            "still_overflow": still_overflow,
        }
        meta["component_count"] = meta["component_count"] - len(removed)
        for r in removed:
            if r in meta["components"]:
                meta["components"].remove(r)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        status = "🔧" if removed else ("⚠️" if still_overflow else "✅")
        return f"{os.path.basename(html_path)} {status} scroll:{overflow_info['scrollW']}x{overflow_info['scrollH']} removed:{len(removed)}"

    except Exception as e:
        return f"{os.path.basename(html_path)} ❌ {e}"
    finally:
        context.close()


def main():
    parser = argparse.ArgumentParser(description="Playwright 后处理验证器")
    parser.add_argument("-i", "--html-dir", default=None,
                        help="HTML 目录 (直接指定)")
    parser.add_argument("-m", "--meta-dir", default=None,
                        help="Meta 目录 (直接指定)")
    parser.add_argument("-b", "--base-path", default=None,
                        help="输出根目录 (自动发现 html_onlytext_*/html_withpic_* 子目录)")
    parser.add_argument("-d", "--date", default=None,
                        help="日期标识 (配合 -b 使用, 默认今天)")
    parser.add_argument("-j", "--jobs", type=int, default=1,
                        help="Playwright 并发数 (单线程最稳定)")
    parser.add_argument("--sample", type=int, default=0,
                        help="仅验证前 N 个 (0=全部)")
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    # ── 收集 (html_dir, meta_dir) 对 ──
    dir_pairs = []
    if args.html_dir and args.meta_dir:
        dir_pairs.append((args.html_dir, args.meta_dir))
    elif args.base_path:
        import time as _t
        date = args.date or _t.strftime("%y%m%d")
        for sub in ("onlytext", "withpic"):
            hd = os.path.join(args.base_path, f"html_{sub}_{date}")
            md = os.path.join(args.base_path, f"meta_{sub}_{date}")
            if os.path.isdir(hd) and os.path.isdir(md):
                dir_pairs.append((hd, md))
        if not dir_pairs:
            print(f"在 {args.base_path} 下未找到 html_*_{date} 目录")
            return
    else:
        parser.error("请指定 -i/-m 或 -b")

    # ── 收集所有文件 ──
    tasks = []
    for hd, md in dir_pairs:
        for f in sorted(os.listdir(hd)):
            if f.endswith(".html"):
                idx = f.split("_")[0]
                mp = os.path.join(md, f"{idx}_meta.json")
                if os.path.exists(mp):
                    tasks.append((os.path.join(hd, f), mp))

    if args.sample > 0:
        tasks = tasks[:args.sample]

    print(f"验证 {len(tasks)} 个文件 (来自 {len(dir_pairs)} 个目录)")
    tic = time.time()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        ok = fix = err = still = 0
        for i, (hp, mp) in enumerate(tasks):
            result = validate_one(hp, mp, browser)
            if "✅" in result: ok += 1
            elif "🔧" in result: fix += 1
            elif "⚠️" in result: still += 1
            else: err += 1

            if (i+1) % 10 == 0 or i == len(tasks)-1:
                print(f"  [{i+1}/{len(tasks)}] {result}")

        browser.close()

    elapsed = time.time() - tic
    print(f"\n完成: ✅{ok} 🔧{fix} ⚠️{still} ❌{err} | {elapsed:.1f}s")


if __name__ == "__main__":
    main()