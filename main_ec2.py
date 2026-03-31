"""
GUI 数据生成主入口 v4.3
======================
- 直接使用 CorpusManager.sample(length) 作为语料接口
- HTML 按是否含图片分目录:
    html_onlytext_{date}/  +  meta_onlytext_{date}/
    html_withpic_{date}/   +  meta_withpic_{date}/
"""
import json, os, time, random
from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_CONCURRENT = 4
DEFAULT_SAMPLE_COUNT = 100


# ============================================================
# ID 管理 — 同时扫描两个目录
# ============================================================

def get_existing_ids(base_path, date_str):
    ids = set()
    for prefix in ("html_onlytext_", "html_withpic_"):
        d = os.path.join(base_path, f"{prefix}{date_str}")
        if os.path.exists(d):
            for f in os.listdir(d):
                try:
                    ids.add(int(f.split("_")[0]))
                except (ValueError, IndexError):
                    pass
    return ids


# ============================================================
# 单页面生成
# ============================================================

def generate_single_page(idx, date_str, corpus_manager, base_path, **kw):
    from assembler import assemble_page

    try:
        html, meta = assemble_page(corpus_fn=corpus_manager.sample, **kw)

        # ── 按是否有图片决定子目录 ──
        has_pic = meta["image_count"] > 0
        sub = "withpic" if has_pic else "onlytext"

        html_dir = os.path.join(base_path, f"html_{sub}_{date_str}")
        meta_dir = os.path.join(base_path, f"meta_{sub}_{date_str}")
        os.makedirs(html_dir, exist_ok=True)
        os.makedirs(meta_dir, exist_ok=True)

        # ── 保存 HTML ──
        fname = f"{idx:06d}_{meta['resolution']}_{meta['page_type']}.html"
        hp = os.path.join(html_dir, fname)
        with open(hp, "w", encoding="utf-8") as f:
            f.write(html)

        # ── 保存元数据 ──
        meta["id"] = idx
        meta["date"] = date_str
        meta["has_pic"] = has_pic
        mp = os.path.join(meta_dir, f"{idx:06d}_meta.json")
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        tag = "🖼" if has_pic else "📝"
        return (
            f"ID {idx:06d} ✅{tag} {meta['page_type']} | {meta['layout']} | "
            f"{meta['theme']} | {meta['resolution']} | "
            f"{meta['component_count']}comp {meta['image_count']}img | "
            f"h:{meta['height_used']:.0f}/{meta['height_budget']:.0f}rem"
        )
    except Exception as e:
        import traceback
        return f"ID {idx:06d} ❌ {e}\n{traceback.format_exc()}"


# ============================================================
# 批量生成
# ============================================================

def generate_batch(tasks, date_str, corpus_manager, base_path, max_concurrent):
    s = f = 0
    total = len(tasks)
    with ThreadPoolExecutor(max_workers=max_concurrent) as ex:
        fm = {
            ex.submit(generate_single_page, t, date_str, corpus_manager, base_path): t
            for t in tasks
        }
        done = 0
        for fut in as_completed(fm):
            try:
                r = fut.result()
                done += 1
                if "✅" in r:
                    s += 1
                else:
                    f += 1
                if done % 100 == 0 or "❌" in r or done == total:
                    print(f"  [{done}/{total}] {r}")
            except Exception as e:
                f += 1
                done += 1
                print(f"  [{done}/{total}] ID {fm[fut]:06d} 💥 {e}")
    return s, f


# ============================================================
# 主入口
# ============================================================

def main(
    sample_count=DEFAULT_SAMPLE_COUNT,
    date_str=None,
    id_start=0,
    base_path=None,
    corpus_manager=None,
    max_concurrent=MAX_CONCURRENT,
    batch_size=10000,
):
    """
    Args:
        corpus_manager: CorpusManager 实例, 必须提供 .sample(length) 方法
    """
    from components_v4 import COMPONENT_REGISTRY, DAISY_THEMES
    from layouts import LAYOUT_TEMPLATES
    from page_types import PAGE_TYPES
    from resolutions import RESOLUTIONS, ROOT_FONT_SIZE, SCALE_FACTOR

    if corpus_manager is None:
        raise ValueError("必须传入 corpus_manager (CorpusManager 实例)")

    if date_str is None:
        date_str = time.strftime("%y%m%d")

    print("=" * 70)
    print("  GUI 数据生成器 v4.3 — 视口感知 + 高度预算 + 分目录存储")
    print("=" * 70)
    print(f"  数量: {sample_count:,} | 并发: {max_concurrent} | 批次: {batch_size:,}")
    print(f"  组件: {len(COMPONENT_REGISTRY)} | 主题: {len(DAISY_THEMES)} | "
          f"布局: {len(LAYOUT_TEMPLATES)} | 页面类型: {len(PAGE_TYPES)} | "
          f"分辨率: {len(RESOLUTIONS)}")
    print(f"  缩放: ×{SCALE_FACTOR} | 根字号: {ROOT_FONT_SIZE}px | "
          f"最小文字: {int(ROOT_FONT_SIZE * 0.75)}px")
    print(f"  语料: {type(corpus_manager).__name__} "
          f"(已加载 {len(corpus_manager.corpora)} 个语料库)")
    print(f"  输出: html_onlytext_{date_str}/  +  html_withpic_{date_str}/")
    print("-" * 70)

    # ── ID 分配 (扫描两个目录) ──
    existing = get_existing_ids(base_path, date_str)
    if existing:
        print(f"  已有 {len(existing)} 个文件, 跳过已有 ID")

    tasks = []
    cid = id_start
    for _ in range(sample_count):
        while cid in existing:
            cid += 1
        tasks.append(cid)
        existing.add(cid)
        cid += 1

    # ── 批量生成 ──
    tic = time.time()
    ts = tf = 0
    nb = (len(tasks) + batch_size - 1) // batch_size
    for b in range(nb):
        bt = tasks[b * batch_size : min((b + 1) * batch_size, len(tasks))]
        if nb > 1:
            print(f"\n📦 批次 {b+1}/{nb}")
        s, f = generate_batch(bt, date_str, corpus_manager, base_path, max_concurrent)
        ts += s
        tf += f

    elapsed = time.time() - tic

    # ── 统计分布 ──
    cnt_text = cnt_pic = 0
    for prefix in ("meta_onlytext_", "meta_withpic_"):
        d = os.path.join(base_path, f"{prefix}{date_str}")
        if os.path.exists(d):
            c = len([f for f in os.listdir(d) if f.endswith(".json")])
            if "onlytext" in prefix:
                cnt_text = c
            else:
                cnt_pic = c

    print(f"\n{'=' * 70}")
    print(f"  🎉 完成!")
    print(f"  成功: {ts:,} | 失败: {tf:,} | 耗时: {elapsed:.1f}s | "
          f"{sample_count / max(elapsed, .001):.0f} 页/秒")
    print(f"  📝 纯文本: {cnt_text:,} | 🖼 带图: {cnt_pic:,}")
    print(f"  输出: {os.path.abspath(base_path)}")
    print("=" * 70)
    return ts, tf


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    import ast
    parser = argparse.ArgumentParser(
        description="GUI 数据生成器 v4.3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:

  python3 main_ec2.py -n 5000 --id-start 0 --output /data/wuwenhao/data/GUI_data_EN/ --corpus "['english-wiki_part_1', 'english-wiki_part_2', 'english-wiki_part_3', 'english-wiki_part_4', 'english-wiki_part_5','english-wiki_part_6', 'english-wiki_part_7','english-wiki_part_8', 'english-wiki_part_9','english-wiki_part_10']"
  python3 main_ec2.py -n 5000 --id-start 0 --output /data/wuwenhao/data/GUI_data_ZH/ --corpus "['chinese-laws', 'chinese-news', 'chinese-novel']"
        """,
    )
    parser.add_argument("-n", "--count", type=int, default=10000,
                        help="生成数量 (默认 10)")
    parser.add_argument("-d", "--date", type=str, default=None,
                        help="日期标识 (默认 YYMMDD)")
    parser.add_argument("--id-start", type=int, default=0,
                        help="起始 ID")
    parser.add_argument("-o", "--output", type=str, default="/data/wuwenhao/data/GUI_data_EN/",
                        help="输出目录")
    parser.add_argument("-j", "--jobs", type=int, default=4,
                        help="并发线程数")
    parser.add_argument("--batch-size", type=int, default=10000,
                        help="每批次大小")
    parser.add_argument("--corpus", type=str, default=['vocab_zh'],
                        help="语料库名称, 例: chinese-news chinese-chatgpt THUCNews")

    args = parser.parse_args()

    # ── 加载语料库 ──
    from sample_from_corpus import CorpusManager
    cm = CorpusManager(ast.literal_eval(args.corpus))

    main(
        sample_count=args.count,
        date_str=args.date,
        id_start=args.id_start,
        base_path=args.output,
        corpus_manager=cm,
        max_concurrent=args.jobs,
        batch_size=args.batch_size,
    )