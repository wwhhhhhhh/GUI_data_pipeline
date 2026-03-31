"""
LanceDB 批量关键词检索替换图片占位
================================

假设你的生成 HTML 里有图片占位（来自 components_v4.py 的 img-placeholder div）：
  <div id="img_0000" class="img-placeholder w-full aspect-video ..." data-img-keyword="xxx"></div>

流程：
1) 每批处理 N 个 HTML（默认 10000）
2) 从 meta json 里读取 keyword（优先）或从 HTML 解析 data-img-keyword
3) 从 parquet 加载候选集，并用 parquet[level1] 构建 LanceDB 向量索引
4) keyword -> 向量 -> LanceDB 搜索 top1，拿到候选行里的图片路径
5) 将占位 div 替换为 <img src="...">，输出到 html_processed 目录，并可选择复制图片到 output/images

注意：
- 这里用的是 TF-IDF char ngram + LanceDB 的向量检索，向量维度由 --n-features 控制。
- 真实项目里 level1/图片路径字段名可能不同：脚本会尝试常见列名，并支持你在参数里指定。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import lancedb  # type: ignore
except Exception:  # pragma: no cover
    lancedb = None


IMG_PLACEHOLDER_RE = re.compile(
    r'<div id="(img_\d+)" class="img-placeholder ([^"]+)" style="background:#FF0000;" '
    r'data-img-id="\1" data-img-keyword="([^"]*)"></div>'
)


def _safe_relpath(p: str) -> str:
    p = p.replace("\\", "/")
    p = p.lstrip("/")
    return p


def _detect_img_path_col(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    candidates = [
        "train_relative_img_path",
        "img_path",
        "image_path",
        "relative_img_path",
        "path",
        "img",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"无法从 parquet 列中找到图片路径列。现有列: {list(df.columns)}")


def _collect_html_files(html_dir: str) -> List[str]:
    html_files = sorted([f for f in os.listdir(html_dir) if f.endswith(".html")])
    return html_files


def _meta_path_for_html(meta_dir: str, html_filename: str) -> Optional[str]:
    idx = html_filename.split("_", 1)[0]
    mp = os.path.join(meta_dir, f"{idx}_meta.json")
    return mp if os.path.exists(mp) else None


def _load_keywords_from_meta(meta_path: str) -> Dict[str, str]:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    images = meta.get("images", [])
    # images: [{"id": "img_0000", "keyword": "..."}]
    id_to_kw = {}
    for it in images:
        if not isinstance(it, dict):
            continue
        img_id = it.get("id")
        kw = it.get("keyword")
        if img_id and kw:
            id_to_kw[str(img_id)] = str(kw)
    return id_to_kw


def _load_keywords_from_html(html_content: str) -> Dict[str, str]:
    id_to_kw: Dict[str, str] = {}
    for m in IMG_PLACEHOLDER_RE.finditer(html_content):
        img_id, _cls, kw = m.group(1), m.group(2), m.group(3)
        id_to_kw[img_id] = kw
    return id_to_kw


def _build_lancedb_index_from_parquet(
    *,
    parquet_dir: str,
    lancedb_dir: str,
    table_name: str,
    level1_col: str,
    img_path_col: str,
    n_features: int,
    candidate_limit: Optional[int],
    rebuild_index: bool,
    vectorizer_cache_path: Optional[str],
):
    # 1) 读 parquet 候选
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "**/*.parquet"), recursive=True))
    if not parquet_files:
        raise FileNotFoundError(f"未在 {parquet_dir} 找到 parquet 文件")

    dfs = []
    for pf in parquet_files:
        df = pd.read_parquet(pf, columns=[level1_col, img_path_col])
        dfs.append(df)
        if candidate_limit is not None and sum(len(x) for x in dfs) >= candidate_limit:
            break

    df_all = pd.concat(dfs, ignore_index=True)
    if candidate_limit is not None:
        df_all = df_all.iloc[:candidate_limit].copy()

    df_all[level1_col] = df_all[level1_col].astype(str)
    df_all[img_path_col] = df_all[img_path_col].astype(str)

    # 2) 向量化（TF-IDF char ngram）
    if vectorizer_cache_path and (not rebuild_index) and os.path.exists(vectorizer_cache_path):
        import pickle

        with open(vectorizer_cache_path, "rb") as f:
            vectorizer = pickle.load(f)
    else:
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(1, 3),
            max_features=n_features,
        )
        vectorizer.fit(df_all[level1_col].tolist())
        if vectorizer_cache_path:
            import pickle

            os.makedirs(os.path.dirname(vectorizer_cache_path), exist_ok=True)
            with open(vectorizer_cache_path, "wb") as f:
                pickle.dump(vectorizer, f)

    X = vectorizer.transform(df_all[level1_col].tolist())
    vectors = X.toarray().astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / (norms + 1e-10)

    # 3) 写入 LanceDB
    os.makedirs(lancedb_dir, exist_ok=True)
    db = lancedb.connect(lancedb_dir)

    # LanceDB 向量列名必须叫 vector
    records = []
    for i in range(len(df_all)):
        records.append(
            {
                "id": int(i),
                "level1": df_all.iloc[i][level1_col],
                "img_path": df_all.iloc[i][img_path_col],
                "vector": vectors[i],
            }
        )

    if rebuild_index:
        db.drop_table_if_exists(table_name)
        table = db.create_table(table_name, records, mode="overwrite")
    else:
        # table 不存在则创建
        try:
            table = db.open_table(table_name)
        except Exception:
            table = db.create_table(table_name, records, mode="overwrite")

    return db, table, vectorizer


def _vectorize_keywords(vectorizer: TfidfVectorizer, keywords: List[str]) -> np.ndarray:
    X = vectorizer.transform(keywords)
    q = X.toarray().astype(np.float32)
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    return q / (norms + 1e-10)


def _load_candidate_matrix_from_parquet_numpy(
    *,
    parquet_dir: str,
    level1_col: str,
    img_path_col: str,
    candidate_limit: Optional[int],
    n_features: int,
    vectorizer_cache_path: Optional[str],
    rebuild_index: bool,
):
    """
    NumPy 后端：
    - 从 parquet 读取候选（level1 + img_path）
    - TF-IDF 向量化 level1
    - 返回 (vectorizer, candidate_vectors_normed, candidate_img_paths)
    """
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "**/*.parquet"), recursive=True))
    if not parquet_files:
        raise FileNotFoundError(f"未在 {parquet_dir} 找到 parquet 文件")

    dfs = []
    for pf in parquet_files:
        df = pd.read_parquet(pf, columns=[level1_col, img_path_col])
        dfs.append(df)
        if candidate_limit is not None and sum(len(x) for x in dfs) >= candidate_limit:
            break

    df_all = pd.concat(dfs, ignore_index=True)
    if candidate_limit is not None:
        df_all = df_all.iloc[:candidate_limit].copy()

    df_all[level1_col] = df_all[level1_col].astype(str)
    df_all[img_path_col] = df_all[img_path_col].astype(str)

    if vectorizer_cache_path and (not rebuild_index) and os.path.exists(vectorizer_cache_path):
        import pickle

        with open(vectorizer_cache_path, "rb") as f:
            vectorizer = pickle.load(f)
    else:
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(1, 3),
            max_features=n_features,
        )
        vectorizer.fit(df_all[level1_col].tolist())
        if vectorizer_cache_path:
            import pickle

            os.makedirs(os.path.dirname(vectorizer_cache_path), exist_ok=True)
            with open(vectorizer_cache_path, "wb") as f:
                pickle.dump(vectorizer, f)

    X = vectorizer.transform(df_all[level1_col].tolist())
    V = X.toarray().astype(np.float32)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    V = V / (norms + 1e-10)

    candidate_paths = df_all[img_path_col].tolist()
    return vectorizer, V, candidate_paths


def _retrieve_kw_to_candidates_numpy(
    *,
    vectorizer: TfidfVectorizer,
    candidate_vectors: np.ndarray,
    candidate_img_paths: List[str],
    keywords: List[str],
    topn: int,
    q_block_size: int = 256,
):
    """
    合并关键词批检索（NumPy）：
    - 一次性向量化 unique keywords
    - 按相似度 top-n 取候选（rank 从高到低）
    返回 kw -> [img_path_0, img_path_1, ...]（相似度降序）
    """
    unique_keywords = list(dict.fromkeys(keywords))
    if not unique_keywords:
        return {}

    topn = min(topn, candidate_vectors.shape[0])

    kw_to_candidates: Dict[str, List[str]] = {}
    # 向量维度应一致
    for start in range(0, len(unique_keywords), q_block_size):
        block = unique_keywords[start : start + q_block_size]
        Q = _vectorize_keywords(vectorizer, block)
        # scores: (b, n_candidates)
        scores = Q @ candidate_vectors.T

        # 先用 argpartition 拿 topn 的无序集合，再按分数排序
        idx_part = np.argpartition(scores, -topn, axis=1)[:, -topn:]
        scores_part = np.take_along_axis(scores, idx_part, axis=1)
        order = np.argsort(-scores_part, axis=1)
        idx_sorted = np.take_along_axis(idx_part, order, axis=1)

        for i, kw in enumerate(block):
            ids = idx_sorted[i].tolist()
            kw_to_candidates[kw] = [candidate_img_paths[j] for j in ids]

    return kw_to_candidates


def _search_best_img_paths(
    *,
    table,
    vectorizer: TfidfVectorizer,
    keywords: List[str],
    img_path_col: str = "img_path",
    top_k: int = 1,
) -> Dict[str, str]:
    unique_keywords = list(dict.fromkeys(keywords))
    if not unique_keywords:
        return {}

    q_vecs = _vectorize_keywords(vectorizer, unique_keywords)

    kw_to_img_path: Dict[str, str] = {}
    for i, kw in enumerate(unique_keywords):
        q = q_vecs[i]
        res = table.search(q).limit(top_k).to_pandas()
        if len(res) == 0:
            continue
        # LanceDB 返回的是距离，距离越小相似度越高（这里用归一化向量，L2/cosine 单调）
        row = res.iloc[0]
        kw_to_img_path[kw] = str(row[img_path_col])
    return kw_to_img_path


def _search_kw_to_candidates_lancedb(
    *,
    table,
    vectorizer: TfidfVectorizer,
    keywords: List[str],
    img_path_col: str = "img_path",
    topn: int = 10,
) -> Dict[str, List[str]]:
    """
    LanceDB 后端（保守实现）：
    每个 keyword 单独检索 topn；再做去重分配放在外层完成。
    """
    unique_keywords = list(dict.fromkeys(keywords))
    if not unique_keywords:
        return {}

    kw_to_candidates: Dict[str, List[str]] = {}
    for kw in unique_keywords:
        q_vec = _vectorize_keywords(vectorizer, [kw])[0]
        res = table.search(q_vec).limit(topn).to_pandas()
        if len(res) == 0:
            continue
        kw_to_candidates[kw] = [str(x) for x in res[img_path_col].tolist()]
    return kw_to_candidates


def _ensure_copied(
    *,
    img_rel_path: str,
    images_root: str,
    output_dir: str,
    copy_images: bool,
) -> Optional[str]:
    img_rel_path = _safe_relpath(img_rel_path)
    if images_root:
        src_path = os.path.join(images_root, img_rel_path)
    else:
        src_path = img_rel_path

    if not os.path.exists(src_path):
        return None

    if not copy_images:
        # 不复制时直接引用源路径（在 file:// 场景下通常可用）
        return src_path

    dest_path = os.path.join(output_dir, "images", img_rel_path)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copy2(src_path, dest_path)
    # HTML 中引用 output_dir 里的相对路径
    return f"./images/{img_rel_path}"


def _replace_placeholders_with_imgs(
    *,
    html_content: str,
    id_to_src: Dict[str, str],
) -> str:
    def _repl(m: re.Match) -> str:
        img_id = m.group(1)
        cls = m.group(2)
        kw = m.group(3)
        src = id_to_src.get(img_id)
        if not src:
            return m.group(0)
        # 占位 div 尺寸由 aspect-* 类决定，这里保留原 class 并补 object-cover
        return f'<img id="{img_id}" class="{cls} object-cover" src="{src}" alt="{kw}"/>'

    return IMG_PLACEHOLDER_RE.sub(_repl, html_content)


def process_batch(
    batch_html_files: List[str],
    *,
    html_dir: str,
    meta_dir: Optional[str],
    output_dir: str,
    images_root: str,
    backend: str,
    table=None,
    vectorizer: Optional[TfidfVectorizer] = None,
    candidate_vectors: Optional[np.ndarray] = None,
    candidate_img_paths: Optional[List[str]] = None,
    top_k: int = 10,
    allow_reuse: bool = False,
    q_block_size: int = 256,
    copy_images: bool,
):
    if backend == "numpy":
        if vectorizer is None or candidate_vectors is None or candidate_img_paths is None:
            raise ValueError("backend=numpy 时需要 vectorizer/candidate_vectors/candidate_img_paths")
    elif backend == "lancedb":
        if table is None or vectorizer is None:
            raise ValueError("backend=lancedb 时需要 table/vectorizer")
    else:
        raise ValueError(f"未知 backend: {backend}")

    # 1) 收集每个 HTML 的占位 keyword，并生成“出现顺序”的 occurrences 列表
    html_to_id_to_kw: Dict[str, Dict[str, str]] = {}
    occurrences: List[Tuple[str, str, str]] = []  # (html_filename, img_id, keyword)
    all_keywords: List[str] = []  # 给检索去重用
    for html_filename in batch_html_files:
        in_path = os.path.join(html_dir, html_filename)
        meta_path = _meta_path_for_html(meta_dir, html_filename) if meta_dir else None
        if meta_path:
            id_to_kw = _load_keywords_from_meta(meta_path)
        else:
            with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            id_to_kw = _load_keywords_from_html(content)
        html_to_id_to_kw[html_filename] = id_to_kw

        # 遵循 id_to_kw 的插入顺序（从 meta/json 或 regex 命中顺序得到）
        for img_id, kw in id_to_kw.items():
            kw = str(kw)
            occurrences.append((html_filename, img_id, kw))
            all_keywords.append(kw)

    if not occurrences:
        return

    # 2) merged retrieval：unique keywords -> top-n 候选路径列表
    unique_keywords = list(dict.fromkeys(all_keywords))
    if backend == "numpy":
        assert vectorizer is not None
        assert candidate_vectors is not None
        assert candidate_img_paths is not None
        kw_to_candidates = _retrieve_kw_to_candidates_numpy(
            vectorizer=vectorizer,
            candidate_vectors=candidate_vectors,
            candidate_img_paths=candidate_img_paths,
            keywords=unique_keywords,
            topn=top_k,
            q_block_size=q_block_size,
        )
    else:
        kw_to_candidates = _search_kw_to_candidates_lancedb(
            table=table,
            vectorizer=vectorizer,  # type: ignore[arg-type]
            keywords=unique_keywords,
            topn=top_k,
        )

    # 2) 输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 3) 按出现顺序做全局去重分配（关键逻辑）
    global_used_paths = set()
    id_to_src_by_html: Dict[str, Dict[str, str]] = {}
    for html_filename, img_id, kw in occurrences:
        cands = kw_to_candidates.get(kw, [])
        if not cands:
            continue

        chosen: Optional[str] = None
        for cand_path in cands:
            if cand_path not in global_used_paths:
                chosen = cand_path
                break

        if chosen is None and allow_reuse:
            chosen = cands[0]

        if chosen is None:
            continue

        global_used_paths.add(chosen)

        src = _ensure_copied(
            img_rel_path=chosen,
            images_root=images_root,
            output_dir=output_dir,
            copy_images=copy_images,
        )
        if not src:
            continue

        id_to_src_by_html.setdefault(html_filename, {})[img_id] = src

    # 4) 对每个 HTML 做替换
    for html_filename in batch_html_files:
        in_path = os.path.join(html_dir, html_filename)
        out_path = os.path.join(output_dir, html_filename)

        with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        id_to_src = id_to_src_by_html.get(html_filename, {})
        new_content = _replace_placeholders_with_imgs(html_content=content, id_to_src=id_to_src)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(new_content)


def main():
    parser = argparse.ArgumentParser("LanceDB keyword retrieval -> html_processed")
    parser.add_argument("--html-dir", required=True, help="输入 html_withpic_*/ 目录")
    parser.add_argument("--meta-dir", default=None, help="输入 meta_withpic_*/ 目录（建议提供）")
    parser.add_argument("--parquet-dir", required=True, help="候选 parquet 目录（含 level1 字段与图片路径字段）")
    parser.add_argument("--images-root", default="", help="图片本地根目录（和 parquet img_path 对齐）。不想复制图片可留空")
    parser.add_argument("--output-dir", required=True, help="输出 html_processed 目录")
    parser.add_argument("--lancedb-dir", default=None, help="LanceDB 存储目录（默认 output-dir/lancedb）")

    parser.add_argument("--level1-col", default="level1", help="parquet 里 level1 字段名")
    parser.add_argument("--img-path-col", default=None, help="parquet 里图片路径字段名（可不填，脚本会自动猜）")
    parser.add_argument("--top-k", type=int, default=1, help="每个 keyword 取 top1/更高")

    parser.add_argument("--batch-html-size", type=int, default=10000, help="每批处理多少 HTML（默认 10000）")
    parser.add_argument("--candidate-limit", type=int, default=100000, help="从 parquet 取多少候选（默认 100000 ~= 10w 图片）")

    parser.add_argument("--n-features", type=int, default=1024, help="TF-IDF 向量维度（默认 1024）")
    parser.add_argument("--q-block-size", type=int, default=256, help="NumPy 后端检索时的 query block 大小")
    parser.add_argument("--backend", choices=["numpy", "lancedb"], default="numpy", help="检索后端：numpy 合并检索更快；lancedb 更通用")
    parser.add_argument("--rebuild-index", action="store_true", help="强制重建 LanceDB 索引")
    parser.add_argument("--vectorizer-cache", default=None, help="向量器缓存路径（pickle），减少重复 fit")
    parser.add_argument("--copy-images", action="store_true", help="将匹配到的图片复制到 output_dir/images")
    parser.add_argument("--allow-reuse", action="store_true", help="当所有候选都被用完时，允许复用图片（默认不复用）")
    args = parser.parse_args()
    backend = args.backend

    # 先读一次 parquet 来猜 img_path 列名
    parquet_files = sorted(glob.glob(os.path.join(args.parquet_dir, "**/*.parquet"), recursive=True))
    if not parquet_files:
        raise FileNotFoundError(f"未在 {args.parquet_dir} 找到 parquet 文件")
    sample_df = pd.read_parquet(parquet_files[0], columns=[args.level1_col])
    # 猜 img_path 列
    if args.img_path_col is None:
        # 读全列名用于猜测（只读列名不读数据）
        cols_df = pd.read_parquet(parquet_files[0])
        img_path_col = _detect_img_path_col(cols_df, None)
    else:
        img_path_col = args.img_path_col

    html_files = _collect_html_files(args.html_dir)
    if not html_files:
        raise FileNotFoundError(f"在 {args.html_dir} 未找到 html 文件")

    batch_size = max(1, args.batch_html_size)

    table = None
    vectorizer: Optional[TfidfVectorizer] = None
    candidate_vectors: Optional[np.ndarray] = None
    candidate_img_paths: Optional[List[str]] = None

    if backend == "numpy":
        vectorizer, candidate_vectors, candidate_img_paths = _load_candidate_matrix_from_parquet_numpy(
            parquet_dir=args.parquet_dir,
            level1_col=args.level1_col,
            img_path_col=img_path_col,
            candidate_limit=args.candidate_limit,
            n_features=args.n_features,
            vectorizer_cache_path=args.vectorizer_cache,
            rebuild_index=args.rebuild_index,
        )
    else:
        if lancedb is None:
            raise RuntimeError("backend=lancedb 但 lancedb 未安装，请安装 lancedb 或切换到 backend=numpy")
        lancedb_dir = args.lancedb_dir or os.path.join(args.output_dir, "lancedb")
        os.makedirs(lancedb_dir, exist_ok=True)
        table_name = "level1_images"
        _db, table, vectorizer = _build_lancedb_index_from_parquet(
            parquet_dir=args.parquet_dir,
            lancedb_dir=lancedb_dir,
            table_name=table_name,
            level1_col=args.level1_col,
            img_path_col=img_path_col,
            n_features=args.n_features,
            candidate_limit=args.candidate_limit,
            rebuild_index=args.rebuild_index,
            vectorizer_cache_path=args.vectorizer_cache,
        )

    for start in range(0, len(html_files), batch_size):
        batch = html_files[start : start + batch_size]
        print(f"[Batch] {start}/{len(html_files)} => {len(batch)} html")
        process_batch(
            batch,
            html_dir=args.html_dir,
            meta_dir=args.meta_dir,
            output_dir=args.output_dir,
            images_root=args.images_root,
            backend=backend,
            table=table,
            vectorizer=vectorizer,
            candidate_vectors=candidate_vectors,
            candidate_img_paths=candidate_img_paths,
            top_k=args.top_k,
            allow_reuse=args.allow_reuse,
            q_block_size=args.q_block_size,
            copy_images=args.copy_images,
        )

    print("✅ 完成：html_processed 目录已写入。")


if __name__ == "__main__":
    main()

# python3 retrieval_lancedb_batch.py \
#   --backend numpy \
#   --html-dir /path/html_withpic_YYMMDD \
#   --meta-dir /path/meta_withpic_YYMMDD \
#   --parquet-dir /path/parquet_dir \
#   --output-dir /path/html_processed_YYMMDD \
#   --batch-html-size 10000 \
#   --candidate-limit 100000 \
#   --top-k 20 \
#   --copy-images \
#   --images-root /path/to/local_img_root