import argparse
import os
import re
import numpy as np
import pandas as pd
import faiss  # faiss-cpu
import time
import torch
import gc
import collections
from multiprocessing import Pool
from typing import List, Dict, Tuple
import moxing as mox

# ===================== 1. 配置与参数 =====================

def parse_args():
    parser = argparse.ArgumentParser(description="Keyword-First Distributed Retrieval")
    parser.add_argument('--feat_range', nargs=2, type=float, default=[0.0, 0.10])
    parser.add_argument('--html_range', nargs=2, type=float, default=[0.0, 1.0])
    parser.add_argument('--inner_shards', type=int, default=5)
    parser.add_argument('--emb_batch_size', type=int, default=512)
    parser.add_argument('--search_buffer', type=int, default=20,
                        help="每个 keyword 额外多检索的 buffer 数量，防止路径冲突时不够用")
    return parser.parse_args()


class CFG:
    FEAT_ROOT = "/cache/01.USERS/w00878018/data/L5_Embedding_V2d0d0/QwenVL_Embedding_Feature_V0/"
    PQ_ROOT = "/cache/01.USERS/w00878018/data/L5_Embedding_V2d0d0/QwenVL_Embedding_Pq_V0/"
    EMBEDDING_MODEL_PATH = "/cache/01.USERS/w00878018/data/L5_Embedding_V2d0d0/models/Qwen3-VL-Embedding-2B/"
    HTML_INPUT_DIR = '/cache/GUI_data/html_260227/'
    HTML_OUTPUT_DIR = '/cache/GUI_data/html_260227_processed/'

    os.makedirs(HTML_INPUT_DIR, exist_ok=True)
    os.makedirs(HTML_OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(FEAT_ROOT):
        mox.file.copy_parallel(
            FEAT_ROOT.replace('/cache', 's3://bucket-green-huadong2-711/'), FEAT_ROOT)
        mox.file.copy_parallel(
            PQ_ROOT.replace('/cache', 's3://bucket-green-huadong2-711/'), PQ_ROOT, threads=64)
        mox.file.copy_parallel(
            EMBEDDING_MODEL_PATH.replace('/cache', 's3://bucket-green-huadong2-711/'),
            EMBEDDING_MODEL_PATH)

    OLD_PREFIX = "bucket-6824-huanan/data/AIGC/TRAIN_DATA_WHOLE_PACK/Pangu-T2I_Filter_pack/"
    NEW_PREFIX = "bucket-green-huadong2-711/02.DATA/T2I/01.Release/TRAIN_DATA_WHOLE_PACK/Pangu-T2I_Filter_pack/"

    FULL_DIM = 2048
    VECTOR_DIM = 1024
    DEVICE = "npu"

    # ---- 模糊匹配正则 ----
    _SPLIT_TOK = r'<\s*split\s*>?'
    PROTOCOL_RE = re.compile(
        r'(' +
        _SPLIT_TOK + r'\s*(\d+)\s*' +
        _SPLIT_TOK + r'\s*(\d+)\s*' +
        _SPLIT_TOK + r'\s*(.*?)\s*' +
        _SPLIT_TOK +
        r')',
        re.IGNORECASE | re.DOTALL
    )


# ===================== 2. IO 加载 =====================

def _load_single_entry(name_key, npy_path, pq_path, vector_dim, full_dim, old_prefix, new_prefix):
    try:
        feats = np.load(npy_path).squeeze()
        if feats.ndim == 1:
            feats = feats.reshape(1, -1)
        if feats.shape[-1] != full_dim:
            return None

        feats_reduced = feats[:, :vector_dim].astype(np.float32)
        norms = np.linalg.norm(feats_reduced, axis=1, keepdims=True)
        feats_reduced /= (norms + 1e-10)

        df = pd.read_parquet(pq_path, columns=["train_relative_img_path"])
        img_paths = [p.replace(old_prefix, new_prefix) for p in df["train_relative_img_path"]]
        return feats_reduced, img_paths
    except Exception:
        return None


# ===================== 3. 搜索引擎 =====================

class MiniSearchEngine:
    def __init__(self, keys_to_load, npy_map, pq_map):
        start_t = time.time()
        self.index = faiss.IndexFlatIP(CFG.VECTOR_DIM)
        self.metadata: List[str] = []

        tasks = [
            (k, npy_map[k], pq_map[k], CFG.VECTOR_DIM, CFG.FULL_DIM, CFG.OLD_PREFIX, CFG.NEW_PREFIX)
            for k in keys_to_load if k in pq_map
        ]

        if tasks:
            with Pool(processes=16) as pool:
                results = pool.starmap(_load_single_entry, tasks)

            temp_features = []
            for res in results:
                if res:
                    temp_features.append(res[0])
                    self.metadata.extend(res[1])

            if temp_features:
                self.index.add(np.ascontiguousarray(np.vstack(temp_features).astype(np.float32)))

        self.load_cost = time.time() - start_t
        print(f"   [Index] 构建完成: 向量数 {self.index.ntotal} | 元数据 {len(self.metadata)} | 耗时 {self.load_cost:.2f}s")


# ===================== 4. 核心流程：Keyword-First =====================

def scan_all_htmls(html_files: List[str]) -> Tuple[Dict[str, str], Dict[str, List[dict]]]:
    """
    Phase 1: 扫描 HTML，提取 keyword 及引用位置（模糊正则）。
    """
    file_contents = {}
    keyword_refs = collections.defaultdict(list)
    total_matches = 0
    fuzzy_count = 0
    no_match_count = 0

    for fi, fname in enumerate(html_files):
        # 进度日志
        if (fi + 1) % 5000 == 0:
            print(f"      [Scan] {fi + 1}/{len(html_files)} files | "
                  f"匹配: {total_matches} | 无匹配文件: {no_match_count}")

        in_path = os.path.join(CFG.HTML_INPUT_DIR, fname)
        try:
            with open(in_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"❌ Read Error {fname}: {e}")
            continue

        matches = CFG.PROTOCOL_RE.findall(content)
        if not matches:
            no_match_count += 1
            continue

        file_contents[fname] = content
        for full_str, h, w, keyword in matches:
            keyword = keyword.strip()
            if not keyword:
                continue
            hw = f"{h}_{w}"
            keyword_refs[keyword].append({
                'fname': fname,
                'hw': hw,
                'full_match_str': full_str,
            })
            total_matches += 1
            canonical = f"<split>{h}<split>{w}<split>{keyword}<split>"
            if full_str != canonical:
                fuzzy_count += 1

    print(f"      [Scan Done] 总匹配: {total_matches} | 无匹配文件: {no_match_count}/{len(html_files)}")
    if fuzzy_count > 0:
        print(f"      ⚠️ 模糊匹配命中 {fuzzy_count}/{total_matches} 条")

    return file_contents, keyword_refs


def embed_keywords(unique_keywords: List[str], model, batch_size: int = 512) -> np.ndarray:
    """
    Phase 2: 批量向量化（带进度日志）。
    """
    all_feats = []
    instruction = "Retrieve images or text relevant to the user's query."
    total = len(unique_keywords)
    t_start = time.time()

    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch_qs = unique_keywords[i:i + batch_size]
            batch_data = [{'text': q, 'instruction': instruction} for q in batch_qs]
            embeddings = model.process(batch_data)
            feats = embeddings.cpu().numpy().astype(np.float32)[:, :CFG.VECTOR_DIM]
            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            all_feats.append(feats / (norms + 1e-10))

            done = min(i + batch_size, total)
            if done % (batch_size * 5) == 0 or done == total:
                elapsed = time.time() - t_start
                speed = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / speed if speed > 0 else 0
                print(f"      [Embed] {done}/{total} | "
                      f"{speed:.0f} kw/s | ETA {eta:.0f}s")

    return np.vstack(all_feats)


def retrieve_and_replace(
    file_contents: Dict[str, str],
    keyword_refs: Dict[str, List[dict]],
    query_vectors: np.ndarray,
    unique_keywords: List[str],
    engine: MiniSearchEngine,
    search_buffer: int = 20,
):
    """
    Phase 3: 对每个 keyword 做一次 top-K 检索，按顺序分发，天然去重。
    """
    global_used = set()
    total_assigned = 0
    total_failed = 0
    t_start = time.time()

    for idx, keyword in enumerate(unique_keywords):
        refs = keyword_refs[keyword]
        needed = len(refs)

        search_k = min(needed + search_buffer + len(global_used) // 10, engine.index.ntotal)
        search_k = max(search_k, needed + search_buffer)
        search_k = min(search_k, engine.index.ntotal)

        if search_k == 0:
            total_failed += needed
            continue

        D, I = engine.index.search(query_vectors[idx:idx + 1], search_k)

        available = []
        for cand_idx in I[0]:
            if cand_idx == -1:
                break
            path = engine.metadata[cand_idx]
            if path not in global_used:
                available.append(path)
                if len(available) >= needed:
                    break

        for i, ref in enumerate(refs):
            if i < len(available):
                img_path = available[i]
                global_used.add(img_path)
                total_assigned += 1

                save_name = f"{keyword}__{ref['hw']}__{img_path}.jpg"
                file_contents[ref['fname']] = file_contents[ref['fname']].replace(
                    ref['full_match_str'], f"./{save_name}"
                )
            else:
                total_failed += 1

        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t_start
            speed = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (len(unique_keywords) - idx - 1) / speed if speed > 0 else 0
            print(f"      [Retrieve] {idx + 1}/{len(unique_keywords)} | "
                  f"分配: {total_assigned} | 失败: {total_failed} | "
                  f"{speed:.0f} kw/s | ETA {eta:.0f}s")

    print(f"      [Retrieve Done] 总分配: {total_assigned} | 总失败: {total_failed} | "
          f"全局已用: {len(global_used)} | 耗时: {time.time() - t_start:.1f}s")
    return file_contents


def write_outputs(file_contents: Dict[str, str]):
    written = 0
    for fname, content in file_contents.items():
        out_path = os.path.join(CFG.HTML_OUTPUT_DIR, fname)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(content)
        written += 1
    print(f"      [Write Done] 写入 {written} 个文件")


def process_shard(html_files: List[str], model, engine: MiniSearchEngine, args):
    shard_start = time.time()

    # Phase 1
    print(f"   [Phase 1] 扫描 {len(html_files)} 个 HTML...")
    file_contents, keyword_refs = scan_all_htmls(html_files)
    unique_keywords = list(keyword_refs.keys())
    total_refs = sum(len(v) for v in keyword_refs.values())
    print(f"   [Phase 1 Done] 有效文件: {len(file_contents)} | "
          f"去重 keyword: {len(unique_keywords)} | 总需求: {total_refs} | "
          f"耗时: {time.time() - shard_start:.1f}s")

    if not unique_keywords:
        print("   [Skip] 无有效 keyword，跳过")
        return

    # Phase 2
    print(f"   [Phase 2] 批量向量化 {len(unique_keywords)} 个 keyword...")
    t0 = time.time()
    query_vectors = embed_keywords(unique_keywords, model, batch_size=args.emb_batch_size)
    print(f"   [Phase 2 Done] 耗时 {time.time() - t0:.1f}s")

    # Phase 3
    print(f"   [Phase 3] 检索与分发 (索引量={engine.index.ntotal}, buffer={args.search_buffer})...")
    t0 = time.time()
    file_contents = retrieve_and_replace(
        file_contents, keyword_refs, query_vectors,
        unique_keywords, engine, search_buffer=args.search_buffer
    )
    print(f"   [Phase 3 Done] 耗时 {time.time() - t0:.1f}s")

    # Phase 4
    print(f"   [Phase 4] 写入磁盘...")
    write_outputs(file_contents)

    print(f"   [Shard Total] 耗时: {time.time() - shard_start:.1f}s")


# ===================== 5. 主程序 =====================

def main():
    args = parse_args()
    total_start = time.time()

    # --- 1. 扫描特征文件 ---
    print("🔍 正在扫描文件库...")
    npy_map = {
        os.path.splitext(f)[0]: os.path.join(r, f)
        for r, _, fs in os.walk(CFG.FEAT_ROOT)
        for f in fs if f.endswith('.npy')
    }
    pq_map = {
        os.path.splitext(f)[0]: os.path.join(r, f)
        for r, _, fs in os.walk(CFG.PQ_ROOT)
        for f in fs if f.endswith('.parquet')
    }

    all_keys = sorted(set(npy_map.keys()) & set(pq_map.keys()))
    sampled_keys = all_keys[
        int(len(all_keys) * args.feat_range[0]):int(len(all_keys) * args.feat_range[1])
    ]

    all_htmls = sorted(f for f in os.listdir(CFG.HTML_INPUT_DIR) if f.endswith('.html'))
    sampled_htmls = all_htmls[
        int(len(all_htmls) * args.html_range[0]):int(len(all_htmls) * args.html_range[1])
    ]

    print(f"📊 特征文件: {len(sampled_keys)} | HTML 文件: {len(sampled_htmls)}")
    print(f"⚙️  内部分片: {args.inner_shards} (特征和HTML同步切分，前10%对前10%)")

    # --- 2. 加载模型 ---
    from Embedding_sim_qwen3vl.qwen3_vl_embedding import Qwen3VLEmbedder
    model = Qwen3VLEmbedder(model_name_or_path=CFG.EMBEDDING_MODEL_PATH, device=CFG.DEVICE)

    # --- 3. 分片循环：特征和 HTML 一一对应切分 ---
    f_step = max(1, len(sampled_keys) // args.inner_shards)
    h_step = max(1, len(sampled_htmls) // args.inner_shards)

    for i in range(args.inner_shards):
        loop_start = time.time()

        # 对齐切分：第 i 片特征 对应 第 i 片 HTML
        f_s = i * f_step
        f_e = len(sampled_keys) if i == args.inner_shards - 1 else (i + 1) * f_step
        h_s = i * h_step
        h_e = len(sampled_htmls) if i == args.inner_shards - 1 else (i + 1) * h_step

        curr_keys = sampled_keys[f_s:f_e]
        curr_htmls_full = sampled_htmls[h_s:h_e]

        if not curr_keys or not curr_htmls_full:
            continue

        print(f"\n{'='*60}")
        print(f"🚀 [分片 {i + 1}/{args.inner_shards}]")
        print(f"   特征: [{f_s}, {f_e}) = {len(curr_keys)} files")
        print(f"   HTML:  [{h_s}, {h_e}) = {len(curr_htmls_full)} files")

        # 跳过已处理的 HTML
        exist_htmls = set(os.listdir(CFG.HTML_OUTPUT_DIR))
        curr_htmls = [f for f in curr_htmls_full if f not in exist_htmls]
        print(f"   去除已处理后剩余: {len(curr_htmls)}/{len(curr_htmls_full)} files")

        if not curr_htmls:
            print(f"   [Skip] 本分片全部已处理")
            continue

        # 3.1 构建索引
        engine = MiniSearchEngine(curr_keys, npy_map, pq_map)

        if engine.index.ntotal == 0:
            print(f"   [Skip] 索引为空，跳过")
            del engine
            continue

        # 3.2 处理
        process_shard(curr_htmls, model, engine, args)

        # 3.3 清理
        del engine
        gc.collect()
        if CFG.DEVICE == "npu":
            torch.npu.empty_cache()

        print(f"⏰ [分片 {i + 1}] 总耗时: {time.time() - loop_start:.1f}s")

    # --- 4. 上传结果 ---
    print("\n📤 上传结果到 OBS...")
    mox.file.copy_parallel(
        CFG.HTML_OUTPUT_DIR,
        CFG.HTML_OUTPUT_DIR.replace('/cache', 's3://bucket-green-huadong2-711/01.USERS/w00878018/data/')
    )

    print(f"\n✨ 全部完成! 总耗时: {(time.time() - total_start) / 60:.2f} 分钟")


if __name__ == "__main__":
    main()

# ===================== 启动命令示例 =====================
# python retrieval_pic_dist_new_260227.py --feat_range 0.0 0.10 --html_range 0.0 1.0 --inner_shards 10 --search_buffer 20

# 多卡并行 (每张卡各跑一段 feat_range + html_range):
# ASCEND_RT_VISIBLE_DEVICES=0 nohup python retrieval_pic_dist_260227.py --feat_range 0.0 0.02 --html_range 0.0 0.2 > log_27_0.txt 2>&1
# ASCEND_RT_VISIBLE_DEVICES=1 nohup python retrieval_pic_dist_260227.py --feat_range 0.02 0.04 --html_range 0.2 0.4 > log_27_1.txt 2>&1
# ASCEND_RT_VISIBLE_DEVICES=2 nohup python retrieval_pic_dist_260227.py --feat_range 0.04 0.06 --html_range 0.4 0.6 > log_27_2.txt 2>&1
# ASCEND_RT_VISIBLE_DEVICES=3 nohup python retrieval_pic_dist_260227.py --feat_range 0.06 0.08 --html_range 0.6 0.8 > log_27_3.txt 2>&1
# ASCEND_RT_VISIBLE_DEVICES=4 nohup python retrieval_pic_dist_260227.py --feat_range 0.08 0.10 --html_range 0.8 1 > log_27_4.txt 2>&1
# ...