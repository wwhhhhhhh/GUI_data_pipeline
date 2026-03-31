"""
LanceDB + TF-IDF 最简检索实现

流程：
  1. 用 sklearn TfidfVectorizer 把文本转成向量
  2. 存入 LanceDB（向量 + 原文）
  3. 查询时同样转向量，用 LanceDB 的向量检索找 top-k
"""

import lancedb
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# ============ 1. 准备数据 ============

docs = [
    "猫喜欢吃鱼，也喜欢晒太阳",
    "狗是人类最忠诚的朋友",
    "深度学习在自然语言处理中应用广泛",
    "TF-IDF 是一种经典的文本特征提取方法",
    "LanceDB 是一个轻量级向量数据库",
    "Python 是最流行的编程语言之一",
    "机器学习模型需要大量数据进行训练",
    "猫和狗都是常见的宠物动物",
]

# ============ 2. TF-IDF 向量化 ============

# 用字级别分词（中文友好，不依赖jieba）
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3))
tfidf_matrix = vectorizer.fit_transform(docs)  # sparse matrix
vectors = tfidf_matrix.toarray().astype(np.float32)  # 转 dense

print(f"文档数: {len(docs)}, 向量维度: {vectors.shape[1]}")

# ============ 3. 存入 LanceDB ============

db = lancedb.connect("/tmp/tfidf_demo")

data = [
    {"id": i, "text": doc, "vector": vec}
    for i, (doc, vec) in enumerate(zip(docs, vectors))
]

# 建表（如已存在则覆盖）
table = db.create_table("docs", data, mode="overwrite")
print(f"已存入 {table.count_rows()} 条记录")

# ============ 4. 查询 ============

def search(query: str, top_k: int = 3):
    """TF-IDF 向量化 query → LanceDB 向量检索"""
    q_vec = vectorizer.transform([query]).toarray().astype(np.float32)[0]
    results = table.search(q_vec).limit(top_k).to_pandas()
    return results

# 测试
queries = ["猫和鱼", "编程语言", "向量数据库", "机器学习训练数据"]

for q in queries:
    print(f"\n🔍 查询: {q}")
    results = search(q, top_k=3)
    for _, row in results.iterrows():
        print(f"   [{row['_distance']:.4f}] {row['text']}")