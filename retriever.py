"""
retriever.py — 在线检索

职责：拿到 query → embed → 在 FAISS 里搜 top-K
v0.2 升级：加 BM25 混合检索（#016）+ Re-ranking（#017）

填写顺序：
- 读完 #014 + #015 → 实现 load_index() + search()
- 读完 #016 → 加 BM25 兜底
- 读完 #017 → 加 reranker
"""

from indexer import embed_text, INDEX_PATH


def load_index(index_path: str = INDEX_PATH):
    """加载已建好的 FAISS 索引 + chunk 元数据"""
    # TODO: 实现 FAISS index 反序列化 + chunks metadata 加载
    raise NotImplementedError("等读完 #014 向量数据库再填")


def search(query: str, top_k: int = 3) -> list[dict]:
    """检索：query → embedding → 在 index 里找 top-K 相似 chunks

    返回: [{"text": "...", "score": 0.8, "source": "docs/xxx.md"}, ...]
    """
    query_vec = embed_text(query)
    # TODO: 调 FAISS 的 search，拿到 top-K 索引 + 距离
    # TODO: 把索引映射回 chunks，组装返回结构
    raise NotImplementedError("等读完 #014 向量数据库再填")


def search_with_fallback(query: str, top_k: int = 3) -> list[dict]:
    """带兜底的检索（参考 tiny-agent-loop v0.4 学到的错误恢复思路）

    护栏：
    - 检索返回 0 条 → 降级为关键词检索（BM25）或扩大 top-K
    - embedding 超时 → 重试 / 切片重切 / 关键词兜底
    """
    # TODO: 等 v0.2 读完 #016 混合检索再实现
    return search(query, top_k)


if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "测试 query"
    results = search(query)
    for r in results:
        print(f"[{r['score']:.3f}] {r['source']}: {r['text'][:80]}...")
