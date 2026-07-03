"""
retriever.py — 在线检索（混合检索 + RRF 融合 + Cross-Encoder 精排）

────────────────────────────────────────────────────────────
  整体数据流
────────────────────────────────────────────────────────────
  用户问题 "什么是 HyDE？"
       │
       ├─→ ① 向量检索（BAAI/bge-m3 1024d）
       │      query ──bge-m3──→ 1024d 向量
       │      Chroma 里找 top-20 最相似的 chunk
       │
       ├─→ ② 关键词检索（BM25）
       │      query ──jieba 分词──→ 跟 109 个 chunk 算 TF-IDF 得分
       │      取 top-20
       │
       └─→ ③ RRF 融合
              两路各 top-20 → 按排名位置加权合并 → 取 top-10
              │
              └─→ ④ Cross-Encoder 精排（bge-reranker-v2-m3）
                    把 query + 每个候选 chunk 拼接后逐对打分
                    → 取 top-3 返回

  概念对照（Phase 2 #016 #017）
  ─────────────────────────────
  粗排 = ① + ② + ③（向量检索 + 关键词检索 + RRF 融合）
  精排 = ④（Cross-Encoder 逐对打分）
────────────────────────────────────────────────────────────
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from rank_bm25 import BM25Okapi
import jieba
from sentence_transformers import CrossEncoder

load_dotenv()

# 硅基流动客户端（复用 bge-m3 embedding，跟 indexer.py 同一配置）
client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
)

PERSIST_DIR = "chroma_store"      # indexer.py 建的 Chroma 数据目录
RRF_K = 60                        # RRF 公式里的 k 常数（行业默认 60）


# ═══════════════════════════════════════════════════════════════
#  ① BM25 关键词检索
# ═══════════════════════════════════════════════════════════════
#
#  BM25 是 TF-IDF 的现代变种，核心思想：
#    - 一个词在文档中出现越多次 → 得分越高（但会饱和，不会无限加分）
#    - 一个词在越少文档中出现 → 权重越高（"HyDE"只出现在 #019，权重极大）
#    - 完全不用 embedding，纯统计
#
#  为什么中文必须分词？
#    英文天然以空格分词     "What is RAG?" → ["What", "is", "RAG", "?"]
#    中文没有空格分隔       "什么是RAG？"   → jieba → ["什么", "是", "RAG", "？"]
#    不分词的后果：BM25 把整句话当成一个"词"，匹配率趋近零


def _tokenize(text: str) -> list[str]:
    """jieba 中文分词——把连续汉字切成有意义的词语"""
    return list(jieba.cut(text))


class Bm25Searcher:
    """
    BM25 关键词检索器

    工作流程：
      1. 初始化时：对全部 109 个 chunk 逐条分词 → 训练 BM25Okapi 索引
      2. 查询时：query 分词 → 跟每条 chunk 算 BM25 得分 → 排名 → 返回 top-K
    """

    def __init__(self, chunks: list[dict]):
        """
        chunks: [{"text": "文档内容...", "source": "011-what-is-rag.md"}, ...]

        建索引的三步：
          1. 提取纯文本列表     corpus = ["chunk0文本", "chunk1文本", ...]
          2. 逐条 jieba 分词    tokenized = [["chunk0", "的", "分词"], [...], ...]
          3. 训练 BM25 模型     BM25Okapi 内部计算每条文档的词频和 IDF
        """
        self.chunks = chunks
        self.corpus = [c["text"] for c in chunks]                # 109 个文本
        self.tokenized = [_tokenize(t) for t in self.corpus]     # 109 个分词列表
        self.bm25 = BM25Okapi(self.tokenized)                    # 训练 BM25 索引

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """
        关键词检索

        参数:
          query: 用户问题原文，如 "什么是 RAG？"
          top_k: 返回前多少条

        返回:
          [(chunk_index, bm25_score), ...]
          例: [(42, 4.32), (7, 2.15), ...]  ← 索引 42 的 chunk 得分最高

        步骤:
          1. jieba 切 query      "什么是 RAG？" → ["什么", "是", "RAG", "？"]
          2. 对每条 chunk 算分   BM25 内部：词频 × 逆文档频率
          3. 按得分降序排列      得分高的排前面
        """
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        # enumerate: 同时拿到"索引"和"得分"  [(0, 1.2), (1, 3.4), ...]
        # sorted 按得分降序
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])
        return ranked[:top_k]


# ═══════════════════════════════════════════════════════════════
#  ② 向量检索（Bi-Encoder）
# ═══════════════════════════════════════════════════════════════
#
#  Bi-Encoder = query 和 doc 分别嵌成向量，然后余弦比相似度
#  bge-m3 把任意文本映射到 1024 维空间——
#  语义相近的文本在这个空间里的向量也靠近


def _embed_query(query: str) -> list[float]:
    """
    把一条 query 嵌成 1024 维向量

    调用硅基流动 API（跟 indexer.py 建索引用的同一个模型 BAAI/bge-m3）
    为什么必须同一个模型？→ Day 6 hard truth：嵌入和检索用不同模型 = 向量空间不一致 = 检索全废
    """
    resp = client.embeddings.create(
        model="BAAI/bge-m3",
        input=[query],
    )
    return resp.data[0].embedding  # [0.12, -0.34, 0.56, ...] 1024 个 float


def _dense_search(query: str, collection: chromadb.Collection, top_k: int = 20
                  ) -> list[tuple[int, float]]:
    """
    向量检索——在 Chroma 里找跟 query 语义最接近的 top-K chunk

    参数:
      query: 用户问题
      collection: Chroma collection（已存 109 条 1024d 向量）
      top_k: 返回前多少条

    返回:
      [(chunk_index, distance), ...]  ← 注意：distance 越小越相似（L2 距离）
      例: [(42, 0.23), (7, 0.58), ...]

    步骤:
      1. 嵌 query 成 1024d 向量
      2. 调 Chroma.query() 在 109 条向量里找距离最近的 top-K
      3. 把 Chroma 返回的 id 字符串 → int（Chroma 里存的是 "0", "5", "12"...）
    """
    qv = _embed_query(query)                                    # 1024d 向量
    result = collection.query(
        query_embeddings=[qv],                                   # 拿嵌好的向量去搜
        n_results=min(top_k, collection.count()),                # 不会超过总条数
    )
    ids = result["ids"][0]                                       # ["0", "5", "12", ...]
    distances = result["distances"][0]                           # [0.23, 0.58, ...]
    return [(int(i), d) for i, d in zip(ids, distances)]


# ═══════════════════════════════════════════════════════════════
#  ③ RRF 融合
# ═══════════════════════════════════════════════════════════════
#
#  RRF 核心公式：score = 1 / (k + rank)
#
#  为什么用排名而不是原始分数？
#    向量检索返回的是"距离"（0.23），BM25 返回的是"相关性分"（4.32）
#    → 单位不同、量纲不同、不可直接相加
#    → RRF 绕开分数绝对值，只关心"排第几"——排名单位统一
#
#  为什么 k=60？
#    k 越大 → 排名靠后的文档被压得越狠，第 1 名和第 10 名的差距变小
#    k=60 是经验值，鲁棒性好，基本不需要调
#
#  示例（两个 chunk A 和 B 在两路里的排名）：
#    chunk A：dense 排第 2，sparse 排第 1 → rrf = 1/62 + 1/61 = 0.0161 + 0.0164 = 0.0325
#    chunk B：dense 排第 1，sparse 排第 5 → rrf = 1/61 + 1/65 = 0.0164 + 0.0154 = 0.0318
#    → A 最终得分更高，因为它在两路里都靠前


def _rrf_fusion(
    dense_ranked: list[tuple[int, float]],       # 向量检索结果
    sparse_ranked: list[tuple[int, float]],      # BM25 检索结果
    k: int = RRF_K,                               # RRF 常数
    top_k: int = 10,                              # 融合后保留多少条
) -> list[dict]:
    """
    RRF 融合：把两路"排名"合并成一个"得分"

    参数:
      dense_ranked:  [(chunk_idx, distance), ...]  ← 按距离排序（越近越前）
      sparse_ranked: [(chunk_idx, bm25_score), ...] ← 按 BM25 分排序（越高越前）
      k: RRF 公式常数
      top_k: 返回前多少条

    返回:
      [{"idx": chunk_index, "rrf_score": 综合得分}, ...]

    步骤:
      1. 遍历 dense 排名：位置 0（第 1 名）= 1/(k+0+1)，位置 1（第 2 名）= 1/(k+1+1)...
      2. 遍历 sparse 排名，同理累加
      3. 按综合得分降序排列
    """
    # scores 是 {chunk_idx: rrf_总分} 的字典
    scores: dict[int, float] = {}

    # 向量检索这路：排第几给几分
    for rank, (idx, _) in enumerate(dense_ranked):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)

    # BM25 这路：同样公式累加（同一 chunk 在两路都出现 = 两路得分相加）
    for rank, (idx, _) in enumerate(sparse_ranked):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)

    # 按 RRF 总分降序
    merged = sorted(scores.items(), key=lambda x: -x[1])
    return [{"idx": idx, "rrf_score": sc} for idx, sc in merged[:top_k]]


# ═══════════════════════════════════════════════════════════════
#  启动时一次性初始化（模块 import 时跑一次，后续复用）
# ═══════════════════════════════════════════════════════════════

# ① 加载 Chroma 持久化索引（indexer.py 建的那份）
_chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
_collection = _chroma_client.get_or_create_collection("tiny-rag")

# ② 把 109 条 chunk 拉出来，同时给 BM25 建索引
#    Chroma 的 get() 返回三个列表，同一索引位置对应同一条 chunk：
#      _all["ids"]       = ["0", "1", "2", ...]
#      _all["documents"]  = ["chunk文本0", "chunk文本1", ...]
#      _all["metadatas"]  = [{"source": "011-xxx.md"}, ...]
_all = _collection.get()
CHUNKS = [
    {
        "text": doc,
        "source": meta["source"] if meta else "?",
    }
    for doc, meta in zip(_all["documents"], _all["metadatas"])
]
_bm25 = Bm25Searcher(CHUNKS)      # 用 109 条 chunk 训练 BM25 索引

# ③ 精排模型——延迟加载（首次调 rerank() 才下载 ~2.27GB）
#    用 _get_reranker() 而不是模块级初始化，避免 import retriever 时就崩
_reranker = None

def _get_reranker():
    global _reranker
    if _reranker is None:
        model_path = os.path.expanduser(
            "~/.cache/modelscope/hub/models/BAAI/bge-reranker-v2-m3"
        )
        _reranker = CrossEncoder(model_path, trust_remote_code=True)
    return _reranker


# ═══════════════════════════════════════════════════════════════
#  对外接口 —— rag.py 调这两个函数就够
# ═══════════════════════════════════════════════════════════════

def search(query: str, top_k: int = 5) -> list[dict]:
    """
    混合检索入口（粗排）——向量 + BM25 + RRF 融合

    参数:
      query: 用户问题原文
      top_k: 返回前多少条（这些会交给 rerank 做精排，或直接喂给 LLM）

    返回:
      [{"text": "chunk内容", "source": "011-xxx.md", "score": 0.032}, ...]
      score 是 RRF 综合得分，越大越好

    内部流程:
      ① _dense_search → 向量 top-20
      ② _bm25.search   → 关键词 top-20
      ③ _rrf_fusion    → 融合两路 → top-K
    """
    # 两路各取 top-20（候选池放大，交给 RRF 融合后缩到 top_k）
    dense = _dense_search(query, _collection, top_k=20)
    sparse = _bm25.search(query, top_k=20)

    # RRF 融合两路排名
    merged = _rrf_fusion(dense, sparse, k=RRF_K, top_k=top_k)

    # 把 {idx, rrf_score} 转成 {text, source, score}
    results = []
    for m in merged:
        c = CHUNKS[m["idx"]]
        results.append({
            "text": c["text"],
            "source": c["source"],
            "score": round(m["rrf_score"], 3),
        })
    return results


def rerank(query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
    """
    Cross-Encoder 精排 —— bge-reranker-v2-m3 逐对打分

    跟粗排的本质区别（Cross-Encoder vs Bi-Encoder）：
      粗排（Bi-Encoder）：  query 嵌一下 / doc 嵌一下 / 余弦比一下 → 毫秒级
      精排（Cross-Encoder）：query+doc 拼一起喂给 transformer → 逐对跑 → 慢但准

    参数:
      query: 用户问题原文
      candidates: search() 返回的 top-K 候选 chunk
      top_k: 精排后保留几条

    返回:
      同 search() 格式，但 score 被替换为 cross-encoder 的相关性分
    """
    if not candidates:
        return []

    # 构造 (query, doc) 逐对列表
    # [("什么是HyDE？", "doc1文本"), ("什么是HyDE？", "doc2文本"), ...]
    pairs = [[query, c["text"]] for c in candidates]

    # 逐对打分
    scores = _get_reranker().predict(pairs)

    # 把精排分写回 candidates 的 score 字段
    for i, c in enumerate(candidates):
        c["score"] = round(float(scores[i]), 3)

    # 按精排分降序
    ranked = sorted(candidates, key=lambda x: -x["score"])
    return ranked[:top_k]


def search_with_expand(query: str, first_top_k: int = 10, second_top_k: int = 5):
    results_0 = search(query,first_top_k)
    results = rerank(query,results_0,second_top_k)
    sources = {result["source"] for result in results}
    last_result = [
        chunk
        for chunk in CHUNKS
        if chunk["source"] in sources
    ]
    return last_result
# ═══════════════════════════════════════════════════════════════
#  命令行调试入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "什么是 RAG？"
    print(f"🔍 查询: {q}\n")
    results_0 = search(q)
    print("-------------------------------------------- 粗排序 top5-------------------------------------------------------------------")
    for i, r in enumerate(results_0):
        print(f"{i+1}. [{r['score']:.3f}] {r['source']}")
        print(f"   {r['text'][:100]}...\n")
    results = rerank(q,results_0)
    sources = {result["source"] for result in results}
    last_result = [
        chunk
        for chunk in CHUNKS
        if chunk["source"] in sources
    ]
    print("-------------------------------------------- 精排序 top3-------------------------------------------------------------------")
    for i, r in enumerate(last_result):
        print(f"{i+1}. {r['source']}")
        print(f"   {r['text'][:100]}...\n")
