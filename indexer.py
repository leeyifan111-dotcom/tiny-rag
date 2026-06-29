"""
indexer.py — 离线建索引

职责：把 docs/ 下的文档变成可检索的向量索引
流程：load → chunk → embed → store

填写顺序（按 Phase 2 阅读节奏）：
- 读完 #013 分块策略 → 实现 chunk_text()
- 读完 #014 向量数据库 → 决定用 FAISS 还是别的，实现 store
- 读完 #015 Embedding 模型 → 选模型，实现 embed_text()
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
)

DOCS_DIR = "docs"
PERSIST_DIR = "chroma_store"


def load_documents(docs_dir: str = DOCS_DIR) -> list[dict]:
    """读取 docs/ 下所有 .md / .txt 文件，返回 [{path, text}, ...]"""
    docs = []
    for filename in os.listdir(docs_dir):
        if not filename.endswith((".md", ".txt")):
            continue
        filepath = os.path.join(docs_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append({"path": filename, "text": text})
    return docs


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """固定长度分块，尽可能在句子边界切（v0.1，v0.2 升级递归/语义分块）

    做法：按句号切 → 逐句累积 → 接近 chunk_size 就截断 → overlap 粘合衔接
    """
    sentences = text.replace("\n", " ").split("。")
    chunks = []
    current = ""

    for s in sentences:
        if not s.strip():
            continue
        seg = s + "。"

        if len(current) + len(seg) <= chunk_size:
            current += seg
        else:
            if current.strip():
                chunks.append(current.strip())
            current = current[-overlap:] + seg if current else seg

    if current.strip():
        chunks.append(current.strip())

    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量调用 DeepSeek embedding API，返回 1024 维向量列表"""
    resp = client.embeddings.create(
        model="BAAI/bge-m3",
        input=texts,
    )
    return [d.embedding for d in resp.data]


def build_index(docs_dir: str = DOCS_DIR, persist_dir: str = PERSIST_DIR):
    """主流程：load → chunk → embed（DeepSeek API） → store（Chroma）

    v0.1: DeepSeek embedding（1024 维，中文友好，复用已有 key）。
    v0.2: 可选换 bge-m3 本地模型，降低 API 成本。
    """
    docs = load_documents(docs_dir)
    print(f"📂 加载 {len(docs)} 个文档")

    all_chunks = []
    for doc in docs:
        chunks = chunk_text(doc["text"])
        for chunk in chunks:
            all_chunks.append({"text": chunk, "source": doc["path"]})
    print(f"✂️  切成 {len(all_chunks)} 个 chunks")

    # 批量 embedding（DeepSeek 一次最多 32 条，这里 109 条分 4 批）
    BATCH = 32
    texts = [c["text"] for c in all_chunks]
    all_vectors = []
    print(f"🧮 正在嵌入 {len(texts)} 个 chunks（model: deepseek-embedding）...")
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        vectors = embed_texts(batch)
        all_vectors.extend(vectors)
        print(f"   {min(i + BATCH, len(texts))}/{len(texts)}")

    # 持久化客户端
    client_db = chromadb.PersistentClient(path=persist_dir)
    collection = client_db.get_or_create_collection("tiny-rag")

    # 清空旧数据
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    # 写入：预先算好的 embeddings（不用 Chroma 内置英文模型）
    ids = [str(i) for i in range(len(all_chunks))]
    metadatas = [{"source": c["source"]} for c in all_chunks]

    collection.add(
        ids=ids,
        embeddings=all_vectors,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"💾 索引已保存到 {persist_dir}/（{collection.count()} 条，维度 1024）")


if __name__ == "__main__":
    build_index()
