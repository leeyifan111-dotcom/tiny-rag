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

load_dotenv()

DOCS_DIR = "docs"
INDEX_PATH = "tiny.index"


def load_documents(docs_dir: str = DOCS_DIR) -> list[dict]:
    """读取 docs/ 下所有 .md / .txt 文件，返回 [{path, text}, ...]"""
    # TODO: 实现文件遍历 + 内容读取
    raise NotImplementedError("等读完 #012 RAG Pipeline 组件再填")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """把长文本切成有重叠的 chunks（v0.1 用固定长度，v0.2 升级语义分块）"""
    # TODO: 实现固定长度切分 + overlap
    raise NotImplementedError("等读完 #013 分块策略再填")


def embed_text(text: str) -> list[float]:
    """调用 embedding API，把文本变成向量"""
    # TODO: 实现 DeepSeek / OpenAI embedding 调用
    raise NotImplementedError("等读完 #015 Embedding 模型再填")


def build_index(docs_dir: str = DOCS_DIR, index_path: str = INDEX_PATH):
    """主流程：load → chunk → embed → store"""
    docs = load_documents(docs_dir)
    print(f"📂 加载 {len(docs)} 个文档")

    all_chunks = []
    for doc in docs:
        chunks = chunk_text(doc["text"])
        for chunk in chunks:
            all_chunks.append({"text": chunk, "source": doc["path"]})
    print(f"✂️  切成 {len(all_chunks)} 个 chunks")

    # TODO: 批量 embed + 存 FAISS
    raise NotImplementedError("等读完 #014 向量数据库选型再填")


if __name__ == "__main__":
    build_index()
