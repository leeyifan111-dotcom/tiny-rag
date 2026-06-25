# 🔎 tiny-rag

> 给 LLM 装上「长期记忆」的最小可用 RAG 系统
> Phase 2 配套项目，做中读 RAG 全栈

---

## 🎯 目标

- 跑通最小 RAG pipeline：**文档 → 分块 → embedding → 检索 → 生成**
- 用代码把 Phase 2 的 10 篇 RAG 文档（#011-#020）每一篇都落到至少一行代码
- 跟 [tiny-agent-loop](../tiny-agent-loop) 串联，最终拼成「Agentic RAG」（#018）

## 🗺️ 路线

| 版本 | 内容                                                                | 配套阅读        |
| ---- | ------------------------------------------------------------------- | --------------- |
| v0.1 | 最小可用：固定分块 + FAISS 内存版 + 单轮检索生成                    | #011-#015       |
| v0.2 | 混合检索（BM25 + 向量）+ Re-ranking                                 | #016-#017       |
| v0.3 | 加 JSON Mode eval（5 分量表）+ 高级查询变体（HyDE / Decomposition） | #018-#020       |
| v0.4 | 跟 tiny-agent-loop 集成 → Agentic RAG（LLM 自主决定要不要检索）     | #018 + 旗舰项目 |

## 📂 结构

```
tiny-rag/
├── docs/                # 测试语料（放 markdown / txt）
├── indexer.py           # 离线：文档 → chunks → embeddings → FAISS index
├── retriever.py         # 在线：query → embedding → top-K chunks
├── rag.py               # 主入口：retriever + LLM 生成
├── requirements.txt
├── .env.example         # 复制为 .env 并填入 API key
└── .gitignore
```

## ▶️ 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env       # 填入 DEEPSEEK_API_KEY
# 把你的 markdown 文档放进 docs/

python indexer.py          # 建索引
python rag.py "你的问题"   # 问答
```

## 🧠 设计原则

- **v0.1 故意选最简单的实现**（FAISS 内存版、固定分块）—— 跑通整个 pipeline 再优化，避免在某一环节钻太深
- **每个组件独立可测**——indexer / retriever / rag 各自能单独 import 跑
- **错误显式可见**——参考 tiny-agent-loop v0.4 学到的「软硬约束双轨」：错误返回结构化结果，不静默吞掉
