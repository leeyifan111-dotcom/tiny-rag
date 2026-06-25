"""
rag.py — RAG 主入口

职责：用户提问 → 检索 top-K chunks → 拼 prompt → LLM 生成
v0.3 加：JSON Mode 给答案打分（#020 + #061）

填写顺序：
- 读完 #011 + #012 → 实现 ask() 主流程
- 读完 #020 → 加 evaluate() 给答案打分
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

from retriever import search_with_fallback

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


SYSTEM_PROMPT = """你是一个基于检索的问答助手。

# 回答规则
- 只能基于"参考资料"回答问题
- 如果参考资料里没有相关信息，明确说"参考资料中未包含相关内容"
- 禁止用内置知识补充资料里没有的实体名称
- 引用资料原文时标注来源

# 输出格式
答案: ...
来源: docs/xxx.md
"""


def ask(question: str, top_k: int = 3) -> str:
    """RAG 主流程：检索 → 拼 prompt → 生成"""
    # 1. 检索相关 chunks
    chunks = search_with_fallback(question, top_k=top_k)

    if not chunks:
        return "参考资料中未包含相关内容"

    # 2. 拼接上下文
    context = "\n\n".join(
        f"[来源: {c['source']}]\n{c['text']}" for c in chunks
    )

    # 3. 喂给 LLM 生成
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"参考资料:\n{context}\n\n问题: {question}"},
        ],
    )
    return resp.choices[0].message.content


def evaluate(question: str, answer: str, contexts: list[str]) -> dict:
    """评估答案质量（#020 RAG 评估指标 + #061 JSON Mode）

    返回: {"faithfulness": 1-5, "relevance": 1-5, "reason": "..."}
    """
    # TODO: 等读完 #020 + #061 再实现
    raise NotImplementedError("等读完 #020 RAG 评估再填")


if __name__ == "__main__":
    import sys
    question = sys.argv[1] if len(sys.argv) > 1 else "请先在命令行传入问题"
    print(f"\n❓ 问题: {question}\n")
    answer = ask(question)
    print(f"\n💬 回答:\n{answer}\n")
