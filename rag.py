"""
rag.py — RAG 主入口

职责：用户提问 → 检索 top-K chunks → 拼 prompt → LLM 生成
v0.3 加：JSON Mode 给答案打分（#020 + #061）

填写顺序：
- 读完 #011 + #012 → 实现 ask() 主流程
- 读完 #020 → 加 evaluate() 给答案打分
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

from retriever import search_with_expand

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


def ask(question: str, first_top_k: int = 10, second_top_k: int = 5) -> str:
    """RAG 主流程：检索 → 拼 prompt → 生成"""
    # 1. 检索相关 chunks
    chunks = search_with_expand(question, first_top_k, second_top_k)

    if not chunks:
        return "参考资料中未包含相关内容"

    print(f"获取到{len(chunks)}个chunk")

    # 2. 拼接上下文（最多 6000 字符，防爆上下文窗口）
    context = ""
    total = 0
    for c in chunks:
        line = f"[来源: {c['source']}]\n{c['text']}\n\n"
        if total + len(line) > 6000:
            break
        context += line
        total += len(line)

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
    """用 JSON Mode 评估答案的 faithfulness + relevance（#020 + #061）

    返回: {"faithfulness": 1-5, "relevance": 1-5, "reason": "..."}

    faithfulness: 答案是不是基于资料写的，有没有编造？
    relevance:   资料跟用户问题有没有关系？
    """
    eval_prompt = f"""你是一个 RAG 质量评估专家。根据以下信息，用 1-5 分评估生成的答案。

用户问题：
{question}

参考资料（最多 5 条）：
{chr(10).join(f"- {c[:300]}" for c in contexts[:5])}

生成的答案：
{answer[:500]}

输出 JSON：{{"faithfulness": 1-5, "relevance": 1-5, "reason": "简述依据"}}

- faithfulness：1=完全编造，3=部分有据，5=严格基于资料
- relevance：1=资料跟问题无关，3=部分相关，5=高度匹配"""

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": eval_prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


if __name__ == "__main__":
    import sys
    question = sys.argv[1] if len(sys.argv) > 1 else "请先在命令行传入问题"
    print(f"\n❓ 问题: {question}\n")
    answer = ask(question)
    print(f"\n💬 回答:\n{answer}\n")

    # 评估
    chunks = search_with_expand(question)
    if chunks:
        texts = [c["text"] for c in chunks]
        result = evaluate(question, answer, texts)
        print(f"📊 评估: faithfulness={result['faithfulness']}/5, relevance={result['relevance']}/5")
        print(f"   理由: {result['reason']}")
