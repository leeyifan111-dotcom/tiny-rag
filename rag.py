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


def ask(question: str, first_top_k: int = 10, second_top_k: int = 5
        ) -> tuple[str, list[dict]]:
    """RAG 主流程：检索 → 拼 prompt → 生成

    返回: (答案文本, 使用的 chunk 列表)——chunk 供 evaluate() 复用，保证同源
    """
    # 1. 检索相关 chunks
    chunks = search_with_expand(question, first_top_k, second_top_k)

    if not chunks:
        return "参考资料中未包含相关内容", []

    print(f"获取到{len(chunks)}个chunk")

    # 2. 拼接上下文（最多 6000 字符，防爆上下文窗口）
    used = []
    context = ""
    total = 0
    for c in chunks:
        line = f"[来源: {c['source']}]\n{c['text']}\n\n"
        if total + len(line) > 6000:
            break
        context += line
        used.append(c)
        total += len(line)

    # 3. 喂给 LLM 生成
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"参考资料:\n{context}\n\n问题: {question}"},
        ],
    )
    return resp.choices[0].message.content, used


def evaluate(question: str, answer: str, contexts: list[str]) -> dict:
    """RAGAS 风格评估：faithfulness（claim 检验）+ relevance（反问匹配）

    返回: {"faithfulness": 0.0~1.0, "answer_relevance": 0.0~1.0}

    ── Faithfulness (#020) ──────────────────────────────
    不是让 LLM 直接打 1-5 分，而是：
      1. 把答案拆成多个 claim（断言句）
      2. 逐条判读：这个 claim 在参考资料中有依据吗？
      3. faithfulness = 有依据的 claim 数 / 总 claim 数

    ── Answer Relevance (#020) ──────────────────────────
      1. 基于答案让 LLM 生成 3 个反问句（"这个答案回答了哪些问题？"）
      2. 把每个反问句嵌成向量，算它们跟原 question 向量的平均余弦相似度
      3. answer_relevance = mean(cos(q_vec, rq_vec) for rq in reverse_questions)
    """
    # ── Faithfulness：拆 claim → 逐条判 ──
    claims_prompt = f"""把以下答案拆成多个独立的断言句（claim）。每行一个，不要编号。
如果答案说"我不确定"或"不知道"，直接返回空。

答案：
{answer}"""

    claims_resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": claims_prompt}],
    )
    claims_text = claims_resp.choices[0].message.content.strip()
    claims = [c.strip() for c in claims_text.split("\n") if c.strip()]

    if not claims:
        return {"faithfulness": 0.0, "answer_relevance": 0.0}

    # 逐条判：这条 claim 有没有被 context 支撑？
    context_block = "\n".join(c[:500] for c in contexts[:5])
    supported = 0
    for claim in claims:
        verdict_prompt = f"""参考资料：
{context_block}

断言：{claim}

这个断言是否完全能被参考资料证实？输出 JSON：{{"supported": true/false}}"""
        v_resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": verdict_prompt}],
            response_format={"type": "json_object"},
        )
        if json.loads(v_resp.choices[0].message.content).get("supported", False):
            supported += 1

    faithfulness = supported / len(claims)

    # ── Answer Relevance：反问匹配 ──
    reverse_prompt = f"""基于以下答案，生成 3 个不同角度的问题，这些问题是这个答案在回答的。
输出 JSON：{{"questions": ["问题1", "问题2", "问题3"]}}

答案：
{answer}"""

    rev_resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": reverse_prompt}],
        response_format={"type": "json_object"},
    )
    reverse_questions = json.loads(rev_resp.choices[0].message.content)["questions"]

    # 向量余 xian 相似度（复用 retriever 的 bge-m3 embedding）
    from retriever import _embed_query
    q_vec = _embed_query(question)
    similarities = []
    for rq in reverse_questions:
        rq_vec = _embed_query(rq)
        dot = sum(a * b for a, b in zip(q_vec, rq_vec))
        similarities.append(dot)  # bge-m3 返回归一化向量，点积 = 余弦相似度
    answer_relevance = sum(similarities) / len(similarities)

    return {
        "faithfulness": round(faithfulness, 3),
        "answer_relevance": round(answer_relevance, 3),
    }


if __name__ == "__main__":
    import sys
    question = sys.argv[1] if len(sys.argv) > 1 else "请先在命令行传入问题"
    print(f"\n❓ 问题: {question}\n")
    answer, chunks = ask(question)
    print(f"\n💬 回答:\n{answer}\n")

    # 评估——用 ask() 返回的同批 chunk，保证上下文一致
    if chunks:
        texts = [c["text"] for c in chunks]
        result = evaluate(question, answer, texts)
        print(f"📊 faithfulness={result['faithfulness']:.0%}, answer_relevance={result['answer_relevance']:.0%}")

