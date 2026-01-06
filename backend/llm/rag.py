import os
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple

def _build_prompt(
    question: str,
    contexts: List[Dict[str, Any]],
) -> str:
    # 컨텍스트 블록 만들기(지금 형태 유지)
    blocks = []
    for i, r in enumerate(contexts[:8], start=1):
        name = r.get("whisky_name", "") or ""
        score = float(r.get("score", 0.0) or 0.0)
        text = (r.get("text") or "").strip()
        tags = (r.get("tags") or "").strip()
        link = (r.get("link") or "").strip()

        ctx = text
        if tags:
            ctx = ctx + f"\nTags: {tags}"

        source = link if link else ""
        blocks.append(f"[{i}] {name} (score={score:.3f})\n{ctx}\nSource: {source}")

    ctx_text = "\n\n".join(blocks)

    return f"""
당신은 위스키를 전문적으로 추천 및 분석하는 AI 어시스턴트입니다.
[중요 규칙 - 반드시 준수]
- 아래 [컨텍스트]에 등장하는 정보(위스키명/향미/특징)만 사용하세요.
- 컨텍스트에 없는 위스키명, 향미, 숙성/캐스크/연도 등의 추측은 절대 금지합니다.
- 응답은 사용자의 질문에 맞게 위스키를 추천 혹은 위스키의 설명을 해야합니다.
- 답변은 질문과 동일한 언어로 자연스럽게 작성합니다.
- 길게 구조화된 목록/설명(장문, 과한 불릿)은 금지합니다.
- 사용자가 맛, 향, 피니쉬 등을 제공하면, 해당 특정 기준을 사용해서 추천을 다각화해야합니다.
- 응답은 간결하고 확실한 정보를 통해 사용자의 선호도에 맞는 위스키를 최대 2개, 2~3 줄 이내로 추천해야합니다.
- 추천된 위스키에는 컨테스트를 기반으로 상세한 설명이 뒷받침되어야 합니다.
- 질문이 "특정 위스키 설명"이면 주어진 컨텍스트의 정보를 토대로 해당 설명을 주어진 답변에 따라 전문적으로 답변합니다.
- 답변이 나올 떄 사용자가 질문한 언어에 맞게 적절하게 컨텍스트도 해당 언어와 동일하게 자연스럽게 번역되어 답변하여야 합니다.
<question>
{question}
</question>

<context>
{ctx_text}
</context>
"""


def retrieve(qdrant: QdrantClient, embedder: SentenceTransformer, question: str, top_k: int = 20) -> List[Dict[str, Any]]:
    collection_name = os.getenv("QDRANT_COLLECTION")
    qv = embedder.encode(question, normalize_embeddings=True).tolist()

    hits = qdrant.query_points(
        collection_name=collection_name,
        query=qv,
        limit=top_k,
        with_payload=True,
    )    

    results = []
    for p in hits.points:
        payload = p.payload or {}
        results.append({
            "score": float(p.score),
            "whisky_name": payload.get("whisky_name", ""),
            "link": payload.get("link", ""),
            "text": payload.get("text", ""),
            "tags": payload.get("tags", ""),
        })        
    return results

def ctxs_to_strings(ctxs: List[Dict[str, Any]], max_ctx: int = 8) -> List[str]:
    out = []
    for r in (ctxs or [])[:max_ctx]:
        name = (r.get("whisky_name") or "").strip()
        text = (r.get("text") or "").strip()
        tags = (r.get("tags") or "").strip()
        link = (r.get("link") or "").strip()
        score = r.get("score", None)

        block = ""
        if name:
            block += f"Whisky: {name}\n"
        if score is not None:
            block += f"Score: {float(score):.4f}\n"
        if text:
            block += text + "\n"
        if tags:
            block += f"Tags: {tags}\n"
        if link:
            block += f"Source: {link}\n"

        block = block.strip()
        if block:
            out.append(block)
    return out

def generate_answer(
    qdrant: QdrantClient,
    embedder: SentenceTransformer,
    llm: OpenAI,
    question: str,
    model: str = "llama-3.1-8b-instant"
) -> tuple[str, list[dict], list[str]]:
    ctxs = retrieve(qdrant=qdrant, embedder=embedder, question=question)
    prompt = _build_prompt(question, ctxs)
    prompt = "당신은 위스키를 전문적으로 추천 및 분석하는 AI 어시스턴트입니다."

    resp = llm.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )
    answer = resp.choices[0].message.content or ""

    ctx_texts = ctxs_to_strings(ctxs, max_ctx=8)
    return answer, ctxs, ctx_texts
