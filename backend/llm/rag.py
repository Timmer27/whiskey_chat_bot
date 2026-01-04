import os
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

def _build_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, r in enumerate(contexts[:8], start=1):
        name = r.get("whisky_name", "") or ""
        score = float(r.get("score", 0.0) or 0.0)
        text = (r.get("text") or "").strip()
        tags = (r.get("tags") or "").strip()
        link = (r.get("link") or "").strip()

        # text + tags를 한 블록으로
        ctx = text
        if tags:
            ctx = ctx + f"\nTags: {tags}"

        source = link if link else ""

        blocks.append(
            f"[{i}] {name} (score={score:.3f})\n{ctx}\nSource: {source}"
        )

    ctx_text = "\n\n".join(blocks)
    return f"""당신은 위스키 추천/설명 전문가입니다.
아래 컨텍스트(리뷰 발췌)만 근거로 답하세요. 컨텍스트에 없는 추측은 금지합니다.
사용자가 질문한 언어와 같은 언어로 답변하세요.

[사용자 질문]
{question}

[컨텍스트]
{ctx_text}

[답변 지침]
- 먼저 사용자의 취향/요구를 한 문장으로 재정의
- 추천/설명은 3~5개 (각 항목마다 근거: [번호] 1개 이상)
- 가능하면 'Nose/Taste/Finish'를 구분해 설명
- 마지막에 '추가로 물어볼 질문' 2개
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

def generate_answer(
    qdrant: QdrantClient,
    embedder: SentenceTransformer,
    llm: OpenAI,
    question: str,
    model: str = "llama-3.1-8b-instant"
) -> str:
    results = retrieve(qdrant=qdrant, embedder=embedder, question=question)

    prompt = _build_prompt(question, results)

    resp = llm.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )
    return resp.choices[0].message.content