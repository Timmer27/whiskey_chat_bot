import os
import uuid
import pandas as pd
from tqdm import tqdm 
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

def get_qdrant_client():
    url = os.getenv("QDRANT_URL")
    if not url:
        raise ValueError("QDRANT_URL is missing")
    qdrant = QdrantClient(url=url)
    return qdrant

def reset_collection(client: QdrantClient, collection: str, vector_size: int):
    try:
        client.delete_collection(collection_name=collection)
    except Exception:
        pass

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

def stable_point_id(name: str, link: str = "") -> str:
    base = f"{(name or '').strip()}|{(link or '').strip()}"
    if base.strip("|").strip() == "":
        base = str(uuid.uuid4())
    return str(uuid.uuid5(uuid.NAMESPACE_URL, base))


def upsert_qdrant(
    client: QdrantClient,
    collection: str,
    df: pd.DataFrame,
    model: SentenceTransformer,
    batch_size: int,
) -> None:
    total = len(df)

    for start in tqdm(range(0, total, batch_size), desc=f"Upserting to Qdrant[{collection}]"):
        chunk = df.iloc[start : start + batch_size]

        texts = chunk["embed_text"].tolist()

        vectors = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        points = []
        for (_, row), vec, text in zip(chunk.iterrows(), vectors, texts):
            name = str(row.get("Whisky Name", "") or "").strip()
            link = str(row.get("Link", "") or "").strip()
            pid = stable_point_id(name, link)

            payload = {
                "whisky_name": name,
                "link": link,
                "text": text,
            }

            points.append(PointStruct(id=pid, vector=vec.tolist(), payload=payload))

        client.upsert(collection_name=collection, points=points)

    cnt = client.count(collection_name=collection, exact=True).count
    print(f"Done. collection='{collection}', points={cnt}")        
