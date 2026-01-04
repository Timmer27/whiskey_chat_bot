from sentence_transformers import SentenceTransformer

def get_embeddings(embed_model: str) -> SentenceTransformer:
    embedder = SentenceTransformer(embed_model)
    return embedder