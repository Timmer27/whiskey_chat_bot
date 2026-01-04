import os
from qdrant_client import QdrantClient

def get_qdrant_client():
    url = os.getenv("QDRANT_URL")
    if not url:
        raise ValueError("QDRANT_URL is missing")
    qdrant = QdrantClient(url=url)
    return qdrant