import os
import pandas as pd
import argparse
from llm.embeddings import process_embedding_data
from db.qdrant import QdrantClient, upsert_qdrant, reset_collection
from sentence_transformers import SentenceTransformer

def main():
    DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    DEFAULT_BATCH_SIZE = int(os.getenv("BATCH_SIZE", "128"))

    p = argparse.ArgumentParser(description="Reset Qdrant collection and ingest whisky CSV embeddings.")
    p.add_argument("--csv", default="data/whisky_reviews.csv", help="Path to CSV file")
    p.add_argument("--collections", default="whisky_reviews", help="Qdrant collection name")
    p.add_argument("--reset", action="store_true", help="If set, delete & recreate the collection")
    p.add_argument("--reset_only", action="store_true", help="Reset collection and exit")
    p.add_argument("--qdrant_url", default=DEFAULT_QDRANT_URL, help="Qdrant URL")
    p.add_argument("--embed_model", default=DEFAULT_EMBED_MODEL, help="SentenceTransformer model name/path")
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Upsert batch size")
    args = p.parse_args()

    client = QdrantClient(url=args.qdrant_url)
    model = SentenceTransformer(args.embed_model)
    vector_size = model.get_sentence_embedding_dimension()

    if args.reset_only:
        reset_collection(client, args.collections, vector_size)
        print(f"Reset done. collection='{args.collections}'")
        return

    if args.reset:
        reset_collection(client, args.collections, vector_size)

    df = pd.read_csv(args.csv)
    process_df = process_embedding_data(df)

    upsert_qdrant(
        client=client,
        collection=args.collections,
        df=process_df,
        model=model,
        batch_size=args.batch_size,
    )

    cnt = client.count(collection_name=args.collections, exact=True).count
    print(f"Done. collection='{args.collections}', points={cnt}")


if __name__ == "__main__":
    main()
