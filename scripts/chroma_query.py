import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Query a ChromaDB collection (local embeddings).")
    ap.add_argument("--chroma-dir", default="out/chroma_db", help="Persistent Chroma directory")
    ap.add_argument("--collection", default="phis_book", help="Collection name")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base", help="SentenceTransformer model id")
    ap.add_argument("--q", required=True, help="Query text")
    ap.add_argument("--k", type=int, default=5, help="Top K results")
    args = ap.parse_args()

    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(path=str(Path(args.chroma_dir)), settings=Settings(anonymized_telemetry=False))
    col = client.get_collection(args.collection)

    embed_model = SentenceTransformer(args.model)
    q_emb = embed_model.encode([f"query: {args.q}"], normalize_embeddings=True).tolist()

    # Chroma `include` doesn't accept `ids` (ids are always returned).
    res = col.query(query_embeddings=q_emb, n_results=args.k, include=["documents", "metadatas", "distances"])
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    for i in range(len(ids)):
        meta = metas[i] or {}
        page = meta.get("page")
        print(f"\n#{i+1} id={ids[i]} page={page} dist={dists[i]:.4f}")
        # strip "passage: "
        doc = docs[i] or ""
        if doc.startswith("passage: "):
            doc = doc[len("passage: ") :]
        print(doc[:800].strip())


if __name__ == "__main__":
    main()
