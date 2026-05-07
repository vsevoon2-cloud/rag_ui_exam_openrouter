import argparse
import json
import os
from pathlib import Path
from typing import Any

import requests

from clean_book_md import clean_text
from chroma_build import chunk_text, iter_page_files


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def openrouter_chat(api_key: str, model: str, messages: list[dict[str, Any]], timeout_s: int) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": 900}
    r = requests.post(OPENROUTER_URL, headers=headers, data=json.dumps(payload), timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()


def tokenize(text: str) -> list[str]:
    import re

    return [token.lower() for token in re.findall(r"[A-Za-zА-Яа-яЁё0-9]{2,}", text)]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description="Grounded RAG answer: retrieve from ChromaDB, answer via OpenRouter.")
    ap.add_argument("--chroma-dir", default="out/chroma_db")
    ap.add_argument("--collection", default="phis_book")
    ap.add_argument("--embed-model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--llm-model", default="google/gemini-2.0-flash-001")
    ap.add_argument("--retrieval", choices=["vector", "hybrid"], default="vector")
    ap.add_argument("--pages-dir", default="out/or_md_flash2/pages")
    ap.add_argument("--target-chars", type=int, default=2000)
    ap.add_argument("--overlap-chars", type=int, default=250)
    ap.add_argument("--q", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing OPENROUTER_API_KEY env var.")

    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(path=str(Path(args.chroma_dir)), settings=Settings(anonymized_telemetry=False))
    col = client.get_collection(args.collection)
    embed_model = SentenceTransformer(args.embed_model)

    q_emb = embed_model.encode([f"query: {args.q}"], normalize_embeddings=True).tolist()
    # Chroma `include` doesn't accept `ids` (ids are always returned).
    res = col.query(query_embeddings=q_emb, n_results=max(args.k * 3, 20), include=["documents", "metadatas", "distances"])

    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    results: list[dict[str, Any]] = []
    if args.retrieval == "vector":
        for i in range(min(len(ids), args.k)):
            meta = metas[i] or {}
            doc = docs[i] or ""
            if doc.startswith("passage: "):
                doc = doc[len("passage: ") :]
            results.append(
                {
                    "id": ids[i],
                    "page": meta.get("page"),
                    "distance": float(dists[i]),
                    "text": doc,
                }
            )
    else:
        from rank_bm25 import BM25Okapi

        candidates: list[dict[str, Any]] = []
        for page_no, md_path in iter_page_files(Path(args.pages_dir)):
            raw = md_path.read_text(encoding="utf-8", errors="replace")
            text = clean_text(raw, strict=True).strip()
            if not text:
                continue
            parts = chunk_text(text, target_chars=args.target_chars, overlap_chars=args.overlap_chars)
            for index, part in enumerate(parts, start=1):
                candidates.append({"id": f"p{page_no:04d}_c{index:03d}", "page": page_no, "text": part})

        bm25 = BM25Okapi([tokenize(item["text"]) for item in candidates])
        bm25_scores = bm25.get_scores(tokenize(args.q))
        bm25_ids = [
            candidates[index]["id"]
            for index in sorted(range(len(candidates)), key=lambda idx: bm25_scores[idx], reverse=True)[: max(args.k * 3, 20)]
        ]

        by_id: dict[str, dict[str, Any]] = {item["id"]: item for item in candidates}
        for i in range(len(ids)):
            doc = docs[i] or ""
            if doc.startswith("passage: "):
                doc = doc[len("passage: ") :]
            by_id[ids[i]] = {
                "id": ids[i],
                "page": int((metas[i] or {}).get("page") or 0),
                "distance": float(dists[i]),
                "text": doc,
            }

        fused = reciprocal_rank_fusion([bm25_ids, ids])
        final_ids = [doc_id for doc_id, _score in sorted(fused.items(), key=lambda item: item[1], reverse=True)[: args.k]]
        for chunk_id in final_ids:
            item = by_id[chunk_id]
            results.append(
                {
                    "id": chunk_id,
                    "page": item.get("page"),
                    "distance": float(item.get("distance", 999.0)),
                    "text": item["text"],
                }
            )

    ctx_lines = []
    for i, item in enumerate(results, start=1):
        ctx_lines.append(
            f"[source {i+1} | page {item['page']} | id {item['id']} | dist {item['distance']:.4f}]\n{item['text']}"
        )

    context = "\n\n---\n\n".join(ctx_lines)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful assistant. Answer ONLY using the provided sources. "
                "If the sources are insufficient, say you don't know. "
                "Cite sources by page number like (стр. 123)."
            ),
        },
        {"role": "user", "content": f"Question: {args.q}\n\nSources:\n{context}"},
    ]

    answer = openrouter_chat(api_key, args.llm_model, messages, timeout_s=args.timeout)
    print(answer)


if __name__ == "__main__":
    main()
