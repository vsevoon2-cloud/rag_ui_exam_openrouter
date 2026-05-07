import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clean_book_md import clean_text
from chroma_build import chunk_text, iter_page_files


@dataclass
class Candidate:
    chunk_id: str
    page: int
    text: str


def tokenize(text: str) -> list[str]:
    import re

    return [token.lower() for token in re.findall(r"[A-Za-zА-Яа-яЁё0-9]{2,}", text)]


def build_candidates(pages_dir: Path, target_chars: int, overlap_chars: int) -> list[Candidate]:
    candidates: list[Candidate] = []
    for page_no, md_path in iter_page_files(pages_dir):
        raw = md_path.read_text(encoding="utf-8", errors="replace")
        text = clean_text(raw, strict=True).strip()
        if not text:
            continue
        parts = chunk_text(text, target_chars=target_chars, overlap_chars=overlap_chars)
        for index, part in enumerate(parts, start=1):
            candidates.append(Candidate(chunk_id=f"p{page_no:04d}_c{index:03d}", page=page_no, text=part))
    return candidates


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid BM25 + vector query over cleaned textbook chunks.")
    ap.add_argument("--pages-dir", default="out/or_md_flash2/pages")
    ap.add_argument("--chroma-dir", default="out/chroma_db_v2")
    ap.add_argument("--collection", default="phis_book_v2")
    ap.add_argument("--embed-model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--q", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--target-chars", type=int, default=2000)
    ap.add_argument("--overlap-chars", type=int, default=250)
    args = ap.parse_args()

    import chromadb
    from chromadb.config import Settings
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

    pages_dir = Path(args.pages_dir)
    candidates = build_candidates(pages_dir, args.target_chars, args.overlap_chars)
    if not candidates:
        raise SystemExit(f"No candidates built from {pages_dir}")

    tokenized_corpus = [tokenize(candidate.text) for candidate in candidates]
    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = tokenize(args.q)
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_ranked = [
        candidates[index].chunk_id
        for index in sorted(range(len(candidates)), key=lambda idx: bm25_scores[idx], reverse=True)[: max(args.k * 3, 20)]
    ]

    client = chromadb.PersistentClient(path=str(Path(args.chroma_dir)), settings=Settings(anonymized_telemetry=False))
    col = client.get_collection(args.collection)
    embedder = SentenceTransformer(args.embed_model)
    q_emb = embedder.encode([f"query: {args.q}"], normalize_embeddings=True).tolist()
    vec = col.query(query_embeddings=q_emb, n_results=max(args.k * 3, 20), include=["documents", "metadatas", "distances"])

    vec_ids = vec.get("ids", [[]])[0]
    vec_docs = vec.get("documents", [[]])[0]
    vec_metas = vec.get("metadatas", [[]])[0]
    vec_dists = vec.get("distances", [[]])[0]

    by_id: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        by_id[candidate.chunk_id] = {"page": candidate.page, "text": candidate.text}
    for idx, chunk_id in enumerate(vec_ids):
        text = vec_docs[idx] or ""
        if text.startswith("passage: "):
            text = text[len("passage: ") :]
        by_id[chunk_id] = {
            "page": int((vec_metas[idx] or {}).get("page") or 0),
            "text": text,
            "distance": float(vec_dists[idx]),
        }

    fused = reciprocal_rank_fusion([bm25_ranked, vec_ids])
    final_ids = [doc_id for doc_id, _score in sorted(fused.items(), key=lambda item: item[1], reverse=True)[: args.k]]

    for rank, chunk_id in enumerate(final_ids, start=1):
        item = by_id.get(chunk_id, {})
        page = item.get("page", 0)
        text = str(item.get("text", "")).strip()
        vec_distance = item.get("distance")
        dist_str = f"{vec_distance:.4f}" if isinstance(vec_distance, float) and not math.isnan(vec_distance) else "n/a"
        print(f"\n#{rank} id={chunk_id} page={page} vec_dist={dist_str} rrf={fused[chunk_id]:.4f}")
        print(text[:900])


if __name__ == "__main__":
    main()

