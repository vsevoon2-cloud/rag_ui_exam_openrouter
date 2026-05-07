import json
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi


@dataclass
class Hit:
    chunk_id: str
    page: int
    distance: float
    text: str


def _tokenize(text: str) -> list[str]:
    import re

    return [t.lower() for t in re.findall(r"[A-Za-zА-Яа-яЁё0-9]{2,}", text)]


def _normalize_token(token: str) -> str:
    token = token.lower().replace("ё", "е")
    suffixes = [
        "иями", "ями", "ами", "ией", "ией", "иях", "ях", "ого", "ему", "ому", "ыми", "ими",
        "ать", "ять", "ить", "ость", "ести", "ение", "ения", "ений", "ости", "ов", "ев", "ей",
        "ия", "ие", "ий", "ый", "ой", "ая", "ое", "ые", "ам", "ям", "ах", "ях", "ом", "ем", "ую",
        "юю", "ть", "ти", "ет", "ют", "ут", "ит", "ат", "ят", "а", "я", "ы", "и", "е", "у", "ю", "о",
    ]
    if len(token) <= 4:
        return token
    for suffix in suffixes:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def _normalized_tokens(text: str) -> list[str]:
    return [_normalize_token(token) for token in _tokenize(text)]


def _keyword_overlap(query_tokens: set[str], text: str) -> int:
    if not query_tokens:
        return 0
    text_tokens = set(_normalized_tokens(text))
    return len(query_tokens & text_tokens)


def _rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


class LocalIndex:
    def __init__(self, index_dir: Path, collection: str, embed_model: str) -> None:
        self.index_dir = index_dir
        self.collection = collection
        self.embed_model = embed_model

        self.client = None
        self.col = None
        self.embedder = None
        try:
            import chromadb
            from chromadb.config import Settings
            from sentence_transformers import SentenceTransformer

            # Support two layouts:
            # 1) index root with "chroma" subfolder
            # 2) direct chroma directory
            chroma_path = index_dir / "chroma"
            if not chroma_path.exists():
                chroma_path = index_dir
            self.client = chromadb.PersistentClient(path=str(chroma_path), settings=Settings(anonymized_telemetry=False))
            self.col = self.client.get_collection(collection)
            self.embedder = SentenceTransformer(embed_model)
        except Exception:
            # Allow manifest/chunks-only fallback mode (BM25 without vectors).
            self.client = None
            self.col = None
            self.embedder = None

        chunks_path = index_dir / "chunks.jsonl"
        self._chunks = self._load_chunks(chunks_path)
        if not self._chunks and self.col is not None:
            self._chunks = self._load_chunks_from_chroma()
        if not self._chunks:
            self._chunks = self._load_chunks_from_manifest(index_dir / "manifest.jsonl")
        self._bm25 = BM25Okapi([_tokenize(c["text"]) for c in self._chunks]) if self._chunks else None

    @staticmethod
    def _load_chunks(path: Path) -> list[dict]:
        chunks = []
        if not path.exists():
            return chunks
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "id" in obj and "text" in obj and "page" in obj:
                        chunks.append(obj)
                except Exception:
                    continue
        return chunks

    def _load_chunks_from_chroma(self) -> list[dict]:
        if self.col is None:
            return []
        res = self.col.get(include=["documents", "metadatas"])
        ids = res.get("ids", [])
        docs = res.get("documents", [])
        metas = res.get("metadatas", [])
        chunks: list[dict] = []
        for i, chunk_id in enumerate(ids):
            text = (docs[i] if i < len(docs) else "") or ""
            if text.startswith("passage: "):
                text = text[len("passage: ") :]
            meta = (metas[i] if i < len(metas) else {}) or {}
            page = int(meta.get("page") or 0)
            chunks.append({"id": str(chunk_id), "text": str(text), "page": page})
        return chunks

    @staticmethod
    def _load_chunks_from_manifest(path: Path) -> list[dict]:
        chunks: list[dict] = []
        if not path.exists():
            return chunks
        # manifest source_file paths are relative to workspace root (e.g. out\or_md_flash2\...)
        root = path.parent.parent.parent
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    chunk_id = str(obj.get("id", "")).strip()
                    meta = obj.get("metadata", {}) or {}
                    page = int(meta.get("page") or 0)
                    source_file = str(meta.get("source_file") or "").strip()
                    if not chunk_id or not source_file:
                        continue
                    source_path = root / Path(source_file)
                    if not source_path.exists():
                        continue
                    text = source_path.read_text(encoding="utf-8", errors="ignore").strip()
                    if not text:
                        continue
                    chunks.append({"id": chunk_id, "text": text, "page": page})
                except Exception:
                    continue
        return chunks

    def query_vector(self, q: str, k: int) -> list[Hit]:
        if self.col is None or self.embedder is None:
            return []
        q_emb = self.embedder.encode([f"query: {q}"], normalize_embeddings=True).tolist()
        res = self.col.query(query_embeddings=q_emb, n_results=k, include=["documents", "metadatas", "distances"])
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        hits = []
        for i in range(len(ids)):
            text = docs[i] or ""
            if text.startswith("passage: "):
                text = text[len("passage: ") :]
            page = int((metas[i] or {}).get("page") or 0)
            hits.append(Hit(chunk_id=ids[i], page=page, distance=float(dists[i]), text=text))
        return hits

    def query_hybrid(self, q: str, k: int) -> list[Hit]:
        if not self._chunks or self._bm25 is None:
            return self.query_vector(q, k)
        # BM25 ranking over stored chunk texts
        scores = self._bm25.get_scores(_tokenize(q))
        bm25_ranked_ids = [self._chunks[i]["id"] for i in sorted(range(len(self._chunks)), key=lambda idx: scores[idx], reverse=True)[: max(k * 3, 20)]]

        vec_hits = self.query_vector(q, max(k * 3, 20))
        vec_ids = [h.chunk_id for h in vec_hits]

        by_id: dict[str, Hit] = {h.chunk_id: h for h in vec_hits}
        for c in self._chunks:
            if c["id"] not in by_id:
                by_id[c["id"]] = Hit(chunk_id=c["id"], page=int(c["page"]), distance=999.0, text=str(c["text"]))

        fused = _rrf([bm25_ranked_ids, vec_ids])
        final_ids = [doc_id for doc_id, _score in sorted(fused.items(), key=lambda item: item[1], reverse=True)[:k]]
        return [by_id[i] for i in final_ids]

    def _query_bm25_filtered(self, q: str, k: int, *, min_overlap: int = 2) -> list[Hit]:
        if not self._chunks or self._bm25 is None:
            return []
        query_tokens = set(_normalized_tokens(q))
        scores = self._bm25.get_scores(list(query_tokens))
        ranked_idx = sorted(range(len(self._chunks)), key=lambda idx: scores[idx], reverse=True)
        hits: list[Hit] = []
        for idx in ranked_idx:
            chunk = self._chunks[idx]
            overlap = _keyword_overlap(query_tokens, str(chunk.get("text") or ""))
            if overlap < min_overlap:
                continue
            hits.append(Hit(chunk_id=str(chunk["id"]), page=int(chunk.get("page") or 0), distance=999.0, text=str(chunk.get("text") or "")))
            if len(hits) >= k:
                break
        return hits

    def _query_keyword_ranked(self, q: str, k: int) -> list[Hit]:
        if not self._chunks:
            return []
        query_tokens = set(_normalized_tokens(q))
        ranked: list[tuple[int, int, dict]] = []
        for chunk in self._chunks:
            text = str(chunk.get("text") or "")
            overlap = _keyword_overlap(query_tokens, text)
            if overlap <= 0:
                continue
            ranked.append((overlap, len(text), chunk))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [
            Hit(
                chunk_id=str(chunk["id"]),
                page=int(chunk.get("page") or 0),
                distance=999.0,
                text=str(chunk.get("text") or ""),
            )
            for overlap, _len, chunk in ranked[:k]
        ]

    def search(self, query: str, top_k: int = 5, mode: str = "hybrid") -> list[dict]:
        hits = self.query_hybrid(query, top_k) if mode == "hybrid" else self.query_vector(query, top_k)
        # Heuristic: if results look like table-of-contents/front-matter and have weak keyword overlap,
        # re-run a stricter BM25 filter to avoid always returning page 1..3.
        if hits:
            query_tokens = set(_normalized_tokens(query))
            first = hits[0]
            overlap0 = _keyword_overlap(query_tokens, first.text)
            if first.page and first.page <= 5 and overlap0 < 2 and mode == "hybrid":
                bm25_hits = self._query_bm25_filtered(query, top_k, min_overlap=2)
                if bm25_hits:
                    hits = bm25_hits
            elif overlap0 < 2:
                lexical_hits = self._query_keyword_ranked(query, top_k)
                if lexical_hits:
                    hits = lexical_hits
        return [
            {
                "id": hit.chunk_id,
                "page": hit.page,
                "distance": hit.distance,
                "text": hit.text,
            }
            for hit in hits
        ]
