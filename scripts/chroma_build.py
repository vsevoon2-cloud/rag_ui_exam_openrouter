import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from clean_book_md import clean_text


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any]


def clean_md(md: str) -> str:
    return clean_text(md, strict=True).strip()


def iter_page_files(pages_dir: Path) -> Iterable[tuple[int, Path]]:
    for p in sorted(pages_dir.glob("page_*.md")):
        m = re.match(r"page_(\d+)\.md$", p.name)
        if not m:
            continue
        yield int(m.group(1)), p


def chunk_text(text: str, *, target_chars: int, overlap_chars: int) -> list[str]:
    text = text.strip()
    if not text:
        return []

    # split on paragraph boundaries first
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur, cur_len
        if not cur:
            return
        chunks.append("\n\n".join(cur).strip())
        cur = []
        cur_len = 0

    for part in parts:
        plen = len(part)
        if cur_len and cur_len + plen + 2 > target_chars:
            flush()
        cur.append(part)
        cur_len += plen + 2

    flush()

    # add overlap by re-splitting chunks into windows (simple char-based overlap)
    if overlap_chars <= 0 or len(chunks) <= 1:
        return chunks

    out: list[str] = []
    for i, ch in enumerate(chunks):
        if i == 0:
            out.append(ch)
            continue
        prev = out[-1]
        overlap = prev[-overlap_chars:] if len(prev) > overlap_chars else prev
        out.append((overlap + "\n" + ch).strip())
    return out


def page_chunk(text: str) -> list[str]:
    text = text.strip()
    return [text] if text else []


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a persistent ChromaDB from per-page markdown.")
    ap.add_argument("--pages-dir", default="out/or_md_flash2/pages", help="Directory with page_XXXX.md")
    ap.add_argument("--chroma-dir", default="out/chroma_db", help="Persistent Chroma directory")
    ap.add_argument("--collection", default="phis_book", help="Chroma collection name")
    ap.add_argument("--model", default="intfloat/multilingual-e5-large", help="SentenceTransformer model id")
    ap.add_argument("--chunking-mode", choices=["page", "chars"], default="page", help="Chunking strategy")
    ap.add_argument("--target-chars", type=int, default=4500, help="Chunk target size (chars)")
    ap.add_argument("--overlap-chars", type=int, default=400, help="Chunk overlap (chars)")
    ap.add_argument("--batch", type=int, default=64, help="Upsert batch size")
    ap.add_argument("--start", type=int, default=1, help="Start page number")
    ap.add_argument("--end", type=int, default=0, help="End page number (0 = no limit)")
    ap.add_argument("--reset", action="store_true", help="Delete existing chroma-dir before rebuild")
    ap.add_argument("--write-manifest", action="store_true", help="Write manifest of chunks to chroma-dir/manifest.jsonl")
    args = ap.parse_args()

    pages_dir = Path(args.pages_dir)
    chroma_dir = Path(args.chroma_dir)
    if args.reset and chroma_dir.exists():
        shutil.rmtree(chroma_dir, ignore_errors=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer

    embed_model = SentenceTransformer(args.model)

    client = chromadb.PersistentClient(path=str(chroma_dir), settings=Settings(anonymized_telemetry=False))
    col = client.get_or_create_collection(name=args.collection, metadata={"hnsw:space": "cosine"})

    chunks: list[Chunk] = []
    for page_no, md_path in iter_page_files(pages_dir):
        if page_no < args.start:
            continue
        if args.end and page_no > args.end:
            continue
        raw = md_path.read_text(encoding="utf-8", errors="replace")
        text = clean_md(raw)
        if not text:
            continue
        parts = page_chunk(text) if args.chunking_mode == "page" else chunk_text(text, target_chars=args.target_chars, overlap_chars=args.overlap_chars)
        for i, part in enumerate(parts, start=1):
            cid = f"p{page_no:04d}_c{i:03d}"
            # e5 recommends prefixing documents with "passage: "
            doc = f"passage: {part}"
            chunks.append(
                Chunk(
                    id=cid,
                    text=doc,
                    metadata={
                        "page": page_no,
                        "chunk": i,
                        "source_file": str(md_path),
                        "text_chars": len(part),
                    },
                )
            )

    if not chunks:
        raise SystemExit(f"No chunks found in {pages_dir}")

    manifest_path = chroma_dir / "manifest.jsonl"
    if args.write_manifest:
        with manifest_path.open("w", encoding="utf-8") as f:
            for ch in chunks:
                f.write(json.dumps({"id": ch.id, "metadata": ch.metadata}, ensure_ascii=False) + "\n")

    # Upsert in batches
    for off in range(0, len(chunks), args.batch):
        batch = chunks[off : off + args.batch]
        texts = [c.text for c in batch]
        embs = embed_model.encode(texts, normalize_embeddings=True).tolist()
        col.upsert(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[c.metadata for c in batch],
            embeddings=embs,
        )
        print(f"upsert {off + len(batch)}/{len(chunks)}")

    print(f"done: collection={args.collection} chunks={len(chunks)} chroma_dir={chroma_dir}")


if __name__ == "__main__":
    main()
