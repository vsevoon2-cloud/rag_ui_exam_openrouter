import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import shutil

import fitz

from .text_clean import strict_clean_markdown


@dataclass
class BuildProgress:
    stage: str
    current: int
    total: int
    message: str = ""
    cost_usd: float = 0.0


def chunk_text(text: str, target_chars: int = 2000, overlap_chars: int = 250) -> list[str]:
    text = text.strip()
    if not text:
        return []
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= target_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= target_chars:
            current = paragraph
            continue
        start = 0
        while start < len(paragraph):
            end = min(start + target_chars, len(paragraph))
            chunk = paragraph[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(paragraph):
                current = ""
                break
            start = max(end - overlap_chars, start + 1)
    if current:
        chunks.append(current)
    return chunks


def page_chunks(text: str) -> list[str]:
    text = text.strip()
    return [text] if text else []


def _render_page_jpeg(doc: fitz.Document, page_index: int, scale: float, quality: int) -> bytes:
    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    return pix.tobytes("jpeg", jpg_quality=quality)


def build_index_from_pdf(
    *,
    pdf_path: Path,
    out_dir: Path,
    openrouter_client,
    ocr_model: str,
    embed_model: str,
    collection: str,
    chunking_mode: str,
    target_chars: int,
    overlap_chars: int,
    render_scale: float = 3.0,
    jpeg_quality: int = 85,
    progress_cb: Callable[[BuildProgress], None] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = out_dir / "pages_md"
    pages_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = out_dir / "chunks.jsonl"
    meta_path = out_dir / "index_meta.json"

    doc = fitz.open(str(pdf_path))
    total_pages = doc.page_count
    total_cost = 0.0

    for page_index in range(total_pages):
        page_no = page_index + 1
        md_path = pages_dir / f"page_{page_no:04d}.md"
        if md_path.exists():
            continue
        if progress_cb:
            progress_cb(BuildProgress(stage="ocr", current=page_no, total=total_pages, cost_usd=total_cost))
        img_bytes = _render_page_jpeg(doc, page_index, render_scale, jpeg_quality)
        md_raw, usage = openrouter_client.ocr_page_to_markdown(img_bytes, "image/jpeg", ocr_model)
        md_path.write_text(strict_clean_markdown(md_raw) + "\n", encoding="utf-8")
        if usage.cost:
            total_cost += usage.cost

    chroma_dir = out_dir / "chroma"
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir, ignore_errors=True)
    if chunks_path.exists():
        chunks_path.unlink()

    chunks: list[dict] = []
    for page_index in range(total_pages):
        page_no = page_index + 1
        text = strict_clean_markdown((pages_dir / f"page_{page_no:04d}.md").read_text(encoding="utf-8", errors="replace"))
        if chunking_mode == "page":
            parts = page_chunks(text)
        else:
            parts = chunk_text(text, target_chars=target_chars, overlap_chars=overlap_chars)
        for chunk_index, part in enumerate(parts, start=1):
            chunks.append({"id": f"p{page_no:04d}_c{chunk_index:03d}", "page": page_no, "text": part})

    with chunks_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    meta_path.write_text(
        json.dumps(
            {
                "collection": collection,
                "embed_model": embed_model,
                "chunking_mode": chunking_mode,
                "target_chars": target_chars,
                "overlap_chars": overlap_chars,
                "pages": total_pages,
                "chunks": len(chunks),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if progress_cb:
        progress_cb(BuildProgress(stage="embed", current=0, total=len(chunks), cost_usd=total_cost))

    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(path=str(chroma_dir), settings=Settings(anonymized_telemetry=False))
    collection_obj = client.get_or_create_collection(name=collection, metadata={"hnsw:space": "cosine"})
    embedder = SentenceTransformer(embed_model)

    batch_size = 64
    for offset in range(0, len(chunks), batch_size):
        batch = chunks[offset : offset + batch_size]
        docs = [f"passage: {item['text']}" for item in batch]
        embeddings = embedder.encode(docs, normalize_embeddings=True).tolist()
        collection_obj.upsert(
            ids=[item["id"] for item in batch],
            documents=docs,
            metadatas=[{"page": item["page"]} for item in batch],
            embeddings=embeddings,
        )
        if progress_cb:
            progress_cb(
                BuildProgress(
                    stage="embed",
                    current=min(offset + len(batch), len(chunks)),
                    total=len(chunks),
                    cost_usd=total_cost,
                )
            )
