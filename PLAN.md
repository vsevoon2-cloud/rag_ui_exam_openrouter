# ExamRAG (OpenRouter) — v1 Plan

Ship an open-source, non-coder-friendly desktop app that:
1. lets a user load a PDF and builds a local RAG index (Chroma) from OpenRouter OCR;
2. in “Exam mode” captures a screen region, extracts the question, retrieves from the local index, and answers via OpenRouter with citations.

v1 uses **OpenRouter OCR only** (no local OCR runtime) to keep **PyInstaller onefile** feasible.

Key decisions:
- UI: PySide6
- PDF OCR: OpenRouter vision model → Markdown (strict prompt)
- Retrieval: hybrid (BM25 + vectors, RRF)
- Answer: OpenRouter model selectable, default strongest
- Storage: config + caches under user app data; Chroma persistent directory (multiple files by design)

