import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def app_root() -> Path:
    if getattr(__import__("sys"), "frozen", False):
        return Path(__import__("sys").executable).resolve().parent
    return Path(__file__).resolve().parent.parent


@dataclass
class RetrievalHit:
    chunk_id: str
    page: int
    distance: float
    text: str


@dataclass
class AssistantConfig:
    chroma_dir: Path
    collection: str
    embed_model: str
    ocr_model: str
    answer_model: str
    top_k: int = 8


def image_bytes_to_data_url(image_bytes: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"


class Retriever:
    def __init__(self, chroma_dir: Path, collection: str, embed_model_name: str) -> None:
        import chromadb
        from chromadb.config import Settings
        from sentence_transformers import SentenceTransformer

        self.client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_collection(collection)
        self.embedder = SentenceTransformer(embed_model_name)

    def query(self, question: str, top_k: int) -> list[RetrievalHit]:
        q_emb = self.embedder.encode([f"query: {question}"], normalize_embeddings=True).tolist()
        res = self.collection.query(
            query_embeddings=q_emb,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        hits: list[RetrievalHit] = []
        for i in range(len(ids)):
            text = docs[i] or ""
            if text.startswith("passage: "):
                text = text[len("passage: ") :]
            meta = metas[i] or {}
            hits.append(
                RetrievalHit(
                    chunk_id=ids[i],
                    page=int(meta.get("page") or 0),
                    distance=float(dists[i]),
                    text=text,
                )
            )
        return hits


class OpenRouterClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.session = requests.Session()

    def _post(self, payload: dict[str, Any], timeout_s: int = 120) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        r = self.session.post(OPENROUTER_URL, headers=headers, data=json.dumps(payload), timeout=timeout_s)
        r.raise_for_status()
        return r.json()

    def vision_extract_question(self, image_bytes: bytes, model: str) -> str:
        prompt = (
            "Extract only the exam question from this screenshot in Russian. "
            "Do not answer it. Ignore browser UI, chat UI, watermarks, timers, and unrelated text. "
            "If there is no clear question or the selected region is wrong, return exactly: NO_QUESTION_FOUND"
        )
        payload = {
            "model": model,
            "max_tokens": 500,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_bytes_to_data_url(image_bytes)}},
                    ],
                }
            ],
        }
        data = self._post(payload)
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        if isinstance(content, list):
            parts = [str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text"]
            return "\n".join(parts).strip()
        return str(content).strip()

    def grounded_answer(self, question: str, hits: list[RetrievalHit], model: str) -> str:
        ctx_lines = []
        for i, hit in enumerate(hits, start=1):
            ctx_lines.append(f"[source {i} | page {hit.page} | id {hit.chunk_id} | dist {hit.distance:.4f}]\n{hit.text}")
        context = "\n\n---\n\n".join(ctx_lines)
        payload = {
            "model": model,
            "max_tokens": 900,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a careful assistant. Answer ONLY using the provided textbook sources. "
                        "If the sources are insufficient, say that the answer is not confirmed by the book. "
                        "Write in Russian. Cite source pages like (стр. 50)."
                    ),
                },
                {"role": "user", "content": f"Question: {question}\n\nSources:\n{context}"},
            ],
        }
        data = self._post(payload)
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        if isinstance(content, list):
            parts = [str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text"]
            return "\n".join(parts).strip()
        return str(content).strip()


def default_config() -> AssistantConfig:
    root = app_root()
    return AssistantConfig(
        chroma_dir=root / "out" / "chroma_db_v2",
        collection="phis_book_v2",
        embed_model="intfloat/multilingual-e5-base",
        ocr_model="google/gemini-2.0-flash-001",
        answer_model="google/gemini-2.0-flash-001",
        top_k=8,
    )


def config_path() -> Path:
    return app_root() / "exam_assistant_config.json"


def load_saved_settings() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_settings(data: dict[str, Any]) -> None:
    config_path().write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

