import argparse
import json
import os
import re
import time
from pathlib import Path

import requests


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def iter_pages(pages_dir: Path):
    for path in sorted(pages_dir.glob("page_*.md")):
        match = re.match(r"page_(\d+)\.md$", path.name)
        if not match:
            continue
        yield int(match.group(1)), path


def build_prompt(text: str, questions_per_page: int) -> str:
    sample = (
        "Верни только строки формата:\n"
        "Q: вопрос 1\n"
        "Q: вопрос 2\n"
        "Q: вопрос 3\n"
        "Без нумерации, без JSON, без пояснений."
    )
    return (
        f"Ниже фрагмент учебника. Сгенерируй {questions_per_page} коротких поисковых вопроса на русском, "
        "на которые этот фрагмент прямо отвечает. Вопросы должны быть конкретными, фактологическими, "
        "без местоимений и без ссылок на 'этот текст' или 'эта страница'. "
        "Не придумывай знаний вне фрагмента.\n\n"
        f"{sample}\n\n"
        f"Фрагмент:\n{text[:6000]}"
    )


def call_openrouter(api_key: str, model: str, prompt: str, max_tokens: int, timeout_s: int) -> tuple[str, dict]:
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout_s)
    response.raise_for_status()
    data = response.json()
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        content = "\n".join(str(x.get("text", "")) for x in content if isinstance(x, dict))
    return str(content).strip(), data.get("usage") or {}


def parse_questions(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("q:"):
            question = stripped[2:].strip()
        else:
            question = re.sub(r"^\d+[\).\s-]+", "", stripped).strip()
        if len(question.split()) < 4:
            continue
        if question not in out:
            out.append(question)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic retrieval queries from page markdown via OpenRouter.")
    ap.add_argument("--pages-dir", default="out/or_md_flash2/pages")
    ap.add_argument("--out", default="out/synth_queries.jsonl")
    ap.add_argument("--model", default="google/gemini-2.0-flash-001")
    ap.add_argument("--questions-per-page", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--sleep", type=float, default=0.4)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    api_key = os.environ.get(args.key_env, "").strip()
    if not api_key:
        raise SystemExit(f"Environment variable {args.key_env} is empty")

    pages_dir = Path(args.pages_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_pages: set[int] = set()
    if args.resume and out_path.exists():
        for line in out_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                obj = json.loads(line)
                seen_pages.add(int(obj["page"]))
            except Exception:
                continue

    total_cost = 0.0
    with out_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for page_no, path in iter_pages(pages_dir):
            if page_no < args.start:
                continue
            if args.end and page_no > args.end:
                continue
            if page_no in seen_pages:
                continue

            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                continue

            prompt = build_prompt(text, args.questions_per_page)
            raw, usage = call_openrouter(api_key, args.model, prompt, args.max_tokens, args.timeout)
            questions = parse_questions(raw)
            total_cost += float(usage.get("cost") or 0.0)

            row = {
                "page": page_no,
                "source_file": str(path),
                "questions": questions,
                "raw": raw,
                "usage": usage,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"page {page_no}: questions={len(questions)} cost_total=${total_cost:.6f}")
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()

