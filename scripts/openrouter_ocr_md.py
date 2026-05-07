import argparse
import base64
import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def b64_data_url(jpg_path: Path) -> str:
    data = jpg_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def load_done(out_dir: Path) -> set[int]:
    done = set()
    pages_dir = out_dir / "pages"
    if not pages_dir.exists():
        return done
    for p in pages_dir.glob("page_*.md"):
        try:
            no = int(p.stem.split("_", 1)[1])
        except Exception:
            continue
        done.add(no)
    return done


def call_openrouter(
    *,
    session: requests.Session,
    api_key: str,
    model: str,
    prompt: str,
    data_url: str,
    max_tokens: int,
    timeout_s: int,
    extra_headers: dict[str, str],
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        **extra_headers,
    }
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    }
    r = session.post(OPENROUTER_URL, headers=headers, data=json.dumps(payload), timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def list_models(api_key: str, timeout_s: int, extra_headers: dict[str, str]) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        **extra_headers,
    }
    r = requests.get(OPENROUTER_MODELS_URL, headers=headers, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def extract_markdown(resp: dict[str, Any]) -> str:
    # OpenAI-compatible response: choices[0].message.content
    choices = resp.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if isinstance(content, str):
        return content.strip()
    # Some providers may return list parts
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text", "")))
        return "\n".join(parts).strip()
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="OCR JPG pages into Markdown via OpenRouter (Gemini).")
    ap.add_argument("--in", dest="inp", required=True, help="Directory with page_XXXX.jpg")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument(
        "--model",
        default="google/gemini-2.5-flash-image",
        help="OpenRouter model id (use an image-capable model for OCR)",
    )
    ap.add_argument(
        "--prompt",
        default="OCR this page into Markdown. Keep tables as MD tables. Describe anatomical diagrams briefly.",
        help="User prompt",
    )
    ap.add_argument("--start", type=int, default=1, help="Start page number (1-based)")
    ap.add_argument("--count", type=int, default=0, help="How many pages to process (0 = until end)")
    ap.add_argument("--max-tokens", type=int, default=4096, help="max_tokens for completion")
    ap.add_argument("--sleep", type=float, default=0.4, help="Sleep seconds between requests")
    ap.add_argument("--timeout", type=int, default=180, help="Request timeout seconds")
    ap.add_argument("--resume", action="store_true", help="Skip pages already written to out/pages/")
    ap.add_argument("--write-json", action="store_true", help="Also write raw response JSON per page")
    ap.add_argument("--limit", type=int, default=0, help="Hard cap pages processed this run (0 = no cap)")
    ap.add_argument("--check-model", action="store_true", help="List models once and verify --model exists")
    ap.add_argument("--list-gemini", action="store_true", help="List available google/gemini* model ids and exit")
    ap.add_argument("--retries", type=int, default=6, help="Network retries for transient failures")
    ap.add_argument("--retry-sleep", type=float, default=2.0, help="Base sleep seconds for manual retry backoff")
    ap.add_argument("--site", default="", help="Optional OpenRouter-Referer value")
    ap.add_argument("--title", default="", help="Optional X-Title header for OpenRouter")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()

    in_dir = Path(args.inp)
    out_dir = Path(args.out)
    ensure_dir(out_dir)
    pages_out = out_dir / "pages"
    ensure_dir(pages_out)
    json_out = out_dir / "responses"
    if args.write_json:
        ensure_dir(json_out)

    extra_headers: dict[str, str] = {}
    if args.site:
        extra_headers["HTTP-Referer"] = args.site
    if args.title:
        extra_headers["X-Title"] = args.title

    done = load_done(out_dir) if args.resume else set()

    # Robust HTTP session (retries for 429/5xx). SSL EOF may still happen; we handle it manually too.
    session = requests.Session()
    retry = Retry(
        total=args.retries,
        connect=args.retries,
        read=args.retries,
        status=args.retries,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    if args.list_gemini:
        # models endpoint is public; no key needed
        models = session.get(OPENROUTER_MODELS_URL, timeout=args.timeout).json()
        ids = sorted(
            m.get("id")
            for m in (models.get("data") or [])
            if isinstance(m, dict) and isinstance(m.get("id"), str) and m["id"].startswith("google/gemini")
        )
        print("\n".join(ids))
        return

    if args.check_model:
        try:
            # models endpoint is public; key not required
            models = session.get(OPENROUTER_MODELS_URL, timeout=args.timeout).json()
            model_ids = {m.get("id") for m in (models.get("data") or []) if isinstance(m, dict)}
            if args.model not in model_ids:
                raise SystemExit(
                    f"Model id not found on OpenRouter: {args.model}\n"
                    f"Hint: try `google/gemini-2.5-flash-image` or `google/gemini-2.5-pro`.\n"
                    f"Tip: run without OCR first: python scripts/openrouter_ocr_md.py --check-model ..."
                )
            print(f"model ok: {args.model}")
        except Exception as e:
            raise SystemExit(f"Model check failed: {e}")

    if not api_key:
        raise SystemExit(
            "Missing OPENROUTER_API_KEY env var.\n"
            "Set it in the same shell you run the script, e.g.\n"
            "  $env:OPENROUTER_API_KEY='...'\n"
            "Then rerun."
        )

    # Determine page list
    all_pages = sorted(in_dir.glob("page_*.jpg"))
    if not all_pages:
        raise SystemExit(f"No page_*.jpg found in {in_dir}")

    start = args.start
    # Count can be 0: process until max existing
    max_page = 0
    for p in all_pages:
        try:
            max_page = max(max_page, int(p.stem.split("_", 1)[1]))
        except Exception:
            pass
    end = max_page if args.count == 0 else min(max_page, start + args.count - 1)

    processed = 0
    usage_log: list[dict[str, Any]] = []
    for page_no in range(start, end + 1):
        if args.limit and processed >= args.limit:
            break
        if page_no in done:
            continue
        jpg = in_dir / f"page_{page_no:04d}.jpg"
        if not jpg.exists():
            continue

        md_path = pages_out / f"page_{page_no:04d}.md"
        if md_path.exists() and args.resume:
            continue

        data_url = b64_data_url(jpg)
        # Manual retry loop to survive SSL EOF and other transient connection errors.
        last_err: Exception | None = None
        resp: dict[str, Any] | None = None
        for attempt in range(1, args.retries + 2):
            try:
                resp = call_openrouter(
                    session=session,
                    api_key=api_key,
                    model=args.model,
                    prompt=args.prompt,
                    data_url=data_url,
                    max_tokens=args.max_tokens,
                    timeout_s=args.timeout,
                    extra_headers=extra_headers,
                )
                break
            except requests.HTTPError as e:
                last_err = e
                # Non-transient errors: don't retry too long.
                status = getattr(getattr(e, "response", None), "status_code", None)
                print(f"page {page_no}: HTTP error (attempt {attempt}): {e}")
                if status in (400, 401, 402, 403, 404):
                    if status == 404:
                        print(
                            "hint: 404 is often a bad model id; try --list-gemini then pick an id (image-capable)."
                        )
                    break
            except requests.RequestException as e:
                last_err = e
                print(f"page {page_no}: network error (attempt {attempt}): {e}")

            # backoff
            sleep_s = args.retry_sleep * (2 ** max(0, attempt - 1))
            time.sleep(min(sleep_s, 60.0))

        if resp is None:
            # Do NOT write an .md placeholder; allow clean resume.
            err_line = f"page {page_no}: FAILED after retries: {last_err}"
            print(err_line)
            (out_dir / "failures.log").open("a", encoding="utf-8").write(err_line + "\n")
            processed += 1
            continue

        md = extract_markdown(resp)
        md_path.write_text(md + "\n", encoding="utf-8")

        if args.write_json:
            (json_out / f"page_{page_no:04d}.json").write_text(
                json.dumps(resp, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        usage = resp.get("usage") or {}
        usage_log.append({"page": page_no, "usage": usage})
        (out_dir / "usage.json").write_text(json.dumps(usage_log, ensure_ascii=False, indent=2), encoding="utf-8")

        print(
            f"page {page_no}: wrote {md_path} (usage={usage if usage else 'n/a'})"
        )
        processed += 1
        time.sleep(args.sleep)

    # Merge pages into one markdown
    merged = []
    for p in sorted(pages_out.glob("page_*.md")):
        merged.append(f"\n\n<!-- {p.name} -->\n\n")
        merged.append(p.read_text(encoding="utf-8"))
    (out_dir / "book.md").write_text("".join(merged).strip() + "\n", encoding="utf-8")
    print(f"done. merged -> {out_dir / 'book.md'}")


if __name__ == "__main__":
    main()
