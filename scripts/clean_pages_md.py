import argparse
from pathlib import Path

from clean_book_md import clean_text


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean per-page markdown files from common LLM wrapper chatter.")
    ap.add_argument("--pages-dir", required=True, help="Directory with page_XXXX.md")
    ap.add_argument("--backup", action="store_true", help="Write .bak next to each page file before overwriting")
    ap.add_argument("--strict", action="store_true", help="Drop likely OCR/diagram noise more aggressively")
    args = ap.parse_args()

    pages_dir = Path(args.pages_dir)
    if not pages_dir.exists():
        raise SystemExit(f"Not found: {pages_dir}")

    changed = 0
    total = 0
    for md_path in sorted(pages_dir.glob("page_*.md")):
        total += 1
        raw = md_path.read_text(encoding="utf-8", errors="replace")
        cleaned = clean_text(raw, strict=bool(args.strict))
        if cleaned != raw:
            if args.backup:
                md_path.with_suffix(md_path.suffix + ".bak").write_text(raw, encoding="utf-8")
            md_path.write_text(cleaned, encoding="utf-8")
            changed += 1

    print(f"files_total={total} files_changed={changed}")


if __name__ == "__main__":
    main()
