import argparse
import json
import re
from pathlib import Path


IMG_MD_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
IMG_HTML_RE = re.compile(r"<img[^>]+src=[\"']([^\"']+)[\"'][^>]*>", re.IGNORECASE)


def page_no_from_name(p: Path) -> int | None:
    m = re.match(r"page_(\d+)\.md$", p.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Find markdown/html image references in per-page markdown files.")
    ap.add_argument("--pages-dir", required=True, help="Directory with page_XXXX.md")
    ap.add_argument("--out", required=True, help="Output json path")
    args = ap.parse_args()

    pages_dir = Path(args.pages_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for md_path in sorted(pages_dir.glob("page_*.md")):
        page_no = page_no_from_name(md_path)
        if page_no is None:
            continue
        text = md_path.read_text(encoding="utf-8", errors="replace")
        refs = []
        for m in IMG_MD_RE.finditer(text):
            refs.append({"kind": "md", "src": m.group(1).strip()})
        for m in IMG_HTML_RE.finditer(text):
            refs.append({"kind": "html", "src": m.group(1).strip()})
        if refs:
            results.append({"page": page_no, "file": str(md_path), "refs": refs})

    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    txt = "\n".join(f"{r['page']:04d}\t{len(r['refs'])}\t{Path(r['file']).name}" for r in results)
    (out_path.with_suffix(".txt")).write_text(txt + ("\n" if txt else ""), encoding="utf-8")
    print(f"pages_with_images={len(results)} wrote={out_path}")


if __name__ == "__main__":
    main()

