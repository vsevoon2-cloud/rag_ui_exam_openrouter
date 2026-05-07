import argparse
import re
from pathlib import Path


IMG_MD_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
IMG_HTML_RE = re.compile(r"<img[^>]+>", re.IGNORECASE)


def is_probably_broken(src: str) -> bool:
    s = src.strip()
    if s.startswith("http://") or s.startswith("https://"):
        return False
    if s.startswith("data:image/"):
        return False
    # common placeholders / nonsense paths
    if s.startswith("attachment:"):
        return True
    if s.startswith("/tmp/") or s.startswith("/static/"):
        return True
    if "imgur.com/example" in s:
        return True
    # relative paths we almost certainly don't have
    if s.startswith("crops/") or s.endswith(".png") or s.endswith(".jpg"):
        return True
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Remove/replace broken image links in per-page markdown.")
    ap.add_argument("--pages-dir", required=True, help="Directory with page_XXXX.md")
    ap.add_argument("--mode", choices=["comment", "remove"], default="comment")
    args = ap.parse_args()

    pages_dir = Path(args.pages_dir)
    changed = 0
    for md_path in sorted(pages_dir.glob("page_*.md")):
        text = md_path.read_text(encoding="utf-8", errors="replace")

        def repl_md(m: re.Match) -> str:
            alt = (m.group(1) or "").strip()
            src = (m.group(2) or "").strip()
            if not is_probably_broken(src):
                return m.group(0)
            if args.mode == "remove":
                return f"{alt}\n" if alt else ""
            label = alt if alt else "figure"
            return f"<!-- IMAGE REMOVED: {label} (src={src}) -->"

        new_text = IMG_MD_RE.sub(repl_md, text)
        # HTML images: always remove/comment (they won't resolve locally)
        if args.mode == "remove":
            new_text = IMG_HTML_RE.sub("", new_text)
        else:
            new_text = IMG_HTML_RE.sub("<!-- IMAGE REMOVED: <img ...> -->", new_text)

        if new_text != text:
            md_path.write_text(new_text, encoding="utf-8")
            changed += 1

    print(f"changed_files={changed}")


if __name__ == "__main__":
    main()

