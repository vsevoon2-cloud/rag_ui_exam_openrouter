import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


CONTROL_CHARS_RE = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F]")
MULTISPACE_RE = re.compile(r"[ \t\u00A0]{2,}")


def clean_unicode(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text)
    text = CONTROL_CHARS_RE.sub("", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    return text


def fix_hyphenation(text: str) -> str:
    text = text.replace("\u00ad", "")
    return re.sub(r"([A-Za-zА-Яа-яЁё])-\n([A-Za-zА-Яа-яЁё])", r"\1\2", text)


def join_boxes(text: str) -> str:
    """
    `text_body` is produced as one OCR box per line.
    For embeddings/search we want readable prose:
    - keep explicit paragraph breaks if they exist (blank lines)
    - otherwise merge single newlines into spaces
    """
    text = text.strip()
    if not text:
        return ""

    lines = [ln.strip() for ln in text.split("\n")]
    paras: list[str] = []
    buf: list[str] = []
    for ln in lines:
        if ln == "":
            if buf:
                paras.append(" ".join(buf))
                buf = []
            continue
        buf.append(ln)
    if buf:
        paras.append(" ".join(buf))
    out = "\n\n".join(paras)
    out = MULTISPACE_RE.sub(" ", out)
    # punctuation spacing
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    return out.strip()


def normalize_page(text_body: str) -> str:
    t = clean_unicode(text_body)
    t = fix_hyphenation(t)
    t = join_boxes(t)
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize body.json produced by reocr_exclude_margins.py")
    ap.add_argument("--in", dest="inp", required=True, help="Input body.json")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--field", default="text_body", help="Input field name (default: text_body)")
    ap.add_argument("--min-len", type=int, default=20, help="Min normalized length for txt output")
    args = ap.parse_args()

    in_path = Path(args.inp)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pages: list[dict[str, Any]] = json.loads(in_path.read_text(encoding="utf-8"))
    out_pages: list[dict[str, Any]] = []
    txt_lines: list[str] = []

    for rec in pages:
        page_no = int(rec.get("page"))
        raw = str(rec.get(args.field) or "")
        norm = normalize_page(raw)
        rec2 = dict(rec)
        rec2["text_body_normalized"] = norm
        rec2["text_body_normalized_len"] = len(norm)
        out_pages.append(rec2)

        if len(norm) >= args.min_len:
            txt_lines.append(f"\n\n===== PAGE {page_no} =====\n")
            txt_lines.append(norm)

    (out_dir / "body_normalized.json").write_text(
        json.dumps(out_pages, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "body_normalized.txt").write_text("".join(txt_lines).strip() + "\n", encoding="utf-8")
    print(f"pages={len(out_pages)} out={(out_dir / 'body_normalized.json')}")


if __name__ == "__main__":
    main()

