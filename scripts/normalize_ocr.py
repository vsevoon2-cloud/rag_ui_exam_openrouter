import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


CONTROL_CHARS_RE = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F]")
MULTISPACE_RE = re.compile(r"[ \t\u00A0]{2,}")


def clean_unicode(text: str) -> str:
    # Normalize common OCR oddities and remove control chars.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text)
    text = CONTROL_CHARS_RE.sub("", text)
    # Drop zero-width chars
    text = text.replace("\u200b", "").replace("\ufeff", "")
    return text


def drop_leading_page_number(text: str, page_no: int) -> str:
    lines = [ln.strip() for ln in text.split("\n")]
    while lines and lines[0] == "":
        lines.pop(0)
    if lines and lines[0] == str(page_no):
        lines = lines[1:]
        # drop one extra empty line if exists
        while lines and lines[0] == "":
            lines.pop(0)
    return "\n".join(lines).strip()


def fix_hyphenation(text: str) -> str:
    # Join hyphenated line breaks: "энер-\nгии" -> "энергии"
    # Also handle soft hyphen.
    text = text.replace("\u00ad", "")
    text = re.sub(r"([A-Za-zА-Яа-яЁё])-\n([A-Za-zА-Яа-яЁё])", r"\1\2", text)
    return text


def join_lines(text: str) -> str:
    # Convert arbitrary line breaks into paragraphs:
    # - blank lines keep paragraph breaks
    # - otherwise join lines with spaces
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
    return out.strip()


def normalize_text(raw_text: str, page_no: int) -> str:
    text = clean_unicode(raw_text)
    text = drop_leading_page_number(text, page_no)
    text = fix_hyphenation(text)
    text = join_lines(text)
    # normalize punctuation spacing
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]])", r"\1", text)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize OCR output produced by scripts/ocr_sample.py")
    ap.add_argument("--in", dest="inp", required=True, help="Input JSON (out/.../result.json)")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--min-len", type=int, default=20, help="Skip pages with normalized text shorter than this")
    args = ap.parse_args()

    in_path = Path(args.inp)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data: list[dict[str, Any]] = json.loads(in_path.read_text(encoding="utf-8"))
    normalized: list[dict[str, Any]] = []

    txt_lines: list[str] = []
    for rec in data:
        page_no = int(rec.get("page"))
        raw_text = str(rec.get("text") or "")
        norm = normalize_text(raw_text, page_no)
        rec2 = dict(rec)
        rec2["text_normalized"] = norm
        rec2["text_normalized_len"] = len(norm)
        normalized.append(rec2)

        if len(norm) >= args.min_len:
            txt_lines.append(f"\n\n===== PAGE {page_no} =====\n")
            txt_lines.append(norm)

    (out_dir / "normalized.json").write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "normalized.txt").write_text("".join(txt_lines).strip() + "\n", encoding="utf-8")

    kept = sum(1 for r in normalized if int(r.get("text_normalized_len", 0)) >= args.min_len)
    print(f"pages={len(normalized)} kept(min_len={args.min_len})={kept}")


if __name__ == "__main__":
    main()

