import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


WS_RE = re.compile(r"\s+")
DIGIT_RE = re.compile(r"\d")
EDGE_PUNCT_RE = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)
try:
    import regex as _regex  # type: ignore

    LETTERS_RE = _regex.compile(r"[^\p{L}]+", _regex.UNICODE)
except Exception:
    # Python stdlib `re` doesn't support \p{L}; fallback: keep latin/cyrillic only.
    LETTERS_RE = re.compile(r"[^A-Za-zА-Яа-яЁё]+")


def keyify(s: str) -> str:
    s = s.strip().lower()
    s = WS_RE.sub(" ", s)
    s = DIGIT_RE.sub("#", s)
    s = EDGE_PUNCT_RE.sub("", s)
    return s


def sigify(s: str, *, take: int = 48) -> str:
    """
    Fuzzier signature to survive OCR noise:
    - keep letters only
    - lowercase
    - take first N chars
    """
    s = s.strip().lower()
    s = LETTERS_RE.sub("", s)
    return s[:take]


def split_paras(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paras = [p.strip() for p in text.split("\n\n")]
    return [p for p in paras if p]


def detect_patterns(
    pages: list[dict[str, Any]],
    *,
    min_pages: int,
    max_para_len: int,
    sample_first_n: int = 2,
    sample_last_n: int = 2,
) -> tuple[set[str], set[str], dict[str, int], dict[str, int]]:
    top = Counter()
    bottom = Counter()

    for rec in pages:
        text = str(rec.get("text_normalized") or "")
        paras = split_paras(text)
        if not paras:
            continue

        for p in paras[:sample_first_n]:
            if len(p) <= max_para_len:
                k = sigify(p)
                if len(k) >= 16:
                    top[k] += 1

        for p in paras[-sample_last_n:]:
            if len(p) <= max_para_len:
                k = sigify(p)
                if len(k) >= 16:
                    bottom[k] += 1

    top_patterns = {k for k, c in top.items() if c >= min_pages}
    bottom_patterns = {k for k, c in bottom.items() if c >= min_pages}
    return top_patterns, bottom_patterns, dict(top), dict(bottom)


def strip_page(
    text: str, top_patterns: set[str], bottom_patterns: set[str], *, max_para_len: int
) -> str:
    paras = split_paras(text)
    if not paras:
        return ""

    # strip repeated short paras from the beginning
    changed = True
    while paras and changed:
        changed = False
        if len(paras[0]) <= max_para_len and sigify(paras[0]) in top_patterns:
            paras.pop(0)
            changed = True

    # strip repeated short paras from the end
    changed = True
    while paras and changed:
        changed = False
        if len(paras[-1]) <= max_para_len and sigify(paras[-1]) in bottom_patterns:
            paras.pop()
            changed = True

    return "\n\n".join(paras).strip()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Detect and strip repeating OCR headers/footers in normalized.json"
    )
    ap.add_argument("--in", dest="inp", required=True, help="Input normalized.json")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument(
        "--min-pages",
        type=int,
        default=40,
        help="Min pages a short paragraph must repeat to be considered header/footer",
    )
    ap.add_argument(
        "--max-para-len",
        type=int,
        default=80,
        help="Only paragraphs up to this length are considered for header/footer stripping",
    )
    args = ap.parse_args()

    in_path = Path(args.inp)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pages: list[dict[str, Any]] = json.loads(in_path.read_text(encoding="utf-8"))

    top_patterns, bottom_patterns, top_counts, bottom_counts = detect_patterns(
        pages, min_pages=args.min_pages, max_para_len=args.max_para_len
    )

    cleaned: list[dict[str, Any]] = []
    stripped_top = 0
    stripped_bottom = 0

    txt_lines: list[str] = []
    for rec in pages:
        page_no = int(rec.get("page"))
        text = str(rec.get("text_normalized") or "")
        before_paras = split_paras(text)
        after = strip_page(text, top_patterns, bottom_patterns, max_para_len=args.max_para_len)
        after_paras = split_paras(after)

        if before_paras and after_paras:
            if before_paras[0] != after_paras[0]:
                stripped_top += 1
            if before_paras[-1] != after_paras[-1]:
                stripped_bottom += 1
        elif before_paras and not after_paras:
            stripped_top += 1
            stripped_bottom += 1

        rec2 = dict(rec)
        rec2["text_clean"] = after
        rec2["text_clean_len"] = len(after)
        cleaned.append(rec2)

        if after:
            txt_lines.append(f"\n\n===== PAGE {page_no} =====\n")
            txt_lines.append(after)

    (out_dir / "cleaned.json").write_text(
        json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "cleaned.txt").write_text("".join(txt_lines).strip() + "\n", encoding="utf-8")

    report = {
        "min_pages": args.min_pages,
        "max_para_len": args.max_para_len,
        "top_patterns_count": len(top_patterns),
        "bottom_patterns_count": len(bottom_patterns),
        "stripped_top_pages": stripped_top,
        "stripped_bottom_pages": stripped_bottom,
        "top_patterns": sorted(
            [{"key": k, "count": top_counts.get(k, 0)} for k in top_patterns],
            key=lambda x: (-x["count"], x["key"]),
        ),
        "bottom_patterns": sorted(
            [{"key": k, "count": bottom_counts.get(k, 0)} for k in bottom_patterns],
            key=lambda x: (-x["count"], x["key"]),
        ),
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"pages={len(pages)} top_patterns={len(top_patterns)} bottom_patterns={len(bottom_patterns)} "
        f"stripped_top_pages={stripped_top} stripped_bottom_pages={stripped_bottom}"
    )


if __name__ == "__main__":
    main()
