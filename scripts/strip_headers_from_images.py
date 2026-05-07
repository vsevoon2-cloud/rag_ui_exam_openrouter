import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import cv2


WS_RE = re.compile(r"\s+")
LETTERS_ONLY_RE = re.compile(r"[^A-Za-zА-Яа-яЁё]+")


def normalize_header_text(s: str) -> str:
    s = s.strip().lower()
    s = WS_RE.sub(" ", s)
    s = LETTERS_ONLY_RE.sub("", s)
    return s


def ocr_strip(reader: Any, bgr: Any, top_frac: float, bottom_frac: float) -> tuple[str, str]:
    h, w = bgr.shape[:2]
    top_h = max(1, int(h * top_frac))
    bot_h = max(1, int(h * bottom_frac))

    top = bgr[0:top_h, 0:w]
    bot = bgr[h - bot_h : h, 0:w]

    def read(img: Any) -> str:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        items = reader.readtext(rgb, detail=1, paragraph=False)
        # items: [bbox, text, conf]
        parts: list[str] = []
        for _bbox, text, conf in items:
            try:
                c = float(conf)
            except Exception:
                c = 0.0
            t = str(text).strip()
            if t and c >= 0.3:
                parts.append(t)
        return " ".join(parts).strip()

    return read(top), read(bot)


def strip_prefix(text: str, header_norm: str) -> str:
    """
    Remove a header string if it appears early in the page text (OCR noise tolerant).
    We match by letters-only normalized prefix.
    """
    if not text.strip() or not header_norm:
        return text.strip()

    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras:
        return ""

    first = paras[0]
    first_norm = normalize_header_text(first)[: max(32, len(header_norm))]
    target = header_norm[:32]
    # if header is a prefix of first paragraph after normalization, drop it
    if target and first_norm.startswith(target):
        paras = paras[1:]
    return "\n\n".join(paras).strip()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Strip repeating chapter headers/footers using OCR of top/bottom image strips."
    )
    ap.add_argument("--in", dest="inp", required=True, help="Input normalized.json")
    ap.add_argument(
        "--base",
        required=True,
        help="Base directory that contains page images referenced by 'image' field (e.g. out/ocr_all)",
    )
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--top-frac", type=float, default=0.10, help="Top strip fraction of page height")
    ap.add_argument("--bottom-frac", type=float, default=0.08, help="Bottom strip fraction of page height")
    ap.add_argument(
        "--min-pages",
        type=int,
        default=20,
        help="Min pages a header must repeat to be considered removable",
    )
    ap.add_argument("--sig-len", type=int, default=24, help="Header signature length")
    ap.add_argument("--min-sig-chars", type=int, default=8, help="Min signature chars to count")
    ap.add_argument("--langs", default="ru,en", help="EasyOCR languages (comma-separated)")
    args = ap.parse_args()

    in_path = Path(args.inp)
    base_dir = Path(args.base)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pages: list[dict[str, Any]] = json.loads(in_path.read_text(encoding="utf-8"))

    import torch
    import easyocr

    langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    reader = easyocr.Reader(langs, gpu=bool(torch.cuda.is_available()))
    print(f"EasyOCR device: {'gpu' if torch.cuda.is_available() else 'cpu'}")

    header_counts: Counter[str] = Counter()

    extracted: list[dict[str, Any]] = []
    for rec in pages:
        page_no = int(rec.get("page"))
        rel = rec.get("image")
        if not rel:
            extracted.append({**rec, "header_raw": "", "header_norm": "", "footer_raw": ""})
            continue
        img_path = (base_dir / rel).resolve()
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            extracted.append({**rec, "header_raw": "", "header_norm": "", "footer_raw": ""})
            continue

        header_raw, footer_raw = ocr_strip(reader, bgr, args.top_frac, args.bottom_frac)
        header_norm = normalize_header_text(header_raw)
        header_sig = header_norm[: args.sig_len]
        if len(header_sig) >= args.min_sig_chars:
            header_counts[header_sig] += 1

        extracted.append({**rec, "header_raw": header_raw, "header_norm": header_sig, "footer_raw": footer_raw})

        if page_no % 25 == 0:
            print(f"scanned headers: page {page_no}")

    removable = {k for k, c in header_counts.items() if c >= args.min_pages}

    cleaned: list[dict[str, Any]] = []
    removed_pages = 0
    for rec in extracted:
        header_sig = str(rec.get("header_norm") or "")
        text = str(rec.get("text_normalized") or "")
        if header_sig and header_sig in removable:
            new_text = strip_prefix(text, header_sig)
            if new_text != text.strip():
                removed_pages += 1
        else:
            new_text = text.strip()
        rec2 = dict(rec)
        rec2["text_clean2"] = new_text
        rec2["text_clean2_len"] = len(new_text)
        cleaned.append(rec2)

    report = {
        "top_frac": args.top_frac,
        "bottom_frac": args.bottom_frac,
        "min_pages": args.min_pages,
        "distinct_headers": len(header_counts),
        "removable_headers": len(removable),
        "removed_pages": removed_pages,
        "top_headers": header_counts.most_common(30),
    }

    (out_dir / "clean2.json").write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"pages={len(cleaned)} removable_headers={len(removable)} removed_pages={removed_pages}")


if __name__ == "__main__":
    main()
