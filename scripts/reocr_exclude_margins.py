import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2


def bbox_y_bounds(bbox: Any) -> tuple[float, float]:
    ys = [float(p[1]) for p in bbox]
    return min(ys), max(ys)


def bbox_x_bounds(bbox: Any) -> tuple[float, float]:
    xs = [float(p[0]) for p in bbox]
    return min(xs), max(xs)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Re-run EasyOCR on saved page images, excluding top/bottom margins (headers/footers)."
    )
    ap.add_argument("--in", dest="inp", required=True, help="Input normalized.json (for page list + rel image paths)")
    ap.add_argument("--base", required=True, help="Base dir that contains the images folder (e.g. out/ocr_all)")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--top-frac", type=float, default=0.10, help="Exclude boxes whose y_max is within top fraction")
    ap.add_argument(
        "--bottom-frac", type=float, default=0.08, help="Exclude boxes whose y_min is within bottom fraction"
    )
    ap.add_argument("--min-conf", type=float, default=0.3, help="Min confidence to keep a text box")
    ap.add_argument("--langs", default="ru,en", help="EasyOCR languages (comma-separated)")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N pages (0 = all)")
    ap.add_argument("--start", type=int, default=1, help="Start page number (1-based)")
    ap.add_argument("--count", type=int, default=0, help="How many pages to process from start (0 = all)")
    args = ap.parse_args()

    in_path = Path(args.inp)
    base_dir = Path(args.base)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pages: list[dict[str, Any]] = json.loads(in_path.read_text(encoding="utf-8"))
    start0 = max(args.start - 1, 0)
    if args.count and args.count > 0:
        pages = pages[start0 : start0 + args.count]
    else:
        pages = pages[start0:]
    if args.limit and args.limit > 0:
        pages = pages[: args.limit]

    import torch
    import easyocr

    langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    use_gpu = bool(torch.cuda.is_available())
    reader = easyocr.Reader(langs, gpu=use_gpu)
    print(f"EasyOCR device: {'gpu' if use_gpu else 'cpu'}")

    out_pages: list[dict[str, Any]] = []
    txt_lines: list[str] = []

    for i, rec in enumerate(pages, start=1):
        page_no = int(rec.get("page"))
        rel = rec.get("image")
        if not rel:
            out_pages.append({**rec, "text_body": "", "text_body_len": 0})
            continue

        img_path = (base_dir / rel).resolve()
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            out_pages.append({**rec, "text_body": "", "text_body_len": 0})
            continue

        h, w = bgr.shape[:2]
        top_y = h * args.top_frac
        bottom_y = h * (1.0 - args.bottom_frac)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        items = reader.readtext(rgb, detail=1, paragraph=False)

        kept: list[tuple[float, float, str]] = []
        for bbox, text, conf in items:
            try:
                c = float(conf)
            except Exception:
                c = 0.0
            t = str(text).strip()
            if not t or c < args.min_conf:
                continue

            y0, y1 = bbox_y_bounds(bbox)
            # Exclude header/footer boxes
            if y1 <= top_y:
                continue
            if y0 >= bottom_y:
                continue

            x0, _x1 = bbox_x_bounds(bbox)
            kept.append((y0, x0, t))

        kept.sort(key=lambda z: (z[0], z[1]))
        text_body = "\n".join(t for _y, _x, t in kept).strip()

        rec2 = dict(rec)
        rec2["text_body"] = text_body
        rec2["text_body_len"] = len(text_body)
        out_pages.append(rec2)

        if text_body:
            txt_lines.append(f"\n\n===== PAGE {page_no} =====\n")
            txt_lines.append(text_body)

        if page_no % 25 == 0:
            print(f"processed page {page_no} ({i}/{len(pages)})")

    (out_dir / "body.json").write_text(json.dumps(out_pages, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "body.txt").write_text("".join(txt_lines).strip() + "\n", encoding="utf-8")
    print(f"pages={len(out_pages)} out={(out_dir / 'body.json')}")


if __name__ == "__main__":
    main()
