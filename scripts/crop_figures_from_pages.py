import argparse
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMG_MD_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def page_no_from_md(md_path: Path) -> int | None:
    m = re.match(r"page_(\d+)\.md$", md_path.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def find_candidate_regions(bgr: np.ndarray, *, top_frac: float, bottom_frac: float) -> list[tuple[int, int, int, int]]:
    """
    Heuristic region detector for scanned pages:
    - binarize foreground
    - merge nearby ink with morphology
    - find connected components
    - keep larger rectangular regions (likely figures/tables), exclude full-page blocks
    """
    h, w = bgr.shape[:2]
    y_top = int(h * top_frac)
    y_bottom = int(h * (1.0 - bottom_frac))

    roi = bgr[y_top:y_bottom, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Merge text into blocks; figures/tables become solid-ish blocks too.
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    merged = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, k1, iterations=2)
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, k2, iterations=1)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[tuple[int, int, int, int]] = []
    page_area = float(w * h)
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        # shift back to full image coords
        y += y_top
        area = float(ww * hh)
        if area < page_area * 0.02:
            continue
        if ww < w * 0.25 or hh < h * 0.06:
            continue
        if area > page_area * 0.75:
            continue
        boxes.append((x, y, ww, hh))

    # Sort top-to-bottom
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def expand_box(box: tuple[int, int, int, int], w: int, h: int, pad: int) -> tuple[int, int, int, int]:
    x, y, ww, hh = box
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + ww + pad)
    y1 = min(h, y + hh + pad)
    return (x0, y0, x1 - x0, y1 - y0)


def write_crops(
    *,
    page_jpg: Path,
    out_dir: Path,
    page_no: int,
    top_frac: float,
    bottom_frac: float,
    pad: int,
    max_crops: int,
) -> list[str]:
    bgr = cv2.imread(str(page_jpg))
    if bgr is None:
        return []
    h, w = bgr.shape[:2]
    boxes = find_candidate_regions(bgr, top_frac=top_frac, bottom_frac=bottom_frac)
    if max_crops > 0:
        boxes = boxes[:max_crops]

    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    rel_paths: list[str] = []
    for i, box in enumerate(boxes, start=1):
        x, y, ww, hh = expand_box(box, w, h, pad)
        crop = bgr[y : y + hh, x : x + ww]
        crop_path = crops_dir / f"page_{page_no:04d}_fig_{i:02d}.png"
        cv2.imwrite(str(crop_path), crop)
        rel_paths.append(f"crops/{crop_path.name}")
    return rel_paths


def rewrite_md(md_path: Path, crop_paths: list[str], *, keep_alt: bool) -> bool:
    text = md_path.read_text(encoding="utf-8", errors="replace")
    matches = list(IMG_MD_RE.finditer(text))
    if not matches:
        return False

    out = []
    last = 0
    crop_i = 0
    for m in matches:
        out.append(text[last : m.start()])
        alt = (m.group(1) or "").strip()
        if crop_i < len(crop_paths):
            new_src = crop_paths[crop_i]
            crop_i += 1
            new_alt = alt if (keep_alt and alt) else "Figure"
            out.append(f"![{new_alt}]({new_src})")
        else:
            # No crop available: comment out the image
            out.append(f"<!-- IMAGE REMOVED: alt={alt} src={m.group(2).strip()} -->")
        last = m.end()
    out.append(text[last:])
    new_text = "".join(out)
    if new_text != text:
        md_path.write_text(new_text, encoding="utf-8")
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Crop figures/tables from selected pages and rewrite MD image links.")
    ap.add_argument("--image-refs", required=True, help="Path to image_refs.json produced by find_image_refs.py")
    ap.add_argument("--pages-jpg", required=True, help="Directory with page_XXXX.jpg (rendered pages)")
    ap.add_argument("--md-pages", required=True, help="Directory with page_XXXX.md")
    ap.add_argument("--out", required=True, help="Output directory (will create crops/ inside)")
    ap.add_argument("--top-frac", type=float, default=0.10, help="Exclude header area from detection")
    ap.add_argument("--bottom-frac", type=float, default=0.08, help="Exclude footer area from detection")
    ap.add_argument("--pad", type=int, default=10, help="Crop padding (pixels)")
    ap.add_argument("--max-crops", type=int, default=6, help="Max crops per page (0 = no limit)")
    ap.add_argument("--keep-alt", action="store_true", help="Keep original alt text if present")
    args = ap.parse_args()

    refs_path = Path(args.image_refs)
    pages_jpg = Path(args.pages_jpg)
    md_pages = Path(args.md_pages)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    refs: list[dict[str, Any]] = json.loads(refs_path.read_text(encoding="utf-8"))
    changed = 0
    cropped_pages = 0

    for rec in refs:
        page_no = int(rec.get("page"))
        jpg = pages_jpg / f"page_{page_no:04d}.jpg"
        md = md_pages / f"page_{page_no:04d}.md"
        if not jpg.exists() or not md.exists():
            continue

        crop_paths = write_crops(
            page_jpg=jpg,
            out_dir=out_dir,
            page_no=page_no,
            top_frac=args.top_frac,
            bottom_frac=args.bottom_frac,
            pad=args.pad,
            max_crops=args.max_crops,
        )
        if crop_paths:
            cropped_pages += 1
        if rewrite_md(md, crop_paths, keep_alt=bool(args.keep_alt)):
            changed += 1

    print(f"pages_with_refs={len(refs)} pages_cropped={cropped_pages} md_changed={changed} crops_dir={out_dir/'crops'}")


if __name__ == "__main__":
    main()

