import argparse
from pathlib import Path

import fitz  # pymupdf
from PIL import Image


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render PDF pages to JPG files.")
    ap.add_argument("--pdf", required=True, help="Path to PDF")
    ap.add_argument("--out", default="out/pages_jpg", help="Output directory")
    ap.add_argument("--start", type=int, default=1, help="Start page (1-based)")
    ap.add_argument("--count", type=int, default=0, help="Number of pages (0 = until end)")
    ap.add_argument("--scale", type=float, default=4.0, help="Render scale (e.g. 4.0)")
    ap.add_argument("--quality", type=int, default=85, help="JPEG quality (1-95)")
    ap.add_argument("--dpi", type=int, default=0, help="If set, override scale to approximate this DPI (A4 baseline)")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    doc = fitz.open(str(pdf_path))
    start0 = max(args.start - 1, 0)
    end0 = doc.page_count if not args.count else min(start0 + args.count, doc.page_count)

    # A4 at 72 dpi in PDF points; crude conversion: pixels ~= points * (dpi/72)
    # We'll keep scale unless user provides dpi.
    scale = float(args.scale)
    if args.dpi and args.dpi > 0:
        scale = float(args.dpi) / 72.0

    for page_index in range(start0, end0):
        page_no = page_index + 1
        out_path = out_dir / f"page_{page_no:04d}.jpg"
        if out_path.exists():
            continue
        page = doc.load_page(page_index)
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        mode = "RGB" if pix.n >= 3 else "L"
        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        if mode != "RGB":
            img = img.convert("RGB")
        img.save(out_path, format="JPEG", quality=int(args.quality), optimize=True)
        print(f"rendered page {page_no} -> {out_path}")

    print(f"done pages {args.start}..{end0} into {out_dir}")


if __name__ == "__main__":
    main()

