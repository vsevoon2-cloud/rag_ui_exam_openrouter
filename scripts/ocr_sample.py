import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import fitz  # pymupdf
import numpy as np


def render_page(doc: fitz.Document, page_index: int, scale: float) -> np.ndarray:
    page = doc.load_page(page_index)
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def preprocess_for_ocr(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # mild denoise + contrast normalize
    gray = cv2.fastNlMeansDenoising(gray, None, h=12, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # binarize (works reasonably for low-DPI scans)
    thr = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )

    # remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    return thr


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render + preprocess + OCR sample pages from a scanned PDF.")
    ap.add_argument("--pdf", required=True, help="Path to PDF")
    ap.add_argument("--out", default="out/ocr_sample", help="Output directory")
    ap.add_argument("--start", type=int, default=1, help="Start page (1-based)")
    ap.add_argument("--count", type=int, default=5, help="Number of pages")
    ap.add_argument("--scale", type=float, default=4.0, help="Render scale (4.0 ~= 4x pixels)")
    ap.add_argument("--langs", default="ru,en", help="Comma-separated EasyOCR languages, e.g. ru,en")
    ap.add_argument(
        "--gpu",
        action="store_true",
        help="Force GPU for EasyOCR (requires CUDA-enabled PyTorch). Defaults to auto-detect.",
    )
    ap.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU for EasyOCR (overrides --gpu).",
    )
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    out_dir = Path(args.out)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "images")
    ensure_dir(out_dir / "preprocessed")

    import easyocr  # lazy import (downloads models on first run)

    langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    use_gpu = False
    if args.cpu:
        use_gpu = False
    elif args.gpu:
        use_gpu = True
    else:
        try:
            import torch

            use_gpu = bool(torch.cuda.is_available())
        except Exception:
            use_gpu = False

    reader = easyocr.Reader(langs, gpu=use_gpu)
    print(f"EasyOCR device: {'gpu' if use_gpu else 'cpu'}")

    doc = fitz.open(str(pdf_path))
    start0 = max(args.start - 1, 0)
    end0 = min(start0 + args.count, doc.page_count)

    results: list[dict[str, Any]] = []
    text_out_lines: list[str] = []

    for page_index in range(start0, end0):
        page_no = page_index + 1
        bgr = render_page(doc, page_index, args.scale)
        img_path = out_dir / "images" / f"page_{page_no:04d}.png"
        cv2.imwrite(str(img_path), bgr)

        pre = preprocess_for_ocr(bgr)
        pre_path = out_dir / "preprocessed" / f"page_{page_no:04d}_pre.png"
        cv2.imwrite(str(pre_path), pre)

        # EasyOCR expects RGB or grayscale; provide RGB for best detector performance
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # paragraph=True changes return shape across versions; keep it stable.
        ocr = reader.readtext(rgb, detail=1, paragraph=False)

        # Sort boxes top-to-bottom then left-to-right for a readable plain text output.
        def sort_key(entry: Any) -> tuple[float, float]:
            bbox = entry[0]
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            return (float(min(ys)), float(min(xs)))

        ocr_sorted = sorted(ocr, key=sort_key)

        page_text_parts: list[str] = []
        for bbox, text, conf in ocr_sorted:
            text = str(text).strip()
            try:
                conf_f = float(conf)
            except Exception:
                conf_f = 0.0
            if text and conf_f >= 0.3:
                page_text_parts.append(text)

        page_text = "\n".join(page_text_parts).strip()

        results.append(
            {
                "page": page_no,
                "image": os.path.relpath(img_path, out_dir),
                "preprocessed": os.path.relpath(pre_path, out_dir),
                "text_len": len(page_text),
                "text": page_text,
            }
        )

        text_out_lines.append(f"\n\n===== PAGE {page_no} =====\n")
        text_out_lines.append(page_text)

        print(f"page {page_no}: text_len={len(page_text)} blocks={len(page_text_parts)}")

    (out_dir / "result.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "result.txt").write_text("".join(text_out_lines), encoding="utf-8")

    md = ["# OCR sample\n"]
    for r in results:
        md.append(f"\n## Page {r['page']}\n")
        md.append(f"\n- image: `{r['image']}`\n- preprocessed: `{r['preprocessed']}`\n- text_len: {r['text_len']}\n\n")
        md.append("```text\n")
        md.append(r["text"][:5000])
        md.append("\n```\n")
    (out_dir / "result.md").write_text("".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
