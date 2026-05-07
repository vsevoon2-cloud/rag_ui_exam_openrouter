import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import fitz  # pymupdf
import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def estimate_skew_angle(gray: np.ndarray) -> float:
    # Estimate skew from foreground pixels via minAreaRect.
    # Returns angle in degrees; positive -> rotate clockwise (OpenCV convention differs).
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr > 0))
    if coords.size == 0:
        return 0.0
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    # angle in [-90, 0)
    if angle < -45:
        angle = 90 + angle
    return float(angle)


def rotate(image: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 0.2:
        return image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def unsharp(gray: np.ndarray, amount: float = 1.2, sigma: float = 1.0) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    sharp = cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)
    return sharp


def preprocess(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      - pre_bgr: image to feed OCR (BGR)
      - debug_bin: binarized debug view
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    angle = estimate_skew_angle(gray)
    bgr2 = rotate(bgr, angle)
    gray2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2GRAY)
    gray2 = unsharp(gray2, amount=1.4, sigma=1.0)

    # adaptive bin for debug/optional OCR
    bin_img = cv2.adaptiveThreshold(
        gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9
    )
    return bgr2, bin_img


def ocr_paddle(ocr: Any, bgr: np.ndarray) -> list[dict[str, Any]]:
    # PaddleOCR expects RGB ndarray
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out = ocr.ocr(rgb, cls=True)
    items: list[dict[str, Any]] = []
    if not out:
        return items
    # out is list (per image) of lines; for ndarray input, often out[0] is lines
    lines = out[0] if isinstance(out, list) and len(out) == 1 and isinstance(out[0], list) else out
    for line in lines or []:
        try:
            bbox = line[0]
            text = line[1][0]
            conf = float(line[1][1])
        except Exception:
            continue
        items.append({"bbox": bbox, "text": str(text), "conf": conf})
    return items


def sort_ocr_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(item: dict[str, Any]) -> tuple[float, float]:
        bbox = item.get("bbox") or []
        try:
            ys = [float(p[1]) for p in bbox]
            xs = [float(p[0]) for p in bbox]
            return (min(ys) if ys else 0.0, min(xs) if xs else 0.0)
        except Exception:
            return (0.0, 0.0)

    return sorted(items, key=key)


def main() -> None:
    ap = argparse.ArgumentParser(description="Heavy OCR with PaddleOCR on scanned PDF pages.")
    ap.add_argument("--pdf", required=True, help="Path to PDF")
    ap.add_argument("--out", default="out/ocr_paddle", help="Output directory")
    ap.add_argument("--start", type=int, default=1, help="Start page (1-based)")
    ap.add_argument("--count", type=int, default=10, help="Number of pages")
    ap.add_argument("--scale", type=float, default=4.0, help="Render scale")
    ap.add_argument("--lang", default="ru", help="PaddleOCR lang (e.g. ru, en, ...)")  # ru covers Cyrillic
    ap.add_argument("--use-gpu", action="store_true", help="Use GPU if PaddlePaddle GPU is installed")
    ap.add_argument("--min-conf", type=float, default=0.3, help="Min confidence to keep a line")
    args = ap.parse_args()

    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception as e:
        raise SystemExit(
            "PaddleOCR is not installed. Install first, e.g.\n"
            "  python -m pip install -U paddleocr\n"
            "and PaddlePaddle (CPU or GPU) per official instructions.\n"
            f"Import error: {e}"
        )

    out_dir = Path(args.out)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "images")
    ensure_dir(out_dir / "preprocessed")

    ocr = PaddleOCR(
        lang=args.lang,
        use_angle_cls=True,
        use_gpu=bool(args.use_gpu),
        show_log=False,
    )

    pdf_path = Path(args.pdf)
    doc = fitz.open(str(pdf_path))
    start0 = max(args.start - 1, 0)
    end0 = min(start0 + args.count, doc.page_count)

    results: list[dict[str, Any]] = []
    txt_lines: list[str] = []

    for page_index in range(start0, end0):
        page_no = page_index + 1
        bgr = render_page(doc, page_index, args.scale)
        img_path = out_dir / "images" / f"page_{page_no:04d}.png"
        cv2.imwrite(str(img_path), bgr)

        pre_bgr, debug_bin = preprocess(bgr)
        pre_path = out_dir / "preprocessed" / f"page_{page_no:04d}_pre.png"
        cv2.imwrite(str(pre_path), pre_bgr)
        bin_path = out_dir / "preprocessed" / f"page_{page_no:04d}_bin.png"
        cv2.imwrite(str(bin_path), debug_bin)

        items = ocr_paddle(ocr, pre_bgr)
        items = [it for it in items if float(it.get("conf", 0.0)) >= args.min_conf and str(it.get("text", "")).strip()]
        items = sort_ocr_items(items)

        page_text = "\n".join(it["text"].strip() for it in items).strip()
        results.append(
            {
                "page": page_no,
                "image": os.path.relpath(img_path, out_dir),
                "preprocessed": os.path.relpath(pre_path, out_dir),
                "binarized": os.path.relpath(bin_path, out_dir),
                "lines": items,
                "text_len": len(page_text),
                "text": page_text,
            }
        )

        txt_lines.append(f"\n\n===== PAGE {page_no} =====\n")
        txt_lines.append(page_text)
        print(f"page {page_no}: text_len={len(page_text)} lines={len(items)}")

    (out_dir / "result.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "result.txt").write_text("".join(txt_lines).strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

