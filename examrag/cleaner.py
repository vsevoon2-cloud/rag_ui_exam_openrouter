import re


FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*$", re.MULTILINE)
WS_RE = re.compile(r"[ \t\u00A0]{2,}")
BROKEN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
WRAPPER_LINE_RE = re.compile(r"^\s*(okay|sure|alright)[, ]+here('s| is)\b.*$", re.IGNORECASE)
WRAPPER_LINE_RE_ALT = re.compile(r"^\s*here('s| is)\s+the\s+(ocr|ocred|ocr'd|extracted|converted)\b.*$", re.IGNORECASE)
WRAPPER_LINE_RE2 = re.compile(r"^\s*I('ve| have)\s+(kept|preserved)\s+the\s+table\s+structure\b.*$", re.IGNORECASE)
WRAPPER_LINE_RE3 = re.compile(r"^\s*I('ve| have)\s+(also\s+)?added\b.*(description|descriptions|brief)\b.*$", re.IGNORECASE)
WRAPPER_LINE_RE4 = re.compile(r"^\s*certainly!?\s+here('s| is)\b.*$", re.IGNORECASE)
WRAPPER_LINE_RE5 = re.compile(r"^\s*\*?\s*\*\*\s*content\s+from\s+the\s+image\s*:.*$", re.IGNORECASE)
STRICT_NOISE_PATTERNS = [
    re.compile(r"^\s*(figure|diagram|image|anatomical diagram)\b.*$", re.IGNORECASE),
    re.compile(r"^\s*\*?\s*(figure|diagram|image)\s*[:.-].*$", re.IGNORECASE),
    re.compile(r"^\s*(label|description)\s*\|\s*.*$", re.IGNORECASE),
    re.compile(r"^\s*\*?\s*the image shows\b.*$", re.IGNORECASE),
    re.compile(r"^\s*\*?\s*content from the cells\b.*$", re.IGNORECASE),
]
BROKEN_IMAGE_SRC_PATTERNS = (
    "attachment:",
    "/tmp/",
    "/static/",
    "image.png",
    "image-url",
    "placeholder_image",
    "diagram_description_needed",
    "original_image_crop_",
)
ENGLISH_HEAVY_RE = re.compile(r"[A-Za-z]")
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")


def _looks_like_english_noise(line: str) -> bool:
    lower = line.lower()
    if "ocr" in lower or "markdown" in lower:
        return True
    latin = len(ENGLISH_HEAVY_RE.findall(line))
    cyr = len(CYRILLIC_RE.findall(line))
    if latin >= 12 and cyr == 0:
        return True
    if latin >= 18 and latin > cyr * 3:
        return True
    return False


def clean_text(s: str, *, strict: bool = False) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = FENCE_RE.sub("", s)
    s = BROKEN_IMAGE_RE.sub("", s)

    out_lines: list[str] = []
    for line in s.split("\n"):
        stripped = line.strip()
        if not stripped:
            out_lines.append("")
            continue
        if WRAPPER_LINE_RE.match(stripped):
            continue
        if WRAPPER_LINE_RE_ALT.match(stripped):
            continue
        if WRAPPER_LINE_RE2.match(stripped):
            continue
        if WRAPPER_LINE_RE3.match(stripped):
            continue
        if WRAPPER_LINE_RE4.match(stripped):
            continue
        if WRAPPER_LINE_RE5.match(stripped):
            continue
        if strict:
            if any(pattern.match(stripped) for pattern in STRICT_NOISE_PATTERNS):
                continue
            if any(token in stripped for token in BROKEN_IMAGE_SRC_PATTERNS):
                continue
            if _looks_like_english_noise(stripped):
                continue
        out_lines.append(line.rstrip())

    cleaned = "\n".join(out_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = WS_RE.sub(" ", cleaned)
    return cleaned.strip() + "\n"
