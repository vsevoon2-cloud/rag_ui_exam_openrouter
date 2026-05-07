from .cleaner import clean_text


def strict_clean_markdown(md: str) -> str:
    return clean_text(md, strict=True).strip()
