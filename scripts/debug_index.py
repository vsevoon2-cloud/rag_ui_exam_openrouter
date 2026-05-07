import sys
from pathlib import Path

# Allow running from scripts/ without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examrag.retrieval import LocalIndex  # noqa: E402


def main() -> None:
    index_dir = Path("out/chroma_db_v3")
    if not index_dir.exists():
        raise SystemExit(f"Not found: {index_dir} (adjust path in scripts/debug_index.py)")

    index = LocalIndex(
        index_dir=index_dir,
        collection="examrag",
        embed_model="intfloat/multilingual-e5-base",
    )

    q = "поджелудочная железа смешанная секреция"
    results = index.search(q, top_k=5, mode="hybrid")
    for r in results:
        page = r.get("page")
        text = (r.get("text") or "").replace("\n", " ").strip()
        print(f"стр. {page}: {text[:240]}")


if __name__ == "__main__":
    main()

