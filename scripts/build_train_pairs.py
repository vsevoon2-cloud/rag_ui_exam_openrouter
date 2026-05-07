import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examrag.retrieval import LocalIndex  # noqa: E402


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build train pairs and hard negatives from synthetic queries.")
    ap.add_argument("--synth", default="out/synth_queries.jsonl")
    ap.add_argument("--index-dir", default="out/chroma_db_v3")
    ap.add_argument("--collection", default="examrag")
    ap.add_argument("--embed-model", default="intfloat/multilingual-e5-large")
    ap.add_argument("--out-pairs", default="out/train_pairs.jsonl")
    ap.add_argument("--out-triplets", default="out/train_triplets.jsonl")
    ap.add_argument("--negatives", type=int, default=3)
    args = ap.parse_args()

    index = LocalIndex(Path(args.index_dir), args.collection, args.embed_model)
    pairs_path = Path(args.out_pairs)
    triplets_path = Path(args.out_triplets)
    pairs_path.parent.mkdir(parents=True, exist_ok=True)

    pair_count = 0
    triplet_count = 0
    with open(args.synth, "r", encoding="utf-8") as src, pairs_path.open("w", encoding="utf-8") as pairs_out, triplets_path.open("w", encoding="utf-8") as triplets_out:
        for line in src:
            try:
                row = json.loads(line)
            except Exception:
                continue
            page = int(row.get("page") or 0)
            questions = row.get("questions") or []
            source_file = Path(str(row.get("source_file") or ""))
            if not page or not source_file.exists():
                continue
            positive_text = normalize(source_file.read_text(encoding="utf-8", errors="replace"))
            if not positive_text:
                continue

            for question in questions:
                q = normalize(str(question))
                if len(q.split()) < 4:
                    continue
                pair = {"query": q, "positive": positive_text, "page": page}
                pairs_out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                pair_count += 1

                hits = index.search(q, top_k=max(args.negatives * 3, 10), mode="hybrid")
                negatives = []
                for hit in hits:
                    hit_page = int(hit.get("page") or 0)
                    hit_text = normalize(str(hit.get("text") or ""))
                    if not hit_text or hit_page == page:
                        continue
                    negatives.append(hit_text)
                    if len(negatives) >= args.negatives:
                        break
                for negative in negatives:
                    triplets_out.write(json.dumps({"query": q, "positive": positive_text, "negative": negative, "page": page}, ensure_ascii=False) + "\n")
                    triplet_count += 1

    print(f"pairs={pair_count} triplets={triplet_count}")


if __name__ == "__main__":
    main()

