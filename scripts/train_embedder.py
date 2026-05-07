import argparse
import json
from pathlib import Path

import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader


def load_pairs(path: Path) -> list[InputExample]:
    examples: list[InputExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                query = str(row["query"]).strip()
                positive = str(row["positive"]).strip()
            except Exception:
                continue
            if not query or not positive:
                continue
            examples.append(InputExample(texts=[f"query: {query}", f"passage: {positive}"]))
    return examples


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune a retrieval embedder on synthetic query-positive pairs.")
    ap.add_argument("--pairs", default="out/train_pairs.jsonl")
    ap.add_argument("--base-model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--out-model", default="out/models/e5-base-synth")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    pairs = load_pairs(Path(args.pairs))
    if not pairs:
        raise SystemExit(f"No pairs found in {args.pairs}")

    model = SentenceTransformer(args.base_model)
    train_loader = DataLoader(pairs, shuffle=True, batch_size=args.batch_size, drop_last=False)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = int(len(train_loader) * max(args.epochs, 1) * args.warmup_ratio)
    use_amp = bool(args.fp16 and torch.cuda.is_available())

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        show_progress_bar=True,
        output_path=args.out_model,
        use_amp=use_amp,
        checkpoint_path=None,
        checkpoint_save_steps=0,
    )

    print(f"saved {args.out_model}")


if __name__ == "__main__":
    main()

