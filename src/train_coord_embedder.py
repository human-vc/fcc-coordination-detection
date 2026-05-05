"""Contrastive fine-tune SBERT on uploader-pair supervision from BuzzFeed.

Uses MultipleNegativesRankingLoss (Henderson et al. 2017): for each (anchor,
positive) pair in a batch, all *other* positives in the same batch are
in-batch hard negatives. This is the standard contrastive recipe for
sentence-transformers (cf. Reimers & Gurevych SBERT paper).

The novel objective for this paper: positive pairs are comments from the
*same Box.com uploader account* in the FCC 17-108 bulk-upload FOIA file.
Two comments uploaded by the same account are taken to be from the same
coordination source — paraphrases of a shared template, identical text
copies, or campaign variants.

Output:
  models/coord_embedder/  — fine-tuned SBERT (drop-in for all-MiniLM-L6-v2)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main(*, base_model: str = "all-MiniLM-L6-v2",
         batch_size: int = 64,
         epochs: int = 1,
         max_seq_length: int = 128,
         output_dir: Path | None = None,
         max_pairs: int | None = None) -> None:
    output_dir = output_dir or (MODELS / "coord_embedder")
    pairs_path = PROC / "contrastive_pairs.parquet"
    if not pairs_path.exists():
        raise SystemExit(f"missing {pairs_path}; run build_contrastive_pairs.py first")

    dev = device()
    print(f"device: {dev}")

    print(f"loading pairs from {pairs_path}...")
    df = pq.read_table(pairs_path).to_pandas()
    if max_pairs:
        df = df.sample(n=min(max_pairs, len(df)), random_state=0).reset_index(drop=True)
    print(f"  using {len(df):,} pairs")

    examples = [InputExample(texts=[a, p]) for a, p in
                zip(df["anchor"].tolist(), df["positive"].tolist())]
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size,
                        drop_last=True)

    print(f"loading base model {base_model}...")
    model = SentenceTransformer(base_model, device=dev)
    model.max_seq_length = max_seq_length

    loss = losses.MultipleNegativesRankingLoss(model)

    n_steps = len(loader) * epochs
    warmup = max(100, int(0.1 * n_steps))
    print(f"training: {epochs} epochs × {len(loader)} batches = {n_steps} steps; "
          f"warmup={warmup}")

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=warmup,
        show_progress_bar=True,
        output_path=str(output_dir),
        save_best_model=True,
    )
    print(f"\nwrote fine-tuned model to {output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="all-MiniLM-L6-v2")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-seq-length", type=int, default=128)
    p.add_argument("--max-pairs", type=int, default=None,
                   help="cap training set size for fast experimentation")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()
    main(base_model=args.base_model, batch_size=args.batch_size,
         epochs=args.epochs, max_seq_length=args.max_seq_length,
         max_pairs=args.max_pairs, output_dir=args.output_dir)
