"""LLM-as-judge for cluster-level coordination classification.

For each candidate cluster, sample 3 representative comments and ask Claude
Haiku 4.5 to classify the cluster as a coordinated paraphrase campaign or
organic. Returns P(coordinated) per cluster.

Calibration against the FOIA-attributed gold set is done in
src/ppi_calibration.py. Together they yield PPI-calibrated compound e-values
that constitute the new ML methodology contribution.

Cost: ~1000 calls × (~500 in tokens + ~50 out tokens) on Haiku 4.5 ≈ $0.80.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from anthropic import Anthropic

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"

PROMPT_TEMPLATE = """You are reviewing comments submitted to a U.S. federal regulatory docket. \
Below are 3 comments that an automated clustering algorithm grouped together because \
their text embeddings were highly similar.

Your task: classify whether these comments are part of a *coordinated paraphrase \
campaign* (a single template circulated to multiple submitters, possibly with \
mechanical paraphrase variation), versus *organic similar opinions* (separate \
people expressing related but independently composed views).

Coordinated indicators: identical phrases, near-duplicate sentence structure, \
boilerplate "Dear FCC Chairman" template framing, leftover instructions like \
"please enter your comment here", form-letter signatures, mass-mailing artifacts.

Organic indicators: similar sentiment expressed in clearly different words, \
personal anecdotes, independent reasoning, varied structure and length.

Comment 1:
{c1}

Comment 2:
{c2}

Comment 3:
{c3}

Respond with exactly two lines:
COORDINATED: yes|no
CONFIDENCE: 0.0 to 1.0 (your confidence in the classification)"""


def parse_response(text: str) -> tuple[bool | None, float | None]:
    """Extract (coordinated, confidence) from response."""
    coord = None
    conf = None
    for line in text.splitlines():
        line_l = line.strip().lower()
        if line_l.startswith("coordinated:"):
            val = line_l.split(":", 1)[1].strip()
            if val.startswith("y"):
                coord = True
            elif val.startswith("n"):
                coord = False
        elif line_l.startswith("confidence:"):
            try:
                conf = float(line_l.split(":", 1)[1].strip().split()[0])
            except Exception:
                pass
    return coord, conf


def sample_cluster_texts(cl_df: pd.DataFrame, cm_df: pd.DataFrame,
                        cluster_id: int, n_sample: int = 3,
                        rng: random.Random | None = None) -> list[str]:
    """Sample n_sample comment texts from a cluster."""
    rng = rng or random.Random(0)
    members = cl_df[cl_df["cluster_id"] == cluster_id]["comment_id"].tolist()
    if len(members) < n_sample:
        sampled_ids = members
    else:
        sampled_ids = rng.sample(members, n_sample)
    texts = []
    for cid in sampled_ids:
        row = cm_df[cm_df["comment_id"] == cid]
        if len(row):
            txt = str(row["comment_text"].iloc[0])
            # truncate excessive lengths to keep prompt costs sane
            if len(txt) > 1200:
                txt = txt[:1200] + " [truncated]"
            texts.append(txt)
    while len(texts) < n_sample:
        texts.append("[no text available]")
    return texts


def judge_cluster(client: Anthropic, texts: list[str],
                 model: str = "claude-haiku-4-5-20251001") -> dict:
    """Single LLM-judge call. Returns dict with raw + parsed."""
    prompt = PROMPT_TEMPLATE.format(c1=texts[0], c2=texts[1], c3=texts[2])
    msg = client.messages.create(
        model=model,
        max_tokens=120,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text
    coord, conf = parse_response(text)
    return {
        "raw": text,
        "coordinated": coord,
        "confidence": conf,
        "input_tokens": msg.usage.input_tokens,
        "output_tokens": msg.usage.output_tokens,
    }


def run_pipeline(*, cluster_path: Path, n_clusters: int = 1000,
                 min_cluster_size: int = 8,
                 model: str = "claude-haiku-4-5-20251001",
                 output_path: Path | None = None,
                 stratify_path: Path | None = None,
                 seed: int = 42) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY") or \
        Path(ROOT / ".env_anthropic").read_text().strip()
    client = Anthropic(api_key=api_key)

    print(f"loading clusters: {cluster_path}")
    cl = pq.read_table(cluster_path).to_pandas()
    print(f"loading comments...")
    cm = pq.read_table(PROC / "comments.parquet",
                       columns=["comment_id", "comment_text"]).to_pandas()

    # Filter to candidate clusters
    cand = cl[(cl["cluster_id"] >= 0) & (cl["cluster_size"] >= min_cluster_size)]
    cluster_ids = sorted(cand["cluster_id"].unique())
    print(f"  {len(cluster_ids):,} candidate clusters")

    # Stratified sampling by FOIA astroturf attribution
    rng = random.Random(seed)
    if stratify_path is not None and stratify_path.exists():
        att = pd.read_csv(stratify_path)
        astro = set(att.loc[att["frac_astroturf"].fillna(0) >= 0.5,
                            "cluster_id"].astype(int).tolist())
        astro_ids = [c for c in cluster_ids if int(c) in astro]
        nonastro_ids = [c for c in cluster_ids if int(c) not in astro]
        n_each = n_clusters // 2
        sampled_astro = rng.sample(astro_ids, min(n_each, len(astro_ids)))
        sampled_other = rng.sample(nonastro_ids,
                                    min(n_clusters - len(sampled_astro),
                                        len(nonastro_ids)))
        sampled = sampled_astro + sampled_other
        rng.shuffle(sampled)
        print(f"  stratified sample: {len(sampled_astro)} astroturf + "
              f"{len(sampled_other)} non-astroturf = {len(sampled)} total")
    else:
        sampled = rng.sample(cluster_ids, min(n_clusters, len(cluster_ids)))
        print(f"  uniform sample: {len(sampled)} clusters")

    # Run LLM judge
    rows = []
    in_total = 0
    out_total = 0
    t0 = time.time()
    for i, cid in enumerate(sampled):
        texts = sample_cluster_texts(cl, cm, int(cid), n_sample=3, rng=rng)
        try:
            result = judge_cluster(client, texts, model=model)
        except Exception as e:
            print(f"  [{i}] cluster {cid} ERROR: {e}")
            continue
        in_total += result["input_tokens"]
        out_total += result["output_tokens"]
        rows.append({
            "cluster_id": int(cid),
            "coordinated": result["coordinated"],
            "confidence": result["confidence"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "raw_response": result["raw"],
        })
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i+1:4d}/{len(sampled)}] {rate:.1f} clusters/s | "
                  f"in={in_total:,} out={out_total:,} tokens "
                  f"(~${(in_total/1e6 * 1.0 + out_total/1e6 * 5.0):.2f})")

    out_df = pd.DataFrame(rows)
    out_path = output_path or PROC / "llm_judge_results.parquet"
    out_df.to_parquet(out_path, compression="zstd", index=False)
    print(f"\nwrote {out_path}  ({len(out_df):,} clusters classified)")
    print(f"total tokens: {in_total:,} in, {out_total:,} out")
    if model.startswith("claude-haiku"):
        cost = (in_total / 1e6) * 1.0 + (out_total / 1e6) * 5.0
    elif model.startswith("claude-sonnet"):
        cost = (in_total / 1e6) * 3.0 + (out_total / 1e6) * 15.0
    else:
        cost = 0
    print(f"approx cost: ${cost:.2f}")
    coord_rate = out_df["coordinated"].mean()
    print(f"\nLLM coordination rate: {coord_rate:.3f}")
    print(f"confidence distribution: mean={out_df['confidence'].mean():.3f}, "
          f"std={out_df['confidence'].std():.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cluster-path", type=Path,
                   default=PROC / "clusters_leiden_r0.9.parquet")
    p.add_argument("--n-clusters", type=int, default=1000)
    p.add_argument("--min-cluster-size", type=int, default=8)
    p.add_argument("--model", default="claude-haiku-4-5-20251001")
    p.add_argument("--output-path", type=Path, default=None)
    p.add_argument("--stratify-path", type=Path,
                   default=RES / "attribution_table_r0.9.csv",
                   help="if given, stratifies sampling by attribution")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_pipeline(cluster_path=args.cluster_path,
                 n_clusters=args.n_clusters,
                 min_cluster_size=args.min_cluster_size,
                 model=args.model,
                 output_path=args.output_path,
                 stratify_path=args.stratify_path,
                 seed=args.seed)
