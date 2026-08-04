"""Run the full methodology bake-off and save results."""
import time
import pandas as pd
from pathlib import Path

from aice.bakeoff import run_grid, summarize


OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(exist_ok=True)


def main():
    print("=== Methodology bake-off (Korthauer 2019 / Lee-Ren / Bashari standard) ===")
    print()

    # Full grid following Korthauer 2019 / Lee-Ren standard
    K_list = [200, 500, 1000, 2000, 5000]
    pi_0_list = [0.5, 0.7, 0.9, 0.95]
    dependence_list = [
        ("independent", 0.0),
        ("prds_block", 0.3),
        ("prds_ar1", 0.5),
    ]
    alpha_list = [0.01, 0.05, 0.10, 0.20]
    n_reps = 200  # Lee-Ren uses 500; reducing to 200 for time, ≥200 is Korthauer floor

    n_cells = len(K_list) * len(pi_0_list) * len(dependence_list)
    print(f"Grid: {len(K_list)}K × {len(pi_0_list)}π_0 × {len(dependence_list)}dep "
          f"× {len(alpha_list)}α × {n_reps} reps = {n_cells * n_reps} trials × 7 methods")
    print()

    t0 = time.time()
    df = run_grid(
        K_list=K_list,
        pi_0_list=pi_0_list,
        dependence_list=dependence_list,
        alpha_list=alpha_list,
        n_reps=n_reps,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    df.to_parquet(OUT_DIR / "bakeoff_trials.parquet")
    print(f"saved {len(df):,} trial rows to {OUT_DIR}/bakeoff_trials.parquet")

    agg = df.groupby(
        ["method", "K", "pi_0", "dependence", "rho", "alpha"]
    ).agg(
        mean_fdr=("fdp", "mean"),
        sd_fdr=("fdp", "std"),
        mean_power=("power", "mean"),
        sd_power=("power", "std"),
        mean_k=("k_rejected", "mean"),
        mean_ap=("ap", "mean"),
        n_reps=("rep", "count"),
    ).reset_index()
    agg.to_parquet(OUT_DIR / "bakeoff_summary.parquet")
    agg.to_csv(OUT_DIR / "bakeoff_summary.csv", index=False)
    print(f"saved summary to {OUT_DIR}/bakeoff_summary.csv")

    # Headline table: method × alpha at K=2000, pi_0=0.7, independent
    headline = agg[
        (agg.K == 2000) & (agg.pi_0 == 0.7) & (agg.dependence == "independent")
    ].pivot_table(
        index="method",
        columns="alpha",
        values=["mean_fdr", "mean_power"],
    )
    print("\n=== Headline at K=2000, π_0=0.7, independent ===")
    print(headline.round(3).to_string())


if __name__ == "__main__":
    main()
