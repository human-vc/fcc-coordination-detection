# FCC Coordination Detection

FDR-controlled coordination cluster detection on partially-labeled signed graphs, evaluated on the FCC 2017 "Restoring Internet Freedom" comment corpus (~24M comments) with NY AG investigation labels as ground truth.

## Working hypothesis

Given a signed similarity graph G = (V, E+, E-) constructed from regulatory comments, a partial label set L from publicly disclosed coordination campaigns, and a candidate-cluster generator producing S_1, ..., S_m, an e-BH procedure on cluster-level e-values yields a rejection set R with FDR ≤ α, finite-sample valid under arbitrary cluster-overlap dependence.

## Data sources

- **Comment corpus.** `slnader/fcc-comments` on HuggingFace (Stanford RegLab). 24M comments from FCC Docket 17-108 (Restoring Internet Freedom), with metadata, commenter-type annotations, and citation flags. License: CC-BY-NC-SA-4.0.
- **Coordination labels.** NY Attorney General 2021 report, "Fake Comments: How U.S. Companies and Partisans Hack Democracy" — identifies ~18M fraudulent submissions, including 7,700 stolen identities used for fake congressional letters.
- **Held-out evaluation.** EPA endangerment-finding 2025 docket (EPA-HQ-OAR-2025-0194) with 169 EPA-disclosed mass campaigns. Pulled separately via Regulations.gov v4 API.

## Repository layout

```
fcc-coordination-detection/
├── data/
│   ├── raw/          # source archives, not committed
│   ├── processed/    # embeddings, similarity graphs, not committed
│   └── labels/       # NY AG ground truth merged with corpus
├── src/
│   ├── ingest.py     # unpack fcc.tar.gz, load into duckdb
│   ├── embed.py      # MiniLM embeddings of express comments
│   ├── graph.py      # similarity graph construction (signed)
│   ├── clusters.py   # candidate cluster generation
│   ├── evalues.py    # e-value computation per cluster
│   ├── ebh.py        # e-BH procedure with FDR control
│   └── eval.py       # precision/recall vs NY AG labels
├── notebooks/        # exploratory analysis
├── results/          # tables, figures
└── requirements.txt
```

## Quick start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Ingest (assumes fcc.tar.gz already downloaded to data/raw/)
python src/ingest.py

# Pilot: embed a 100K comment subsample and visualize similarity structure
python src/embed.py --subsample 100000
python src/graph.py --threshold 0.95
```

## References

- Nader, S. (2022). "Information Retrieval from Public Consultations." *Policy & Internet* 14(4). DOI: 10.1002/poi3.327
- Marandon, A. (2024). "Conformal link prediction for false discovery rate control." *TEST* 33(4).
- Chugg et al. (2023). "Auditing Fairness by Betting." NeurIPS spotlight.
- Blohm, Chen, Neumann, Gionis (2025). "Discovering Opinion Intervals from Conflicts in Signed Graphs." NeurIPS oral.
- NY AG (2021). "Fake Comments: How U.S. Companies and Partisans Hack Democracy."

## License

Code: MIT. Data: see upstream licenses (CC-BY-NC-SA-4.0 for the FCC corpus).
