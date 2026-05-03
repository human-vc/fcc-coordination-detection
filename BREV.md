# Running the embedding stage on Brev

The 3.8M comment embedding job takes ~6 hours on M3 Pro MPS but ~3–5 minutes on a single A100. Run it on Brev.

## What you sync up

You only need two parquet files for embedding:

```
data/processed/comments.parquet      400 MB  (3.8M unique comment texts)
data/processed/submissions.parquet  1.0 GB  (24M submissions, joined for template_size labels)
```

Total upload: ~1.4 GB. Don't sync anything else.

## What you sync back

```
data/processed/embeddings.npy           ~3 GB  (3.8M × 384 × float16)
data/processed/embedding_index.parquet  ~50 MB
```

## Brev workflow (~10 min including transfer)

```bash
# 1. Spin up a single-A100 instance (or any CUDA GPU, MiniLM is small)
brev create fcc-embed --gpu A100

# 2. SSH in
brev shell fcc-embed

# 3. On the Brev box: clone + install (do not push the repo to GitHub yet,
#    just rsync the source files you need)
mkdir -p ~/work && cd ~/work
# from your laptop, in another terminal:
#   rsync -avz --exclude='.venv' --exclude='data' \
#     /Users/jacobcrainic/fcc-coordination-detection/ \
#     brev:~/work/fcc-coordination-detection/
#   rsync -avz \
#     /Users/jacobcrainic/fcc-coordination-detection/data/processed/comments.parquet \
#     /Users/jacobcrainic/fcc-coordination-detection/data/processed/submissions.parquet \
#     brev:~/work/fcc-coordination-detection/data/processed/

# 4. On the Brev box: setup
cd ~/work/fcc-coordination-detection
pip install -r requirements.txt

# 5. Run embedding (auto-detects CUDA, defaults to batch_size=1024)
python src/embed.py

# 6. From your laptop: pull results back
#   rsync -avz \
#     brev:~/work/fcc-coordination-detection/data/processed/embeddings.npy \
#     brev:~/work/fcc-coordination-detection/data/processed/embedding_index.parquet \
#     /Users/jacobcrainic/fcc-coordination-detection/data/processed/

# 7. Tear down to stop billing
brev stop fcc-embed
brev delete fcc-embed
```

## Expected throughput

| Device | Batch size | Throughput | 3.8M rows |
|---|---|---|---|
| A100 80GB | 1024 | ~15K/s | 4 min |
| L40S | 1024 | ~10K/s | 6 min |
| L4 | 512 | ~5K/s | 13 min |
| M3 Pro MPS | 256 | ~170/s | ~6 hours |

L4 is the cheapest Brev option that still beats local by 30x.

## After embedding

The downstream pipeline (kNN graph, clustering, e-values, e-BH) is CPU-heavy and runs in minutes locally on the M3 Pro. Don't keep the GPU box running for those stages — pull the embeddings back and continue locally.
