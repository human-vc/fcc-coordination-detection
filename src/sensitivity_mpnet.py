from __future__ import annotations
import time
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from scipy.stats import beta as beta_dist
from sentence_transformers import SentenceTransformer
from sklearn.metrics import average_precision_score
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data' / 'processed'
RES = ROOT / 'results'
MPNET_DIR = PROC / 'mpnet'
MPNET_DIR.mkdir(exist_ok=True)


def device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def fit_beta_2mix_em(f, n_iter=80, tol=1e-5):
    eps = 1e-6
    f = np.clip(f.astype(np.float64), eps, 1 - eps)
    logit = np.log(f / (1 - f)).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=0, n_init=5)
    gmm.fit(logit)
    z = gmm.predict(logit)
    if f[z == 0].mean() > f[z == 1].mean():
        z = 1 - z

    def fit(mu, var):
        if var <= eps:
            return 1.0, 1.0
        c = mu * (1 - mu) / var - 1
        if c <= 0:
            return 1.0, 1.0
        return max(mu * c, 1e-3), max((1 - mu) * c, 1e-3)

    mu0, var0 = float(f[z == 0].mean()), float(f[z == 0].var())
    mu1, var1 = float(f[z == 1].mean()), float(f[z == 1].var())
    a0, b0 = fit(mu0, var0)
    a1, b1 = fit(mu1, var1)
    w0 = float((z == 0).mean())
    log_lik_prev = -np.inf
    for _ in range(n_iter):
        log_p0 = beta_dist.logpdf(f, a0, b0) + np.log(max(w0, 1e-9))
        log_p1 = beta_dist.logpdf(f, a1, b1) + np.log(max(1 - w0, 1e-9))
        log_total = np.logaddexp(log_p0, log_p1)
        gamma0 = np.exp(log_p0 - log_total)
        log_lik = log_total.sum()
        if abs(log_lik - log_lik_prev) < tol * abs(log_lik_prev):
            break
        log_lik_prev = log_lik
        n0 = gamma0.sum()
        n1 = (1 - gamma0).sum()
        if n0 < 1 or n1 < 1:
            break
        mu0 = (gamma0 * f).sum() / n0
        mu1 = ((1 - gamma0) * f).sum() / n1
        var0 = (gamma0 * (f - mu0) ** 2).sum() / n0
        var1 = ((1 - gamma0) * (f - mu1) ** 2).sum() / n1
        a0, b0 = fit(mu0, var0)
        a1, b1 = fit(mu1, var1)
        w0 = n0 / (n0 + n1)
    if a0 / (a0 + b0) > a1 / (a1 + b1):
        a0, b0, a1, b1, w0 = a1, b1, a0, b0, 1 - w0
    return ((a0, b0, w0), (a1, b1, 1 - w0))


def main():
    print('=== MPNet sensitivity for fragmentation ===')
    print(f'device: {device()}')

    print('\n[1/4] Loading candidate-cluster member texts...')
    coarse = pq.read_table(PROC / 'clusters_leiden_r0.9.parquet').to_pandas()
    candidates = coarse[(coarse['cluster_id'] >= 0) & (coarse['cluster_size'] >= 8)]
    cand_row_ids = candidates['row_id'].tolist()
    print(f'  candidate clusters: {candidates["cluster_id"].nunique():,}')
    print(f'  total member rows: {len(cand_row_ids):,}')

    con = duckdb.connect()
    con.execute(f"CREATE VIEW c AS SELECT * FROM read_parquet('{PROC / 'comments.parquet'}')")
    con.execute(f"CREATE VIEW i AS SELECT * FROM read_parquet('{PROC / 'embedding_index.parquet'}')")
    sql = """
        SELECT i.row_id, c.comment_text
        FROM i JOIN c USING (comment_id)
        WHERE i.row_id IN (SELECT * FROM cand_ids)
        ORDER BY i.row_id
    """
    con.register('cand_ids', pd.DataFrame({'row_id': cand_row_ids}))
    df = con.execute(sql).fetchdf()
    print(f'  fetched {len(df):,} text rows')

    emb_path = MPNET_DIR / 'embeddings_candidates.npy'
    rowmap_path = MPNET_DIR / 'row_id_map.npy'
    if emb_path.exists() and rowmap_path.exists():
        print(f'  found existing {emb_path.name}, reusing')
        emb = np.load(emb_path, mmap_mode='r')
        rowmap = np.load(rowmap_path)
    else:
        print('\n[2/4] Embedding with MPNet (all-mpnet-base-v2)...')
        dev = device()
        model = SentenceTransformer('all-mpnet-base-v2', device=dev)
        bs = {'cuda': 256, 'mps': 64, 'cpu': 16}[dev]
        texts = df['comment_text'].fillna('').tolist()
        rowmap = df['row_id'].to_numpy(dtype=np.int64)
        n = len(texts)
        out = np.zeros((n, 768), dtype=np.float16)
        t0 = time.time()
        for i in range(0, n, bs):
            batch = texts[i:i + bs]
            with torch.inference_mode():
                e = model.encode(batch, batch_size=bs, convert_to_numpy=True,
                                  show_progress_bar=False, normalize_embeddings=True)
            out[i:i + len(batch)] = e.astype(np.float16)
            if (i // bs) % 50 == 0:
                rate = (i + len(batch)) / max(time.time() - t0, 1)
                eta = (n - i - len(batch)) / max(rate, 1)
                print(f'    {i+len(batch):,}/{n:,}  rate={rate:.0f}/s  eta={eta:.0f}s', flush=True)
        elapsed = time.time() - t0
        print(f'  embedded {n:,} in {elapsed:.0f}s ({n/elapsed:.0f}/s)')
        np.save(emb_path, out)
        np.save(rowmap_path, rowmap)
        emb = out
        print(f'  wrote {emb_path}')

    print('\n[3/4] Computing fragmentation rate per candidate using MPNet sub-clustering...')
    member_to_cluster = candidates.set_index('row_id')['cluster_id']
    rowmap_to_idx = {int(r): i for i, r in enumerate(rowmap)}

    sims_threshold = 0.97
    rows = []
    t0 = time.time()
    for cid, sub in candidates.groupby('cluster_id'):
        if len(sub) < 8:
            continue
        idx = [rowmap_to_idx.get(int(r)) for r in sub['row_id'].to_numpy()]
        idx = [i for i in idx if i is not None]
        if len(idx) < 8:
            continue
        sub_emb = np.asarray(emb[idx], dtype=np.float32)
        sub_emb /= np.maximum(np.linalg.norm(sub_emb, axis=1, keepdims=True), 1e-9)
        sims = sub_emb @ sub_emb.T
        adj = sims >= sims_threshold
        np.fill_diagonal(adj, True)
        n_pts = len(sub_emb)
        comp = np.full(n_pts, -1)
        comp_id = 0
        for i in range(n_pts):
            if comp[i] != -1:
                continue
            stack = [i]
            while stack:
                v = stack.pop()
                if comp[v] != -1:
                    continue
                comp[v] = comp_id
                stack.extend(np.where(adj[v] & (comp == -1))[0].tolist())
            comp_id += 1
        n_distinct = comp_id
        rows.append({'cluster_id': int(cid), 'n': n_pts, 'n_distinct_fine': n_distinct,
                     'fragmentation_rate': n_distinct / n_pts})
    elapsed = time.time() - t0
    print(f'  {len(rows):,} candidate clusters scored in {elapsed:.0f}s')

    frag = pd.DataFrame(rows)
    labels = pd.read_csv(RES / 'fragmentation_scores.csv')[['cluster_id', 'y_astro', 'y_adv']]
    frag = frag.merge(labels, on='cluster_id', how='inner')

    print('\n[4/4] Beta-mixture EM + AP at α=0.10...')
    f = frag['fragmentation_rate'].to_numpy()
    (a0, b0, w0), (a1, b1, w1) = fit_beta_2mix_em(f)
    f_clipped = np.clip(f, 1e-6, 1 - 1e-6)
    log_e = beta_dist.logpdf(f_clipped, a1, b1) - beta_dist.logpdf(f_clipped, a0, b0)
    e = np.exp(np.clip(log_e, -700, 700))
    order = np.argsort(-e)
    e_sorted = e[order]
    K = len(frag)
    threshold = K / (0.10 * np.arange(1, K + 1))
    rej_idx = np.where(e_sorted >= threshold)[0]
    k_hat = int(rej_idx.max() + 1) if rej_idx.size else 0
    if k_hat > 0:
        mask = frag.iloc[order[:k_hat]]
        astro_pct = float(mask['y_astro'].mean())
        adv_pct = float(mask['y_adv'].mean())
        astro_recall = mask['y_astro'].sum() / max(int(frag['y_astro'].sum()), 1)
    else:
        astro_pct = adv_pct = astro_recall = 0.0
    ap = average_precision_score(frag['y_astro'], log_e)

    print(f'\n=== MPNet result (cosine sub-cluster threshold τ={sims_threshold}) ===')
    print(f'  K = {K:,}')
    print(f'  Beta mixture: g0 mean={a0/(a0+b0):.3f} (w={w0:.3f}), g1 mean={a1/(a1+b1):.3f} (w={w1:.3f})')
    print(f'  AP against NYAG: {ap:.3f}')
    print(f'  e-BH at α=0.10: rejects {k_hat:,} ({100*k_hat/K:.1f}%)')
    print(f'  precision astroturf: {astro_pct:.3f}')
    print(f'  precision advocacy:  {adv_pct:.3f}')
    print(f'  recall astroturf:    {astro_recall:.3f}')

    out_path = RES / 'sensitivity_mpnet.csv'
    pd.DataFrame([{
        'embedding': 'mpnet-base-v2',
        'gamma_fine_or_tau': sims_threshold,
        'g0_mean': a0 / (a0 + b0),
        'g1_mean': a1 / (a1 + b1),
        'g0_weight': w0,
        'AP': ap,
        'k_rejected': k_hat,
        'precision_astro': astro_pct,
        'precision_adv': adv_pct,
        'recall_astro': astro_recall,
    }]).to_csv(out_path, index=False)
    print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()
