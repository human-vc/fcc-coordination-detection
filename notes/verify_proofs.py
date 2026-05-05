from __future__ import annotations
import numpy as np
from scipy.special import gammaln, ive
from scipy.stats import vonmises_fisher

def log_vmf_norm(d: int, kappa: float) -> float:
    if kappa < 1e-12:
        return -log_unit_sphere_area(d)
    nu = d / 2 - 1
    log_I = np.log(ive(nu, kappa)) + kappa
    return nu * np.log(kappa) - d / 2 * np.log(2 * np.pi) - log_I

def log_unit_sphere_area(d: int) -> float:
    return np.log(2.0) + d / 2 * np.log(np.pi) - gammaln(d / 2)

def log_vmf_pdf(x: np.ndarray, mu: np.ndarray, kappa: float) -> np.ndarray:
    d = x.shape[1]
    return log_vmf_norm(d, kappa) + kappa * (x @ mu)

def log_uniform_pdf(d: int) -> float:
    return -log_unit_sphere_area(d)

def vmf_mle(x: np.ndarray) -> tuple[np.ndarray, float]:
    d = x.shape[1]
    mean = x.mean(axis=0)
    r = np.linalg.norm(mean)
    r = min(r, 0.9999)
    if r < 1e-09:
        return (mean / max(r, 1e-12), 0.001)
    mu_hat = mean / r
    kappa = r * (d - r * r) / (1 - r * r)
    for _ in range(2):
        nu = d / 2 - 1
        A = ive(nu + 1, kappa) / ive(nu, kappa)
        A_prime = 1.0 - A * A - (d - 1) / max(kappa, 1e-09) * A
        delta = (A - r) / max(A_prime, 1e-09)
        kappa = max(kappa - delta, 0.001)
    return (mu_hat, float(kappa))

def e_value(x: np.ndarray, *, log_q: callable, rng: np.random.Generator) -> float:
    n = x.shape[0]
    perm = rng.permutation(n)
    half = n // 2
    A = x[perm[:half]]
    B = x[perm[half:half * 2]]
    (mu_hat, kappa_hat) = vmf_mle(A)
    log_p = log_vmf_pdf(B, mu_hat, kappa_hat)
    log_e = (log_p - log_q(B)).sum()
    return float(np.exp(log_e))

def verify_N_normalizer(*, d: int=20, kappas: tuple[float, ...]=(0.5, 1.0, 3.0, 10.0, 30.0), n_mc: int=200000, seed: int=0) -> list[dict]:
    rng = np.random.default_rng(seed)
    mu = np.zeros(d)
    mu[0] = 1.0
    log_unif = log_uniform_pdf(d)
    out = []
    for kappa in kappas:
        z = rng.standard_normal((n_mc, d))
        x = z / np.linalg.norm(z, axis=1, keepdims=True)
        log_ratio = log_vmf_pdf(x, mu, kappa) - log_unif
        ratio = np.exp(log_ratio)
        m_emp = ratio.mean()
        se_emp = ratio.std() / np.sqrt(n_mc)
        out.append({'kappa': float(kappa), 'd': d, 'n_mc': n_mc, 'E[p_vMF/p_unif]': float(m_emp), 'se': float(se_emp), 'claim': '= 1', 'passed': bool(abs(m_emp - 1.0) < 6 * se_emp)})
    return out

def verify_T1_validity(*, d: int=20, n: int=100, n_trials: int=5000, seed: int=0) -> dict:
    rng = np.random.default_rng(seed)
    log_q_const = log_uniform_pdf(d)

    def log_q(x):
        return np.full(x.shape[0], log_q_const)
    ev_log = np.empty(n_trials)
    for t in range(n_trials):
        z = rng.standard_normal((n, d))
        x = z / np.linalg.norm(z, axis=1, keepdims=True)
        ev = e_value(x, log_q=log_q, rng=rng)
        ev_log[t] = np.log(max(ev, 1e-300))
    e_mean = np.exp(ev_log).mean()
    e_se = np.exp(ev_log).std() / np.sqrt(n_trials)
    log_e_mean = ev_log.mean()
    return {'n_trials': n_trials, 'd': d, 'n': n, 'E[E_c]_empirical': float(e_mean), 'se': float(e_se), 'log_E_c_mean': float(log_e_mean), 'log_E_c_p50': float(np.median(ev_log)), 'log_E_c_p99': float(np.quantile(ev_log, 0.99)), 'claim': 'E[E_c | H_c] = 1', 'passed': bool(abs(e_mean - 1.0) < 4 * e_se)}

def verify_T1m_markov(*, d: int=20, n: int=100, n_trials: int=20000, thresholds: tuple[float, ...]=(1.0, 2.0, 5.0, 10.0, 100.0, 1000.0), seed: int=0) -> dict:
    rng = np.random.default_rng(seed)
    log_q_const = log_uniform_pdf(d)

    def log_q(x):
        return np.full(x.shape[0], log_q_const)
    ev_vals = np.empty(n_trials)
    for t in range(n_trials):
        z = rng.standard_normal((n, d))
        x = z / np.linalg.norm(z, axis=1, keepdims=True)
        ev_vals[t] = e_value(x, log_q=log_q, rng=rng)
    rows = []
    all_passed = True
    for t in thresholds:
        emp = float((ev_vals >= t).mean())
        bound = 1.0 / t
        passed = emp <= bound + 3 * np.sqrt(emp * (1 - emp) / n_trials)
        all_passed = all_passed and passed
        rows.append({'t': float(t), 'P(E_c>=t)_emp': emp, 'Markov_bound_1/t': float(bound), 'passed': bool(passed)})
    return {'n_trials': n_trials, 'd': d, 'n': n, 'claim': 'P(E_c >= t | H_c) <= 1/t  for all t > 0', 'rows': rows, 'all_passed': bool(all_passed)}

def verify_T2_graceful(*, d: int=20, n: int=100, eps: float=0.1, n_trials: int=5000, seed: int=0) -> dict:
    rng = np.random.default_rng(seed)
    log_q_const = log_uniform_pdf(d)
    ref = rng.standard_normal(d)
    ref /= np.linalg.norm(ref)

    def log_qhat(x):
        cos_theta = x @ ref
        return log_q_const + np.log1p(eps * cos_theta)
    ev_vals = np.empty(n_trials)
    for t in range(n_trials):
        z = rng.standard_normal((n, d))
        x = z / np.linalg.norm(z, axis=1, keepdims=True)
        ev = e_value(x, log_q=log_qhat, rng=rng)
        ev_vals[t] = ev
    m = n // 2
    bound = (1 + eps) ** m
    e_mean = ev_vals.mean()
    e_se = ev_vals.std() / np.sqrt(n_trials)
    return {'n_trials': n_trials, 'd': d, 'n': n, 'eps_sup': eps, 'm': m, 'E[Ê_c]_empirical': float(e_mean), 'se': float(e_se), 'bound_(1+eps)^m': float(bound), 'bound_exp(m*eps)': float(np.exp(m * eps)), 'claim': 'E[Ê_c | H_c] <= (1+eps)^m', 'passed': bool(e_mean <= bound + 4 * e_se)}

def verify_KL_expansion(*, d: int=20, kappas: tuple[float, ...]=(0.1, 0.3, 1.0, 3.0, 10.0), n_mc: int=200000, seed: int=0) -> list[dict]:
    rng = np.random.default_rng(seed)
    mu = np.zeros(d)
    mu[0] = 1.0
    log_unif = log_uniform_pdf(d)
    out = []
    for kappa in kappas:
        vmf = vonmises_fisher(mu, kappa, seed=rng)
        x = vmf.rvs(n_mc)
        log_p = log_vmf_pdf(x, mu, kappa)
        kl_mc = (log_p - log_unif).mean()
        kl_se = (log_p - log_unif).std() / np.sqrt(n_mc)
        kl_taylor = kappa * kappa / (2 * d)
        out.append({'kappa': float(kappa), 'd': d, 'KL_MC': float(kl_mc), 'KL_SE': float(kl_se), 'KL_taylor_k2_over_2d': float(kl_taylor), 'ratio_MC/taylor': float(kl_mc / max(kl_taylor, 1e-12))})
    return out

def verify_direction_concentration(*, d: int=20, m: int=50, kappas: tuple[float, ...]=(0.5, 1.0, 3.0, 10.0), n_trials: int=2000, seed: int=0) -> list[dict]:
    rng = np.random.default_rng(seed)
    mu = np.zeros(d)
    mu[0] = 1.0
    out = []
    for kappa in kappas:
        vmf = vonmises_fisher(mu, kappa, seed=rng)
        nu = d / 2 - 1
        A_d = ive(nu + 1, kappa) / ive(nu, kappa)
        cosines = np.empty(n_trials)
        kappa_hats = np.empty(n_trials)
        for t in range(n_trials):
            x = vmf.rvs(m)
            (mu_hat, kappa_hat) = vmf_mle(x)
            cosines[t] = float(mu_hat @ mu)
            kappa_hats[t] = kappa_hat
        out.append({'kappa_true': float(kappa), 'A_d(kappa)': float(A_d), '<mu_hat,mu>_mean': float(cosines.mean()), '<mu_hat,mu>_p10': float(np.quantile(cosines, 0.1)), '<mu_hat,mu>_p50': float(np.median(cosines)), 'kappa_hat_mean': float(kappa_hats.mean()), 'writeup_says_near_A_d': float(A_d), 'truth_should_be_near_1': True})
    return out

def verify_power_threshold(*, d: int=20, n: int=100, K_clusters: int=20000, alpha: float=0.1, kappas: tuple[float, ...]=(0.5, 1.0, 2.0, 4.0, 8.0, 16.0), n_trials: int=1000, seed: int=0) -> list[dict]:
    rng = np.random.default_rng(seed)
    mu = np.zeros(d)
    mu[0] = 1.0
    log_unif = log_uniform_pdf(d)

    def log_q(x):
        return np.full(x.shape[0], log_unif)
    threshold = K_clusters / alpha
    log_threshold = np.log(threshold)
    out = []
    for kappa in kappas:
        vmf = vonmises_fisher(mu, kappa, seed=rng)
        rej_count = 0
        log_evs = []
        for t in range(n_trials):
            x = vmf.rvs(n)
            ev = e_value(x, log_q=log_q, rng=rng)
            log_e = np.log(max(ev, 1e-300))
            log_evs.append(log_e)
            if log_e >= log_threshold:
                rej_count += 1
        log_evs = np.array(log_evs)
        beta = 0.1
        kstar_pred = np.sqrt(d * np.log(K_clusters / (alpha * beta)) / n)
        out.append({'kappa': float(kappa), 'd': d, 'n': n, 'K': K_clusters, 'alpha': alpha, 'rejection_rate': rej_count / n_trials, 'log_E_c_mean': float(log_evs.mean()), 'log_E_c_p10': float(np.quantile(log_evs, 0.1)), 'log_threshold': float(log_threshold), 'kappa*_predicted_C1=1': float(kstar_pred)})
    return out
if __name__ == '__main__':
    import json
    print('=' * 70)
    print('N — NORMALIZER: ∫ p_vMF dσ = 1   (the m=1 reduction of T1)')
    print('=' * 70)
    for r in verify_N_normalizer(d=20):
        print(json.dumps(r, indent=2))
    print()
    print('=' * 70)
    print('T1 — VALIDITY (heavy-tailed mean): E[E_c | H_c] = 1')
    print('    NOTE: log-normal heavy tails make the direct mean unverifiable')
    print('    at this sample size; the right test is T1m (Markov) below.')
    print('=' * 70)
    r = verify_T1_validity(d=20, n=100, n_trials=5000)
    print(json.dumps(r, indent=2))
    print()
    print('=' * 70)
    print('T1m — MARKOV: P(E_c >= t | H_c) <= 1/t (proper validity check)')
    print('=' * 70)
    r = verify_T1m_markov(d=20, n=100, n_trials=20000)
    print(json.dumps(r, indent=2))
    print()
    print('=' * 70)
    print('T2 — GRACEFUL DEGRADATION: E[Ê_c] <= (1+eps)^m under sup-norm q̂')
    print('=' * 70)
    for eps in (0.05, 0.1, 0.2):
        r = verify_T2_graceful(d=20, n=100, eps=eps, n_trials=2000)
        print(json.dumps(r, indent=2))
    print()
    print('=' * 70)
    print('K — KL(vMF || Unif) vs κ²/(2d) Taylor')
    print('=' * 70)
    for r in verify_KL_expansion(d=20):
        print(json.dumps(r, indent=2))
    print()
    print('=' * 70)
    print('D — DIRECTION CONCENTRATION: <μ̂,μ> behavior')
    print('=' * 70)
    for r in verify_direction_concentration(d=20, m=50, n_trials=500):
        print(json.dumps(r, indent=2))
    print()
    print('=' * 70)
    print('P — POWER THRESHOLD: rejection rate vs κ')
    print('=' * 70)
    for r in verify_power_threshold(d=20, n=100, n_trials=500):
        print(json.dumps(r, indent=2))
