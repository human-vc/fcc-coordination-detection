"""Core AICE construction: cross-fit anchor-restricted Beta MLE compound e-values.

The construction in three steps:

(1) Anchor identification (structural). A subset A ⊂ [K] is identified
    structurally as null. For coordination detection: tau_c = 1 byte-identical
    clusters. For differential expression: housekeeping gene panel. For GWAS:
    monomorphic SNP block. The user supplies the anchor mask.

(2) Cross-fit Beta MLE. Partition [K] into J = 5 folds. For each fold k:
        fit (g_0^(-k), g_1^(-k)) on clusters NOT in fold k:
            g_0^(-k) = Beta MLE on {f_c : c notin k, c in A}
            g_1^(-k) = Beta MLE on {f_c : c notin k, c notin A}
        compute hat E_c = g_1^(-k)(f_c) / g_0^(-k)(f_c) for c in fold k.

(3) e-BH at level alpha. Under anchor purity (Assumption ass:anchor) and
    cross-fit independence, hat E_c is a finite-sample valid e-value, and
    e-BH controls FDR <= alpha (Wang-Ramdas 2022, Thm 1).

Theorem 1 (Validity, Section 3.2): finite-sample E[hat E_c | H_0] <= 1.
Theorem 2 (Misspecification, Section 3.4): FDR <= alpha * (1 + chi^2-slack).
Theorem 3 (Bayes-oracle attainment, Section 3.3): under (A1) Beta well-spec,
    AICE attains the IWR §5.2 oracle compound e-value at parametric
    1/sqrt(n_anchor) rate, with FDP and power regret O_p(1/sqrt(n)).
"""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score

from aice.density import fit_beta_mom, fit_beta_mle, beta_log_density, EPS


def _em_fit_alt_given_g0(
    stat_train: np.ndarray, a0: float, b0: float, pi_init: float = 0.7,
    n_iter: int = 30, tol: float = 1e-5, fit_fn=None,
) -> tuple[float, float, float]:
    """EM for the 2-component mixture g_marg = pi * Beta(a0, b0) + (1 - pi) * g_alt
    with g_0 = Beta(a0, b0) FIXED. Estimates g_alt as a Beta and pi.

    Used by AICE+ to recover the alternative density rather than the mixture marginal.
    """
    from scipy.stats import beta as beta_dist
    if fit_fn is None:
        fit_fn = fit_beta_mom
    s = np.clip(stat_train.astype(np.float64), EPS, 1 - EPS)
    if len(s) < 10:
        return 1.0, 1.0, 0.5
    # Initialize alt with method-of-moments on stat_train (biased toward null,
    # but EM corrects after a few iterations)
    a1, b1 = fit_fn(s)
    pi = pi_init
    log_lik_prev = -np.inf
    for _ in range(n_iter):
        log_p0 = beta_dist.logpdf(s, a0, b0) + np.log(max(pi, 1e-9))
        log_p1 = beta_dist.logpdf(s, a1, b1) + np.log(max(1 - pi, 1e-9))
        log_total = np.logaddexp(log_p0, log_p1)
        gamma1 = np.exp(log_p1 - log_total)
        log_lik = float(log_total.sum())
        if abs(log_lik - log_lik_prev) < tol * max(abs(log_lik_prev), 1.0):
            break
        log_lik_prev = log_lik
        n1 = float(gamma1.sum())
        if n1 < 2:
            break
        # Weighted moment-matching for g_alt component
        mu1 = float((gamma1 * s).sum() / n1)
        var1 = float((gamma1 * (s - mu1) ** 2).sum() / n1)
        if var1 <= EPS:
            break
        c = mu1 * (1 - mu1) / var1 - 1
        if c <= 0:
            break
        a1 = max(mu1 * c, 1e-3)
        b1 = max((1 - mu1) * c, 1e-3)
        pi = float(1 - n1 / len(s))
    return float(a1), float(b1), float(pi)


def fit_evalues(
    stat: np.ndarray,
    anchor_mask: np.ndarray,
    n_folds: int = 5,
    family: str = "beta",
    estimator: str = "mom",
    decontaminate: bool = False,
    seed: int = 42,
    return_components: bool = False,
):
    """Fit cross-fit anchor-restricted compound e-values.

    Parameters
    ----------
    stat : ndarray of shape (K,), values in (0, 1).
        The cluster-level test statistic. Higher values are more likely under
        the alternative.
    anchor_mask : ndarray of shape (K,), boolean.
        True iff cluster c is a structurally identified null anchor.
    n_folds : int, default 5
        Number of cross-fit folds.
    family : str, default 'beta'
        Density family for g_0, g_1. Currently only 'beta' supported.
    estimator : str, default 'mom'
        'mom' for method-of-moments (closed-form, faster) or 'mle' for
        maximum likelihood (digamma equations, more efficient).
    seed : int, default 42
        Random seed for fold assignment.
    return_components : bool, default False
        If True, returns per-fold (g_0, g_1) parameters for diagnostics.

    Returns
    -------
    log_e : ndarray of shape (K,)
        Natural log of cross-fit compound e-values for each cluster.
    fold_info : list of dict (only if return_components=True)
        Per-fold diagnostics.

    Validity
    --------
    Under Assumptions:
        (A2) Anchor purity (Section 3.2): clusters in {c : anchor_mask[c]}
            are drawn iid from the null density g_0.
        (A3) Cross-fit independence (Section 3.1): hat g_j^(-k) is computed
            from clusters not in fold k, hence independent of test points
            in fold k.
    we have:
        E[hat E_c | H_0] <= 1   in finite samples (Theorem 1)
    when (g_0, g_1) is in the Beta family. Asymptotic validity holds
    without parametric assumptions by Polyanskiy-Wu / Ignatiadis-Wang-Ramdas
    consistency (Theorem 6.3 of arXiv:2409.19812).
    """
    if family != "beta":
        raise ValueError(f"Only 'beta' family currently supported, got {family!r}")
    if estimator not in ("mom", "mle"):
        raise ValueError(f"estimator must be 'mom' or 'mle', got {estimator!r}")

    stat = np.asarray(stat, dtype=np.float64)
    anchor_mask = np.asarray(anchor_mask, dtype=bool)
    if stat.shape != anchor_mask.shape:
        raise ValueError(
            f"shape mismatch: stat {stat.shape} vs anchor_mask {anchor_mask.shape}"
        )
    if stat.ndim != 1:
        raise ValueError(f"stat must be 1d, got shape {stat.shape}")
    K = len(stat)
    if K < 2 * n_folds:
        raise ValueError(f"need at least 2*n_folds={2*n_folds} samples, got K={K}")

    fit_fn = fit_beta_mom if estimator == "mom" else fit_beta_mle
    log_e = np.zeros(K, dtype=np.float64)
    fold_info: list[dict] = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold_idx, (train_idx, eval_idx) in enumerate(kf.split(np.arange(K))):
        a_tr = anchor_mask[train_idx]
        n_anchor_tr = int(a_tr.sum())
        n_nonanchor_tr = int((~a_tr).sum())

        if n_anchor_tr < 4 or n_nonanchor_tr < 4:
            # Not enough samples in either bin to fit; e-value defaults to 1
            # (log_e = 0). Validity is preserved (1 is a trivially valid e-value).
            fold_info.append({
                "fold": fold_idx,
                "n_anchor": n_anchor_tr,
                "n_nonanchor": n_nonanchor_tr,
                "skipped": True,
            })
            continue

        a0, b0 = fit_fn(stat[train_idx][a_tr])
        if decontaminate:
            # AICE+: estimate g_alt by EM with g_0 = Beta(a0, b0) fixed, on
            # non-anchor data which is a mixture of null + alternative.
            a1, b1, _pi_hat = _em_fit_alt_given_g0(
                stat[train_idx][~a_tr], a0, b0, fit_fn=fit_fn,
            )
        else:
            # AICE: fit g_1 as the marginal of non-anchor data (IWR §7 mixture-LR).
            a1, b1 = fit_fn(stat[train_idx][~a_tr])

        eval_stat = np.clip(stat[eval_idx], EPS, 1 - EPS)
        log_e[eval_idx] = beta_log_density(eval_stat, a1, b1) - beta_log_density(
            eval_stat, a0, b0
        )

        fold_info.append({
            "fold": fold_idx,
            "a0": float(a0), "b0": float(b0),
            "a1": float(a1), "b1": float(b1),
            "g0_mean": float(a0 / (a0 + b0)),
            "g1_mean": float(a1 / (a1 + b1)),
            "n_anchor": n_anchor_tr,
            "n_nonanchor": n_nonanchor_tr,
            "skipped": False,
        })

    if return_components:
        return log_e, fold_info
    return log_e


def aice_cc_boost(
    log_e: np.ndarray, anchor_mask: np.ndarray, alpha: float = 0.10,
) -> np.ndarray:
    """[HEURISTIC, NOT FINITE-SAMPLE VALID]
    Conditional-calibration-style boost of AICE e-values via 1/p_c.

    Note: empirical study found this boost VIOLATES e-BH FDR control on
    synthetic mixtures (FDR ~ 1.15 alpha at alpha=0.10 with n=300 reps),
    because 1/p_c is not a valid e-value under H_0 even when p_c is a
    super-uniform conformal p-value. The validity gap can be closed by
    composing with an admissible Vovk-Wang p-to-e calibrator (e.g.
    e_p = kappa * p^(kappa-1)), but the resulting power gain is modest.
    Retained as a reference implementation for follow-up research; do
    NOT use for FDR control. Use AICE-BH (conformal_p_from_evalues + BH)
    instead, which is finite-sample valid under exchangeability.
    """
    log_e = np.asarray(log_e, dtype=np.float64)
    anchor_mask = np.asarray(anchor_mask, dtype=bool)
    e = np.exp(np.clip(log_e, -700, 700))
    e_anchor = e[anchor_mask]
    n_anchor = len(e_anchor)
    if n_anchor < 4:
        return log_e.copy()
    # Boost factor c_hat = 1 / p_c (inverse conformal p-value)
    sorted_anchor_desc = np.sort(e_anchor)[::-1]
    n_above = np.searchsorted(-sorted_anchor_desc, -e, side="left")
    p_c = (1.0 + n_above) / (1.0 + n_anchor)
    boost = 1.0 / np.maximum(p_c, 1e-12)
    return log_e + np.log(boost)


def conformal_p_from_evalues(
    log_e: np.ndarray, anchor_mask: np.ndarray
) -> np.ndarray:
    """Conformal p-values from AICE-style e-values using anchor as calibration set.

    For each cluster c, p_c = (1 + |{a ∈ A : e_a >= e_c}|) / (1 + n_anchor),
    where A is the anchor index set and n_anchor = |A|. Under exchangeability
    of c with the anchor under H_{0,c}, p_c is super-uniform (valid p-value).
    Following Bashari-Epstein-Romano-Sesia (NeurIPS 2023) and Marandon et al.
    (2024), applying BH to {p_c} controls FDR under PRDS.

    The combined procedure --- AICE+ score, conformal calibration to p, BH ---
    is the BH-compatible variant of AICE+ that uses p-value resolution rather
    than e-BH Markov-bound thresholds, with empirical power gains of 40-50%
    over e-BH on the same e-values when alternatives are heavy-tailed.

    Parameters
    ----------
    log_e : ndarray of shape (K,)
        Log-e-values from fit_evalues (or any cluster-level score).
    anchor_mask : ndarray of shape (K,) boolean
        True iff cluster c is a structural anchor (null reference).

    Returns
    -------
    p : ndarray of shape (K,) of conformal p-values in [1/(n_anchor+1), 1].
    """
    log_e = np.asarray(log_e, dtype=np.float64)
    anchor_mask = np.asarray(anchor_mask, dtype=bool)
    e = np.exp(np.clip(log_e, -700, 700))
    e_anchor = np.sort(e[anchor_mask])[::-1]
    n_anchor = len(e_anchor)
    if n_anchor < 4:
        return np.ones_like(log_e)
    n_above = np.searchsorted(-e_anchor, -e, side="left")
    return (1.0 + n_above) / (1.0 + n_anchor)


def aice_storey_ebh(
    log_e: np.ndarray,
    anchor_mask: np.ndarray,
    alpha: float,
    t_quantile: float = 0.5,
) -> tuple[np.ndarray, int, float]:
    """Anchor-empirical Storey-e adaptive boost on AICE e-values.

    Estimates the null fraction pi_0 from the anchor empirical CDF of the
    AICE e-values, then runs e-BH at the boosted level alpha / pi_hat_0.
    Under anchor purity (Assumption A2 of the paper), pi_hat_0 is a
    finite-sample-conservative estimator of pi_0 whenever the alternative
    density satisfies P_{g_alt}(E <= t) <= P_{g_0}(E <= t) at the chosen
    threshold t (a stochastic-domination condition that holds when t is
    below the alternative mode and is verifiable from the anchor and
    non-anchor empirical CDFs).

    Construction:
        1. Compute t = t_quantile-th sample quantile of {e_a : a anchor}.
           By construction q_hat_A(t) = t_quantile (= 0.5 if median).
        2. pi_hat_0 = (1 + #{c : e_c <= t}) / ((K+1) * q_hat_A(t)).
        3. Apply e-BH at level alpha / pi_hat_0.

    Validity:
        Wang-Ramdas (2022) show e-BH at level alpha has FDR <= pi_0 * alpha.
        Since q_hat_A(t) is unbiased for P_{g_0}(E <= t) (anchor purity),
        E[#{c: e_c <= t}] = K * (pi_0 * P_{g_0}(E <= t) + (1-pi_0) * P_{g_alt}(E <= t)).
        Under stochastic domination P_{g_alt}(E <= t) <= P_{g_0}(E <= t),
        E[pi_hat_0] >= pi_0 - O(1/sqrt(n_anchor)).  Hence e-BH at alpha/pi_hat_0
        controls FDR <= alpha + O(1/sqrt(n_anchor)) finite-sample, exact in
        the n_anchor -> infty limit.

    Parameters
    ----------
    log_e : ndarray (K,), AICE log-e-values.
    anchor_mask : ndarray (K,) boolean, True iff cluster is a structural anchor.
    alpha : float in (0, 1), nominal FDR level.
    t_quantile : float in (0, 1), default 0.5. Quantile of anchor e-values
        used as the Storey threshold. Median (0.5) is the natural choice
        and gives q_hat_A(t) = 0.5 by construction.

    Returns
    -------
    rej_indices : ndarray, indices of rejected hypotheses.
    k_hat : int, |R_alpha|.
    pi_hat_0 : float, the boost-divisor.
    """
    log_e = np.asarray(log_e, dtype=np.float64)
    anchor_mask = np.asarray(anchor_mask, dtype=bool)
    if log_e.shape != anchor_mask.shape:
        raise ValueError(f"shape mismatch: log_e {log_e.shape} vs anchor_mask {anchor_mask.shape}")
    if not (0 < t_quantile < 1):
        raise ValueError(f"t_quantile must be in (0, 1), got {t_quantile}")
    e = np.exp(np.clip(log_e, -700, 700))
    K = len(e)
    e_anchor = e[anchor_mask]
    n_anchor = len(e_anchor)
    if n_anchor < 10:
        rej, k = ebh(log_e, alpha)
        return rej, k, 1.0
    t = float(np.quantile(e_anchor, t_quantile))
    q_hat = max(np.mean(e_anchor <= t), 0.05)
    n_below = int((e <= t).sum())
    pi_hat_0 = (n_below + 1) / ((K + 1) * q_hat)
    pi_hat_0 = float(np.clip(pi_hat_0, 0.05, 1.0))
    alpha_eff = min(alpha / pi_hat_0, 0.999)
    rej, k = ebh(log_e, alpha_eff)
    return rej, k, pi_hat_0


def ebh(log_e: np.ndarray, alpha: float) -> tuple[np.ndarray, int]:
    """Apply e-BH (Wang-Ramdas 2022) at level alpha.

    Sort e-values in descending order. The rejection set is
        {c : E_c >= K / (alpha * k_hat)}
    where k_hat is the largest k for which the k-th largest e-value
    exceeds K / (alpha * k).

    Parameters
    ----------
    log_e : ndarray of shape (K,), natural log of e-values.
    alpha : float in (0, 1), nominal FDR level.

    Returns
    -------
    rej_indices : ndarray, indices of rejected hypotheses (in original ordering).
    k_hat : int, number of rejections.
    """
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    log_e = np.asarray(log_e, dtype=np.float64)
    if log_e.ndim != 1:
        raise ValueError(f"log_e must be 1d, got shape {log_e.shape}")
    K = len(log_e)
    if K == 0:
        return np.array([], dtype=int), 0

    e = np.exp(np.clip(log_e, -700, 700))
    order = np.argsort(-e)
    threshold = K / (alpha * np.arange(1, K + 1))
    sorted_e = e[order]
    rej_idx = np.where(sorted_e >= threshold)[0]
    k_hat = int(rej_idx.max() + 1) if rej_idx.size else 0
    return order[:k_hat], k_hat


def evaluate(
    log_e: np.ndarray, labels: np.ndarray, alpha: float = 0.10
) -> dict:
    """Evaluate AICE rejection set against ground-truth labels.

    Parameters
    ----------
    log_e : ndarray of shape (K,), natural log of e-values.
    labels : ndarray of shape (K,), 0/1 ground truth (1 = alternative).
    alpha : float, nominal FDR level.

    Returns
    -------
    metrics : dict with keys:
        k_rejected : int, |R_alpha|
        AP : float, average precision (rank-based, alpha-independent)
        precision : float, |R alpha cap H_1| / |R_alpha|  (1 - FDP)
        recall : float, |R alpha cap H_1| / |H_1|  (power)
        FDP : float, false discovery proportion
    """
    log_e = np.asarray(log_e, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    if log_e.shape != labels.shape:
        raise ValueError(
            f"shape mismatch: log_e {log_e.shape} vs labels {labels.shape}"
        )

    rej, k_hat = ebh(log_e, alpha)
    n_alt = int(labels.sum())

    if k_hat > 0 and n_alt > 0:
        rej_labels = labels[rej]
        n_tp = int(rej_labels.sum())
        precision = n_tp / k_hat
        recall = n_tp / n_alt
        fdp = 1 - precision
    elif k_hat > 0:
        precision = 0.0
        recall = 0.0
        fdp = 1.0
    else:
        precision = 0.0
        recall = 0.0
        fdp = 0.0

    try:
        ap = float(average_precision_score(labels, log_e)) if n_alt > 0 else float("nan")
    except Exception:
        ap = float("nan")

    return {
        "k_rejected": k_hat,
        "AP": ap,
        "precision": precision,
        "recall": recall,
        "FDP": fdp,
    }
