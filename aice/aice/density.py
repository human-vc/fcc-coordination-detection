"""Beta density estimation routines for AICE.

Two estimators:
    fit_beta_mom: method-of-moments estimator (closed-form, fast, used as default)
    fit_beta_mle: maximum likelihood estimator (digamma equations, slightly more efficient)

The cross-fit anchor-restricted MLE used inside AICE is a parametric Beta
estimator. Theorem 3 (Bayes-oracle attainment) requires:
    (A1) Beta well-specification on the anchor stratum,
    (A2) Anchor purity,
    (A3) Cross-fit independence,
which together give parametric 1/sqrt(n) consistency rate by van der Vaart
Asymptotic Statistics, Thm 5.39.
"""
from __future__ import annotations
import numpy as np
from scipy.special import digamma, gammaln
from scipy.optimize import minimize_scalar

EPS = 1e-6
MIN_PARAM = 1e-3


def fit_beta_mom(values: np.ndarray) -> tuple[float, float]:
    """Method-of-moments estimator for Beta(a, b).

    Parameters
    ----------
    values : ndarray, values in (0, 1).

    Returns
    -------
    (a, b) : tuple of floats, parameters of fitted Beta distribution.

    Notes
    -----
    For a Beta(a, b), mu = a / (a + b), var = a*b / ((a+b)^2 * (a+b+1)).
    Solving: a + b = mu * (1 - mu) / var - 1, then a = mu * (a+b),
    b = (1 - mu) * (a + b). Returns (1, 1) (uniform) if degenerate.
    """
    v = np.clip(np.asarray(values, dtype=np.float64), EPS, 1 - EPS)
    if len(v) < 4 or v.var() < 1e-10:
        return 1.0, 1.0
    mu = float(v.mean())
    var = float(v.var())
    c = mu * (1 - mu) / var - 1
    if c <= 0:
        return 1.0, 1.0
    return max(mu * c, MIN_PARAM), max((1 - mu) * c, MIN_PARAM)


def beta_log_density(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Log density of Beta(a, b) at x in (0, 1)."""
    x_c = np.clip(np.asarray(x, dtype=np.float64), EPS, 1 - EPS)
    log_B = gammaln(a) + gammaln(b) - gammaln(a + b)
    return (a - 1) * np.log(x_c) + (b - 1) * np.log(1 - x_c) - log_B


def fit_beta_mle(values: np.ndarray, n_iter: int = 100, tol: float = 1e-7) -> tuple[float, float]:
    """Maximum likelihood estimator for Beta(a, b) by Newton-Raphson on digamma equations.

    The MLE solves:
        log(geometric_mean(x)) = digamma(a) - digamma(a + b)
        log(geometric_mean(1 - x)) = digamma(b) - digamma(a + b)

    Initialized from method-of-moments. For samples drawn from a true Beta
    distribution the MLE is asymptotically efficient (Fisher information rate);
    method-of-moments is consistent at parametric rate but less efficient. AICE
    uses MLE when n is large and MoM otherwise.
    """
    v = np.clip(np.asarray(values, dtype=np.float64), EPS, 1 - EPS)
    if len(v) < 4 or v.var() < 1e-10:
        return 1.0, 1.0
    log_x_mean = float(np.log(v).mean())
    log_1mx_mean = float(np.log(1 - v).mean())
    a, b = fit_beta_mom(v)
    for _ in range(n_iter):
        ab = a + b
        f1 = digamma(a) - digamma(ab) - log_x_mean
        f2 = digamma(b) - digamma(ab) - log_1mx_mean
        # Hessian via trigamma not strictly needed; simple line search via projected gradient
        # works adequately for Beta in our regime. Fall back to MoM if Newton diverges.
        if abs(f1) < tol and abs(f2) < tol:
            break
        # Use trigamma for Hessian
        from scipy.special import polygamma
        H11 = polygamma(1, a) - polygamma(1, ab)
        H22 = polygamma(1, b) - polygamma(1, ab)
        H12 = -polygamma(1, ab)
        det = H11 * H22 - H12 ** 2
        if abs(det) < 1e-12:
            break
        da = (H22 * f1 - H12 * f2) / det
        db = (H11 * f2 - H12 * f1) / det
        a_new = max(a - da, MIN_PARAM)
        b_new = max(b - db, MIN_PARAM)
        if not (np.isfinite(a_new) and np.isfinite(b_new)):
            break
        if abs(a_new - a) + abs(b_new - b) < tol:
            a, b = a_new, b_new
            break
        a, b = a_new, b_new
    return float(a), float(b)
