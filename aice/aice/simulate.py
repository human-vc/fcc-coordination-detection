"""Synthetic simulation utilities for AICE bake-off.

Implements the Korthauer 2019 / Lee-Ren 2024 / Bashari-Sesia 2023 standard
2-component mixture FDR benchmark.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import beta as beta_dist, multivariate_normal, norm


def simulate_two_component_mixture(
    K: int,
    pi_0: float,
    a0: float = 2.0,
    b0: float = 5.0,
    a1: float = 8.0,
    b1: float = 2.0,
    anchor_rate: float = 0.10,
    dependence: str = "independent",
    rho: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate K samples from a 2-component Beta mixture with anchor labels.

    Parameters
    ----------
    K : int, number of samples (clusters).
    pi_0 : float in (0, 1), null fraction.
    a0, b0 : Beta parameters for null distribution. Default Beta(2, 5).
    a1, b1 : Beta parameters for alternative distribution. Default Beta(8, 2).
    anchor_rate : float in (0, 1), fraction of nulls that are designated as
        structural anchors. Anchors are guaranteed null (anchor purity).
    dependence : 'independent', 'prds_block', 'prds_ar1', 'arbitrary'
        Dependence structure across the K samples.
    rho : float, dependence parameter (ignored if dependence='independent').
    seed : int, RNG seed.

    Returns
    -------
    stat : ndarray (K,), values in (0, 1), the test statistics.
    is_null : ndarray (K,), boolean True iff cluster is null.
    anchor_mask : ndarray (K,), boolean True iff cluster is a structural anchor.

    Validity guarantee
    ------------------
    Anchors are sampled FROM THE NULL distribution, so anchor purity holds
    by construction: P(is_null | anchor_mask) = 1.
    """
    rng = np.random.default_rng(seed)

    # Step 1: assign null/alternative labels
    is_null = rng.random(K) < pi_0

    # Step 2: generate dependent uniforms (Gaussian copula) if needed
    if dependence == "independent":
        u = rng.random(K)
    elif dependence == "prds_block":
        block_size = max(2, int(np.sqrt(K)))
        n_blocks = (K + block_size - 1) // block_size
        u = np.zeros(K)
        for b in range(n_blocks):
            i0, i1 = b * block_size, min((b + 1) * block_size, K)
            sz = i1 - i0
            cov = (1 - rho) * np.eye(sz) + rho * np.ones((sz, sz))
            z = multivariate_normal.rvs(mean=np.zeros(sz), cov=cov, random_state=rng)
            z = np.atleast_1d(z)
            u[i0:i1] = norm.cdf(z)
    elif dependence == "prds_ar1":
        # AR(1): z_t = rho * z_{t-1} + sqrt(1 - rho^2) * eps_t
        z = np.zeros(K)
        z[0] = rng.standard_normal()
        for t in range(1, K):
            z[t] = rho * z[t - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal()
        u = norm.cdf(z)
    elif dependence == "arbitrary":
        # Toeplitz structure: cov[i,j] = rho^|i-j|, induced via Cholesky
        # For large K we use AR(1) which has the same correlation function
        z = np.zeros(K)
        z[0] = rng.standard_normal()
        for t in range(1, K):
            z[t] = rho * z[t - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal()
        u = norm.cdf(z)
    else:
        raise ValueError(f"unknown dependence: {dependence!r}")

    # Step 3: invert per-sample marginal CDF to get Beta-distributed values
    # null samples ~ Beta(a0, b0), alternative samples ~ Beta(a1, b1)
    stat = np.where(
        is_null,
        beta_dist.ppf(u, a0, b0),
        beta_dist.ppf(u, a1, b1),
    )

    # Step 4: select anchor subset from the nulls (uniform random)
    # NOTE: anchor purity holds by construction
    anchor_mask = np.zeros(K, dtype=bool)
    null_indices = np.where(is_null)[0]
    n_anchor = int(np.round(anchor_rate * len(null_indices)))
    if n_anchor > 0 and len(null_indices) > 0:
        anchor_subset = rng.choice(null_indices, size=min(n_anchor, len(null_indices)),
                                    replace=False)
        anchor_mask[anchor_subset] = True

    return stat, is_null, anchor_mask
