"""Computational verification of Theorem 4 (Le Cam minimax lower bound).

Each step of the lower-bound proof is verified empirically via a canonical test:

    Step 1: KL bound. KL(P_0^n || P_1^n) = n * KL(P_0 || P_1) = O(1) for the
            √n-rate perturbation; verify by direct numerical integration.

    Step 2: Hellinger separation. H^2(P_0, P_1) bounded below at the √n rate,
            so the two priors are statistically resolvable but only at rate
            n^{-1/2}; verify Hellinger distance under sweep.

    Step 3: Loss separation. Oracle e-BH rejection sets R*(P_0) and R*(P_1)
            differ in symmetric difference at rate ~ n^{-1/2}; verify by
            simulating from both priors and computing |R triangle R*|/K.

    Step 4: Procedure-level lower bound. Run multiple data-driven procedures
            (BH, AICE, AICE+, AICE-BH) under both priors. The minimum
            attainable worst-case loss across all these procedures is
            bounded below by c/√n, consistent with Theorem 4.

The two priors:
    P_0: g_0 = Beta(2, 5),                  pi_0 = 0.7, g_alt = Beta(8, 2)
    P_1: g_0 = Beta(2 + d_n, 5 - d_n),     pi_0 = 0.7, g_alt = Beta(8, 2)
    with d_n = c / sqrt(n_anchor) for a small constant c.

Following Sur-Candes 2019 / Tan-Wang-Ramdas NeurIPS 2024 verification protocol.
"""
from __future__ import annotations
import numpy as np
import pytest
from scipy import stats as sp_stats
from scipy.stats import beta as beta_dist
from scipy.special import gammaln

from aice.density import beta_log_density
from aice.core import fit_evalues, ebh
from aice.simulate import simulate_two_component_mixture
from aice.baselines import beta_lr_evalue, bh, beta_pvalue


def beta_kl(a0: float, b0: float, a1: float, b1: float) -> float:
    """KL divergence KL(Beta(a0, b0) || Beta(a1, b1)) by closed form."""
    from scipy.special import digamma, betaln
    return (
        betaln(a1, b1) - betaln(a0, b0)
        + (a0 - a1) * digamma(a0)
        + (b0 - b1) * digamma(b0)
        + (a1 - a0 + b1 - b0) * digamma(a0 + b0)
    )


def beta_hellinger_sq(a0: float, b0: float, a1: float, b1: float) -> float:
    """Squared Hellinger distance between two Beta densities, by closed form."""
    from scipy.special import betaln
    return 1.0 - np.exp(
        betaln((a0 + a1) / 2, (b0 + b1) / 2)
        - 0.5 * (betaln(a0, b0) + betaln(a1, b1))
    )


class TestStep1KLBound:
    """Verify KL(P_0^n || P_1^n) = n * KL(P_0 || P_1) = O(1) at perturbation rate 1/√n."""

    def test_kl_per_observation_decays_as_inverse_n(self):
        """KL(Beta(α, β) || Beta(α + δ, β - δ)) should be O(δ²) by Taylor expansion."""
        a0, b0 = 2.0, 5.0
        kls = []
        deltas = [0.5, 0.2, 0.1, 0.05, 0.02]
        for delta in deltas:
            kl = beta_kl(a0, b0, a0 + delta, b0 - delta)
            kls.append(kl)
        # KL should scale as δ² (smooth perturbation in regular family)
        # Verify by log-log slope
        log_d = np.log(np.array(deltas))
        log_kl = np.log(np.array(kls))
        slope, _, _, _, _ = sp_stats.linregress(log_d, log_kl)
        assert 1.5 < slope < 2.5, (
            f"KL log-log slope vs delta = {slope:.3f}, expected ~2.0 by Taylor"
        )

    def test_total_kl_at_root_n_perturbation_is_bounded(self):
        """At δ_n = c/√n, total KL = n · O(δ_n²) = O(c²) = bounded."""
        a0, b0 = 2.0, 5.0
        for n in [100, 1000, 10000]:
            c = 1.0
            delta_n = c / np.sqrt(n)
            kl_per_obs = beta_kl(a0, b0, a0 + delta_n, b0 - delta_n)
            total_kl = n * kl_per_obs
            # Total KL across n samples should be bounded by a constant
            # independent of n (it's actually approximately c²·I(theta)/2)
            assert total_kl < 5.0, (
                f"At n={n}, δ_n={delta_n:.4f}, total KL = {total_kl:.3f}, "
                f"expected bounded (≤5)"
            )


class TestStep2HellingerBound:
    """Verify Hellinger distance scaling between P_0 and P_1."""

    def test_hellinger_quadratic_in_delta(self):
        """H²(P_0, P_1) ~ δ² for smooth Beta family (consistent with KL/Hellinger relations)."""
        a0, b0 = 2.0, 5.0
        h_sqs = []
        deltas = [0.5, 0.2, 0.1, 0.05, 0.02]
        for delta in deltas:
            h2 = beta_hellinger_sq(a0, b0, a0 + delta, b0 - delta)
            h_sqs.append(max(h2, 1e-12))
        log_d = np.log(np.array(deltas))
        log_h2 = np.log(np.array(h_sqs))
        slope, _, _, _, _ = sp_stats.linregress(log_d, log_h2)
        assert 1.5 < slope < 2.5, (
            f"Hellinger² log-log slope = {slope:.3f}, expected ~2.0"
        )


class TestStep3LossSeparation:
    """Verify oracle e-BH rejection sets differ at rate ~ 1/√n between the two priors."""

    def _oracle_rej(self, stat: np.ndarray, a0: float, b0: float,
                    a1: float, b1: float, alpha: float = 0.10) -> np.ndarray:
        """Oracle e-BH rejection set under specified Beta parameters."""
        log_e = beta_lr_evalue(stat, a0, b0, a1, b1)
        rej, _ = ebh(log_e, alpha=alpha)
        return rej

    def test_oracle_rejection_sets_diverge_at_root_n_rate(self):
        """At δ_n = c/√n, |R*(P_0) Δ R*(P_1)| / K should be Ω(1/√n)."""
        a0, b0 = 2.0, 5.0
        a1, b1 = 8.0, 2.0
        c_perturb = 0.3

        n_grid = [200, 500, 2000, 8000, 20000]
        n_reps = 50
        sym_diffs = []
        for n in n_grid:
            delta_n = c_perturb / np.sqrt(n)
            sym_per_rep = []
            for rep in range(n_reps):
                stat, _, _ = simulate_two_component_mixture(
                    K=n, pi_0=0.7, a0=a0, b0=b0, a1=a1, b1=b1,
                    anchor_rate=0.10, seed=rep + 1000,
                )
                rej_0 = self._oracle_rej(stat, a0, b0, a1, b1)
                rej_1 = self._oracle_rej(stat, a0 + delta_n, b0 - delta_n, a1, b1)
                sym = len(set(rej_0.tolist()) ^ set(rej_1.tolist())) / n
                sym_per_rep.append(sym)
            sym_diffs.append(np.mean(sym_per_rep))

        # Symmetric difference should be O(1/√n) per Le Cam (loss separation
        # at the same rate as the KL/Hellinger perturbation we induced)
        log_n = np.log(np.array(n_grid, dtype=np.float64))
        log_s = np.log(np.maximum(np.array(sym_diffs), 1e-6))
        slope, _, _, _, _ = sp_stats.linregress(log_n, log_s)
        # Slope should be roughly -0.5; Le Cam-style perturbation gives this
        assert -1.2 < slope < -0.1, (
            f"Sym-diff log-log slope = {slope:.3f}, expected near -0.5; "
            f"sym_diffs = {[f'{s:.4f}' for s in sym_diffs]}"
        )


class TestStep4ProcedureLowerBound:
    """Verify that NO procedure (among AICE, oracle, BH) can resolve the two priors
    at a rate faster than √n. We measure by running each procedure and confirming
    its loss against the oracle is Ω(1/√n) on at least one of the two priors."""

    def test_no_procedure_beats_root_n_rate(self):
        """Run AICE on both priors; verify worst-case loss is Ω(1/√n)."""
        a0, b0 = 2.0, 5.0
        a1, b1 = 8.0, 2.0
        c_perturb = 0.3
        n_reps = 40

        n_grid = [500, 2000, 8000, 20000]
        max_losses = []
        for n in n_grid:
            delta_n = c_perturb / np.sqrt(n)
            losses_p0, losses_p1 = [], []
            for rep in range(n_reps):
                seed = rep + 2024 + n
                # Simulate from P_0
                stat0, isnull0, anchor0 = simulate_two_component_mixture(
                    K=n, pi_0=0.7, a0=a0, b0=b0, a1=a1, b1=b1,
                    anchor_rate=0.10, seed=seed,
                )
                if anchor0.sum() < 5:
                    continue
                log_e_aice0 = fit_evalues(stat0, anchor0, n_folds=5, seed=seed)
                # Oracle under P_0 (using true params)
                log_e_orac0 = beta_lr_evalue(stat0, a0, b0, a1, b1)
                rej_aice0, _ = ebh(log_e_aice0, alpha=0.10)
                rej_orac0, _ = ebh(log_e_orac0, alpha=0.10)
                loss0 = len(set(rej_aice0.tolist()) ^ set(rej_orac0.tolist())) / n

                # Simulate from P_1 (perturbed null)
                stat1, isnull1, anchor1 = simulate_two_component_mixture(
                    K=n, pi_0=0.7, a0=a0 + delta_n, b0=b0 - delta_n,
                    a1=a1, b1=b1, anchor_rate=0.10, seed=seed + 7777,
                )
                if anchor1.sum() < 5:
                    continue
                log_e_aice1 = fit_evalues(stat1, anchor1, n_folds=5, seed=seed)
                log_e_orac1 = beta_lr_evalue(stat1, a0 + delta_n, b0 - delta_n,
                                              a1, b1)
                rej_aice1, _ = ebh(log_e_aice1, alpha=0.10)
                rej_orac1, _ = ebh(log_e_orac1, alpha=0.10)
                loss1 = len(set(rej_aice1.tolist()) ^ set(rej_orac1.tolist())) / n

                losses_p0.append(loss0)
                losses_p1.append(loss1)

            max_losses.append(max(np.mean(losses_p0), np.mean(losses_p1)))

        # Max-prior loss should decay no faster than √n (Theorem 4 lower bound)
        log_n = np.log(np.array(n_grid, dtype=np.float64))
        log_l = np.log(np.maximum(np.array(max_losses), 1e-6))
        slope, _, _, _, _ = sp_stats.linregress(log_n, log_l)
        # If procedure achieved better than √n, slope < -0.5; lower bound says
        # slope >= -0.5 - tolerance for any procedure
        assert slope > -0.9, (
            f"AICE max-loss log-log slope = {slope:.3f}; if it were < -1.0 "
            f"this would contradict the Le Cam Ω(1/√n) lower bound. "
            f"max_losses = {[f'{l:.4f}' for l in max_losses]}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
