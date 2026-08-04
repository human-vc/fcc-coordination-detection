"""Tests for aice.baselines: BH, Storey, e-BH, Lee-Ren CC, group-BH."""
import numpy as np
import pytest

from aice.baselines import (
    bh, by, storey_qvalue, ebh, ebh_lr_naive, ebh_cc,
    group_bh, beta_pvalue, beta_lr_evalue,
)
from aice.simulate import simulate_two_component_mixture


class TestBH:
    def test_no_rejections(self):
        """All p-values near 1: no rejections."""
        p = np.full(100, 0.99)
        rej, k = bh(p, alpha=0.10)
        assert k == 0

    def test_all_rejections(self):
        """All p-values near 0: all rejections."""
        p = np.full(100, 0.001)
        rej, k = bh(p, alpha=0.10)
        assert k == 100

    def test_against_reference(self):
        """Compare to manual implementation of BH."""
        rng = np.random.default_rng(0)
        K = 100
        p = rng.random(K)
        order = np.argsort(p)
        sorted_p = p[order]
        thr = 0.10 * np.arange(1, K + 1) / K
        valid = np.where(sorted_p <= thr)[0]
        ref_k = int(valid.max() + 1) if valid.size else 0
        _, k = bh(p, alpha=0.10)
        assert k == ref_k


class TestStorey:
    def test_lambda_fixed(self):
        """With lambda_fixed, should not crash."""
        rng = np.random.default_rng(0)
        # Mix of 50% null (uniform) + 50% alternative (small p-values)
        p_null = rng.random(500)
        p_alt = rng.random(500) * 0.01
        p = np.concatenate([p_null, p_alt])
        rej, k, pi_0 = storey_qvalue(p, alpha=0.10, lambda_fixed=0.5)
        # pi_0 estimate should be in (0, 1]; under 50/50 mix expect ~0.5 * 1.0 + 0.5 * (1/(1-0.5) * (close to 0))
        # = 0.5 ish
        assert 0.0 < pi_0 <= 1.0
        # Storey should reject more than vanilla BH (since pi_0 < 1)
        _, k_bh = bh(p, alpha=0.10)
        assert k >= k_bh

    def test_smoke_grid(self):
        """Smoke test with default lambda grid."""
        rng = np.random.default_rng(0)
        p = rng.random(2000)
        # Inject ~10% strong signals
        p[:200] = rng.random(200) * 0.001
        rej, k, pi_0 = storey_qvalue(p, alpha=0.10)
        assert k > 0
        assert 0.0 < pi_0 <= 1.0


class TestEBH:
    def test_basic(self):
        """Smoke test for e-BH."""
        rng = np.random.default_rng(0)
        log_e = rng.standard_normal(1000) * 3
        rej, k = ebh(log_e, alpha=0.10)
        assert isinstance(k, int)
        assert k >= 0

    def test_oracle_lr(self):
        """Vanilla e-BH with oracle LR should achieve some rejections on Beta(2,5)/Beta(8,2)."""
        rng = np.random.default_rng(0)
        K = 2000
        stat, is_null, _ = simulate_two_component_mixture(
            K=K, pi_0=0.7, seed=0,
        )
        rej, k = ebh_lr_naive(stat, 2.0, 5.0, 8.0, 2.0, alpha=0.10)
        # Some rejections expected
        assert k > 0
        # Realized FDR should be controlled
        if k > 0:
            n_false = int(is_null[rej].sum())
            fdp = n_false / k
            # Single-replication FDP can exceed alpha; check at average over reps elsewhere
            assert fdp >= 0


class TestEBHCC:
    def test_smoke(self):
        """Lee-Ren CC e-BH should not crash and should produce valid output."""
        rng = np.random.default_rng(0)
        K = 500
        stat, is_null, anchor_mask = simulate_two_component_mixture(
            K=K, pi_0=0.7, seed=0,
        )
        rej, k = ebh_cc(stat, anchor_mask, alpha=0.10,
                         a0_est=(2.0, 5.0), a1_est=(8.0, 2.0),
                         n_mc=50, seed=0)
        assert isinstance(k, int)
        assert k >= 0


class TestGroupBH:
    def test_smoke(self):
        """Group-BH should run."""
        rng = np.random.default_rng(0)
        K = 1000
        p = rng.random(K)
        cov = rng.random(K)
        rej, k = group_bh(p, cov, alpha=0.10, n_groups=4)
        assert isinstance(k, int)


class TestConverters:
    def test_beta_pvalue(self):
        """p-value of Beta-null distributed value should be uniform."""
        from scipy.stats import beta as beta_dist
        rng = np.random.default_rng(0)
        x = beta_dist.rvs(2.0, 5.0, size=10000, random_state=rng)
        p = beta_pvalue(x, a0=2.0, b0=5.0)
        # KS test for uniform
        from scipy.stats import kstest
        ks_stat, ks_p = kstest(p, "uniform")
        assert ks_p > 0.01, f"KS test rejected uniform null: p = {ks_p}"

    def test_beta_lr_evalue_validity(self):
        """LR e-value evaluated under null should have mean ≤ 1.

        Note: LR mean has heavy-tailed Monte Carlo variance — exact value can
        differ by O(20%) at n=50000 since rare events with x near 1 carry
        most of the mass. The validity claim E[LR | H_0] ≤ 1 is exact in
        expectation; we verify the empirical estimate is bounded.
        """
        from scipy.stats import beta as beta_dist
        rng = np.random.default_rng(0)
        x = beta_dist.rvs(2.0, 5.0, size=200000, random_state=rng)
        log_e = beta_lr_evalue(x, 2.0, 5.0, 8.0, 2.0)
        e = np.exp(log_e)
        mean_e = e.mean()
        # E[LR | H_0] = 1 exactly; empirical estimate is within 20% with high
        # probability for n = 2e5 due to LR heavy-tail variance
        assert mean_e < 1.3, f"LR mean under null = {mean_e}, exceeds 1.3 (validity violated)"


class TestFDRControl:
    """Empirical FDR control verification at α = 0.10 over 50 replications."""

    def _fdr_avg(self, method_fn, n_reps=50, K=1000):
        fdrs = []
        for rep in range(n_reps):
            stat, is_null, anchor_mask = simulate_two_component_mixture(
                K=K, pi_0=0.7, seed=rep,
            )
            rej, k = method_fn(stat, is_null, anchor_mask, rep)
            if k == 0:
                fdrs.append(0.0)
            else:
                fdrs.append(int(is_null[rej].sum()) / k)
        return float(np.mean(fdrs))

    def test_bh_fdr_control(self):
        """BH should control FDR at 0.10."""
        def fn(stat, is_null, anchor, rep):
            p = beta_pvalue(stat, 2.0, 5.0)
            return bh(p, alpha=0.10)
        fdr = self._fdr_avg(fn)
        assert fdr <= 0.12, f"BH realized FDR = {fdr}, exceeds 0.10 + tolerance"

    def test_storey_fdr_control(self):
        """Storey should control FDR at 0.10."""
        def fn(stat, is_null, anchor, rep):
            p = beta_pvalue(stat, 2.0, 5.0)
            rej, k, _ = storey_qvalue(p, alpha=0.10, lambda_fixed=0.5)
            return rej, k
        fdr = self._fdr_avg(fn)
        assert fdr <= 0.13, f"Storey realized FDR = {fdr}, exceeds 0.10 + tolerance"

    def test_ebh_oracle_fdr_control(self):
        """Vanilla e-BH with oracle LR should control FDR at 0.10."""
        def fn(stat, is_null, anchor, rep):
            return ebh_lr_naive(stat, 2.0, 5.0, 8.0, 2.0, alpha=0.10)
        fdr = self._fdr_avg(fn)
        assert fdr <= 0.12, f"e-BH oracle realized FDR = {fdr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
