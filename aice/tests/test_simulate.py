"""Tests for aice.simulate: synthetic 2-component mixture generator."""
import numpy as np
import pytest

from aice.simulate import simulate_two_component_mixture


class TestSimulate:
    def test_basic_shape(self):
        """Output shapes should match K."""
        K = 1000
        stat, is_null, anchor_mask = simulate_two_component_mixture(
            K=K, pi_0=0.7, seed=0,
        )
        assert stat.shape == (K,)
        assert is_null.shape == (K,)
        assert anchor_mask.shape == (K,)

    def test_value_range(self):
        """Test statistics should be in (0, 1)."""
        stat, _, _ = simulate_two_component_mixture(K=10000, pi_0=0.7, seed=0)
        assert np.all(stat >= 0)
        assert np.all(stat <= 1)

    def test_null_fraction(self):
        """Empirical null fraction should match pi_0 within MC error."""
        K = 100000
        for pi_0 in [0.5, 0.7, 0.9]:
            _, is_null, _ = simulate_two_component_mixture(
                K=K, pi_0=pi_0, seed=42,
            )
            empirical = is_null.mean()
            assert abs(empirical - pi_0) < 0.01, (
                f"empirical null frac {empirical:.3f} differs from pi_0={pi_0}"
            )

    def test_anchor_purity(self):
        """Anchors should be a subset of the null clusters (purity = 1)."""
        K = 5000
        for seed in range(20):
            _, is_null, anchor_mask = simulate_two_component_mixture(
                K=K, pi_0=0.7, anchor_rate=0.10, seed=seed,
            )
            assert np.all(is_null[anchor_mask]), (
                f"Anchor purity violated at seed={seed}: "
                f"{(~is_null[anchor_mask]).sum()} non-null anchors"
            )

    def test_anchor_rate(self):
        """Anchor count should approximately match anchor_rate × pi_0 × K."""
        K = 10000
        pi_0 = 0.7
        anchor_rate = 0.15
        _, _, anchor_mask = simulate_two_component_mixture(
            K=K, pi_0=pi_0, anchor_rate=anchor_rate, seed=0,
        )
        expected = anchor_rate * pi_0 * K
        actual = anchor_mask.sum()
        # Allow ~5% tolerance for binomial variation
        assert abs(actual - expected) / expected < 0.10, (
            f"Anchor count {actual} differs from expected {expected:.0f}"
        )

    def test_alternative_distribution(self):
        """Alternative samples should have higher mean than null samples (Beta(8,2) > Beta(2,5))."""
        stat, is_null, _ = simulate_two_component_mixture(
            K=20000, pi_0=0.5, seed=0,
        )
        null_mean = stat[is_null].mean()
        alt_mean = stat[~is_null].mean()
        # Beta(2, 5) mean = 2/7 ≈ 0.286; Beta(8, 2) mean = 8/10 = 0.8
        assert abs(null_mean - 2/7) < 0.02
        assert abs(alt_mean - 0.8) < 0.02

    def test_dependence_structures(self):
        """All dependence options should produce valid output."""
        for dep in ["independent", "prds_block", "prds_ar1", "arbitrary"]:
            stat, is_null, anchor_mask = simulate_two_component_mixture(
                K=1000, pi_0=0.7, dependence=dep, rho=0.3, seed=0,
            )
            assert stat.shape == (1000,)
            assert np.all(np.isfinite(stat))

    def test_dependence_correlation(self):
        """AR(1) dependence should produce visibly correlated samples."""
        stat_ind, _, _ = simulate_two_component_mixture(
            K=5000, pi_0=0.99, dependence="independent", seed=0,
        )
        stat_dep, _, _ = simulate_two_component_mixture(
            K=5000, pi_0=0.99, dependence="prds_ar1", rho=0.7, seed=0,
        )
        # AR(1) should have higher lag-1 autocorrelation
        ac_ind = np.corrcoef(stat_ind[:-1], stat_ind[1:])[0, 1]
        ac_dep = np.corrcoef(stat_dep[:-1], stat_dep[1:])[0, 1]
        assert ac_dep > ac_ind + 0.1, (
            f"Expected AR(1) autocorr > independent: {ac_dep:.3f} vs {ac_ind:.3f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
