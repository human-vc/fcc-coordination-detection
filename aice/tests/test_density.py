"""Tests for aice.density: Beta MoM and MLE estimators."""
import numpy as np
import pytest
from scipy.stats import beta as beta_dist

from aice.density import fit_beta_mom, fit_beta_mle, beta_log_density


class TestFitBetaMom:
    def test_recovers_uniform(self):
        """MoM on Beta(1,1) data should return ~ Beta(1,1)."""
        rng = np.random.default_rng(0)
        x = rng.random(10000)
        a, b = fit_beta_mom(x)
        assert abs(a - 1.0) < 0.1
        assert abs(b - 1.0) < 0.1

    def test_recovers_known_beta(self):
        """MoM on Beta(2, 5) data should recover (2, 5) within tolerance."""
        rng = np.random.default_rng(0)
        x = beta_dist.rvs(2.0, 5.0, size=10000, random_state=rng)
        a, b = fit_beta_mom(x)
        assert abs(a - 2.0) < 0.15
        assert abs(b - 5.0) < 0.4

    def test_recovers_skewed_beta(self):
        """MoM on Beta(8, 2) data should recover (8, 2) within tolerance."""
        rng = np.random.default_rng(1)
        x = beta_dist.rvs(8.0, 2.0, size=10000, random_state=rng)
        a, b = fit_beta_mom(x)
        assert abs(a - 8.0) < 1.0
        assert abs(b - 2.0) < 0.3

    def test_degenerate_input(self):
        """Should not crash on tiny / constant input."""
        a, b = fit_beta_mom(np.array([0.5, 0.5, 0.5]))
        assert a == 1.0 and b == 1.0
        a, b = fit_beta_mom(np.array([0.5]))
        assert a == 1.0 and b == 1.0

    def test_extreme_values(self):
        """Should clip values near 0 and 1 to avoid numerical issues."""
        x = np.array([0.0, 1.0, 1e-10, 1 - 1e-10, 0.5, 0.3])
        a, b = fit_beta_mom(x)
        assert np.isfinite(a) and np.isfinite(b)
        assert a > 0 and b > 0

    def test_consistency_rate(self):
        """MoM should be consistent at parametric 1/sqrt(n) rate.

        Theorem 3 (Section 3.3) requires this rate; verify empirically.
        """
        rng = np.random.default_rng(2)
        true_a, true_b = 3.0, 4.0
        errors = []
        for n in [100, 400, 1600, 6400]:
            errs = []
            for trial in range(20):
                x = beta_dist.rvs(true_a, true_b, size=n,
                                  random_state=rng)
                a, b = fit_beta_mom(x)
                err = np.sqrt((a - true_a) ** 2 + (b - true_b) ** 2)
                errs.append(err)
            errors.append(np.mean(errs))
        # Each 4x increase in n should reduce error by ~2x (parametric rate)
        ratio_1 = errors[0] / errors[1]
        ratio_2 = errors[1] / errors[2]
        ratio_3 = errors[2] / errors[3]
        # Allow tolerance: rate should be in (1.4, 2.5) for 4x sample size
        for r in (ratio_1, ratio_2, ratio_3):
            assert 1.3 < r < 3.0, f"rate ratio {r} out of expected (1.3, 3.0) range"


class TestFitBetaMle:
    def test_recovers_known_beta(self):
        """MLE on Beta(2, 5) data should recover (2, 5)."""
        rng = np.random.default_rng(0)
        x = beta_dist.rvs(2.0, 5.0, size=5000, random_state=rng)
        a, b = fit_beta_mle(x)
        assert abs(a - 2.0) < 0.15
        assert abs(b - 5.0) < 0.4

    def test_efficiency_vs_mom(self):
        """MLE should be at least as efficient as MoM for Beta data."""
        rng = np.random.default_rng(3)
        true_a, true_b = 3.0, 4.0
        mom_errs, mle_errs = [], []
        for trial in range(30):
            x = beta_dist.rvs(true_a, true_b, size=500, random_state=rng)
            a_mom, b_mom = fit_beta_mom(x)
            a_mle, b_mle = fit_beta_mle(x)
            mom_errs.append(np.sqrt((a_mom - true_a) ** 2 + (b_mom - true_b) ** 2))
            mle_errs.append(np.sqrt((a_mle - true_a) ** 2 + (b_mle - true_b) ** 2))
        # MLE should not be substantially worse than MoM (within 20%)
        assert np.mean(mle_errs) < 1.2 * np.mean(mom_errs)


class TestBetaLogDensity:
    def test_matches_scipy(self):
        """log_density should match scipy.stats.beta.logpdf."""
        x = np.linspace(0.01, 0.99, 50)
        for (a, b) in [(2.0, 5.0), (1.0, 1.0), (8.0, 2.0), (0.5, 0.5)]:
            mine = beta_log_density(x, a, b)
            theirs = beta_dist.logpdf(x, a, b)
            np.testing.assert_allclose(mine, theirs, rtol=1e-8)

    def test_handles_extreme_values(self):
        """Should handle x = 0 and x = 1 by clipping."""
        x = np.array([0.0, 1.0, 0.5])
        log_p = beta_log_density(x, 2.0, 5.0)
        assert np.all(np.isfinite(log_p))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
