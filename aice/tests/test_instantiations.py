"""Tests for aice.instantiations: DE/housekeeping non-coordination instantiation."""
import numpy as np
import pytest

from aice.instantiations import simulate_de_experiment, run_de_bakeoff


class TestSimulateDE:
    def test_basic_shape(self):
        """Output shapes match G."""
        sim = simulate_de_experiment(G=1000, seed=0)
        for key in ["stat", "anchor_mask", "is_alt", "z", "delta"]:
            assert sim[key].shape == (1000,), f"{key} shape: {sim[key].shape}"

    def test_anchor_purity(self):
        """Housekeeping anchors should never be DE (delta=0)."""
        sim = simulate_de_experiment(G=5000, seed=0)
        # Anchor genes have delta = 0 by construction
        np.testing.assert_array_equal(sim["delta"][sim["anchor_mask"]], 0)
        # Anchor genes are NOT differential
        assert not sim["is_alt"][sim["anchor_mask"]].any()

    def test_de_fraction(self):
        """Empirical DE fraction matches pi_de."""
        sim = simulate_de_experiment(G=10000, pi_de=0.10, seed=0)
        emp_de = sim["is_alt"].mean()
        assert abs(emp_de - 0.10) < 0.02

    def test_hk_fraction(self):
        """Empirical HK fraction matches pi_hk."""
        sim = simulate_de_experiment(G=10000, pi_hk=0.05, pi_de=0.10, seed=0)
        emp_hk = sim["anchor_mask"].mean()
        assert abs(emp_hk - 0.05) < 0.02

    def test_stat_in_unit_interval(self):
        """Test statistic in (0, 1)."""
        sim = simulate_de_experiment(G=1000, seed=0)
        assert sim["stat"].min() > 0
        assert sim["stat"].max() < 1


class TestDEBakeoff:
    def test_runs(self):
        """Smoke test: run_de_bakeoff returns metrics."""
        result = run_de_bakeoff(seed=0, alpha=0.10)
        assert "metrics" in result
        assert len(result["metrics"]) == 4  # AICE + BH + BY + Storey
        for m in result["metrics"]:
            assert "method" in m
            assert "k" in m
            assert "fdp" in m
            assert "power" in m


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
