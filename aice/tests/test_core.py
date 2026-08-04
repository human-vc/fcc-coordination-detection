"""Tests for aice.core: cross-fit AICE construction + e-BH + evaluate."""
import numpy as np
import pytest
from scipy.stats import beta as beta_dist

from aice import fit_evalues, ebh, evaluate
from aice.simulate import simulate_two_component_mixture


class TestFitEvalues:
    def test_basic_run(self):
        """Smoke test: fit_evalues runs without errors."""
        rng = np.random.default_rng(0)
        K = 1000
        stat = rng.random(K)
        anchor_mask = np.zeros(K, dtype=bool)
        anchor_mask[:50] = True
        log_e = fit_evalues(stat, anchor_mask, n_folds=5)
        assert log_e.shape == (K,)
        assert np.all(np.isfinite(log_e))

    def test_returns_components(self):
        """return_components=True should return per-fold info."""
        rng = np.random.default_rng(0)
        K = 1000
        stat = rng.random(K)
        anchor_mask = np.zeros(K, dtype=bool)
        anchor_mask[:50] = True
        log_e, fold_info = fit_evalues(stat, anchor_mask, n_folds=5,
                                        return_components=True)
        assert len(fold_info) == 5
        for info in fold_info:
            assert "fold" in info
            if not info.get("skipped"):
                assert "a0" in info and "b0" in info
                assert "a1" in info and "b1" in info

    def test_input_validation(self):
        """Invalid inputs should raise."""
        with pytest.raises(ValueError, match="shape mismatch"):
            fit_evalues(np.array([0.5, 0.6]), np.array([True]))
        with pytest.raises(ValueError, match="must be 1d"):
            fit_evalues(np.array([[0.5, 0.6]]), np.array([[True, False]]))
        with pytest.raises(ValueError, match="Only 'beta' family"):
            fit_evalues(np.random.random(100), np.zeros(100, dtype=bool),
                        family="gaussian")
        with pytest.raises(ValueError, match="estimator must"):
            fit_evalues(np.random.random(100), np.zeros(100, dtype=bool),
                        estimator="bayes")
        with pytest.raises(ValueError, match="at least"):
            fit_evalues(np.array([0.5, 0.6]), np.array([True, False]), n_folds=5)

    def test_anchor_purity_validity(self):
        """Under correct specification + anchor purity, E[E_c | H_0] should be ≤ 1.

        This is the empirical check of Theorem 1 (validity).
        """
        rng = np.random.default_rng(42)
        K = 5000
        n_reps = 50

        evalues_under_null = []
        for rep in range(n_reps):
            stat, is_null, anchor_mask = simulate_two_component_mixture(
                K=K, pi_0=0.7, anchor_rate=0.10,
                seed=rep,
            )
            log_e = fit_evalues(stat, anchor_mask, n_folds=5,
                                 seed=rep)
            evalues_under_null.append(np.exp(log_e[is_null]).mean())
        mean_e_under_null = np.mean(evalues_under_null)
        # Should be ≤ 1 with high probability under correct specification
        assert mean_e_under_null < 1.2, (
            f"Mean e-value under null = {mean_e_under_null:.3f}, expected ≤ 1 "
            f"under correct specification + anchor purity."
        )


class TestEBH:
    def test_no_rejections(self):
        """Small e-values should produce no rejections."""
        log_e = np.full(100, -10.0)  # all e-values near zero
        rej, k_hat = ebh(log_e, alpha=0.10)
        assert k_hat == 0
        assert len(rej) == 0

    def test_all_rejections(self):
        """Very large e-values should reject everything."""
        log_e = np.full(100, 100.0)
        rej, k_hat = ebh(log_e, alpha=0.10)
        assert k_hat == 100
        assert len(rej) == 100

    def test_correctness_against_naive(self):
        """e-BH output should match a naive O(K^2) reference implementation."""
        rng = np.random.default_rng(0)
        K = 200
        log_e = rng.standard_normal(K) * 3
        e = np.exp(log_e)
        # Naive reference
        order = np.argsort(-e)
        sorted_e = e[order]
        valid_k = []
        for k in range(1, K + 1):
            if sorted_e[k - 1] >= K / (0.10 * k):
                valid_k.append(k)
        ref_k_hat = max(valid_k) if valid_k else 0

        rej, k_hat = ebh(log_e, alpha=0.10)
        assert k_hat == ref_k_hat
        assert len(rej) == ref_k_hat

    def test_alpha_validation(self):
        """Invalid alpha should raise."""
        with pytest.raises(ValueError):
            ebh(np.array([1.0, 2.0]), alpha=0.0)
        with pytest.raises(ValueError):
            ebh(np.array([1.0, 2.0]), alpha=1.0)
        with pytest.raises(ValueError):
            ebh(np.array([1.0, 2.0]), alpha=-0.1)

    def test_threshold_relationship(self):
        """Each rejected hypothesis should have e-value >= K/(alpha * k_hat)."""
        rng = np.random.default_rng(0)
        K = 1000
        log_e = rng.standard_normal(K) * 2
        rej, k_hat = ebh(log_e, alpha=0.10)
        if k_hat > 0:
            threshold = K / (0.10 * k_hat)
            for idx in rej:
                assert np.exp(log_e[idx]) >= threshold


class TestEvaluate:
    def test_perfect_separation(self):
        """When alternative e-values are huge and null are tiny, recall should be 1."""
        K = 100
        labels = np.zeros(K, dtype=int)
        labels[:30] = 1
        log_e = np.where(labels == 1, 100.0, -100.0)
        result = evaluate(log_e, labels, alpha=0.10)
        assert result["recall"] == 1.0
        assert result["precision"] == 1.0
        assert result["FDP"] == 0.0
        assert result["k_rejected"] == 30

    def test_no_alternative(self):
        """When no alternatives, AP should be NaN, no rejections under null."""
        K = 100
        labels = np.zeros(K, dtype=int)
        log_e = np.full(K, -10.0)
        result = evaluate(log_e, labels, alpha=0.10)
        assert np.isnan(result["AP"])
        assert result["k_rejected"] == 0


class TestValiditySimulation:
    """Empirical validation of Theorem 1 (FDR control in finite samples)."""

    @pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
    def test_fdr_control_correct_specification(self, alpha):
        """Under correct Beta specification + anchor purity, realized FDR ≤ α.

        This is the empirical test of Theorem 1.
        """
        n_reps = 100
        K = 2000
        pi_0 = 0.7
        fdrs = []
        for rep in range(n_reps):
            stat, is_null, anchor_mask = simulate_two_component_mixture(
                K=K, pi_0=pi_0, anchor_rate=0.10, seed=rep,
            )
            log_e = fit_evalues(stat, anchor_mask, n_folds=5, seed=rep)
            rej, k_hat = ebh(log_e, alpha=alpha)
            if k_hat == 0:
                fdrs.append(0.0)
            else:
                fdrs.append(int(is_null[rej].sum()) / k_hat)
        mean_fdr = float(np.mean(fdrs))
        # Realized FDR should be at most alpha + small tolerance for MC error
        assert mean_fdr <= alpha + 0.03, (
            f"Realized FDR = {mean_fdr:.4f} exceeds α + tolerance = "
            f"{alpha + 0.03:.4f} at α = {alpha}"
        )


class TestAnchorEmpiricalStorey:
    """Empirical validation of the anchor-empirical Storey-e adaptive boost."""

    def test_returns_pi_hat_in_unit_interval(self):
        from aice.core import aice_storey_ebh
        stat, _, anchor = simulate_two_component_mixture(K=1000, pi_0=0.7, seed=0)
        log_e = fit_evalues(stat, anchor, seed=0, decontaminate=True)
        rej, k, pi_hat = aice_storey_ebh(log_e, anchor, alpha=0.10)
        assert 0.0 < pi_hat <= 1.0, f"pi_hat_0 = {pi_hat} not in (0, 1]"
        assert k <= len(log_e)

    def test_fallback_with_few_anchors(self):
        from aice.core import aice_storey_ebh
        stat, _, _ = simulate_two_component_mixture(K=200, pi_0=0.7, seed=0)
        anchor = np.zeros(len(stat), dtype=bool)
        anchor[:5] = True  # Below threshold
        log_e = fit_evalues(stat, anchor, seed=0)
        rej, k, pi_hat = aice_storey_ebh(log_e, anchor, alpha=0.10)
        assert pi_hat == 1.0  # No boost when n_anchor too small

    @pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
    def test_storey_fdr_control(self, alpha):
        """Realized FDR of Storey-boosted e-BH ≤ α + tolerance across grid."""
        from aice.core import aice_storey_ebh
        n_reps = 80
        fdrs = []
        for rep in range(n_reps):
            stat, is_null, anchor = simulate_two_component_mixture(
                K=2000, pi_0=0.7, anchor_rate=0.10, seed=rep + 5000)
            log_e = fit_evalues(stat, anchor, seed=rep + 5000, decontaminate=True)
            rej, k, _ = aice_storey_ebh(log_e, anchor, alpha=alpha)
            if k == 0:
                fdrs.append(0.0)
            else:
                fdrs.append(int(is_null[rej].sum()) / k)
        mean_fdr = float(np.mean(fdrs))
        assert mean_fdr <= alpha + 0.03, (
            f"Storey-e FDR = {mean_fdr:.4f} exceeds α + tolerance at α={alpha}"
        )

    def test_storey_dominates_vanilla(self):
        """Storey boost recovers ≥ as many rejections as vanilla e-BH on average."""
        from aice.core import aice_storey_ebh
        n_reps = 30
        k_storey, k_vanilla = [], []
        for rep in range(n_reps):
            stat, _, anchor = simulate_two_component_mixture(
                K=2000, pi_0=0.5, anchor_rate=0.10, seed=rep + 6000)
            log_e = fit_evalues(stat, anchor, seed=rep + 6000, decontaminate=True)
            _, k_s, _ = aice_storey_ebh(log_e, anchor, alpha=0.10)
            _, k_v = ebh(log_e, alpha=0.10)
            k_storey.append(k_s); k_vanilla.append(k_v)
        # Storey should be at least as powerful on average
        assert np.mean(k_storey) >= np.mean(k_vanilla), (
            f"Storey k = {np.mean(k_storey):.1f} < vanilla k = {np.mean(k_vanilla):.1f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
