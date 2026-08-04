"""AICE: Anchor-Identified Cross-fit Compound E-value.

A general framework for constructing compound e-values for FDR control
when the null component admits a structurally identifiable anchor.

Public API:
    fit_evalues(stat, anchor_mask, n_folds=5, family='beta', seed=42)
        Returns log e-values via cross-fit anchor-restricted MLE.

    ebh(log_e, alpha)
        Apply e-BH at level alpha. Returns rejection indices and k_hat.

    evaluate(log_e, labels, alpha)
        Compute AP, FDR, power against ground truth labels.

References:
    Ignatiadis, Wang, Ramdas (2024). Asymptotic and compound e-values.
        arXiv:2409.19812.
    Wang, Ramdas (2022). FDR control with e-values. JRSSB 84(3).
    Chernozhukov et al. (2018). Double/debiased ML. Econom. J. 21(1).
    Vovk, Wang (2021). E-values: calibration, combination, applications.
        Annals of Statistics 49(3).
"""
from aice.core import (
    fit_evalues,
    ebh,
    evaluate,
    conformal_p_from_evalues,
    aice_cc_boost,
    aice_storey_ebh,
)
from aice.density import fit_beta_mom, fit_beta_mle, beta_log_density
from aice.simulate import simulate_two_component_mixture

__version__ = "0.1.0"
__all__ = [
    "fit_evalues",
    "ebh",
    "evaluate",
    "conformal_p_from_evalues",
    "aice_cc_boost",
    "aice_storey_ebh",
    "fit_beta_mom",
    "fit_beta_mle",
    "beta_log_density",
    "simulate_two_component_mixture",
]
