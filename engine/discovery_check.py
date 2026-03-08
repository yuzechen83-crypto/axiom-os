"""
Discovery Check - Symbolic Regression for Meta-Axis Kernel Verification
Train RCLN on galaxy residuals, extract f(r), use DiscoveryEngine to match
the mathematical expansion of the Projection Integral.
"""

from typing import Optional, Tuple
import numpy as np

from .discovery import DiscoveryEngine


def discovery_check_radial_profile(
    r: np.ndarray,
    V_def_sq: np.ndarray,
    V_bary_sq: np.ndarray,
    var_names: Optional[list] = None,
) -> Tuple[Optional[str], np.ndarray, Optional[dict]]:
    """
    Use DiscoveryEngine to fit f(r) for ΔV² = f(r, V_bary²).
    Check if discovered formula matches projection integral expansion:
    ΔV² ∝ ρ(r) · C with ρ ∝ V_bary²/r².
    """
    engine = DiscoveryEngine(use_pysr=False)
    X = np.column_stack([r, V_bary_sq])
    if var_names is None:
        var_names = ["r", "V_bary_sq"]
    formula, pred, coefs = engine.discover_multivariate(
        X, V_def_sq, var_names=var_names, selector="bic",
    )
    return formula, pred, coefs


def discovery_check_kernel_shape(
    r: np.ndarray,
    V_def_sq: np.ndarray,
    V_bary_sq: np.ndarray,
) -> dict:
    """
    Full discovery check: fit f(r), compare to projection integral form.
    Projection predicts: ΔV² ∝ V_bary²/r² (or similar ρ proxy).
    Returns dict with formula, r2, and match_score (heuristic).
    """
    formula, pred, coefs = discovery_check_radial_profile(
        r, V_def_sq, V_bary_sq, var_names=["r", "V_bary_sq"],
    )
    result = {"formula": formula, "pred": pred, "coefs": coefs, "r2": 0.0, "match_score": 0.0}

    if formula is None or pred is None:
        return result

    ss_res = np.sum((pred - V_def_sq) ** 2)
    ss_tot = np.sum((V_def_sq - V_def_sq.mean()) ** 2) + 1e-12
    result["r2"] = float(1 - ss_res / ss_tot)

    # Heuristic: does formula contain V_bary_sq and r? (projection form)
    f_lower = formula.lower()
    has_v_bary = "v_bary" in f_lower or "x1" in f_lower
    has_r = "r" in f_lower or "x0" in f_lower
    result["match_score"] = float(has_v_bary and has_r)

    return result
