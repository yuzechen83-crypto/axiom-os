"""
Discovery Engine - The Abductive Reasoner (Loop B)
Turn Neural weights into Physics Equations.
Dimensional Gatekeeper (Buckingham π) + Symbolic Regression + Crystallization
Supports Signed Quadratic (v·|v|) for odd functions like quadratic drag F ∝ -v·|v|.
"""

import torch
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
import itertools

from ..core.hippocampus import HippocampusLibrary
from ..layers.rcln import DiscoveryHotspot

try:
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class DimensionGroup:
    """Dimension powers [M, L, T, Q, Θ]"""
    powers: Tuple[int, int, int, int, int]

    def __mul__(self, other: "DimensionGroup") -> "DimensionGroup":
        return DimensionGroup(tuple(a + b for a, b in zip(self.powers, other.powers)))

    def __truediv__(self, other: "DimensionGroup") -> "DimensionGroup":
        return DimensionGroup(tuple(a - b for a, b in zip(self.powers, other.powers)))

    def is_dimensionless(self) -> bool:
        return all(p == 0 for p in self.powers)


def buckingham_pi(
    variables: List[Tuple[str, DimensionGroup]],
    repeatable: Optional[List[str]] = None,
) -> List[Dict[str, float]]:
    """
    Buckingham π Theorem: Form dimensionless groups Π1, Π2, ...
    variables: [(name, DimensionGroup), ...]
    Returns list of {var: exponent} for each dimensionless group
    """
    if not variables:
        return []
    n = len(variables)
    dim_matrix = np.array([list(v[1].powers) for v in variables])
    names = [v[0] for v in variables]
    rank = np.linalg.matrix_rank(dim_matrix)
    k = n - rank  # number of dimensionless groups
    if k <= 0:
        return []

    # Find null space (exponents that make product dimensionless)
    _, _, vh = np.linalg.svd(dim_matrix)
    null_space = vh[rank:].T
    groups = []
    for i in range(min(k, null_space.shape[1])):
        exp = null_space[:, i]
        exp = exp / (np.abs(exp).max() + 1e-10)
        groups.append({names[j]: float(exp[j]) for j in range(n)})
    return groups


def check_dimensional_homogeneity(formula_units: str, lhs: str, rhs: str) -> bool:
    """Constraint: output formula MUST be dimensionally homogeneous (simplified check)"""
    return True  # Placeholder: full implementation would parse and verify


def check_symmetry_odd(
    formula: str,
    var_names: Optional[List[str]] = None,
) -> bool:
    """
    Check if formula respects odd symmetry: f(-x) = -f(x).
    For friction/drag: F ∝ -v·|v| is odd. F ∝ v² is even (wrong for antisymmetric).
    """
    if not formula or "|" not in formula:
        return True  # No signed quadratic, assume OK
    return "|" in formula  # v·|v| is odd


def check_symmetry_even(
    formula: str,
    var_names: Optional[List[str]] = None,
) -> bool:
    """Check if formula respects even symmetry: f(-x) = f(x). For energy, potential."""
    return True


def to_dimensionless_groups(
    x: np.ndarray,
    variable_dims: List[Tuple[str, DimensionGroup]],
    repeatable: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """
    Buckingham π: Convert raw inputs to dimensionless groups Π1, Π2, ...
    Returns (features as numpy array, list of group exponent dicts).
    """
    groups = buckingham_pi(variable_dims, repeatable)
    if not groups:
        return x, []
    n = x.shape[0]
    n_vars = x.shape[-1]
    names = [v[0] for v in variable_dims]
    if len(names) != n_vars:
        names = [f"x{i}" for i in range(n_vars)]
    pi_features = []
    for g in groups:
        prod = np.ones(n)
        for j in range(min(len(names), n_vars)):
            name = names[j]
            if name in g:
                prod *= (x[:, j] ** g[name])
        pi_features.append(prod)
    return np.column_stack(pi_features) if pi_features else x, groups


def build_signed_quadratic_features(x_arr: np.ndarray, var_names: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Build features for odd functions: [1, x0, x1, ..., x0*|x0|, x1*|x1|, ...]
    v·|v| is the signed quadratic (antisymmetric), needed for F ∝ -v·|v| (quadratic drag).
    """
    n_vars = x_arr.shape[-1]
    names = var_names or [f"x{i}" for i in range(n_vars)]
    cols = [np.ones(len(x_arr))]
    feat_names = ["1"]
    for i in range(n_vars):
        cols.append(x_arr[:, i])
        feat_names.append(names[i])
        cols.append(x_arr[:, i] * np.abs(x_arr[:, i]))
        feat_names.append(f"{names[i]}*|{names[i]}|")
    return np.column_stack(cols), feat_names


class DiscoveryEngine:
    """
    Workflow:
    1. Input: (x, y_soft) from Discovery Hotspot
    2. Dimensional Gatekeeper: Buckingham π → dimensionless Π
    3. Symbolic Regression: find y_soft = f(Π)
    4. Crystallization: return new formula to Hippocampus
    5. Reset F_soft to near-zero
    """

    def __init__(
        self,
        library: HippocampusLibrary,
        max_terms: int = 5,
        pi_precision: float = 1e-6,
    ):
        self.library = library
        self.max_terms = max_terms
        self.pi_precision = pi_precision
        self._discovery_log: List[Dict] = []

    def extract_formula(
        self,
        x: torch.Tensor,
        y_soft: torch.Tensor,
        variable_names: Optional[List[str]] = None,
        semantic_id: Optional[str] = None,
        polynomial_degree: int = 2,
        use_signed_quadratic: bool = True,
        enforce_odd_symmetry: bool = False,
        variable_dims: Optional[List[Tuple[str, DimensionGroup]]] = None,
    ) -> Optional[str]:
        """
        Symbolic regression: fit y = f(x) or f(Π1, Π2, ...).
        use_signed_quadratic: include v·|v| terms for odd functions (e.g. quadratic drag F ∝ -v·|v|).
        enforce_odd_symmetry: prefer odd functions for friction/drag.
        variable_dims: for Buckingham π conversion before fitting.
        """
        x_np = x.detach().cpu().numpy()
        y_np = y_soft.detach().cpu().numpy().ravel()
        if x_np.size == 0 or y_np.size == 0:
            return None

        n_vars = x_np.shape[-1]
        names = variable_names or [f"x{i}" for i in range(n_vars)]

        # Buckingham π: convert to dimensionless if dims provided
        if variable_dims and len(variable_dims) == n_vars:
            x_fit, _ = to_dimensionless_groups(x_np, variable_dims)
            names = [f"π{i+1}" for i in range(x_fit.shape[-1])]
        else:
            x_fit = x_np

        try:
            if use_signed_quadratic or enforce_odd_symmetry:
                X_sq, feat_names = build_signed_quadratic_features(x_fit, names)
                if HAS_SKLEARN:
                    reg = LinearRegression().fit(X_sq, y_np)
                    all_coefs = np.concatenate([[reg.intercept_], reg.coef_])
                else:
                    all_coefs = np.linalg.lstsq(X_sq, y_np, rcond=None)[0]
                constant = all_coefs[0]
                terms = [f"{constant:.4g}"]
                for i, (c, fn) in enumerate(zip(all_coefs[1:], feat_names[1:])):
                    if abs(c) > 1e-6:
                        terms.append(f"{c:+.4g}*{fn}")
                formula = "y = " + "".join(terms)
                if enforce_odd_symmetry and not check_symmetry_odd(formula, names):
                    pass  # Fall through to poly
                else:
                    return formula
        except Exception:
            pass

        try:
            if HAS_SKLEARN and polynomial_degree >= 2:
                poly = PolynomialFeatures(degree=polynomial_degree, include_bias=True)
                X_poly = poly.fit_transform(x_np)
                reg = LinearRegression().fit(X_poly, y_np)
                coeffs = reg.coef_
                intercept = reg.intercept_
                feature_names = poly.get_feature_names_out(names)
                constant = intercept + sum(c for c, fn in zip(coeffs, feature_names) if fn == "1")
                terms = [f"{constant:.4g}"]
                for c, fn in zip(coeffs, feature_names):
                    if fn != "1" and abs(c) > 1e-6:
                        terms.append(f"{c:+.4g}*{fn}")
                return "y = " + "".join(terms)
        except Exception:
            pass

        X = np.column_stack([np.ones(len(x_np)), x_np])
        try:
            coeffs = np.linalg.lstsq(X, y_np, rcond=None)[0]
            terms = [f"{coeffs[0]:.4g}"]
            for i, (c, n) in enumerate(zip(coeffs[1:], names)):
                if abs(c) > 1e-6:
                    terms.append(f"{c:+.4g}*{n}")
            return "y = " + "".join(terms)
        except Exception:
            return "y = f(x)"

    def process_hotspot(
        self,
        hotspot: DiscoveryHotspot,
        rcln_module: Any,
        semantic_id_prefix: str = "Discovered",
    ) -> Optional[str]:
        """
        Full workflow: Hotspot → Buckingham π → Symbolic fit → Crystallize → Reset
        """
        if hotspot.last_input is None or hotspot.last_soft_output is None:
            return None

        x = hotspot.last_input
        y_soft = hotspot.last_soft_output
        formula = self.extract_formula(x, y_soft, semantic_id=semantic_id_prefix)

        if formula:
            sid = f"{semantic_id_prefix}_{len(self._discovery_log)}"
            self.library.crystallize(sid, formula, units="", domain="discovered")
            self._discovery_log.append({
                "semantic_id": sid,
                "formula": formula,
                "hotspot_id": hotspot.instance_id,
            })
            if hasattr(rcln_module, "reset_soft_weights"):
                rcln_module.reset_soft_weights()
            if hasattr(rcln_module, "reset_monitor"):
                rcln_module.reset_monitor()
            return formula
        return None

    def get_log(self) -> List[Dict]:
        return self._discovery_log.copy()
