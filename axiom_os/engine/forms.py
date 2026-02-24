"""
Parametric Form Candidates for Discovery.
Universal interface: fit(X, y) -> (pred, coefs, formula_str).
Selection by AIC/BIC - no Lasso gate.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.linear_model import Lasso, Ridge, ElasticNet
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

FEATURE_NAMES = ["t", "t^2", "t^3", "sqrt(t)", "log(1+t)", "exp(-t)", "exp(-2t)", "exp(-3t)"]


def _curve_fit_multi_start(
    model_func,
    t: np.ndarray,
    y: np.ndarray,
    p0_list: List[tuple],
    bounds: Tuple[Tuple, Tuple],
    sigma: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Multi-start curve_fit: try multiple initial guesses, return first success.
    Reduces sensitivity to bad p0.
    """
    if not HAS_SCIPY:
        return None, None
    for p0 in p0_list:
        try:
            popt, _ = curve_fit(
                model_func, t, y, sigma=sigma, absolute_sigma=False,
                p0=p0, bounds=bounds, maxfev=2000,
            )
            pred = model_func(t, *popt)
            if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                continue
            return popt, pred
        except Exception:
            continue
    return None, None


def _build_aging_features(t: np.ndarray) -> np.ndarray:
    """Build feature matrix for poly/linear forms. t in [0,1]."""
    t = np.asarray(t, dtype=np.float64).ravel()
    t = np.clip(t, 1e-8, 1.0)
    return np.column_stack([
        t, t ** 2, t ** 3, np.sqrt(t), np.log1p(t),
        np.exp(-t), np.exp(-2 * t), np.exp(-3 * t),
    ])


def aic(n: int, mse: float, k: int) -> float:
    """AIC = n*ln(MSE) + 2*k for model selection."""
    if mse <= 0:
        return np.inf
    return n * np.log(mse + 1e-12) + 2 * k


def bic(n: int, mse: float, k: int) -> float:
    """BIC = n*ln(MSE) + k*ln(n). Stronger penalty for complexity."""
    if mse <= 0:
        return np.inf
    return n * np.log(mse + 1e-12) + k * np.log(n)


class FormCandidate(ABC):
    """Abstract form for parametric discovery. Extensible per scenario."""

    name: str = "base"

    @abstractmethod
    def fit(
        self,
        t: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any], str]:
        """
        Fit form to data. Returns (predictions, coefs_dict, formula_str).
        coefs_dict must include "form_type": str.
        sample_weight: optional, higher weight = more influence (sigma = 1/sqrt(w) for curve_fit).
        """
        pass

    @abstractmethod
    def n_params(self) -> int:
        """Number of free parameters for AIC/BIC."""
        pass


class ExpForm(FormCandidate):
    """y = a*exp(-k*t) + c. Learnable k."""

    name = "exp"

    def fit(
        self,
        t: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any], str]:
        t = np.asarray(t, dtype=np.float64).ravel()
        t = np.clip(t, 1e-8, 1.0)
        y = np.asarray(y, dtype=np.float64).ravel()

        def _model(tt, a, k, c):
            return a * np.exp(-k * np.clip(tt, 1e-8, 10.0)) + c

        if not HAS_SCIPY:
            pred = np.full_like(y, np.nan)
            return pred, {"form_type": "exp"}, "exp (scipy required)"

        sigma = None
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=np.float64).ravel()
            sigma = 1.0 / np.sqrt(np.clip(w, 1e-10, 1e10))

        p0_list = [[0.5, 2.0, -0.5], [0.3, 1.0, 0.0], [1.0, 0.5, -0.2]]
        bounds = ([0.1, 0.1, -2], [5, 10, 2])
        popt, pred = _curve_fit_multi_start(_model, t, y, p0_list, bounds, sigma)
        if popt is not None and pred is not None:
            coefs = {"a": popt[0], "k": popt[1], "c": popt[2], "form_type": "exp"}
            formula = f"{popt[0]:.6g}*exp(-{popt[1]:.6g}*t) + {popt[2]:.6g}"
            return pred, coefs, formula
        pred = np.full_like(y, np.nan)
        return pred, {"form_type": "exp"}, "exp (fit failed)"

    def n_params(self) -> int:
        return 3


class PowerForm(FormCandidate):
    """y = a*t^beta + c. Learnable beta."""

    name = "power"

    def fit(
        self,
        t: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any], str]:
        t = np.asarray(t, dtype=np.float64).ravel()
        t = np.clip(t, 1e-8, 1.0)
        y = np.asarray(y, dtype=np.float64).ravel()

        def _model(tt, a, beta, c):
            return a * np.power(np.clip(tt, 1e-8, 10.0), beta) + c

        if not HAS_SCIPY:
            pred = np.full_like(y, np.nan)
            return pred, {"form_type": "power"}, "power (scipy required)"

        sigma = None
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=np.float64).ravel()
            sigma = 1.0 / np.sqrt(np.clip(w, 1e-10, 1e10))

        p0_list = [[-0.5, 0.5, 0.0], [-0.3, 0.7, 0.1], [-1.0, 0.3, -0.1]]
        bounds = ([-5, 0.1, -2], [-0.01, 1.0, 2])
        popt, pred = _curve_fit_multi_start(_model, t, y, p0_list, bounds, sigma)
        if popt is not None and pred is not None:
            coefs = {"a": popt[0], "beta": popt[1], "c": popt[2], "form_type": "power"}
            formula = f"{popt[0]:.6g}*t^{popt[1]:.6g} + {popt[2]:.6g}"
            return pred, coefs, formula
        pred = np.full_like(y, np.nan)
        return pred, {"form_type": "power"}, "power (fit failed)"

    def n_params(self) -> int:
        return 3


class LogForm(FormCandidate):
    """y = a*log(1+t) + c. Early slow, late fast (S-like)."""

    name = "log"

    def fit(
        self,
        t: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any], str]:
        t = np.asarray(t, dtype=np.float64).ravel()
        t = np.clip(t, 1e-8, 1.0)
        y = np.asarray(y, dtype=np.float64).ravel()

        def _model(tt, a, c):
            return a * np.log1p(np.clip(tt, 1e-8, 10.0)) + c

        if not HAS_SCIPY:
            pred = np.full_like(y, np.nan)
            return pred, {"form_type": "log"}, "log (scipy required)"

        sigma = None
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=np.float64).ravel()
            sigma = 1.0 / np.sqrt(np.clip(w, 1e-10, 1e10))

        p0_list = [[-0.5, 0.0], [-0.3, 0.1], [0.5, -0.2]]
        bounds = ([-5, -2], [5, 2])
        popt, pred = _curve_fit_multi_start(_model, t, y, p0_list, bounds, sigma)
        if popt is not None and pred is not None:
            coefs = {"a": popt[0], "c": popt[1], "form_type": "log"}
            formula = f"{popt[0]:.6g}*log(1+t) + {popt[1]:.6g}"
            return pred, coefs, formula
        pred = np.full_like(y, np.nan)
        return pred, {"form_type": "log"}, "log (fit failed)"

    def n_params(self) -> int:
        return 2


class PiecewiseLinearForm(FormCandidate):
    """
    Two-phase: linear zone (t < t_break) + nonlinear zone (t >= t_break).
    Phase 1: f = a1*t + b1. Phase 2: f = a2*t + b2, continuous at t_break.
    For decay: a1 < 0, a2 < 0 (typically |a2| > |a1| for accelerated aging).
    """

    name = "piecewise"

    def fit(
        self,
        t: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any], str]:
        t = np.asarray(t, dtype=np.float64).ravel()
        t = np.clip(t, 1e-8, 1.0)
        y = np.asarray(y, dtype=np.float64).ravel()

        def _model(tt, a1, b1, a2, t_break):
            t_break = np.clip(t_break, 0.1, 0.9)
            b2 = (a1 - a2) * t_break + b1
            return np.where(tt <= t_break, a1 * tt + b1, a2 * tt + b2)

        if not HAS_SCIPY:
            pred = np.full_like(y, np.nan)
            return pred, {"form_type": "piecewise"}, "piecewise (scipy required)"

        sigma = None
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=np.float64).ravel()
            sigma = 1.0 / np.sqrt(np.clip(w, 1e-10, 1e10))

        p0_list = [[-0.3, 0.0, -0.8, 0.5], [-0.2, 0.1, -0.5, 0.4], [-0.5, -0.1, -1.0, 0.6]]
        bounds = ([-1.0, -0.5, -1.0, 0.1], [-0.01, 0.5, -0.01, 0.9])
        popt, pred = _curve_fit_multi_start(_model, t, y, p0_list, bounds, sigma)
        if popt is not None and pred is not None:
            coefs = {
                "a1": popt[0], "b1": popt[1], "a2": popt[2],
                "t_break": float(np.clip(popt[3], 0.1, 0.9)),
                "form_type": "piecewise",
            }
            coefs["b2"] = (popt[0] - popt[2]) * coefs["t_break"] + popt[1]
            formula = (
                f"phase1: {popt[0]:.4g}*t+{popt[1]:.4g} (t<{coefs['t_break']:.2g}); "
                f"phase2: {popt[2]:.4g}*t+{coefs['b2']:.4g}"
            )
            return pred, coefs, formula
        pred = np.full_like(y, np.nan)
        return pred, {"form_type": "piecewise"}, "piecewise (fit failed)"

    def n_params(self) -> int:
        return 4


class PolyForm(FormCandidate):
    """y = c0*t + c1*t^2 + ... + c7*exp(-3t) + intercept. Lasso/OLS."""

    name = "poly"

    def __init__(self, alpha: float = 1e-3):
        self.alpha = alpha

    def fit(
        self,
        t: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any], str]:
        t = np.asarray(t, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        X = _build_aging_features(t)

        if not HAS_SKLEARN:
            pred = np.full_like(y, np.nan)
            return pred, {"form_type": "poly"}, "poly (sklearn required)"

        try:
            models = [
                Lasso(alpha=self.alpha, max_iter=5000),
                Ridge(alpha=self.alpha * 0.1),
                ElasticNet(alpha=self.alpha, max_iter=5000),
            ]
            coef, intercept, pred = None, None, None
            for model in models:
                try:
                    model.fit(X, y, sample_weight=sample_weight)
                    coef = model.coef_
                    intercept = model.intercept_
                    pred = model.predict(X)
                    break
                except Exception:
                    continue
            if coef is None:
                raise RuntimeError("All regressors failed")

            coefs = {f"c{i}": coef[i] for i in range(8)}
            coefs["intercept"] = intercept
            coefs["form_type"] = "poly"

            terms = [
                f"{coef[i]:.6g}*{FEATURE_NAMES[i]}"
                for i in range(8) if abs(coef[i]) > 1e-6
            ]
            if abs(intercept) > 1e-6:
                terms.append(f"{intercept:.6g}")
            formula = " + ".join(terms) if terms else "0"
            return pred, coefs, formula
        except Exception:
            pred = np.full_like(y, np.nan)
            return pred, {"form_type": "poly"}, "poly (fit failed)"

    def n_params(self) -> int:
        return 9  # 8 coefs + intercept


# Default battery aging form candidates (extensible per scenario)
BATTERY_AGING_FORMS: List[FormCandidate] = [
    ExpForm(),
    PowerForm(),
    PiecewiseLinearForm(),
    PolyForm(),
]


# --- Multivariate forms for spatial/temporal discovery (e.g., wind field) ---

def _build_multivariate_features(
    X: np.ndarray,
    degree: int = 2,
    include_interactions: bool = True,
    include_log_z: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build polynomial features for (t, x, y, z) or similar.
    Returns (feature_matrix, feature_names).
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    names = ["t", "x", "y", "z"][:d] if d <= 4 else [f"x{i}" for i in range(d)]

    feats = [X]
    fnames = [names[i] for i in range(d)]

    if degree >= 2:
        # Quadratic terms
        for i in range(d):
            feats.append((X[:, i] ** 2).reshape(-1, 1))
            fnames.append(f"{names[i]}^2")
        if include_interactions and d >= 2:
            for i in range(d):
                for j in range(i + 1, d):
                    feats.append((X[:, i] * X[:, j]).reshape(-1, 1))
                    fnames.append(f"{names[i]}*{names[j]}")

    if include_log_z and d >= 4:
        z = np.clip(X[:, 3], 1e-6, 1.0)
        feats.append(np.log1p(z).reshape(-1, 1))
        fnames.append("log(1+z)")

    return np.hstack(feats), fnames


class MultivariatePolyForm:
    """
    Multivariate polynomial discovery: y = f(t, x, y, z).
    Uses Lasso for sparse coefficient recovery. Supports BIC selection.
    """

    def __init__(
        self,
        alpha: float = 1e-3,
        degree: int = 2,
        include_interactions: bool = True,
        include_log_z: bool = False,
    ):
        self.alpha = alpha
        self.degree = degree
        self.include_interactions = include_interactions
        self.include_log_z = include_log_z

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        var_names: Optional[List[str]] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any], str]:
        """
        Fit y = f(X). X: (N, D), y: (N,).
        Returns (predictions, coefs_dict, formula_str).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, d = X.shape
        if n < 3:
            return np.full_like(y, np.nan), {"form_type": "multivar_poly"}, "multivar (insufficient data)"

        var_names = var_names if var_names and len(var_names) >= d else (
            ["t", "x", "y", "z"][:d] if d <= 4 else [f"x{i}" for i in range(d)]
        )

        F, fnames = _build_multivariate_features(
            X, degree=self.degree,
            include_interactions=self.include_interactions,
            include_log_z=self.include_log_z,
        )
        # Override fnames with var_names for first d columns
        for i in range(min(d, len(var_names))):
            fnames[i] = var_names[i]
        for i in range(d, min(d + d, len(fnames))):
            idx = i - d
            if idx < len(var_names):
                fnames[i] = f"{var_names[idx]}^2"
        base = d + d
        if self.include_interactions and d >= 2:
            k = 0
            for i in range(d):
                for j in range(i + 1, d):
                    if base + k < len(fnames):
                        fnames[base + k] = f"{var_names[i]}*{var_names[j]}"
                    k += 1

        if not HAS_SKLEARN:
            # Fallback: numpy lstsq for linear fit
            try:
                ones = np.ones((n, 1))
                F_aug = np.hstack([F, ones])
                coef_ls, _, _, _ = np.linalg.lstsq(F_aug, y, rcond=None)
                pred = F_aug @ coef_ls
                terms = []
                for i, c in enumerate(coef_ls[:-1]):
                    if abs(c) > 1e-6:
                        terms.append(f"{c:.4g}*{fnames[i]}")
                if abs(coef_ls[-1]) > 1e-6:
                    terms.append(f"{coef_ls[-1]:.4g}")
                formula = " + ".join(terms) if terms else "0"
                coefs = {fnames[i]: coef_ls[i] for i in range(len(fnames)) if abs(coef_ls[i]) > 1e-6}
                coefs["intercept"] = coef_ls[-1]
                coefs["form_type"] = "multivar_poly"
                return pred, coefs, formula
            except Exception:
                return np.full_like(y, np.nan), {"form_type": "multivar_poly"}, "multivar (fit failed)"

        try:
            models = [
                Lasso(alpha=self.alpha, max_iter=5000),
                Ridge(alpha=self.alpha * 0.1),
                ElasticNet(alpha=self.alpha, max_iter=5000),
            ]
            coef, intercept, pred = None, None, None
            for model in models:
                try:
                    model.fit(F, y, sample_weight=sample_weight)
                    coef = model.coef_
                    intercept = model.intercept_
                    pred = model.predict(F)
                    break
                except Exception:
                    continue
            if coef is None:
                raise RuntimeError("All regressors failed")

            terms = []
            for i, c in enumerate(coef):
                if abs(c) > 1e-6:
                    terms.append(f"{c:.4g}*{fnames[i]}")
            if abs(intercept) > 1e-6:
                terms.append(f"{intercept:.4g}")
            formula = " + ".join(terms) if terms else "0"
            coefs = {fnames[i]: coef[i] for i in range(len(fnames)) if abs(coef[i]) > 1e-6}
            coefs["intercept"] = intercept
            coefs["form_type"] = "multivar_poly"
            return pred, coefs, formula
        except Exception:
            return np.full_like(y, np.nan), {"form_type": "multivar_poly"}, "multivar (fit failed)"

    def n_params(self) -> int:
        d = 4  # typical (t,x,y,z)
        k = d + (d * (d + 1) // 2) if self.include_interactions else d + d
        if self.include_log_z:
            k += 1
        return k + 1  # +1 intercept
