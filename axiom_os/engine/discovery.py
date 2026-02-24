"""
Discovery Engine - Symbolic Regression & Dimensional Analysis
PROPRIETARY - Copyright (c) 2026 yuzechen83-crypto. All Rights Reserved.
Unauthorized copying, modification, or distribution of this algorithm is prohibited.

Extracts symbolic laws from RCLN's Soft Shell using Buckingham Pi constraints.
Universal parametric discovery: fit all form candidates, select by AIC/BIC.
"""

from typing import List, Optional, Any, Tuple, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .forms import FormCandidate
from scipy.linalg import null_space

try:
    from pysr import PySRRegressor
    HAS_PYSR = True
except ImportError:
    HAS_PYSR = False

try:
    from sklearn.linear_model import Lasso, Ridge, ElasticNet
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from pysindy import SINDy
    HAS_PYSINDY = True
except ImportError:
    HAS_PYSINDY = False


class DiscoveryEngine:
    """
    Symbolic regression with dimensional gatekeeper.
    Constrains search to functions mapping π_in → π_out (dimensionless groups).
    """

    def __init__(
        self,
        library: Optional[Any] = None,
        use_pysr: bool = True,
        validation_mse_threshold: float = 0.5,
    ):
        self.library = library
        self.use_pysr = use_pysr and HAS_PYSR
        self.validation_mse_threshold = validation_mse_threshold

    def validate_formula(
        self,
        formula: str,
        X: np.ndarray,
        y_true: np.ndarray,
        output_dim: int = 1,
    ) -> Tuple[bool, float]:
        """
        Validate discovered formula: compare predictions to ground truth.
        Returns (valid, mse). valid = mse < validation_mse_threshold.
        """
        if not formula or len(str(formula).strip()) < 2:
            return False, float("inf")
        X = np.asarray(X, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.float64)
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        n_samples, n_in = X.shape
        safe = {"np": np, "sin": np.sin, "cos": np.cos, "sqrt": np.sqrt, "exp": np.exp, "log": np.log, "abs": np.abs, "pow": np.power}
        safe["x"] = X
        for i in range(min(10, n_in)):
            safe[f"x{i}"] = X[:, i]
            safe[f"x_{i}"] = X[:, i]
        try:
            result = eval(formula, {"__builtins__": {}}, safe)
            y_pred = np.asarray(result, dtype=np.float64)
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)
            if y_pred.shape != y_true.shape:
                y_pred = np.broadcast_to(np.ravel(y_pred)[:1], y_true.shape)
            mse = float(np.mean((y_pred - y_true) ** 2))
            valid = mse < self.validation_mse_threshold
            return valid, mse
        except Exception:
            return False, float("inf")

    def get_dimensionless_groups(
        self,
        inputs: np.ndarray,
        input_units: List[List[int]],
        outputs: Optional[np.ndarray] = None,
        output_units: Optional[List[List[int]]] = None,
        eps: float = 1e-10,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute dimensionless Pi-groups via null space of the dimensional matrix.

        Args:
            inputs: (n_samples, n_inputs)
            input_units: List of 5D vectors [M, L, T, Q, Θ] for each input
            outputs: Optional (n_samples, n_outputs)
            output_units: Optional list of 5D vectors for each output
            eps: Small value to avoid 0^negative, and for null-space rank tolerance

        Returns:
            If outputs is None: pi_inputs (n_samples, n_pi_in)
            Else: (pi_inputs, pi_outputs)
        """
        inputs = np.asarray(inputs, dtype=np.float64)
        n_samples, n_in = inputs.shape
        if len(input_units) != n_in:
            raise ValueError(f"input_units length {len(input_units)} != n_inputs {n_in}")

        # Build dimensional matrix D: (5 base dims) x (n_vars)
        D_in = np.array(input_units, dtype=np.float64).T  # (5, n_in)
        if outputs is None:
            D = D_in
            n_out = 0
        else:
            outputs = np.asarray(outputs, dtype=np.float64)
            if output_units is None:
                output_units = [[0, 0, 0, 0, 0]] * outputs.shape[1]
            D_out = np.array(output_units, dtype=np.float64).T  # (5, n_out)
            D = np.hstack([D_in, D_out])
            n_out = outputs.shape[1]

        # Null space: columns are exponent vectors for dimensionless combinations
        # D @ v = 0  =>  prod(x_j^v_j) is dimensionless
        try:
            N = null_space(D, rcond=1e-10)
        except Exception:
            N = np.zeros((D.shape[1], 1))

        x_safe = np.abs(inputs) + eps

        if outputs is None or n_out == 0:
            if N.size == 0:
                return inputs
            n_pi = N.shape[1]
            pi_inputs = np.ones((n_samples, n_pi))
            for i in range(n_pi):
                for j in range(n_in):
                    pi_inputs[:, i] *= np.power(x_safe[:, j], N[j, i])
            return pi_inputs

        # Partition: π_in (input-only) vs π_out (involves output)
        n_pi = N.shape[1]
        pi_in_list = []
        pi_out_list = []
        y_safe = np.abs(outputs) + eps

        for i in range(n_pi):
            out_exponents = N[n_in:, i]
            if np.all(np.abs(out_exponents) < 1e-10):
                # Input-only group
                pi = np.ones(n_samples)
                for j in range(n_in):
                    pi *= np.power(x_safe[:, j], N[j, i])
                pi_in_list.append(pi)
            else:
                # Group involving output
                pi = np.ones(n_samples)
                for j in range(n_in):
                    pi *= np.power(x_safe[:, j], N[j, i])
                for j in range(n_out):
                    pi *= np.power(y_safe[:, j], N[n_in + j, i])
                pi_out_list.append(pi)

        pi_inputs = np.column_stack(pi_in_list) if pi_in_list else inputs
        pi_outputs = np.column_stack(pi_out_list) if pi_out_list else outputs
        return pi_inputs, pi_outputs

    def distill(
        self,
        rcln_layer: Any,
        data_buffer: Union[List[Tuple[np.ndarray, np.ndarray]], np.ndarray],
        input_units: Optional[List[List[int]]] = None,
        niterations: int = 20,
        use_symplectic_causal: bool = False,
        state_dim: Optional[int] = None,
        output_index: int = 0,
    ) -> Optional[str]:
        """
        Distill symbolic formula from RCLN soft shell data.
        Fit y_soft = f(π_in) where π_in are dimensionless groups.

        Args:
            rcln_layer: RCLNLayer (used to get y_soft via forward if data_buffer is x only)
            data_buffer: List of (x, y_soft) pairs, or (x_array) to compute y_soft from rcln
            input_units: Optional 5D units per input. If None, assume unitless.
            niterations: PySR iterations (if used)

        Returns:
            Best-fit symbolic expression string, or None.
        """
        if isinstance(data_buffer, np.ndarray):
            x_arr = np.asarray(data_buffer)
            if rcln_layer is None:
                return None
            # Need to get y_soft from rcln forward
            import torch
            x_t = torch.from_numpy(x_arr).float()
            with torch.no_grad():
                _ = rcln_layer(x_t)
            y_soft = getattr(rcln_layer, "_last_y_soft", None)
            if y_soft is None:
                return None
            y_arr = y_soft.cpu().numpy()
            if y_arr.ndim == 1:
                y_arr = y_arr.reshape(-1, 1)
        else:
            # data_buffer is [(x, y_soft), ...]
            pairs = list(data_buffer)
            if not pairs:
                return None
            x_arr = np.stack([np.asarray(p[0]).ravel() for p in pairs])
            y_list = [np.asarray(p[1]).ravel() for p in pairs]
            y_arr = np.stack(y_list)
            if y_arr.ndim == 1:
                y_arr = y_arr.reshape(-1, 1)

        n_samples, n_in = x_arr.shape
        n_out = y_arr.shape[1]

        if input_units is None:
            input_units = [[0, 0, 0, 0, 0]] * n_in  # Unitless
        if len(input_units) != n_in:
            input_units = input_units[:n_in]
            while len(input_units) < n_in:
                input_units.append([0, 0, 0, 0, 0])

        # Get dimensionless groups (skip for simple unitless case to avoid Pi transform issues)
        if all(u == [0, 0, 0, 0, 0] for u in input_units) and n_in <= 4:
            pi_in = x_arr
            pi_out = y_arr
        else:
            pi_result = self.get_dimensionless_groups(x_arr, input_units, y_arr, [[0, 0, 0, 0, 0]] * n_out)
            if isinstance(pi_result, tuple):
                pi_in, pi_out = pi_result
            else:
                pi_in = pi_result
                pi_out = y_arr
            if pi_in.shape[1] == 0 or np.allclose(pi_in, pi_in[0:1], atol=1e-12):
                pi_in = x_arr
        # Flatten pi_out for scalar regression (Lasso/PySR expect 1D target)
        if pi_out.ndim > 1:
            pi_out = pi_out[:, 0] if pi_out.shape[1] >= 1 else pi_out.ravel()

        # Causal constraint: allowed input indices (symplectic)
        allowed_indices = None
        if use_symplectic_causal and state_dim is not None:
            try:
                from axiom_os.core.symplectic_causal import get_symplectic_allowed_inputs
                allowed_indices = get_symplectic_allowed_inputs(output_index, state_dim)
            except ImportError:
                pass

        def _filter_terms(terms_dict: dict) -> list:
            """Build formula terms, respecting allowed_indices."""
            out = []
            for i, c in terms_dict.items():
                if allowed_indices is not None and i < n_in and i not in allowed_indices:
                    continue
                if abs(c) > 1e-6:
                    out.append(f"{c:.4f}*x{i}")
            return out

        # Symbolic regression: π_out = f(π_in)
        if self.use_pysr and HAS_PYSR:
            try:
                model = PySRRegressor(
                    niterations=niterations,
                    binary_operators=["+", "*", "-", "/"],
                    unary_operators=["square", "sqrt", "abs"],
                )
                model.fit(pi_in, pi_out)
                eq = model.sympy()
                return str(eq) if eq is not None else None
            except Exception:
                pass

        # Fallback: Lasso -> Ridge -> ElasticNet (robust to collinearity)
        if HAS_SKLEARN:
            for model in [
                Lasso(alpha=0.0001, max_iter=5000),
                Ridge(alpha=0.0001),
                ElasticNet(alpha=0.0001, max_iter=5000),
            ]:
                try:
                    model.fit(pi_in, pi_out)
                    coef = model.coef_
                    intercept = model.intercept_
                    terms_dict = {i: float(c) for i, c in enumerate(coef)}
                    terms = _filter_terms(terms_dict)
                    if abs(intercept) > 1e-6:
                        terms.append(f"{intercept:.4f}")
                    return " + ".join(terms) if terms else "0"
                except Exception:
                    continue

        # Numpy lstsq fallback (no sklearn)
        try:
            ones = np.ones((pi_in.shape[0], 1))
            X_aug = np.hstack([pi_in, ones])
            coef, _, _, _ = np.linalg.lstsq(X_aug, pi_out, rcond=None)
            terms_dict = {i: float(c) for i, c in enumerate(coef[:-1])}
            terms = _filter_terms(terms_dict)
            if abs(coef[-1]) > 1e-6:
                terms.append(f"{coef[-1]:.4f}")
            return " + ".join(terms) if terms else "0"
        except Exception:
            pass
        return None

    def discover_sindy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        degree: int = 2,
        var_names: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        SINDy-style discovery: sparse regression on polynomial library.
        Uses pysindy.PolynomialLibrary if available; else falls back to discover_multivariate.
        """
        if not HAS_PYSINDY:
            return self.discover_multivariate(X, y, var_names=var_names)[0]
        try:
            from pysindy.feature_library import PolynomialLibrary
            lib = PolynomialLibrary(degree=degree, include_bias=True)
            lib.fit(X)
            theta = lib.transform(X)
            if HAS_SKLEARN:
                model = Lasso(alpha=1e-4, max_iter=5000, fit_intercept=False)
                model.fit(theta, y)
                coef = model.coef_
            else:
                coef, _, _, _ = np.linalg.lstsq(theta, y, rcond=None)
                if coef.ndim > 1:
                    coef = coef.ravel()
            names = lib.get_feature_names()
            terms = [f"{coef[i]:.4g}*{names[i]}" for i in range(len(coef)) if abs(coef[i]) > 1e-6]
            return " + ".join(terms) if terms else "0"
        except Exception:
            return self.discover_multivariate(X, y, var_names=var_names)[0]

    def discover_gflownet(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int = 50,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        GFlowNet discovery: sample equations proportional to reward R=exp(-MSE).
        Returns diverse formula candidates.
        """
        from .gflownet import GFlowNetDiscovery
        gf = GFlowNetDiscovery(X, y)
        return gf.discover(n_samples=n_samples, top_k=top_k)

    def discover(
        self,
        x: np.ndarray,
        y: np.ndarray,
        operations: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Legacy: Attempt symbolic regression y ≈ f(x).
        Delegates to distill with a minimal data buffer.
        """
        return self.distill(
            rcln_layer=None,
            data_buffer=list(zip(x, y)),
            input_units=[[0, 0, 0, 0, 0]] * x.shape[1],
        )

    def discover_multivariate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        var_names: Optional[List[str]] = None,
        selector: str = "bic",
        sample_weight: Optional[np.ndarray] = None,
        standardize: bool = False,
        top_k: Optional[int] = None,
    ) -> Tuple[Optional[str], np.ndarray, Optional[dict]]:
        """
        Multivariate symbolic discovery: y = f(t, x, y, z) for wind/spatial fields.
        Tries linear, quadratic, and log-height forms; selects best by AIC/BIC.

        Args:
            X: (N, D) inputs
            y: (N,) target
            var_names: e.g. ["t","x","y","z"]
            selector: "aic" or "bic"
            standardize: If True, z-score X and y before fit. Coefficients become
                comparable; predictions are inverse-transformed to original scale.
            top_k: If set, collect top-k candidates by score and store in
                coefs["_candidates"] as [(formula, pred, coefs), ...].

        Returns:
            (formula_str, predictions, coefs_dict) or (None, zeros, None)
        """
        from .forms import MultivariatePolyForm, aic, bic

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, d = X.shape
        if n < 5:
            return None, np.zeros_like(y), None

        X_mean, X_std, y_mean, y_std = None, None, None, None
        if standardize:
            X_mean = X.mean(axis=0)
            X_std = X.std(axis=0)
            X_std = np.where(X_std < 1e-12, 1.0, X_std)
            X = (X - X_mean) / X_std
            y_mean = y.mean()
            y_std = y.std()
            y_std = y_std if y_std > 1e-12 else 1.0
            y = (y - y_mean) / y_std

        names = var_names if var_names and len(var_names) >= d else (
            ["t", "x", "y", "z"][:d] if d <= 4 else [f"x{i}" for i in range(d)]
        )
        score_fn = bic if selector == "bic" else aic

        configs = [
            {"degree": 1, "include_interactions": False, "include_log_z": False},
            {"degree": 2, "include_interactions": False, "include_log_z": False},
            {"degree": 2, "include_interactions": True, "include_log_z": False},
            {"degree": 2, "include_interactions": True, "include_log_z": True},
        ]

        best_formula: Optional[str] = None
        best_pred: np.ndarray = np.zeros_like(y)
        best_coefs: Optional[dict] = None
        best_score = np.inf
        candidates: List[Tuple[str, np.ndarray, dict, float]] = []

        for cfg in configs:
            try:
                form = MultivariatePolyForm(alpha=1e-3, **cfg)
                pred, coefs, formula = form.fit(X, y, var_names=names, sample_weight=sample_weight)
                if np.any(np.isnan(pred)):
                    continue
                mse = np.mean((pred - y) ** 2)
                k = len([v for kk, v in coefs.items() if kk != "form_type" and isinstance(v, (int, float)) and abs(v) > 1e-6])
                k = max(1, k)
                score = score_fn(n, mse, k)
                if top_k is not None:
                    candidates.append((formula, pred.copy(), dict(coefs), score))
                if score < best_score:
                    best_score = score
                    best_formula = formula
                    best_pred = pred
                    best_coefs = coefs
            except Exception:
                continue

        if best_formula is not None:
            if standardize and y_mean is not None and y_std is not None:
                best_pred = best_pred * y_std + y_mean
            if best_coefs is not None and standardize:
                best_coefs["_y_mean"] = float(y_mean) if y_mean is not None else 0
                best_coefs["_y_std"] = float(y_std) if y_std is not None else 1
            if top_k is not None and candidates:
                candidates.sort(key=lambda x: x[3])
                k = min(top_k, len(candidates))
                top_list: List[Tuple[str, np.ndarray, dict]] = []
                for formula, pred, coefs, _ in candidates[:k]:
                    if standardize and y_mean is not None and y_std is not None:
                        pred = pred * y_std + y_mean
                    top_list.append((formula, pred, coefs))
                if best_coefs is not None:
                    best_coefs["_candidates"] = top_list
            return best_formula, best_pred, best_coefs

        # Fallback: numpy lstsq (works without sklearn)
        try:
            ones = np.ones((n, 1))
            X_aug = np.hstack([X, ones])
            coef, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
            pred = X_aug @ coef
            terms = []
            for i, c in enumerate(coef[:-1]):
                if abs(c) > 1e-6:
                    terms.append(f"{c:.4g}*{names[i]}")
            if abs(coef[-1]) > 1e-6:
                terms.append(f"{coef[-1]:.4g}")
            formula = " + ".join(terms) if terms else "0"
            if standardize and y_mean is not None and y_std is not None:
                pred = pred * y_std + y_mean
            return formula, pred, {"form_type": "linear_fallback"}
        except Exception:
            pass
        return None, np.zeros_like(y), None

    def discover_parametric(
        self,
        t: np.ndarray,
        y: np.ndarray,
        form_candidates: Optional[List["FormCandidate"]] = None,
        selector: str = "aic",
        time_weighted: bool = False,
        balance_early_late: bool = False,
        derivative_constraint: bool = True,
        top_k: Optional[int] = None,
    ) -> Tuple[Optional[str], np.ndarray, Optional[dict]]:
        """
        Universal parametric discovery: fit all form candidates, select by AIC/BIC.
        No Lasso gate - always tries every form.

        Args:
            t: 1D input (e.g. normalized cycles)
            y: 1D target (e.g. soft shell output)
            form_candidates: List of FormCandidate instances. Default: battery aging forms.
            selector: "aic" or "bic"
            time_weighted: If True, weight early points higher (w = 1/(t+0.1))
            balance_early_late: If True, use 0.5*early_MSE + 0.5*late_MSE for selection
            derivative_constraint: If True, reject forms with df/dt >= 0 (capacity must decay)
            top_k: If set, store top-k candidates in coefs["_candidates"].

        Returns:
            (formula_str, predictions, coefs_dict) or (None, zeros, None) on failure.
        """
        from .forms import (
            FormCandidate,
            BATTERY_AGING_FORMS,
            aic,
            bic,
        )

        t = np.asarray(t, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        n = len(y)
        if n < 3:
            return None, np.zeros_like(y), None

        forms = form_candidates if form_candidates is not None else BATTERY_AGING_FORMS
        score_fn = bic if selector == "bic" else aic

        # Time-weighted: early points (small t) get higher weight
        sample_weight = None
        if time_weighted:
            sample_weight = 1.0 / (t + 0.1)
            sample_weight = sample_weight / sample_weight.mean()  # normalize

        # Split for early/late balance
        mid = np.median(t)
        early_mask = t < mid
        late_mask = ~early_mask
        n_early = max(1, early_mask.sum())
        n_late = max(1, late_mask.sum())

        best_formula: Optional[str] = None
        best_pred: np.ndarray = np.zeros_like(y)
        best_coefs: Optional[dict] = None
        best_score = np.inf
        candidates: List[Tuple[str, np.ndarray, dict, float]] = []

        for form in forms:
            try:
                pred, coefs, formula = form.fit(t, y, sample_weight=sample_weight)
                if np.any(np.isnan(pred)):
                    continue
                # Derivative constraint: df/dt < 0 (capacity must decay, not recover)
                if derivative_constraint:
                    sort_idx = np.argsort(t)
                    pred_sorted = pred[sort_idx]
                    if np.any(np.diff(pred_sorted) > 1e-6):
                        continue
                mse = np.mean((pred - y) ** 2)
                if balance_early_late:
                    early_mse = np.mean((pred[early_mask] - y[early_mask]) ** 2)
                    late_mse = np.mean((pred[late_mask] - y[late_mask]) ** 2)
                    mse = 0.5 * early_mse + 0.5 * late_mse
                k = form.n_params()
                score = score_fn(n, mse, k)
                if top_k is not None:
                    candidates.append((formula, pred.copy(), dict(coefs), score))
                if score < best_score:
                    best_score = score
                    best_formula = formula
                    best_pred = pred
                    best_coefs = coefs
            except Exception:
                continue

        if best_coefs is not None and top_k is not None and candidates:
            candidates.sort(key=lambda x: x[3])
            k = min(top_k, len(candidates))
            best_coefs["_candidates"] = [(f, p, c) for f, p, c, _ in candidates[:k]]
        return best_formula, best_pred, best_coefs
