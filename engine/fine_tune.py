"""
Fine-Tune Constants (L-BFGS) - High-precision optimizer for symbolic formulas.
Input: Symbolic string from Discovery Engine (e.g., "x / (1 - exp(-sqrt(x/a)))").
Output: Precise value of a_0 (or g0) minimizing RMSE on calibrated data.
"""

from typing import Optional, Dict, Any, Tuple, Union
import math
import re
import numpy as np
import torch
import torch.nn as nn


# Known RAR formula templates: param_name -> (formula_lambda, init_value, bounds)
RAR_TEMPLATES = {
    "a": ("x / (1 - torch.exp(-torch.sqrt(x / a)))", 3700.0, (100.0, 50000.0)),
    "g0": ("x / (1 - torch.exp(-torch.sqrt(x / g0)))", 3700.0, (100.0, 50000.0)),
    "a0": ("x / (1 - torch.exp(-torch.sqrt(x / a0)))", 3700.0, (100.0, 50000.0)),
}


def _extract_param_name(formula: str) -> Optional[str]:
    """Extract the constant parameter name (a, g0, a0) from formula."""
    formula_lower = formula.lower()
    for name in ["g0", "a0", "a"]:
        if re.search(rf"\b{name}\b", formula_lower):
            return name
    return None


def _clean_formula(formula: str) -> str:
    """Extract the core formula, strip 'y = ', 'g0=...', etc."""
    s = formula.strip()
    for prefix in ["y = ", "y=", "g_obs = ", "pred = "]:
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):].strip()
    if "," in s:
        s = s.split(",")[0].strip()
    return s


def _formula_to_module(
    formula: str,
    param_name: str = "a",
    init_value: float = 3700.0,
    bounds: Optional[Tuple[float, float]] = None,
) -> nn.Module:
    """
    Convert symbolic string to PyTorch Module with learnable parameter.
    formula: e.g. "x / (1 - exp(-sqrt(x/a)))"
    Maps: exp -> torch.exp, sqrt -> torch.sqrt, etc.
    """
    bounds = bounds or (100.0, 50000.0)
    # Normalize formula: use torch.* for math ops (avoid double-replacing torch.exp)
    s = formula.strip()
    if "torch." not in s:
        s = re.sub(r"\bexp\s*\(", "torch.exp(", s, flags=re.IGNORECASE)
        s = re.sub(r"\bsqrt\s*\(", "torch.sqrt(", s, flags=re.IGNORECASE)
    if "torch." not in s:
        s = re.sub(r"\blog\s*\(", "torch.log(", s, flags=re.IGNORECASE)
        s = re.sub(r"\bsin\s*\(", "torch.sin(", s, flags=re.IGNORECASE)
        s = re.sub(r"\bcos\s*\(", "torch.cos(", s, flags=re.IGNORECASE)
        s = re.sub(r"\btanh\s*\(", "torch.tanh(", s, flags=re.IGNORECASE)
        s = re.sub(r"\babs\s*\(", "torch.abs(", s, flags=re.IGNORECASE)
        s = re.sub(r"\bpow\s*\(", "torch.pow(", s, flags=re.IGNORECASE)
    # Ensure x is used (input variable)
    if "x" not in s and "x " not in s:
        s = s.replace("g_bar", "x").replace("gbar", "x")

    class FormulaModule(nn.Module):
        def __init__(self):
            super().__init__()
            val = float(np.clip(init_value, bounds[0], bounds[1]))
            self._param = nn.Parameter(torch.tensor(val, dtype=torch.float64))
            self._bounds = bounds
            self._formula_str = s
            self._param_name = param_name

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            a = self._param.clamp(self._bounds[0], self._bounds[1])
            eps = 1e-14
            x_safe = x.clamp(min=eps)
            try:
                result = eval(
                    self._formula_str,
                    {"torch": torch, "x": x_safe, self._param_name: a, "eps": eps},
                )
            except Exception:
                result = x_safe
            if isinstance(result, (int, float)):
                result = torch.full_like(x_safe, float(result))
            return result.to(x.dtype)

    return FormulaModule()


def fine_tune_constant(
    formula: str,
    g_bar: Union[np.ndarray, torch.Tensor],
    g_obs: Union[np.ndarray, torch.Tensor],
    param_name: Optional[str] = None,
    init_value: Optional[float] = None,
    bounds: Optional[Tuple[float, float]] = None,
    max_iter: int = 100,
    max_epochs: int = 20,
    tol: float = 1e-8,
) -> Dict[str, Any]:
    """
    Fine-tune the constant (a0/g0) in a symbolic formula using L-BFGS.
    Minimizes RMSE on (g_bar, g_obs).

    Args:
        formula: Symbolic string, e.g. "x / (1 - exp(-sqrt(x/a)))"
        g_bar: Input accelerations [(km/s)^2/kpc]
        g_obs: Observed accelerations (target)
        param_name: Override auto-detected param name (a, g0, a0)
        init_value: Override initial value
        bounds: (min, max) for the constant
        max_iter: L-BFGS line-search iterations per epoch
        max_epochs: Number of L-BFGS outer steps
        tol: Convergence tolerance

    Returns:
        dict with a0_precise, rmse_before, rmse_after, n_iters, success
    """
    g_bar = torch.as_tensor(g_bar, dtype=torch.float64)
    g_obs = torch.as_tensor(g_obs, dtype=torch.float64)
    if g_bar.dim() == 1:
        g_bar = g_bar.unsqueeze(-1)
    g_bar_flat = g_bar.ravel()
    g_obs_flat = g_obs.ravel()

    formula_clean = _clean_formula(formula)
    pname = param_name or _extract_param_name(formula_clean) or "a"
    init = init_value if init_value is not None else RAR_TEMPLATES.get(pname, (None, 3700.0, None))[1]
    bnds = bounds or RAR_TEMPLATES.get(pname, (None, None, (100.0, 50000.0)))[2]

    module = _formula_to_module(formula_clean, param_name=pname, init_value=init, bounds=bnds)
    opt = torch.optim.LBFGS(module.parameters(), lr=1.0, max_iter=max_iter, tolerance_grad=tol, tolerance_change=tol)

    # Before
    with torch.no_grad():
        pred_before = module(g_bar_flat)
        rmse_before = math.sqrt(float(((pred_before - g_obs_flat) ** 2).mean()))

    n_iters = 0
    final_loss = float("inf")
    prev_loss = float("inf")

    def closure():
        nonlocal n_iters
        opt.zero_grad()
        pred = module(g_bar_flat)
        loss = ((pred - g_obs_flat) ** 2).mean()
        loss.backward()
        n_iters += 1
        return loss

    try:
        for epoch in range(max_epochs):
            opt.step(closure)
            with torch.no_grad():
                pred = module(g_bar_flat)
                curr_loss = float(((pred - g_obs_flat) ** 2).mean())
            if abs(prev_loss - curr_loss) < tol:
                break
            prev_loss = curr_loss
        with torch.no_grad():
            pred_after = module(g_bar_flat)
            final_loss = float(((pred_after - g_obs_flat) ** 2).mean())
            rmse_after = math.sqrt(final_loss)
            a0_precise = float(module._param.clamp(bnds[0], bnds[1]).item())
    except Exception as e:
        a0_precise = init
        rmse_after = rmse_before
        return {
            "a0_precise": a0_precise,
            "rmse_before": rmse_before,
            "rmse_after": rmse_after,
            "n_iters": n_iters,
            "success": False,
            "error": str(e),
        }

    return {
        "a0_precise": a0_precise,
        "rmse_before": rmse_before,
        "rmse_after": rmse_after,
        "n_iters": n_iters,
        "success": True,
        "improvement": (rmse_before - rmse_after) / (rmse_before + 1e-14) * 100,
    }


def fine_tune_rar(
    g_bar: np.ndarray,
    g_obs: np.ndarray,
    g0_init: Optional[float] = None,
    max_iter: int = 100,
) -> Tuple[float, Dict[str, Any]]:
    """
    Convenience: Fine-tune RAR McGaugh formula.
    Returns: (a0_precise, info_dict)
    """
    formula = "x / (1 - torch.exp(-torch.sqrt(x / a)))"
    info = fine_tune_constant(
        formula,
        g_bar,
        g_obs,
        param_name="a",
        init_value=g0_init,
        bounds=(100.0, 50000.0),
        max_iter=max_iter,
    )
    return info["a0_precise"], info
