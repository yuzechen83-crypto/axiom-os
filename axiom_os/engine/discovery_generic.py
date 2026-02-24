"""
GENERIC Discovery - Extract E(z), S(z), L, M from Trained System
Workflow: Train GENERICSystem on trajectory -> Symbolic regression on E, S -> Inspect L, M
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch

from .discovery import DiscoveryEngine
from ..core.generic import GENERICSystem


def discover_potentials(
    model: GENERICSystem,
    z_data: np.ndarray,
    var_names: Optional[list] = None,
    engine: Optional[DiscoveryEngine] = None,
) -> Tuple[Optional[str], Optional[str], np.ndarray, np.ndarray]:
    """
    Fit symbolic formulas for E(z) and S(z) from trained model outputs.

    Args:
        model: Trained GENERICSystem
        z_data: (n_samples, state_dim) state trajectories
        var_names: e.g. ["x", "v"] or ["q1", "q2", "p1", "p2"]
        engine: DiscoveryEngine instance (creates one if None)

    Returns:
        (formula_E, formula_S, E_values, S_values)
    """
    z_t = torch.from_numpy(z_data).float()
    with torch.no_grad():
        E_vals = model.energy(z_t).numpy().ravel()
        S_vals = model.entropy(z_t).numpy().ravel()

    if engine is None:
        engine = DiscoveryEngine(use_pysr=False)

    state_dim = z_data.shape[1]
    if var_names is None:
        var_names = [f"z{i}" for i in range(state_dim)]

    formula_E = engine.discover(z_data, E_vals)
    formula_S = None
    if model.entropy_net is not None and np.any(np.abs(S_vals) > 1e-8):
        formula_S = engine.discover(z_data, S_vals)

    return formula_E, formula_S, E_vals, S_vals


def discover_generic(
    model: GENERICSystem,
    z_data: np.ndarray,
    var_names: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Full discovery: E formula, S formula, L matrix, M matrix.
    L and M are the learned structure (not symbolic, but interpretable).
    """
    formula_E, formula_S, E_vals, S_vals = discover_potentials(
        model, z_data, var_names=var_names
    )

    with torch.no_grad():
        z0 = torch.from_numpy(z_data[:1]).float()
        L = model.get_L(z0).numpy()
        M = model.get_M(z0).numpy()

    return {
        "formula_E": formula_E,
        "formula_S": formula_S,
        "E_values": E_vals,
        "S_values": S_vals,
        "L": L,
        "M": M,
        "has_friction": np.linalg.norm(M) > 1e-6,
    }
