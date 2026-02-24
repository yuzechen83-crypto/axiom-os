"""
Hippocampus - The Library
Stores crystallized physical laws (Symbolic Truth).
Dynamic Update: accepts new formulas from Discovery Engine.
Automate Crystallization: move discovered terms from Soft Shell to Hard Core.

Einstein Core: Helmholtz-Hodge decomposition to extract conservative (-∇H) part.
Inner Loop: Strictly symplectic validation & imagination.
"""

from typing import Dict, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
import re
import logging

import numpy as np

from .knowledge import SymplecticLaw

_log = logging.getLogger(__name__)

# Threshold for crystallizing a thought (extraction confidence)
CRYSTALLIZE_THRESHOLD = 0.5

# Optional torch for EinsteinCore
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class SymbolicExpression:
    """Symbolic formula: e.g., F = -k*x"""
    id: str
    formula: str
    units: str = ""
    domain: str = ""


class HippocampusLibrary:
    """
    Dictionary mapping Semantic_ID → Symbolic_Expression
    Dynamic Update: add_formula(), remove_formula()
    Symplectic Registry: Stores crystallized Hamiltonian laws (the "Golden" truths).
    """

    def __init__(self):
        self._library: Dict[str, SymbolicExpression] = {}
        self._eval_cache: Dict[str, Callable] = {}
        self.symplectic_registry: Dict[str, SymplecticLaw] = {}

    def register(self, semantic_id: str, formula: str, units: str = "", domain: str = "") -> None:
        """Add or update a symbolic law"""
        self._library[semantic_id] = SymbolicExpression(
            id=semantic_id,
            formula=formula,
            units=units,
            domain=domain,
        )
        self._eval_cache.pop(semantic_id, None)

    def get(self, semantic_id: str) -> Optional[SymbolicExpression]:
        return self._library.get(semantic_id)

    def remove(self, semantic_id: str) -> bool:
        if semantic_id in self._library:
            del self._library[semantic_id]
            self._eval_cache.pop(semantic_id, None)
            return True
        return False

    def list_ids(self) -> list:
        return list(self._library.keys())

    def crystallize(self, semantic_id: str, formula: str, units: str = "", domain: str = "") -> None:
        """Discovery Engine: crystallize new formula into library"""
        self.register(semantic_id, formula, units, domain)

    def __contains__(self, key: str) -> bool:
        return key in self._library

    def __len__(self) -> int:
        return len(self._library)

    def update_physics(self, term_type: str, coefficient: float) -> None:
        """
        Automate Crystallization: move discovered term from Soft Shell to Hard Core.
        If term_type == 'damping', update the hard-coded physics to include -coefficient * velocity.
        """
        if not hasattr(self, "_learned_terms"):
            self._learned_terms: Dict[str, float] = {}
        self._learned_terms[term_type] = coefficient
        if term_type == "damping":
            formula = f"F_damping = -{coefficient:.4g} * v"
            self.register("LearnedDamping", formula, "MLT^-2", "mechanics")
            _log.info("Hippocampus Updated: Learned Damping (c=%s)", coefficient)
            print(f"Hippocampus Updated: Learned Damping (c={coefficient})")

    def get_learned_term(self, term_type: str) -> Optional[float]:
        """Retrieve a crystallized coefficient by term type."""
        return getattr(self, "_learned_terms", {}).get(term_type)

    def crystallize_thought(
        self,
        einstein_model: "EinsteinCore",
        extraction_confidence: float = 1.0,
        name: Optional[str] = None,
        params: Optional[Dict[str, float]] = None,
        threshold: Optional[float] = None,
    ) -> Optional[SymplecticLaw]:
        """
        Memory Consolidation: Commit discovered Hamiltonian into the Hippocampus.
        The "Eye" (RCLN) sees the phenomenon. "Einstein" extracts the essence.
        The "Hippocampus" stores the Law, not the data.

        Input: einstein_model (EinsteinCore after decompose_dynamics).
        Validation: If extraction_confidence > threshold, proceed.
        Extraction: Store the HNN as hamiltonian_func.
        Registration: Create SymplecticLaw, save to symplectic_registry.
        """
        if not HAS_TORCH:
            _log.warning("Crystallize requires torch")
            return None
        thresh = threshold if threshold is not None else CRYSTALLIZE_THRESHOLD
        if extraction_confidence <= thresh:
            _log.warning("Crystallize rejected: confidence %.3f <= threshold %.3f", extraction_confidence, thresh)
            return None

        hnn = einstein_model.get_hnn()
        if hnn is None:
            _log.warning("Crystallize rejected: no HNN in einstein_model")
            return None

        device = getattr(einstein_model, "_device", None) or (torch.device("cpu") if HAS_TORCH else None)

        def H_func(qp):
            if HAS_TORCH and isinstance(qp, torch.Tensor):
                return hnn(qp.to(device))
            qp_t = torch.from_numpy(np.asarray(qp, dtype=np.float32)).float()
            if qp_t.dim() == 1:
                qp_t = qp_t.unsqueeze(0)
            qp_t = qp_t.to(device)
            with torch.no_grad():
                return hnn(qp_t)

        law_name = name or f"HarmonicOscillator_ID_{len(self.symplectic_registry):03d}"
        law = SymplecticLaw(
            name=law_name,
            hamiltonian_func=H_func,
            params=params or {},
            phase_space_dim=2,
            device=device,
        )
        self.symplectic_registry[law_name] = law
        _log.info("[Hippocampus] New Law Crystallized: %s (Energy Conserved)", law_name)
        print(f"[Hippocampus] New Law Crystallized: {law_name} (Energy Conserved)")
        return law

    def recall_symplectic_law(self, name: str) -> Optional[SymplecticLaw]:
        """Retrieve a crystallized SymplecticLaw by name."""
        return self.symplectic_registry.get(name)

    def list_symplectic_laws(self) -> list:
        """List names of all crystallized symplectic laws."""
        return list(self.symplectic_registry.keys())


# -----------------------------------------------------------------------------
# Einstein Core: Symplectic Validation & Imagination
# -----------------------------------------------------------------------------

if HAS_TORCH:

    class HamiltonianNN(nn.Module):
        """Small HNN: H(q, p) -> scalar. Outputs Hamiltonian energy."""

        def __init__(self, dim: int = 2, hidden_dim: int = 32):
            super().__init__()
            self.dim = dim
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
            for p in self.net.parameters():
                p.data.mul_(0.1)

        def forward(self, qp: torch.Tensor) -> torch.Tensor:
            return self.net(qp).squeeze(-1)

    class DissipationNet(nn.Module):
        """R(v): velocity-dependent dissipation. For 2D phase (q,p): outputs (0, R_p)."""

        def __init__(self, dim: int = 2, hidden_dim: int = 16):
            super().__init__()
            self.dim = dim
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
            for p in self.net.parameters():
                p.data.mul_(0.01)

        def forward(self, qp: torch.Tensor) -> torch.Tensor:
            r_p = self.net(qp).squeeze(-1)
            return torch.stack([torch.zeros_like(r_p), r_p], dim=-1)

    class EinsteinCore:
        """
        The Thinking Room: Extract conservative part via Helmholtz-Hodge.
        F = -∇H (Conservative) + Curl A (Solenoidal) + Dissipative
        Einstein focuses ONLY on the -∇H part (via J∇H in phase space).
        """

        def __init__(self, hnn_hidden: int = 32, r_hidden: int = 16, lr: float = 1e-2, epochs: int = 2000):
            self.hnn_hidden = hnn_hidden
            self.r_hidden = r_hidden
            self.lr = lr
            self.epochs = epochs
            self._hnn: Optional[HamiltonianNN] = None
            self._r_net: Optional[DissipationNet] = None
            self._device: Optional[torch.device] = None

        def _make_f_wrapper(self, model: nn.Module, state_dim: int = 2) -> Callable:
            """
            Wrap model so f(x,v) = (dx/dt, dv/dt).
            If model outputs acceleration only: f = (v, model(x,v)).
            If model outputs full dynamics: f = model(x,v).
            """
            def f(qp: torch.Tensor) -> torch.Tensor:
                out = model(qp)
                if out.shape[-1] == 1:
                    v = qp[..., 1:2]
                    return torch.cat([v, out], dim=-1)
                return out
            return f

        def decompose_dynamics(
            self,
            model: Union[nn.Module, Callable],
            data: Optional[torch.Tensor] = None,
            state_dim: int = 2,
            device: Optional[torch.device] = None,
        ) -> Callable[[torch.Tensor], torch.Tensor]:
            """
            Train HNN to approximate conservative part of f(x,v).
            Loss: ||f(x,v) - (J∇H + R(v))||²
            Output: H_func(q, p) -> scalar Hamiltonian
            """
            device = device or torch.device("cpu")
            self._device = device

            if callable(model) and not isinstance(model, nn.Module):
                f_dynamics = model
            else:
                f_dynamics = self._make_f_wrapper(model, state_dim)

            hnn = HamiltonianNN(dim=state_dim, hidden_dim=self.hnn_hidden).to(device)
            r_net = DissipationNet(dim=state_dim, hidden_dim=self.r_hidden).to(device)

            if data is None:
                qp = torch.randn(512, state_dim, device=device) * 2
            else:
                qp = data.to(device).float()
                if qp.shape[0] < 64:
                    qp = qp.repeat(8, 1)
                qp = qp.detach()
                noise = torch.randn_like(qp, device=device) * 0.1
                qp = torch.cat([qp, qp + noise], dim=0)

            qp = qp.detach().requires_grad_(True)
            optimizer = torch.optim.Adam(
                list(hnn.parameters()) + list(r_net.parameters()),
                lr=self.lr,
            )

            for step in range(self.epochs):
                optimizer.zero_grad()
                H = hnn(qp)
                dH = torch.autograd.grad(H.sum(), qp, create_graph=True)[0]
                dH_dq = dH[..., 0:1]
                dH_dp = dH[..., 1:2]
                J_grad_H = torch.cat([dH_dp, -dH_dq], dim=-1)
                R = r_net(qp)
                f_pred = J_grad_H + R
                with torch.no_grad():
                    f_true = f_dynamics(qp.detach())
                loss = torch.nn.functional.mse_loss(f_pred, f_true)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(hnn.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(r_net.parameters(), 1.0)
                optimizer.step()
                if (step + 1) % 500 == 0:
                    _log.info("EinsteinCore decompose step %d loss=%.6e", step + 1, loss.item())

            self._hnn = hnn
            self._r_net = r_net

            def H_func(qp: torch.Tensor) -> torch.Tensor:
                return hnn(qp.to(device))

            return H_func

        def dream_in_symplectic(
            self,
            H_func: Callable[[torch.Tensor], torch.Tensor],
            q0: np.ndarray,
            p0: np.ndarray,
            dt: float = 0.01,
            n_steps: int = 100_000,
            device: Optional[torch.device] = None,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
            """
            Run Leapfrog (strictly symplectic) on H.
            Returns: t, q_traj, p_traj, energy_drift_ok
            Criterion: If energy drifts significantly, law is flawed.
            """
            device = device or self._device or torch.device("cpu")
            q = np.array(q0, dtype=np.float64).ravel()
            p = np.array(p0, dtype=np.float64).ravel()
            state_dim = len(q)

            def grad_H(qp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                t = torch.from_numpy(qp.astype(np.float32)).unsqueeze(0).to(device).requires_grad_(True)
                H = H_func(t)
                H_val = H.sum()
                g = torch.autograd.grad(H_val, t)[0]
                g = g.cpu().numpy().ravel()
                return g[:state_dim], g[state_dim:]

            def energy(qp: np.ndarray) -> float:
                t = torch.from_numpy(qp.astype(np.float32)).unsqueeze(0).to(device)
                with torch.no_grad():
                    return float(H_func(t).item())

            t_arr = np.arange(n_steps + 1) * dt
            q_traj = np.zeros((n_steps + 1, state_dim))
            p_traj = np.zeros((n_steps + 1, state_dim))
            q_traj[0] = q
            p_traj[0] = p

            for i in range(n_steps):
                qp = np.concatenate([q, p])
                dH_dq, dH_dp = grad_H(qp)

                p_half = p - 0.5 * dt * dH_dq
                qp_mid = np.concatenate([q, p_half])
                _, dH_dp_mid = grad_H(qp_mid)

                q = q + dt * dH_dp_mid
                qp_end = np.concatenate([q, p_half])
                dH_dq_end, _ = grad_H(qp_end)

                p = p_half - 0.5 * dt * dH_dq_end

                q_traj[i + 1] = q
                p_traj[i + 1] = p

            E0 = energy(np.concatenate([q_traj[0], p_traj[0]]))
            E_end = energy(np.concatenate([q_traj[-1], p_traj[-1]]))
            drift = abs(E_end - E0)
            drift_ok = drift < 0.05 * (abs(E0) + 1e-8)

            return t_arr, q_traj, p_traj, drift_ok

        def get_hnn(self) -> Optional["HamiltonianNN"]:
            return self._hnn

        def get_dissipation(self) -> Optional["DissipationNet"]:
            return self._r_net


# Default laws
def init_default_library() -> HippocampusLibrary:
    lib = HippocampusLibrary()
    lib.register("HookesLaw", "F = -k * x", "MLT^-2", "mechanics")
    lib.register("Newton2", "F = m * a", "MLT^-2", "mechanics")
    lib.register("OhmsLaw", "V = I * R", "ML^2T^-2Q^-1", "electromagnetism")
    lib.register("PlasmaDensity", "n = n0 * exp(-phi/kT)", "L^-3", "plasma")
    lib.register("NavierStokes_Flux", "flux = -nu * grad(u)", "L^2T^-1", "fluid")
    return lib
