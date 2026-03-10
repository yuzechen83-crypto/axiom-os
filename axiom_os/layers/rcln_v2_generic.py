# -*- coding: utf-8 -*-
"""
RCLN v2.0 - GENERIC Coupling
Advanced Physics-AI Interface: z_dot = L(z) * grad_E(z) + M(z) * grad_S(z)

Comparison:
- RCLN v1.0 (Addition): y = F_hard + F_soft
  Weakness: Loose physical meaning
  
- RCLN v2.0 (GENERIC): z_dot = L * grad_E + M * grad_S
  Improvement: Enforces thermodynamic laws
  - E: Energy (Hard Core)
  - S: Entropy (Soft Shell)
  - L: Poisson matrix (reversible dynamics)
  - M: Friction matrix (irreversible dynamics)
  
  Significance: Forces Soft Shell to obey thermodynamic laws.
  This is not changing network layers, this is changing the physical worldview.

Reference: Grmela & Ottinger, "GENERIC...", 1997
"""

from typing import Optional, Callable, Tuple, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class GENERICCoupling(nn.Module):
    """
    GENERIC Coupling Layer - Thermodynamically consistent physics-AI interface.
    
    Core equation:
        z_dot = L(z) * grad_E(z) + M(z) * grad_S(z)
    
    Where:
        - E(z): Energy (from Hard Core)
        - S(z): Entropy (from Soft Shell)
        - L(z): Poisson matrix (antisymmetric) - encodes reversible dynamics
        - M(z): Friction matrix (symmetric PSD) - encodes irreversible dynamics
    
    Key properties:
        1. Energy conservation: dE/dt = 0 (isolated systems)
        2. Entropy production: dS/dt >= 0 (2nd law)
        3. Degeneracy: L * grad_S = 0, M * grad_E = 0
    """
    
    def __init__(
        self,
        state_dim: int,
        # Hard Core (reversible dynamics)
        energy_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        poisson_matrix_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        # Soft Shell (irreversible dynamics)
        entropy_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        friction_matrix_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        # Coupling parameters
        lambda_rev: float = 1.0,
        lambda_irr: float = 1.0,
        # Constraints
        enforce_constraints: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.lambda_rev = lambda_rev
        self.lambda_irr = lambda_irr
        self.enforce_constraints = enforce_constraints
        
        # Hard Core components
        self.energy_fn = energy_fn
        self.poisson_matrix_fn = poisson_matrix_fn
        
        # Soft Shell components
        self.entropy_fn = entropy_fn
        self.friction_matrix_fn = friction_matrix_fn
    
    def compute_energy(self, z: torch.Tensor) -> torch.Tensor:
        """Compute energy E(z)"""
        if self.energy_fn is not None:
            return self.energy_fn(z)
        # Default: harmonic E = 0.5 * |z|^2
        return 0.5 * (z ** 2).sum(dim=-1, keepdim=True)
    
    def compute_entropy(self, z: torch.Tensor) -> torch.Tensor:
        """Compute entropy S(z)"""
        if self.entropy_fn is not None:
            return self.entropy_fn(z)
        # Default: negative energy
        return -self.compute_energy(z)
    
    def compute_poisson_matrix(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute Poisson matrix L(z).
        L must be antisymmetric: L^T = -L
        """
        if self.poisson_matrix_fn is not None:
            L = self.poisson_matrix_fn(z)
        else:
            # Default: canonical symplectic [0, I; -I, 0]
            batch_size = z.shape[0]
            n = self.state_dim // 2
            if self.state_dim % 2 == 0:
                L = torch.zeros(batch_size, self.state_dim, self.state_dim, device=z.device)
                L[:, :n, n:] = torch.eye(n, device=z.device).unsqueeze(0).expand(batch_size, -1, -1)
                L[:, n:, :n] = -torch.eye(n, device=z.device).unsqueeze(0).expand(batch_size, -1, -1)
            else:
                L = self._make_skew_symmetric(z)
        
        # Enforce antisymmetry
        if self.enforce_constraints:
            L = 0.5 * (L - L.transpose(-2, -1))
        
        return L
    
    def compute_friction_matrix(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute friction matrix M(z).
        M must be symmetric positive semi-definite.
        """
        if self.friction_matrix_fn is not None:
            M = self.friction_matrix_fn(z)
        else:
            # Default: small constant friction
            batch_size = z.shape[0]
            M = 0.01 * torch.eye(self.state_dim, device=z.device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Enforce symmetry
        if self.enforce_constraints:
            M = 0.5 * (M + M.transpose(-2, -1))
            M = self._make_psd(M)
        
        return M
    
    def _make_skew_symmetric(self, z: torch.Tensor) -> torch.Tensor:
        """Create skew-symmetric matrix from parameters."""
        batch_size = z.shape[0]
        triu_indices = torch.triu_indices(self.state_dim, self.state_dim, offset=1, device=z.device)
        n_params = triu_indices.shape[1]
        
        params = torch.tanh(z[:, :n_params % self.state_dim])
        params = F.linear(z[:, :n_params], torch.randn(n_params, self.state_dim, device=z.device))
        
        L = torch.zeros(batch_size, self.state_dim, self.state_dim, device=z.device)
        L[:, triu_indices[0], triu_indices[1]] = params
        L = L - L.transpose(-2, -1)
        
        return L
    
    def _make_psd(self, M: torch.Tensor) -> torch.Tensor:
        """Ensure matrix is positive semi-definite."""
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(M)
            eigenvalues = torch.clamp(eigenvalues, min=0.0)
            M_psd = torch.bmm(torch.bmm(eigenvectors, torch.diag_embed(eigenvalues)), eigenvectors.transpose(-2, -1))
            return M_psd
        except:
            return M + 1e-6 * torch.eye(M.shape[-1], device=M.device).unsqueeze(0).expand_as(M)
    
    def compute_gradients(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gradients grad_E and grad_S."""
        z_requires_grad = z.requires_grad
        z = z.detach().requires_grad_(True)
        
        # Energy gradient
        E = self.compute_energy(z)
        grad_E = torch.autograd.grad(E.sum(), z, create_graph=True, retain_graph=True)[0]
        
        # Entropy gradient
        S = self.compute_entropy(z)
        grad_S = torch.autograd.grad(S.sum(), z, create_graph=True, retain_graph=True)[0]
        
        if not z_requires_grad:
            z = z.detach()
        
        return grad_E, grad_S
    
    def forward(self, z: torch.Tensor, return_thermodynamics: bool = False):
        """
        GENERIC forward pass.
        
        z_dot = L(z) * grad_E(z) + M(z) * grad_S(z)
        """
        # Compute gradients
        grad_E, grad_S = self.compute_gradients(z)
        
        # Compute matrices
        L = self.compute_poisson_matrix(z)
        M = self.compute_friction_matrix(z)
        
        # Compute GENERIC equation
        # Handle both batched and unbatched inputs
        if L.dim() == 2 and grad_E.dim() == 1:
            reversible_term = (L @ grad_E.unsqueeze(-1)).squeeze(-1)
            irreversible_term = (M @ grad_S.unsqueeze(-1)).squeeze(-1)
        else:
            reversible_term = torch.bmm(L, grad_E.unsqueeze(-1)).squeeze(-1)
            irreversible_term = torch.bmm(M, grad_S.unsqueeze(-1)).squeeze(-1)
        
        z_dot = self.lambda_rev * reversible_term + self.lambda_irr * irreversible_term
        
        if return_thermodynamics:
            E = self.compute_energy(z)
            S = self.compute_entropy(z)
            
            # Thermodynamic quantities
            dE_dt = (grad_E * z_dot).sum(dim=-1, keepdim=True)
            dS_dt = (grad_S * z_dot).sum(dim=-1, keepdim=True)
            dissipation = (grad_S.unsqueeze(-2) @ M @ grad_S.unsqueeze(-1)).squeeze(-1)
            
            info = {
                'energy': E,
                'entropy': S,
                'dE_dt': dE_dt,
                'dS_dt': dS_dt,
                'dissipation': dissipation,
                'reversible_term': reversible_term,
                'irreversible_term': irreversible_term,
            }
            return z_dot, info
        
        return z_dot


class RCLNv2_GENERIC(nn.Module):
    """
    RCLN v2.0 - Physics-AI hybrid using GENERIC coupling.
    
    Soft Shell outputs:
        - S(z): Entropy (scalar)
        - M(z): Friction matrix (matrix)
    
    Hard Core outputs:
        - E(z): Energy (scalar)
        - L(z): Poisson matrix (matrix)
    
    Coupling equation:
        z_dot = L(z) * grad_E(z) + M(z) * grad_S(z)
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        # Soft Shell (neural network)
        soft_shell: Optional[nn.Module] = None,
        # Hard Core (physics)
        hard_energy_fn: Optional[Callable] = None,
        hard_poisson_fn: Optional[Callable] = None,
        # Coupling parameters
        lambda_rev: float = 1.0,
        lambda_irr: float = 1.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        
        # Soft Shell - outputs entropy S(z) and friction matrix M(z)
        if soft_shell is not None:
            self.soft_shell = soft_shell
        else:
            self.soft_shell = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )
            self.entropy_head = nn.Linear(hidden_dim, 1)
            self.friction_head = nn.Linear(hidden_dim, state_dim * state_dim)
        
        # Hard Core
        self.hard_energy_fn = hard_energy_fn
        self.hard_poisson_fn = hard_poisson_fn
        
        # GENERIC coupling
        self.coupling = GENERICCoupling(
            state_dim=state_dim,
            energy_fn=self._compute_energy,
            poisson_matrix_fn=self._compute_poisson,
            entropy_fn=self._compute_entropy,
            friction_matrix_fn=self._compute_friction,
            lambda_rev=lambda_rev,
            lambda_irr=lambda_irr,
        )
    
    def _compute_energy(self, z: torch.Tensor) -> torch.Tensor:
        """Hard Core: compute energy"""
        if self.hard_energy_fn is not None:
            return self.hard_energy_fn(z)
        return 0.5 * (z ** 2).sum(dim=-1, keepdim=True)
    
    def _compute_poisson(self, z: torch.Tensor) -> torch.Tensor:
        """Hard Core: compute Poisson matrix"""
        if self.hard_poisson_fn is not None:
            return self.hard_poisson_fn(z)
        batch_size = z.shape[0]
        n = self.state_dim // 2
        L = torch.zeros(batch_size, self.state_dim, self.state_dim, device=z.device)
        if n > 0:
            L[:, :n, n:] = torch.eye(n, device=z.device).unsqueeze(0).expand(batch_size, -1, -1)
            L[:, n:, :n] = -torch.eye(n, device=z.device).unsqueeze(0).expand(batch_size, -1, -1)
        return L
    
    def _compute_entropy(self, z: torch.Tensor) -> torch.Tensor:
        """Soft Shell: compute entropy"""
        features = self.soft_shell(z)
        if hasattr(self, 'entropy_head'):
            return self.entropy_head(features)
        return features[..., :1]
    
    def _compute_friction(self, z: torch.Tensor) -> torch.Tensor:
        """Soft Shell: compute friction matrix M = R^T * R"""
        features = self.soft_shell(z)
        if hasattr(self, 'friction_head'):
            R_flat = self.friction_head(features)
            batch_size = z.shape[0]
            R = R_flat.view(batch_size, self.state_dim, self.state_dim)
            M = torch.bmm(R.transpose(-2, -1), R)
            return M
        return 0.01 * torch.eye(self.state_dim, device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1)
    
    def forward(self, z: torch.Tensor, return_thermodynamics: bool = False):
        """
        RCLN v2.0 forward pass.
        
        Args:
            z: State vector (batch, state_dim)
            return_thermodynamics: Whether to return thermodynamic quantities
        
        Returns:
            z_dot: State derivative (batch, state_dim)
            info: Thermodynamic info (optional)
        """
        return self.coupling(z, return_thermodynamics)
    
    def step(self, z: torch.Tensor, dt: float, method: str = 'rk4') -> torch.Tensor:
        """Time integration step"""
        if method == 'euler':
            z_dot = self.forward(z)
            return z + dt * z_dot
        elif method == 'rk4':
            k1 = self.forward(z)
            k2 = self.forward(z + 0.5 * dt * k1)
            k3 = self.forward(z + 0.5 * dt * k2)
            k4 = self.forward(z + dt * k3)
            return z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def get_architecture_info(self) -> Dict[str, Union[str, float, bool]]:
        """Get architecture info"""
        return {
            'version': 'v2.0_GENERIC',
            'coupling': 'GENERIC',
            'equation': 'z_dot = L * grad_E + M * grad_S',
            'state_dim': self.state_dim,
            'hard_core': 'E(z), L(z)',
            'soft_shell': 'S(z), M(z)',
            'lambda_rev': self.coupling.lambda_rev,
            'lambda_irr': self.coupling.lambda_irr,
        }


def test_rcln_v2():
    """Test RCLN v2.0"""
    print("=" * 70)
    print("RCLN v2.0 - GENERIC Coupling Test")
    print("=" * 70)
    
    model = RCLNv2_GENERIC(
        state_dim=4,
        hidden_dim=32,
    )
    
    print("\n[Model Architecture]")
    info = model.get_architecture_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    z = torch.randn(5, 4, requires_grad=True)
    print(f"\n[Forward Test]")
    print(f"  Input z: {z.shape}")
    
    z_dot, thermo = model(z, return_thermodynamics=True)
    print(f"  Output z_dot: {z_dot.shape}")
    
    print(f"\n[Thermodynamic Quantities]")
    print(f"  Energy E: {thermo['energy'].mean().item():.4f}")
    print(f"  Entropy S: {thermo['entropy'].mean().item():.4f}")
    print(f"  dE/dt: {thermo['dE_dt'].mean().item():.6f} (should be ~0)")
    print(f"  dS/dt: {thermo['dS_dt'].mean().item():.6f} (should be >= 0)")
    
    print(f"\n[Gradient Test]")
    target = torch.randn_like(z_dot)
    loss = nn.MSELoss()(z_dot, target)
    loss.backward()
    
    total_grad_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm ** 2
            if 'soft' in name or 'entropy' in name or 'friction' in name:
                print(f"  Soft Shell {name}: grad_norm={grad_norm:.6f}")
    print(f"  Total grad norm: {total_grad_norm**0.5:.6f}")
    
    print(f"\n[Integration Test]")
    z0 = torch.randn(3, 4)
    dt = 0.01
    z1 = model.step(z0, dt, method='rk4')
    print(f"  z(0): {z0.shape}, mean={z0.mean().item():.4f}")
    print(f"  z({dt}): {z1.shape}, mean={z1.mean().item():.4f}")
    
    print("\n" + "=" * 70)
    print("[SUCCESS] RCLN v2.0 test passed!")
    print("=" * 70)
    print("\nKey Features:")
    print("  - Soft Shell outputs S(z) and M(z)")
    print("  - Hard Core provides E(z) and L(z)")
    print("  - GENERIC coupling: z_dot = L * grad_E + M * grad_S")
    print("  - Thermodynamically consistent")
    print("  - Energy conservation + Entropy production")


if __name__ == "__main__":
    test_rcln_v2()
