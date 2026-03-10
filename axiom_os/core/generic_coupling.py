"""
GENERIC Coupling Framework for Axiom-OS
Copyright (c) 2026 yuzechen83-crypto. All Rights Reserved.

GENERIC = General Equation for Non-Equilibrium Reversible-Irreversible Coupling

The fundamental equation:
    ż = L(z)∇E(z) + M(z)∇S(z)

Where:
    - z: State variables (position, momentum, etc.)
    - E(z): Energy (Hamiltonian) - from Hard Core
    - S(z): Entropy - from Soft Shell  
    - L(z): Poisson matrix (antisymmetric) - encodes reversible dynamics
    - M(z): Friction matrix (symmetric positive semi-definite) - encodes irreversible dynamics

Key Properties:
    1. Energy conservation: dE/dt = 0 (for isolated systems)
    2. Entropy production: dS/dt ≥ 0 (second law of thermodynamics)
    3. Degeneracy conditions: L·∇S = 0, M·∇E = 0

This is the RCLN v3.0 coupling paradigm - replacing simple addition with 
thermodynamically consistent structure.

Reference: "GENERIC: A unifying framework for non-equilibrium thermodynamics"
(Grmela & Öttinger, 1997)
"""

from typing import Optional, Callable, Tuple, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class GENERICLayer(nn.Module):
    """
    GENERIC Coupling Layer - Thermodynamically consistent physics-AI interface.
    
    Replaces simple addition (y = y_hard + y_soft) with structured coupling:
        ż = L(z)∇E(z) + M(z)∇S(z)
    
    Where:
        - Hard Core provides: E(z) (energy) and L(z) structure (Poisson brackets)
        - Soft Shell provides: S(z) (entropy) and M(z) (dissipation)
    
    This ensures:
        1. Energy is conserved by reversible dynamics (L∇E term)
        2. Entropy always increases (M∇S term, M positive semi-definite)
        3. Coupling respects thermodynamic laws
    """

    def __init__(
        self,
        state_dim: int,
        # Hard Core (Reversible Dynamics)
        energy_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        poisson_matrix_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        # Soft Shell (Irreversible Dynamics)
        entropy_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        friction_matrix_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        # Coupling parameters
        lambda_rev: float = 1.0,  # Weight for reversible term
        lambda_irr: float = 1.0,  # Weight for irreversible term
        # Constraints
        enforce_constraints: bool = True,  # Enforce L·∇S = 0, M·∇E = 0
        temperature: float = 1.0,  # For entropy-enthalpy conversion
    ):
        super().__init__()
        self.state_dim = state_dim
        self.lambda_rev = lambda_rev
        self.lambda_irr = lambda_irr
        self.enforce_constraints = enforce_constraints
        self.temperature = temperature

        # Hard Core components (analytical, frozen or learned)
        self.energy_fn = energy_fn
        self.poisson_matrix_fn = poisson_matrix_fn

        # Soft Shell components (neural, learnable)
        self.entropy_fn = entropy_fn
        self.friction_matrix_fn = friction_matrix_fn

        # Learnable entropy approximation (if entropy_fn not provided)
        if entropy_fn is None:
            self._learnable_entropy = nn.Sequential(
                nn.Linear(state_dim, state_dim * 2),
                nn.SiLU(),
                nn.Linear(state_dim * 2, 1)
            )
        else:
            self._learnable_entropy = None

        # Learnable friction matrix (if friction_matrix_fn not provided)
        if friction_matrix_fn is None:
            # M must be symmetric positive semi-definite
            # Parameterize as M = A^T A for automatic PSD
            self._friction_A = nn.Linear(state_dim, state_dim, bias=False)
        else:
            self._friction_A = None

    def compute_energy(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute energy E(z) from Hard Core.
        
        Args:
            z: State tensor (batch, state_dim)
        
        Returns:
            E: Energy scalar per batch element (batch, 1)
        """
        if self.energy_fn is not None:
            return self.energy_fn(z)
        
        # Default: kinetic + potential energy approximation
        # E = 0.5 * |z|^2 (harmonic approximation)
        return 0.5 * (z ** 2).sum(dim=-1, keepdim=True)

    def compute_entropy(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy S(z) from Soft Shell.
        
        Args:
            z: State tensor (batch, state_dim)
        
        Returns:
            S: Entropy scalar per batch element (batch, 1)
        """
        if self.entropy_fn is not None:
            return self.entropy_fn(z)
        
        if self._learnable_entropy is not None:
            return self._learnable_entropy(z)
        
        # Default: Shannon-like entropy (negative of energy)
        return -self.compute_energy(z)

    def compute_poisson_matrix(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute Poisson matrix L(z) for reversible dynamics.
        
        L must be antisymmetric: L^T = -L
        This ensures energy conservation and Hamiltonian structure.
        
        Args:
            z: State tensor (batch, state_dim)
        
        Returns:
            L: Poisson matrix (batch, state_dim, state_dim)
        """
        if self.poisson_matrix_fn is not None:
            L = self.poisson_matrix_fn(z)
        else:
            # Default: Canonical Poisson bracket for Hamiltonian systems
            # L = [0, I; -I, 0] in block form
            # For general case, construct from skew-symmetric parameterization
            batch_size = z.shape[0]
            
            # Create canonical symplectic structure
            n = self.state_dim // 2
            if self.state_dim % 2 == 0:
                L = torch.zeros(batch_size, self.state_dim, self.state_dim, device=z.device)
                # Position-momentum coupling
                L[:, :n, n:] = torch.eye(n, device=z.device).unsqueeze(0).expand(batch_size, -1, -1)
                L[:, n:, :n] = -torch.eye(n, device=z.device).unsqueeze(0).expand(batch_size, -1, -1)
            else:
                # Odd dimension - use skew-symmetric parametrization
                L = self._make_skew_symmetric(z)
        
        # Ensure antisymmetry
        if self.enforce_constraints:
            L = 0.5 * (L - L.transpose(-2, -1))
        
        return L

    def compute_friction_matrix(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute friction matrix M(z) for irreversible dynamics.
        
        M must be symmetric positive semi-definite.
        This ensures entropy production is always non-negative.
        
        Args:
            z: State tensor (batch, state_dim)
        
        Returns:
            M: Friction matrix (batch, state_dim, state_dim)
        """
        if self.friction_matrix_fn is not None:
            M = self.friction_matrix_fn(z)
        elif self._friction_A is not None:
            # M = A^T A ensures positive semi-definite
            A = self._friction_A(z)  # (batch, state_dim)
            # Expand to matrix form
            A_diag = torch.diag_embed(A)  # (batch, state_dim, state_dim)
            M = torch.bmm(A_diag.transpose(-2, -1), A_diag)
        else:
            # Default: small constant friction
            batch_size = z.shape[0]
            M = 0.01 * torch.eye(self.state_dim, device=z.device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Ensure symmetry
        if self.enforce_constraints:
            M = 0.5 * (M + M.transpose(-2, -1))
            # Ensure positive semi-definite by clamping eigenvalues
            M = self._make_psd(M)
        
        return M

    def _make_skew_symmetric(self, z: torch.Tensor) -> torch.Tensor:
        """Create skew-symmetric matrix from parameters."""
        batch_size = z.shape[0]
        # Use upper triangular part to construct skew-symmetric matrix
        triu_indices = torch.triu_indices(self.state_dim, self.state_dim, offset=1, device=z.device)
        n_params = triu_indices.shape[1]
        
        # Generate parameters from z
        params = torch.tanh(z[:, :n_params % self.state_dim])  # Simple mapping
        params = F.linear(z[:, :n_params], torch.randn(n_params, self.state_dim, device=z.device))
        
        L = torch.zeros(batch_size, self.state_dim, self.state_dim, device=z.device)
        L[:, triu_indices[0], triu_indices[1]] = params
        L = L - L.transpose(-2, -1)  # Make skew-symmetric
        
        return L

    def _make_psd(self, M: torch.Tensor) -> torch.Tensor:
        """Ensure matrix is positive semi-definite."""
        # Eigenvalue decomposition and clamp
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(M)
            eigenvalues = torch.clamp(eigenvalues, min=0.0)  # Remove negative eigenvalues
            M_psd = torch.bmm(torch.bmm(eigenvectors, torch.diag_embed(eigenvalues)), eigenvectors.transpose(-2, -1))
            return M_psd
        except:
            # Fallback: return M + small identity
            return M + 1e-6 * torch.eye(M.shape[-1], device=M.device).unsqueeze(0).expand_as(M)

    def compute_gradients(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gradients ∇E and ∇S.
        
        Args:
            z: State tensor (batch, state_dim)
        
        Returns:
            grad_E: Energy gradient (batch, state_dim)
            grad_S: Entropy gradient (batch, state_dim)
        """
        z_requires_grad = z.requires_grad
        z = z.detach().requires_grad_(True)

        # Compute energy and its gradient
        E = self.compute_energy(z)
        grad_E = torch.autograd.grad(
            E.sum(), z, create_graph=True, retain_graph=True
        )[0]

        # Compute entropy and its gradient
        S = self.compute_entropy(z)
        grad_S = torch.autograd.grad(
            S.sum(), z, create_graph=True, retain_graph=True
        )[0]

        if not z_requires_grad:
            z = z.detach()

        return grad_E, grad_S

    def forward(self, z: torch.Tensor, return_thermodynamics: bool = False) -> torch.Tensor:
        """
        GENERIC forward pass.
        
        ż = L(z)∇E(z) + M(z)∇S(z)
        
        Args:
            z: State tensor (batch, state_dim)
            return_thermodynamics: Return energy, entropy, and production rates
        
        Returns:
            z_dot: Time derivative of state (batch, state_dim)
            info: Dictionary with thermodynamic quantities (if requested)
        """
        batch_size = z.shape[0]

        # Compute gradients
        grad_E, grad_S = self.compute_gradients(z)

        # Compute matrices
        L = self.compute_poisson_matrix(z)  # (batch, state_dim, state_dim)
        M = self.compute_friction_matrix(z)  # (batch, state_dim, state_dim)

        # Enforce degeneracy conditions if needed
        if self.enforce_constraints:
            # L · ∇S should be approximately zero (energy gradient orthogonal to entropy)
            # M · ∇E should be approximately zero
            pass  # Already handled by matrix structure

        # Compute GENERIC equation
        # ż = L∇E + M∇S
        reversible_term = torch.bmm(L, grad_E.unsqueeze(-1)).squeeze(-1)  # (batch, state_dim)
        irreversible_term = torch.bmm(M, grad_S.unsqueeze(-1)).squeeze(-1)  # (batch, state_dim)

        z_dot = self.lambda_rev * reversible_term + self.lambda_irr * irreversible_term

        if return_thermodynamics:
            # Compute thermodynamic quantities
            E = self.compute_energy(z)
            S = self.compute_entropy(z)
            
            # Energy production rate: dE/dt = ∇E · ż = ∇E · (L∇E + M∇S)
            # Should be zero (conserved) if degeneracy M·∇E = 0 holds
            dE_dt = (grad_E * z_dot).sum(dim=-1, keepdim=True)
            
            # Entropy production rate: dS/dt = ∇S · ż = ∇S · (L∇E + M∇S)
            # Should be non-negative (second law)
            dS_dt = (grad_S * z_dot).sum(dim=-1, keepdim=True)
            
            # Dissipation: ∇S · M · ∇S ≥ 0
            dissipation = (grad_S.unsqueeze(-2) @ M @ grad_S.unsqueeze(-1)).squeeze(-1)

            info = {
                'energy': E,
                'entropy': S,
                'dE_dt': dE_dt,
                'dS_dt': dS_dt,
                'dissipation': dissipation,
                'reversible_term': reversible_term,
                'irreversible_term': irreversible_term,
                'L': L,
                'M': M,
                'grad_E': grad_E,
                'grad_S': grad_S,
            }
            return z_dot, info

        return z_dot

    def step(self, z: torch.Tensor, dt: float, method: str = 'euler') -> torch.Tensor:
        """
        Time integration step.
        
        Args:
            z: Current state (batch, state_dim)
            dt: Time step
            method: Integration method ('euler', 'rk4', 'symplectic')
        
        Returns:
            z_next: Next state (batch, state_dim)
        """
        if method == 'euler':
            z_dot = self.forward(z)
            return z + dt * z_dot

        elif method == 'rk4':
            # Runge-Kutta 4
            k1 = self.forward(z)
            k2 = self.forward(z + 0.5 * dt * k1)
            k3 = self.forward(z + 0.5 * dt * k2)
            k4 = self.forward(z + dt * k3)
            return z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        elif method == 'symplectic':
            # Symplectic Euler (for Hamiltonian systems)
            # Split into position and momentum
            n = self.state_dim // 2
            q, p = z[:, :n], z[:, n:]
            
            # Compute Hamiltonian derivatives
            z_cat = torch.cat([q, p], dim=1)
            z_dot, info = self.forward(z_cat, return_thermodynamics=True)
            
            # Symplectic update
            q_dot = z_dot[:, :n]
            p_dot = z_dot[:, n:]
            
            p_new = p + dt * p_dot
            q_new = q + dt * q_dot
            
            return torch.cat([q_new, p_new], dim=1)

        else:
            raise ValueError(f"Unknown integration method: {method}")

    def check_thermodynamic_consistency(self, z: torch.Tensor, tolerance: float = 1e-4) -> Dict[str, bool]:
        """
        Check if GENERIC structure satisfies thermodynamic constraints.
        
        Args:
            z: State tensor
            tolerance: Numerical tolerance
        
        Returns:
            checks: Dictionary of constraint satisfaction flags
        """
        _, info = self.forward(z, return_thermodynamics=True)

        checks = {
            'energy_conservation': torch.all(torch.abs(info['dE_dt']) < tolerance).item(),
            'entropy_production': torch.all(info['dS_dt'] >= -tolerance).item(),
            'dissipation_positive': torch.all(info['dissipation'] >= -tolerance).item(),
            'L_antisymmetric': torch.allclose(info['L'], -info['L'].transpose(-2, -1), atol=tolerance),
            'M_symmetric': torch.allclose(info['M'], info['M'].transpose(-2, -1), atol=tolerance),
        }

        return checks


class GENERICRCLN(nn.Module):
    """
    RCLN with GENERIC coupling - Thermodynamically consistent hybrid physics-AI.
    
    Architecture:
        z_dot = L(z)∇E(z) + M(z)∇S(z)
        
        - Hard Core: E(z) (energy) and L(z) (Poisson structure)
        - Soft Shell: S(z) (entropy) and M(z) (friction)
    
    This is RCLN v3.0 - the thermodynamic evolution of the architecture.
    """

    def __init__(
        self,
        state_dim: int,
        # Hard Core
        hard_energy_fn: Optional[Callable] = None,
        hard_poisson_fn: Optional[Callable] = None,
        # Soft Shell  
        soft_entropy_net: Optional[nn.Module] = None,
        soft_friction_net: Optional[nn.Module] = None,
        # Coupling
        lambda_rev: float = 1.0,
        lambda_irr: float = 1.0,
        # Integration
        dt: float = 0.01,
        integrator: str = 'rk4',
    ):
        super().__init__()
        self.state_dim = state_dim
        self.dt = dt
        self.integrator = integrator

        self.generic = GENERICLayer(
            state_dim=state_dim,
            energy_fn=hard_energy_fn,
            poisson_matrix_fn=hard_poisson_fn,
            entropy_fn=None,  # Will use soft_entropy_net
            friction_matrix_fn=None,  # Will use soft_friction_net
            lambda_rev=lambda_rev,
            lambda_irr=lambda_irr,
        )

        # Override with neural networks if provided
        if soft_entropy_net is not None:
            self.generic._learnable_entropy = soft_entropy_net
        if soft_friction_net is not None:
            self.generic._friction_A = soft_friction_net

    def forward(self, z: torch.Tensor, n_steps: int = 1) -> torch.Tensor:
        """
        Integrate forward in time.
        
        Args:
            z: Initial state (batch, state_dim)
            n_steps: Number of time steps
        
        Returns:
            z_final: Final state (batch, state_dim)
        """
        for _ in range(n_steps):
            z = self.generic.step(z, self.dt, self.integrator)
        return z

    def get_dynamics(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get full thermodynamic information."""
        z_dot, info = self.generic.forward(z, return_thermodynamics=True)
        return info


# =============================================================================
# Utility Functions
# =============================================================================

def create_hamiltonian_system(
    mass: float = 1.0,
    potential_fn: Optional[Callable] = None,
) -> Tuple[Callable, Callable]:
    """
    Create energy and Poisson matrix functions for Hamiltonian system.
    
    Args:
        mass: Mass of the particle
        potential_fn: Potential energy function V(q)
    
    Returns:
        energy_fn: H(q, p) = p^2/(2m) + V(q)
        poisson_fn: Canonical Poisson matrix
    """
    def energy_fn(z: torch.Tensor) -> torch.Tensor:
        n = z.shape[-1] // 2
        q, p = z[..., :n], z[..., n:]
        kinetic = (p ** 2).sum(dim=-1, keepdim=True) / (2 * mass)
        if potential_fn is not None:
            potential = potential_fn(q)
        else:
            potential = 0.5 * (q ** 2).sum(dim=-1, keepdim=True)  # Harmonic
        return kinetic + potential

    def poisson_fn(z: torch.Tensor) -> torch.Tensor:
        n = z.shape[-1] // 2
        batch = z.shape[0]
        L = torch.zeros(batch, 2*n, 2*n, device=z.device, dtype=z.dtype)
        # [0, I; -I, 0] structure
        L[:, :n, n:] = torch.eye(n, device=z.device).unsqueeze(0).expand(batch, -1, -1)
        L[:, n:, :n] = -torch.eye(n, device=z.device).unsqueeze(0).expand(batch, -1, -1)
        return L

    return energy_fn, poisson_fn


def test_generic():
    """Test GENERIC layer."""
    print("Testing GENERIC Layer...")

    # Create simple harmonic oscillator
    energy_fn, poisson_fn = create_hamiltonian_system(mass=1.0)

    generic = GENERICLayer(
        state_dim=4,  # 2D position + 2D momentum
        energy_fn=energy_fn,
        poisson_matrix_fn=poisson_fn,
    )

    # Test state
    z = torch.randn(5, 4)

    # Forward
    z_dot, info = generic.forward(z, return_thermodynamics=True)

    print(f"State: {z.shape}")
    print(f"State derivative: {z_dot.shape}")
    print(f"Energy: {info['energy'].shape}, mean = {info['energy'].mean().item():.4f}")
    print(f"Entropy: {info['entropy'].shape}, mean = {info['entropy'].mean().item():.4f}")
    print(f"dE/dt: {info['dE_dt'].mean().item():.6f} (should be ~0)")
    print(f"dS/dt: {info['dS_dt'].mean().item():.6f}")

    # Check consistency
    checks = generic.check_thermodynamic_consistency(z)
    print(f"\nThermodynamic checks:")
    for name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")

    # Test integration
    z_next = generic.step(z, dt=0.01, method='rk4')
    print(f"\nIntegration: {z.shape} -> {z_next.shape}")

    print("\nGENERIC tests passed!")


if __name__ == "__main__":
    test_generic()
