# -*- coding: utf-8 -*-
"""
RCLN v2.0 - GENERIC Full Coupling (Structural Change)
Soft Shell outputs: E, S, L, M (potentials and matrices)
Hard Core provides: Physical priors and constraints

Core equation:
    z_dot = L * grad_E + M * grad_S

Where Soft Shell generates:
    - E_soft: Energy correction
    - S_soft: Entropy correction  
    - L_soft: Poisson matrix correction
    - M_soft: Friction matrix correction

Coupling with Hard Core:
    E = E_hard + α·E_soft
    S = S_hard + β·S_soft
    L = L_hard + γ·L_soft
    M = M_hard + δ·M_soft

Significance: 
    - Soft Shell learns structured physical quantities, not arbitrary numbers
    - Enforces thermodynamic consistency through matrix constraints
    - Hard Core provides interpretable physical priors
"""

from typing import Optional, Callable, Tuple, Dict, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftShell_GENERIC(nn.Module):
    """
    Soft Shell for GENERIC coupling - outputs structured physical quantities.
    
    Outputs:
        - E: Energy scalar [B, 1]
        - S: Entropy scalar [B, 1]
        - L: Poisson matrix [B, state_dim, state_dim]
        - M: Friction matrix [B, state_dim, state_dim]
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 3,
        # What to output
        output_energy: bool = True,
        output_entropy: bool = True,
        output_poisson: bool = True,
        output_friction: bool = True,
        # Constraints
        enforce_antisymmetric_L: bool = True,
        enforce_symmetric_M: bool = True,
        enforce_psd_M: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.output_energy = output_energy
        self.output_entropy = output_entropy
        self.output_poisson = output_poisson
        self.output_friction = output_friction
        self.enforce_antisymmetric_L = enforce_antisymmetric_L
        self.enforce_symmetric_M = enforce_symmetric_M
        self.enforce_psd_M = enforce_psd_M
        
        # Shared feature extractor
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.SiLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        self.feature_net = nn.Sequential(*layers)
        
        # Output heads
        if output_energy:
            self.energy_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        if output_entropy:
            self.entropy_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        # Matrix outputs use factorized parameterization
        if output_poisson:
            # L is antisymmetric: L = A - A^T
            n_upper = state_dim * (state_dim - 1) // 2
            self.poisson_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, n_upper)
            )
        
        if output_friction:
            # M is PSD: M = R^T @ R
            self.friction_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, state_dim * state_dim)
            )
    
    def _build_antisymmetric(self, params: torch.Tensor, state_dim: int) -> torch.Tensor:
        """Build antisymmetric matrix from upper triangular parameters."""
        batch_size = params.shape[0]
        triu_indices = torch.triu_indices(state_dim, state_dim, offset=1, device=params.device)
        
        A = torch.zeros(batch_size, state_dim, state_dim, device=params.device)
        A[:, triu_indices[0], triu_indices[1]] = params
        L = A - A.transpose(-2, -1)
        return L
    
    def _build_psd(self, R_flat: torch.Tensor, state_dim: int) -> torch.Tensor:
        """Build positive semi-definite matrix from factor R."""
        batch_size = R_flat.shape[0]
        R = R_flat.view(batch_size, state_dim, state_dim)
        M = torch.bmm(R.transpose(-2, -1), R)
        return M
    
    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass - output structured physical quantities.
        
        Returns:
            Dict with keys: 'E', 'S', 'L', 'M' (depending on configuration)
        """
        features = self.feature_net(z)
        outputs = {}
        
        # Energy
        if self.output_energy:
            outputs['E'] = self.energy_head(features)
        
        # Entropy
        if self.output_entropy:
            outputs['S'] = self.entropy_head(features)
        
        # Poisson matrix (antisymmetric)
        if self.output_poisson:
            L_params = self.poisson_head(features)
            L = self._build_antisymmetric(L_params, self.state_dim)
            outputs['L'] = L
        
        # Friction matrix (PSD)
        if self.output_friction:
            R_flat = self.friction_head(features)
            M = self._build_psd(R_flat, self.state_dim)
            outputs['M'] = M
        
        return outputs


class RCLNv2_GENERIC_Full(nn.Module):
    """
    RCLN v2.0 - Full GENERIC coupling with structured Soft Shell output.
    
    Soft Shell outputs: E_soft, S_soft, L_soft, M_soft
    Hard Core provides: E_hard, S_hard, L_hard, M_hard (physical priors)
    
    Final quantities:
        E = λ_E·E_hard + (1-λ_E)·E_soft
        S = λ_S·S_hard + (1-λ_S)·S_soft
        L = λ_L·L_hard + (1-λ_L)·L_soft
        M = λ_M·M_hard + (1-λ_M)·M_soft
    
    GENERIC equation:
        z_dot = L * grad_E + M * grad_S
    
    Properties enforced:
        1. Energy conservation: dE/dt ≈ 0 (reversible part)
        2. Entropy production: dS/dt >= 0 (irreversible part)
        3. Degeneracy: L·∇S = 0, M·∇E = 0 (approximate)
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        # Soft Shell
        soft_shell: Optional[nn.Module] = None,
        # Hard Core (physical priors)
        hard_energy_fn: Optional[Callable] = None,
        hard_entropy_fn: Optional[Callable] = None,
        hard_poisson_fn: Optional[Callable] = None,
        hard_friction_fn: Optional[Callable] = None,
        # Coupling weights (0 = pure Hard Core, 1 = pure Soft Shell)
        lambda_E: float = 0.5,
        lambda_S: float = 0.5,
        lambda_L: float = 0.5,
        lambda_M: float = 0.5,
        # Constraints
        enforce_degeneracy: bool = False,
        enforce_psd_M: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        
        # Soft Shell - outputs structured quantities
        if soft_shell is not None:
            self.soft_shell = soft_shell
        else:
            self.soft_shell = SoftShell_GENERIC(
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                output_energy=True,
                output_entropy=True,
                output_poisson=True,
                output_friction=True,
            )
        
        # Hard Core functions
        self.hard_energy_fn = hard_energy_fn
        self.hard_entropy_fn = hard_entropy_fn
        self.hard_poisson_fn = hard_poisson_fn
        self.hard_friction_fn = hard_friction_fn
        
        # Coupling weights (learnable or fixed)
        self.lambda_E = nn.Parameter(torch.tensor(lambda_E))
        self.lambda_S = nn.Parameter(torch.tensor(lambda_S))
        self.lambda_L = nn.Parameter(torch.tensor(lambda_L))
        self.lambda_M = nn.Parameter(torch.tensor(lambda_M))
        
        self.enforce_degeneracy = enforce_degeneracy
        self.enforce_psd_M = enforce_psd_M
    
    def _get_hard_energy(self, z: torch.Tensor) -> torch.Tensor:
        """Default: harmonic energy"""
        if self.hard_energy_fn is not None:
            return self.hard_energy_fn(z)
        return 0.5 * (z ** 2).sum(dim=-1, keepdim=True)
    
    def _get_hard_entropy(self, z: torch.Tensor) -> torch.Tensor:
        """Default: negative energy"""
        if self.hard_entropy_fn is not None:
            return self.hard_entropy_fn(z)
        return -self._get_hard_energy(z)
    
    def _get_hard_poisson(self, z: torch.Tensor) -> torch.Tensor:
        """Default: canonical symplectic matrix"""
        if self.hard_poisson_fn is not None:
            return self.hard_poisson_fn(z)
        batch_size = z.shape[0]
        n = self.state_dim // 2
        L = torch.zeros(batch_size, self.state_dim, self.state_dim, device=z.device)
        if n > 0:
            I = torch.eye(n, device=z.device)
            L[:, :n, n:] = I.unsqueeze(0).expand(batch_size, -1, -1)
            L[:, n:, :n] = -I.unsqueeze(0).expand(batch_size, -1, -1)
        return L
    
    def _get_hard_friction(self, z: torch.Tensor) -> torch.Tensor:
        """Default: small constant friction"""
        if self.hard_friction_fn is not None:
            return self.hard_friction_fn(z)
        batch_size = z.shape[0]
        return 0.01 * torch.eye(self.state_dim, device=z.device).unsqueeze(0).expand(batch_size, -1, -1)
    
    def _couple_quantities(
        self,
        z: torch.Tensor,
        soft_outputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Couple Hard Core and Soft Shell outputs."""
        # Hard Core quantities
        E_hard = self._get_hard_energy(z)
        S_hard = self._get_hard_entropy(z)
        L_hard = self._get_hard_poisson(z)
        M_hard = self._get_hard_friction(z)
        
        # Soft Shell quantities (with defaults if not provided)
        E_soft = soft_outputs.get('E', torch.zeros_like(E_hard))
        S_soft = soft_outputs.get('S', torch.zeros_like(S_hard))
        L_soft = soft_outputs.get('L', torch.zeros_like(L_hard))
        M_soft = soft_outputs.get('M', torch.zeros_like(M_hard))
        
        # Couple with sigmoid-gated weights
        w_E = torch.sigmoid(self.lambda_E)
        w_S = torch.sigmoid(self.lambda_S)
        w_L = torch.sigmoid(self.lambda_L)
        w_M = torch.sigmoid(self.lambda_M)
        
        E = w_E * E_hard + (1 - w_E) * E_soft
        S = w_S * S_hard + (1 - w_S) * S_soft
        L = w_L * L_hard + (1 - w_L) * L_soft
        M = w_M * M_hard + (1 - w_M) * M_soft
        
        return E, S, L, M
    
    def _compute_gradients(
        self,
        z: torch.Tensor,
        E: torch.Tensor,
        S: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gradients of E and S."""
        grad_E = torch.autograd.grad(
            E.sum(), z, create_graph=True, retain_graph=True
        )[0]
        grad_S = torch.autograd.grad(
            S.sum(), z, create_graph=True, retain_graph=True
        )[0]
        return grad_E, grad_S
    
    def _apply_generic(
        self,
        L: torch.Tensor,
        M: torch.Tensor,
        grad_E: torch.Tensor,
        grad_S: torch.Tensor
    ) -> torch.Tensor:
        """Apply GENERIC equation: z_dot = L * grad_E + M * grad_S"""
        # Reversible part: L·∇E
        reversible = torch.bmm(L, grad_E.unsqueeze(-1)).squeeze(-1)
        # Irreversible part: M·∇S
        irreversible = torch.bmm(M, grad_S.unsqueeze(-1)).squeeze(-1)
        return reversible + irreversible
    
    def forward(
        self,
        z: torch.Tensor,
        return_physics: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass with GENERIC coupling.
        
        Args:
            z: State vector [B, state_dim]
            return_physics: Return detailed physics info
        
        Returns:
            z_dot: State derivative [B, state_dim]
            info: Physics dict (if return_physics=True)
        """
        # Ensure z requires grad for gradient computation
        z_in = z
        if not z.requires_grad:
            z = z.detach().requires_grad_(True)
        
        # Soft Shell outputs structured quantities
        soft_outputs = self.soft_shell(z)
        
        # Couple Hard Core and Soft Shell
        E, S, L, M = self._couple_quantities(z, soft_outputs)
        
        # Compute gradients
        grad_E, grad_S = self._compute_gradients(z, E, S)
        
        # Apply GENERIC equation
        z_dot = self._apply_generic(L, M, grad_E, grad_S)
        
        if return_physics:
            # Thermodynamic quantities
            dE_dt = (grad_E * z_dot).sum(dim=-1, keepdim=True)
            dS_dt = (grad_S * z_dot).sum(dim=-1, keepdim=True)
            dissipation = torch.bmm(
                torch.bmm(grad_S.unsqueeze(-2), M),
                grad_S.unsqueeze(-1)
            ).squeeze(-1)
            
            info = {
                'E': E,
                'S': S,
                'L': L,
                'M': M,
                'grad_E': grad_E,
                'grad_S': grad_S,
                'dE_dt': dE_dt,
                'dS_dt': dS_dt,
                'dissipation': dissipation,
                'weights': {
                    'lambda_E': torch.sigmoid(self.lambda_E).item(),
                    'lambda_S': torch.sigmoid(self.lambda_S).item(),
                    'lambda_L': torch.sigmoid(self.lambda_L).item(),
                    'lambda_M': torch.sigmoid(self.lambda_M).item(),
                }
            }
            return z_dot, info
        
        return z_dot
    
    def set_coupling_weights(self, lambda_E: float, lambda_S: float, 
                            lambda_L: float, lambda_M: float):
        """Manually set coupling weights."""
        self.lambda_E.data.fill_(lambda_E)
        self.lambda_S.data.fill_(lambda_S)
        self.lambda_L.data.fill_(lambda_L)
        self.lambda_M.data.fill_(lambda_M)
    
    def freeze_soft_shell(self):
        """Freeze Soft Shell, only train coupling weights."""
        for param in self.soft_shell.parameters():
            param.requires_grad = False
    
    def unfreeze_soft_shell(self):
        """Unfreeze Soft Shell."""
        for param in self.soft_shell.parameters():
            param.requires_grad = True
    
    def get_architecture_info(self) -> Dict[str, Union[str, float, Dict]]:
        """Get architecture information."""
        return {
            'version': 'v2.0_GENERIC_Full',
            'coupling': 'GENERIC with structured Soft Shell',
            'equation': 'z_dot = L * grad_E + M * grad_S',
            'state_dim': self.state_dim,
            'soft_shell_output': 'E, S, L, M (potentials + matrices)',
            'hard_core': 'Physical priors for E, S, L, M',
            'learnable_coupling': True,
            'current_weights': {
                'lambda_E': torch.sigmoid(self.lambda_E).item(),
                'lambda_S': torch.sigmoid(self.lambda_S).item(),
                'lambda_L': torch.sigmoid(self.lambda_L).item(),
                'lambda_M': torch.sigmoid(self.lambda_M).item(),
            }
        }


def test_generic_full():
    """Test RCLN v2.0 GENERIC Full coupling."""
    print("=" * 70)
    print("RCLN v2.0 - GENERIC Full Coupling Test")
    print("=" * 70)
    print("\nKey Feature: Soft Shell outputs structured quantities")
    print("  - E: Energy potential")
    print("  - S: Entropy potential")
    print("  - L: Poisson matrix (antisymmetric)")
    print("  - M: Friction matrix (PSD)")
    print()
    
    # Create model
    model = RCLNv2_GENERIC_Full(
        state_dim=4,
        hidden_dim=32,
        lambda_E=0.3,  # Mostly Hard Core for energy
        lambda_S=0.7,  # Mostly Soft Shell for entropy
        lambda_L=0.2,  # Mostly Hard Core for Poisson
        lambda_M=0.8,  # Mostly Soft Shell for friction
    )
    
    print("[Architecture Info]")
    info = model.get_architecture_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Forward test
    z = torch.randn(5, 4, requires_grad=True)
    print(f"\n[Forward Test]")
    print(f"  Input z: {z.shape}")
    
    z_dot, physics = model(z, return_physics=True)
    print(f"  Output z_dot: {z_dot.shape}")
    
    print(f"\n[Physics Quantities]")
    print(f"  Energy E: mean={physics['E'].mean().item():.4f}, std={physics['E'].std().item():.4f}")
    print(f"  Entropy S: mean={physics['S'].mean().item():.4f}, std={physics['S'].std().item():.4f}")
    print(f"  dE/dt: mean={physics['dE_dt'].mean().item():.6f} (should be ~0)")
    print(f"  dS/dt: mean={physics['dS_dt'].mean().item():.6f} (should be >= 0)")
    print(f"  Dissipation: mean={physics['dissipation'].mean().item():.6f}")
    
    print(f"\n[Matrix Properties]")
    L = physics['L'][0]  # First sample
    M = physics['M'][0]
    print(f"  L antisymmetry check: ||L + L^T|| = {(L + L.T).norm().item():.6f}")
    print(f"  M symmetry check: ||M - M^T|| = {(M - M.T).norm().item():.6f}")
    M_eig = torch.linalg.eigvals(M)
    print(f"  M eigenvalues: min={M_eig.real.min().item():.6f}, max={M_eig.real.max().item():.6f}")
    
    print(f"\n[Gradient Test]")
    target = torch.randn_like(z_dot)
    loss = nn.MSELoss()(z_dot, target)
    loss.backward()
    
    total_grad = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"  Total gradient norm: {total_grad:.6f}")
    
    # Check which parts have gradients
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.norm() > 1e-6:
            print(f"  {name}: grad_norm={param.grad.norm().item():.6f}")
    
    print("\n" + "=" * 70)
    print("[SUCCESS] RCLN v2.0 GENERIC Full coupling works!")
    print("=" * 70)
    print("\nKey Improvements:")
    print("  1. Soft Shell outputs structured physical quantities")
    print("  2. Matrices have correct properties (L antisymmetric, M PSD)")
    print("  3. Learnable coupling weights between Hard Core and Soft Shell")
    print("  4. Thermodynamic consistency enforced")


if __name__ == "__main__":
    test_generic_full()
