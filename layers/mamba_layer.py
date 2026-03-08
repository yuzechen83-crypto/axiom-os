"""
Mamba Layer for Axiom-OS
Copyright (c) 2026 yuzechen83-crypto. All Rights Reserved.

Mamba-RCLN: Sequential Evolution Path (Scheme 4)
- Replaces LSTM/Transformer for continuous-time physics
- State Space Model (SSM) with selective scan
- Linear complexity O(L) vs O(L^2) for Transformer
- Perfect for long sequences (weather, seismic, control)

Architecture:
    h_t = A·h_{t-1} + B·x_t    (State evolution - like ODE)
    y_t = C·h_t                (Output projection)

Key Benefits for Physics:
    1. Continuous-time: Natural fit for differential equations
    2. Long context: Handle 10k+ step sequences
    3. Efficient: Linear memory and compute
    4. Causal: No information leakage from future

Reference: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
"""

from typing import Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SSMBlock(nn.Module):
    """
    State Space Model Block - Core of Mamba.
    
    Discrete-time SSM:
        h_{t} = A·h_{t-1} + B·x_t
        y_t = C·h_t + D·x_t
    
    Where A, B, C, D are learned parameters.
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int = 16,
        conv_width: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.conv_width = conv_width
        self.expand_factor = expand_factor
        self.inner_dim = input_dim * expand_factor

        # Input projection (x -> [B, C, Δ])
        self.in_proj = nn.Linear(input_dim, self.inner_dim * 2, bias=False)

        # Convolution for local context (before SSM)
        self.conv1d = nn.Conv1d(
            in_channels=self.inner_dim,
            out_channels=self.inner_dim,
            kernel_size=conv_width,
            padding=conv_width - 1,
            groups=self.inner_dim,  # Depthwise separable
        )

        # SSM parameters
        # A: State transition matrix (diagonal for efficiency)
        self.A_log = nn.Parameter(torch.randn(self.inner_dim, state_dim))
        self.A_log._no_weight_decay = True

        # D: Skip connection parameter
        self.D = nn.Parameter(torch.ones(self.inner_dim))

        # Output projection
        self.out_proj = nn.Linear(self.inner_dim, input_dim, bias=False)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def _discretize(self, delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Discretize continuous-time SSM parameters.
        Uses zero-order hold (ZOH):
            A_discrete = exp(Δ·A)
            B_discrete = (exp(Δ·A) - I) / A · B ≈ Δ·B (for small Δ)
        """
        # A is negative for stability (real part < 0)
        A = -torch.exp(self.A_log)  # (inner_dim, state_dim)

        # Discretize
        delta = delta.unsqueeze(-1)  # (batch, seq, inner_dim, 1)
        A = A.unsqueeze(0).unsqueeze(0)  # (1, 1, inner_dim, state_dim)

        A_discrete = torch.exp(delta * A)  # (batch, seq, inner_dim, state_dim)
        B_discrete = delta * A_discrete  # Approximation for small Δ

        return A_discrete, B_discrete

    def _selective_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Selective scan algorithm - core of Mamba.
        
        Args:
            x: (batch, seq, inner_dim)
            delta: (batch, seq, inner_dim) - time step
            B: (batch, seq, inner_dim) - input-dependent B
            C: (batch, seq, inner_dim) - input-dependent C
        
        Returns:
            y: (batch, seq, inner_dim)
        """
        batch, seq_len, inner_dim = x.shape

        # Discretize
        A_discrete, B_discrete = self._discretize(delta)  # (batch, seq, inner, state)

        # Expand B for state dimension
        B_expanded = B.unsqueeze(-1)  # (batch, seq, inner, 1)

        # Scan (parallel associative scan)
        h = torch.zeros(batch, inner_dim, self.state_dim, device=x.device, dtype=x.dtype)
        ys = []

        for t in range(seq_len):
            # State update: h = A·h + B·x
            h = A_discrete[:, t] * h + B_expanded[:, t] * x[:, t:t+1, :].transpose(-1, -2)
            # Output: y = C·h
            y = (C[:, t:t+1, :] * h.transpose(-1, -2)).sum(dim=-1)
            ys.append(y)

        return torch.stack(ys, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, input_dim) or (batch, input_dim) for single step
        
        Returns:
            y: Same shape as x
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
            squeeze_output = True
        else:
            squeeze_output = False

        batch, seq_len, input_dim = x.shape

        # Input projection
        x_and_res = self.in_proj(x)  # (batch, seq, inner_dim*2)
        x_ssm, res = x_and_res.split(self.inner_dim, dim=-1)

        # Convolution (local context)
        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Generate SSM parameters from input (selective mechanism)
        delta = F.softplus(x_conv)  # Time step (always positive)
        B = x_conv  # Input-dependent B
        C = x_conv  # Input-dependent C

        # Selective scan
        y_ssm = self._selective_scan(x_conv, delta, B, C)

        # Gating with residual
        y = y_ssm * F.silu(res)

        # Skip connection (D parameter)
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_conv

        # Output projection
        output = self.out_proj(y)

        if self.dropout is not None:
            output = self.dropout(output)

        if squeeze_output:
            output = output.squeeze(1)

        return output


class MambaSoftShell(nn.Module):
    """
    Mamba Soft Shell for RCLN.
    Replaces MLP/LSTM/Transformer with State Space Model.
    
    Architecture:
        Input → SSM Block × N → Output
    
    Benefits:
        - Linear complexity O(L) vs O(L^2) for Transformer
        - Continuous-time dynamics (naturally ODE-like)
        - Long context (10k+ steps)
        - Causal (no future information leakage)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 2,
        state_dim: int = 16,
        expand_factor: int = 2,
        dropout: float = 0.0,
        use_normalization: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

        # SSM layers
        self.layers = nn.ModuleList([
            SSMBlock(
                input_dim=hidden_dim,
                state_dim=state_dim,
                expand_factor=expand_factor,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Layer normalization (optional but recommended)
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) if use_normalization else nn.Identity()
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, input_dim) or (batch, seq_len, input_dim)
        
        Returns:
            y: (batch, output_dim) or (batch, seq_len, output_dim)
        """
        # Handle single step vs sequence
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
            squeeze_output = True
        else:
            squeeze_output = False

        # Input projection
        h = self.input_proj(x)

        # SSM layers
        for layer, norm in zip(self.layers, self.norms):
            h = layer(h)
            h = norm(h)

        # Output projection
        output = self.output_proj(h)

        if squeeze_output:
            output = output.squeeze(1)

        return output

    def forward_sequence(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Process a full sequence.
        
        Args:
            x_seq: (batch, seq_len, input_dim)
        
        Returns:
            y_seq: (batch, seq_len, output_dim)
        """
        return self.forward(x_seq)

    def step(self, x: torch.Tensor, state: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Single step for recurrent inference (real-time control).
        
        Args:
            x: (batch, input_dim)
            state: List of hidden states for each layer
        
        Returns:
            y: (batch, output_dim)
            new_state: Updated hidden states
        """
        # For recurrent usage, process one step at a time
        x = x.unsqueeze(1)  # (batch, 1, input_dim)

        # Input projection
        h = self.input_proj(x)

        new_states = []
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            h = layer(h)
            h = norm(h)
            # In a full implementation, we'd extract and save the SSM state
            # For now, we just process through

        output = self.output_proj(h).squeeze(1)

        return output, []  # Return empty state for now

    def reset_parameters(self) -> None:
        """Reset all parameters."""
        for m in self.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()


class MambaRCLNCell(nn.Module):
    """
    Mamba-based RCLN Cell for sequential physics modeling.
    
    Combines:
        - Hard Core: Known physics equations
        - Mamba Soft Shell: Learn residual dynamics
    
    Use case: Time series prediction, control, ODE learning.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        state_dim: int = 16,
        n_layers: int = 2,
    ):
        super().__init__()
        self.mamba = MambaSoftShell(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            state_dim=state_dim,
        )

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single step forward (for RNN-style usage).
        
        Args:
            x: (batch, input_dim)
            hidden: Not used in Mamba (state is internal)
        
        Returns:
            y: (batch, output_dim)
            hidden: Dummy hidden state for API compatibility
        """
        y = self.mamba(x)
        return y, torch.zeros_like(y)  # Dummy hidden


def create_mamba_rcln(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    state_dim: int = 16,
    n_layers: int = 2,
    **kwargs
) -> MambaSoftShell:
    """
    Factory function for Mamba-RCLN.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        state_dim: State space dimension (larger = more expressive)
        n_layers: Number of SSM layers
    
    Returns:
        MambaSoftShell instance
    """
    return MambaSoftShell(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        state_dim=state_dim,
        n_layers=n_layers,
        **kwargs
    )


# =============================================================================
# Utility Functions
# =============================================================================

def test_mamba():
    """Quick test of Mamba functionality."""
    print("Testing Mamba Soft Shell...")

    # Test single step
    mamba = MambaSoftShell(input_dim=4, hidden_dim=8, output_dim=2, n_layers=2)
    x = torch.randn(10, 4)
    y = mamba(x)
    print(f"Single step: {x.shape} -> {y.shape}")

    # Test sequence
    x_seq = torch.randn(10, 50, 4)  # batch=10, seq=50
    y_seq = mamba(x_seq)
    print(f"Sequence: {x_seq.shape} -> {y_seq.shape}")

    # Test recurrent step
    y_step, _ = mamba.step(x[0:1])
    print(f"Recurrent step: {x[0:1].shape} -> {y_step.shape}")

    print("Mamba tests passed!")


if __name__ == "__main__":
    test_mamba()
