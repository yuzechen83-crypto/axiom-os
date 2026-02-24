"""
Clifford Self-Attention - Spatial Reasoning with Geometric Algebra
Reference: Clifford Group Equivariant Neural Networks (Microsoft Research)
Input: Multivectors (Batch, Seq_Len, 8)
Attention: Q,K,V via Clifford Linear; Score = scalar part of Q·K^T
"""

from typing import Optional
import math
import torch
import torch.nn as nn

from ..core.clifford_ops import geometric_product, CAYLEY, N_BLADES
from .clifford_nn import CliffordLinear


def scalar_part(mv: torch.Tensor) -> torch.Tensor:
    """Extract scalar (grade-0) component of multivector. Index 0."""
    return mv[..., 0:1]


class CliffordSelfAttention(nn.Module):
    """
    Clifford Self-Attention Layer.
    Q, K, V computed via Clifford Linear. Attention score = scalar(Q·K†) / sqrt(d).
    Output = weighted sum of V (geometric product with attention weights).
    """

    def __init__(
        self,
        d_model: int = 8,
        n_heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = max(1, d_model // n_heads)
        self.scale = math.sqrt(self.head_dim)

        # Q, K, V: each maps (B, Seq, 8) -> (B, Seq, 8) via Clifford
        # Input is (B, Seq, 8) - treat as 1 channel of 8 blades
        self.q_proj = CliffordLinear(1, 1)
        self.k_proj = CliffordLinear(1, 1)
        self.v_proj = CliffordLinear(1, 1)
        self.out_proj = CliffordLinear(1, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (B, Seq, 8) multivectors
        Returns: (B, Seq, 8)
        """
        B, S, _ = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x_flat = x.reshape(B * S, 1, N_BLADES)
        Q = self.q_proj(x_flat).reshape(B, S, N_BLADES)
        K = self.k_proj(x_flat).reshape(B, S, N_BLADES)
        V = self.v_proj(x_flat).reshape(B, S, N_BLADES)

        # Attention score: scalar part of Q·K (reversion for K† in Cl(3,0): K† = K for vectors)
        # For each (i,j): score[i,j] = scalar_part(Q[i] * K[j])
        # geometric_product: (B,S,8) x (B,S,8) -> need (B,S,S) scores
        Q_exp = Q.unsqueeze(2)  # (B, S, 1, 8)
        K_exp = K.unsqueeze(1)  # (B, 1, S, 8)
        # Q[i] * K[j] for all i,j: einsum over blades
        scores = torch.einsum("bsik,btjk->bsij", Q_exp, K_exp)
        # Cayley: result_k = sum_ij cayley[i,j,k] * Q_i * K_j
        # Scalar part k=0: scores_scalar[b,s,t] = sum_ij cayley[i,j,0] * Q[b,s,i] * K[b,t,j]
        cayley = CAYLEY.to(x.device)
        attn = torch.einsum("ij,bsi,btj->bst", cayley[..., 0], Q, K) / self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum of V: out[b,s] = sum_t attn[b,s,t] * V[b,t]
        out = torch.einsum("bst,btk->bsk", attn, V)
        out = self.out_proj(out.reshape(B * S, 1, N_BLADES)).reshape(B, S, N_BLADES)
        return out


class CliffordTransformerBlock(nn.Module):
    """Clifford Transformer block: Self-Attention + FFN (Clifford Linear)."""

    def __init__(self, d_model: int = 8, n_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.attn = CliffordSelfAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            CliffordLinear(1, 1),
            nn.GELU(),
            CliffordLinear(1, 1),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, D = x.shape
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        ffn_in = x.reshape(B * S, 1, D)
        x = x + self.dropout(self.ffn(ffn_in).reshape(B, S, D))
        return x


class CliffordTransformerSoftShell(nn.Module):
    """
    Clifford Transformer as RCLN Soft Shell.
    Input: (B, D) flattened -> reshape to (B, Seq, 8) with Seq = D/8.
    Output: (B, output_dim) via projection.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 2,
        n_heads: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = max(1, (input_dim + 7) // 8)
        pad_dim = self.seq_len * 8
        self.pad = nn.Linear(input_dim, pad_dim)

        self.blocks = nn.ModuleList(
            [CliffordTransformerBlock(d_model=8, n_heads=n_heads) for _ in range(n_layers)]
        )
        self.proj = nn.Linear(pad_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B = x.shape[0]
        x = self.pad(x)
        x = x.reshape(B, self.seq_len, 8)
        for block in self.blocks:
            x = block(x)
        x = x.reshape(B, -1)
        return self.proj(x)
