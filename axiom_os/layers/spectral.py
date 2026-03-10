"""
Spectral Layers for Axiom-OS
Copyright (c) 2026 yuzechen83-crypto. All Rights Reserved.

Spectral Convolution: FFT → Learnable Weights → IFFT
- Frequency-domain operations
- Good for periodic physics problems
- Legacy support (FNO and Clifford are preferred)
"""

from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """
    1D Spectral Convolution (Fourier Layer).
    Transforms input via FFT, multiplies by learnable weights in frequency domain,
    transforms back via IFFT. Preserves global frequency structure.
    """

    def __init__(self, in_dim: int, n_modes: Optional[int] = None):
        super().__init__()
        self.in_dim = in_dim
        self.n_modes = n_modes or (in_dim // 2 + 1)
        
        # Learnable complex weights per frequency mode (stored as real, imag)
        scale = 1.0 / math.sqrt(self.n_modes)
        self.weight_real = nn.Parameter(torch.randn(self.n_modes) * scale)
        self.weight_imag = nn.Parameter(torch.randn(self.n_modes) * scale)

    def reset_parameters(self) -> None:
        scale = 1.0 / math.sqrt(self.n_modes)
        nn.init.normal_(self.weight_real, 0, scale)
        nn.init.normal_(self.weight_imag, 0, scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L) real tensor
        Returns:
            y: (B, L) real tensor
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        L = x.shape[-1]
        
        # FFT: (B, L) -> (B, L//2+1) complex
        spectrum = torch.fft.rfft(x.float(), dim=-1)
        n_freq = spectrum.shape[-1]
        
        # Learnable weights: use first n_modes, pad rest with 1+0j (identity)
        n = min(n_freq, self.n_modes)
        w = torch.complex(self.weight_real[:n], self.weight_imag[:n])
        out_spec = spectrum.clone()
        out_spec[..., :n] = spectrum[..., :n] * w
        
        # IFFT: (B, n_freq) -> (B, L)
        return torch.fft.irfft(out_spec, n=L, dim=-1).to(x.dtype)


class SpectralSoftShell(nn.Module):
    """
    Spectral Soft Shell using Spectral Convolution.
    FFT → Learnable frequency weights → IFFT → Projection.
    
    Legacy architecture - consider FNO or Clifford for new projects.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_modes: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_modes = n_modes or min(input_dim // 2 + 1, 64)

        self.spectral = SpectralConv1d(input_dim, n_modes=self.n_modes)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = self.spectral(x)
        h = F.silu(h)
        return self.proj(h)

    def reset_parameters(self) -> None:
        self.spectral.reset_parameters()
        for m in self.proj:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()


# =============================================================================
# 2D Spectral Convolution (for completeness)
# =============================================================================

class SpectralConv2d(nn.Module):
    """2D Spectral Convolution for image-like data."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int = 12, modes2: int = 12):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            y: (B, C_out, H, W)
        """
        batchsize = x.shape[0]
        
        # FFT
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes1, :self.modes2],
            torch.view_as_complex(self.weights1)
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -self.modes1:, :self.modes2],
            torch.view_as_complex(self.weights2)
        )

        # IFFT
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
