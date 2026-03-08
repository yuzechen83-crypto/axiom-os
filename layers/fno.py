"""
Fourier Neural Operator (FNO) - Spectral Convolutions
Reference: Fourier Neural Operator for Parametric Partial Differential Equations (Li et al., ICLR 2021)

Core: Learn mapping v(x) -> u(x) (function to function) via convolutions in frequency domain.
Logic: F^{-1}(R · F(x)) where R is learned in Fourier space.
Resolution invariance: Trained on one grid, generalizes to different resolutions.
"""

from typing import Optional
import math
import torch
import torch.nn as nn


class SpectralConv2d(nn.Module):
    """
    2D Spectral Convolution: FFT -> multiply lowest modes by learned weights -> IFFT.
    High frequencies are filtered out (zero).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Complex multiplication: (B,in_c,x,y), (in_c,out_c,x,y) -> (B,out_c,x,y)."""
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) real
        Returns: (B, out_channels, H, W) real
        """
        B = x.shape[0]
        # FFT2: (B, C, H, W) -> (B, C, H, W//2+1) complex
        x_ft = torch.fft.rfft2(x.float())

        out_ft = torch.zeros(
            B, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        # Positive k1 modes: [0:modes1, 0:modes2]
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2],
            self.weights1,
        )
        # Negative k1 modes: [-modes1:, 0:modes2]
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2],
            self.weights2,
        )

        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1))).to(x.dtype)


class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator.
    Architecture: Lifting -> [SpectralConv + Conv1x1] x n_layers -> Projection
    Expects input (B, C, H, W). Output (B, out_channels, H, W).
    Resolution invariant: H, W can differ from training.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        width: int = 32,
        modes1: int = 12,
        modes2: int = 12,
        n_layers: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        self.n_layers = n_layers

        # Lifting: pointwise
        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)

        # Iterative layers: Spectral (global) + Conv1x1 (local)
        self.spectral_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.spectral_layers.append(
                SpectralConv2d(width, width, modes1, modes2)
            )
            self.conv_layers.append(
                nn.Conv2d(width, width, kernel_size=1),
            )

        # Projection
        self.proj = nn.Sequential(
            nn.Conv2d(width, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) - if (B, H, W, C), will auto-permute to (B, C, H, W)
        Returns: (B, out_channels, H, W)
        """
        # Handle (B, H, W, C) -> (B, C, H, W)
        if x.dim() == 4 and x.shape[-1] in (self.in_channels, 3):
            if x.shape[1] != self.in_channels and x.shape[-1] == self.in_channels:
                x = x.permute(0, 3, 1, 2)

        x = self.lift(x.float())
        for i in range(self.n_layers):
            # Branch A: Spectral (global mixing)
            x_spectral = self.spectral_layers[i](x)
            # Branch B: Local linear
            x_conv = self.conv_layers[i](x)
            x = x_spectral + x_conv
            x = nn.functional.gelu(x)

        return self.proj(x)


class SpectralConv3d(nn.Module):
    """
    3D Spectral Convolution: FFT -> multiply lowest modes by learned weights -> IFFT.
    High frequencies are filtered out (zero).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        modes3: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        scale = 1.0 / (in_channels * out_channels)
        # 4 quadrants for (k1,k2) in 3D rfft: k3 is always non-negative
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights3 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights4 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )

    def compl_mul3d(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Complex multiplication: (B,in_c,h,w,d), (in_c,out_c,h,w,d) -> (B,out_c,h,w,d)."""
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W, D) real
        Returns: (B, out_channels, H, W, D) real
        """
        # rfftn: (B, C, H, W, D) -> (B, C, H, W, D//2+1) complex
        x_ft = torch.fft.rfftn(x.float(), dim=(-3, -2, -1))
        H, W, D = x.size(-3), x.size(-2), x.size(-1)
        D_half = D // 2 + 1

        out_ft = torch.zeros(
            x.size(0), self.out_channels, H, W, D_half,
            dtype=torch.cfloat, device=x.device,
        )
        m1, m2, m3 = min(self.modes1, H), min(self.modes2, W), min(self.modes3, D_half)

        # Positive k1, k2
        out_ft[:, :, :m1, :m2, :m3] = self.compl_mul3d(
            x_ft[:, :, :m1, :m2, :m3], self.weights1[:, :, :m1, :m2, :m3]
        )
        # Negative k1, positive k2
        if m1 < H:
            out_ft[:, :, -m1:, :m2, :m3] = self.compl_mul3d(
                x_ft[:, :, -m1:, :m2, :m3], self.weights2[:, :, :m1, :m2, :m3]
            )
        # Positive k1, negative k2
        if m2 < W:
            out_ft[:, :, :m1, -m2:, :m3] = self.compl_mul3d(
                x_ft[:, :, :m1, -m2:, :m3], self.weights3[:, :, :m1, :m2, :m3]
            )
        # Negative k1, negative k2
        if m1 < H and m2 < W:
            out_ft[:, :, -m1:, -m2:, :m3] = self.compl_mul3d(
                x_ft[:, :, -m1:, -m2:, :m3], self.weights4[:, :, :m1, :m2, :m3]
            )

        return torch.fft.irfftn(out_ft, s=(H, W, D), dim=(-3, -2, -1)).to(x.dtype)


class FNO3d(nn.Module):
    """
    3D Fourier Neural Operator for u(x) -> tau(x) mapping.
    Architecture: Lifting -> [SpectralConv3d + Conv1x1x1] x n_layers -> Projection
    Expects input (B, 3, H, W, D). Output (B, 6, H, W, D) for symmetric stress (tau11,22,33,12,13,23).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 6,
        width: int = 32,
        modes1: int = 4,
        modes2: int = 4,
        modes3: int = 4,
        n_layers: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.n_layers = n_layers

        self.lift = nn.Conv3d(in_channels, width, kernel_size=1)
        self.spectral_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.spectral_layers.append(
                SpectralConv3d(width, width, modes1, modes2, modes3)
            )
            self.conv_layers.append(nn.Conv3d(width, width, kernel_size=1))

        self.proj = nn.Sequential(
            nn.Conv3d(width, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(64, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W, D) or (B, H, W, D, 3)
        Returns: (B, 6, H, W, D)
        """
        if x.dim() == 5 and x.shape[-1] == self.in_channels:
            x = x.permute(0, 4, 1, 2, 3)  # (B,H,W,D,3) -> (B,3,H,W,D)
        x = self.lift(x.float())
        for i in range(self.n_layers):
            x_spectral = self.spectral_layers[i](x)
            x_conv = self.conv_layers[i](x)
            x = x_spectral + x_conv
            x = nn.functional.gelu(x)
        return self.proj(x)
