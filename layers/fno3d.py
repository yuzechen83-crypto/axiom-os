"""
3D Fourier Neural Operator (FNO3d) for Axiom-OS
Copyright (c) 2026 yuzechen83-crypto. All Rights Reserved.

3D Spectral Convolution for volumetric data (turbulence, weather, etc.)
"""

import torch
import torch.nn as nn
import numpy as np


class SpectralConv3d(nn.Module):
    """3D Spectral Convolution layer."""
    
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # First Fourier mode
        self.modes2 = modes2  # Second Fourier mode
        self.modes3 = modes3  # Third Fourier mode
        
        self.scale = 1.0 / (in_channels * out_channels)
        
        # Complex weights for Fourier modes
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, 2)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, 2)
        )
        self.weights3 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, 2)
        )
        self.weights4 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, 2)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, in_channels, D, H, W)
        Returns:
            out: (batch, out_channels, D, H, W)
        """
        batchsize = x.shape[0]
        
        # FFT
        x_ft = torch.fft.rfftn(x, dim=[2, 3, 4])
        
        # Initialize output in Fourier space
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(2), x.size(3), x.size(4)//2 + 1,
            device=x.device, dtype=torch.cfloat
        )
        
        # Multiply relevant Fourier modes
        # Mode 1: Lower frequencies
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = torch.einsum(
            "bixyz,ioxyz->boxyz",
            x_ft[:, :, :self.modes1, :self.modes2, :self.modes3],
            torch.view_as_complex(self.weights1)
        )
        
        # Mode 2: Higher frequencies in first dim
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = torch.einsum(
            "bixyz,ioxyz->boxyz",
            x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3],
            torch.view_as_complex(self.weights2)
        )
        
        # Mode 3: Higher frequencies in second dim
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = torch.einsum(
            "bixyz,ioxyz->boxyz",
            x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3],
            torch.view_as_complex(self.weights3)
        )
        
        # Mode 4: Higher frequencies in both dims
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = torch.einsum(
            "bixyz,ioxyz->boxyz",
            x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3],
            torch.view_as_complex(self.weights4)
        )
        
        # IFFT
        x = torch.fft.irfftn(out_ft, s=(x.size(2), x.size(3), x.size(4)), dim=[2, 3, 4])
        
        return x


class FNO3d(nn.Module):
    """
    3D Fourier Neural Operator.
    
    For volumetric data like 3D turbulence fields.
    """
    
    def __init__(
        self,
        in_channels,
        out_channels,
        width=32,
        modes1=8,
        modes2=8,
        modes3=8,
        n_layers=4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.n_layers = n_layers
        
        # Input lifting
        self.fc0 = nn.Linear(in_channels, width)
        
        # FNO layers
        self.convs = nn.ModuleList()
        self.ws = nn.ModuleList()
        
        for _ in range(n_layers):
            self.convs.append(
                SpectralConv3d(width, width, modes1, modes2, modes3)
            )
            self.ws.append(
                nn.Conv3d(width, width, 1)
            )
        
        # Output projection
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)
    
    def forward(self, x):
        """
        Args:
            x: (batch, in_channels, D, H, W)
        Returns:
            out: (batch, out_channels, D, H, W)
        """
        # Lifting: (B, C_in, D, H, W) -> (B, D, H, W, width)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)  # (B, width, D, H, W)
        
        # FNO layers
        for i in range(self.n_layers):
            x1 = self.convs[i](x)
            x2 = self.ws[i](x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = torch.nn.functional.gelu(x)
        
        # Projection: (B, width, D, H, W) -> (B, D, H, W, C_out)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 4, 1, 2, 3)  # (B, C_out, D, H, W)
        
        return x
