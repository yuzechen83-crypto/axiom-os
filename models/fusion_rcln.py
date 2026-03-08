"""
Axiom-OS Fusion RCLN: Dual-Stream Climate Predictor
===================================================

终极ENSO预测架构：
- Branch A (Surface): SST惯性 - 短期记忆
- Branch B (Deep Ocean): Z20物理 - 长期记忆
- Fusion Gate: 自适应权重分配

Architecture:
    Prediction = alpha * SST_Branch + (1-alpha) * Z20_Branch
    
Where alpha = sigmoid(GateNet(current_state))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Branch A: Surface SST Stream (Short-term Inertia)
# =============================================================================
class SurfaceBranch(nn.Module):
    """
    Surface SST Branch for short-term predictions.
    Captures inertia and immediate trends.
    """
    
    def __init__(self, history_len: int = 6, hidden_dim: int = 32):
        super().__init__()
        self.history_len = history_len
        
        # Simple but effective: LSTM for temporal dynamics
        self.lstm = nn.LSTM(1, hidden_dim, num_layers=2, 
                           batch_first=True, dropout=0.1)
        
        # Output projection
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Learnable persistence coefficient
        self.persistence = nn.Parameter(torch.tensor(0.9))
    
    def forward(self, sst_history: torch.Tensor):
        """
        Args:
            sst_history: (batch, history_len) - past SST values
        
        Returns:
            prediction: (batch,) - SST forecast
        """
        # Add feature dimension
        x = sst_history.unsqueeze(-1)  # (batch, history_len, 1)
        
        # LSTM encoding
        lstm_out, _ = self.lstm(x)
        
        # Use last hidden state
        features = lstm_out[:, -1, :]  # (batch, hidden_dim)
        
        # Predict increment
        increment = self.project(features).squeeze(-1)
        
        # Persistence + learned increment
        alpha = torch.sigmoid(self.persistence)
        last_sst = sst_history[:, -1]
        prediction = alpha * last_sst + (1 - alpha) * increment
        
        return prediction


# =============================================================================
# Branch B: Deep Ocean Z20 Stream (Long-term Physics)
# =============================================================================
class DeepOceanBranch(nn.Module):
    """
    Deep Ocean Z20 Branch for long-term predictions.
    Uses FNO to capture spatial wave propagation.
    """
    
    def __init__(self, nx: int = 64, ny: int = 32, 
                 modes: int = 12, width: int = 32):
        super().__init__()
        self.nx = nx
        self.ny = ny
        
        # Lift to higher dimension
        self.lift = nn.Conv2d(1, width, 1)
        
        # FNO layers
        self.fno_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()
        
        for _ in range(4):
            self.fno_layers.append(
                SpectralConv2d(width, width, modes, modes)
            )
            self.w_layers.append(nn.Conv2d(width, width, 1))
        
        # Global pooling + projection
        self.project = nn.Sequential(
            nn.Linear(width, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, z20_field: torch.Tensor):
        """
        Args:
            z20_field: (batch, 1, ny, nx) - Z20 spatial field
        
        Returns:
            prediction: (batch,) - SST forecast
        """
        x = self.lift(z20_field)
        
        # FNO processing
        for fno, w in zip(self.fno_layers, self.w_layers):
            x = F.gelu(fno(x) + w(x))
        
        # Global average pooling
        x = x.mean(dim=[2, 3])  # (batch, width)
        
        # Project to prediction
        prediction = self.project(x).squeeze(-1)
        
        return prediction


class SpectralConv2d(nn.Module):
    """Spectral convolution for FNO."""
    
    def __init__(self, in_c, out_c, modes1, modes2):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.modes1 = modes1
        self.modes2 = modes2
        
        scale = 1.0 / (in_c * out_c)
        self.weights = nn.Parameter(
            scale * torch.rand(in_c, out_c, modes1, modes2, 2)
        )
    
    def forward(self, x):
        batch = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        
        out_ft = torch.zeros(batch, self.out_c, x.size(-2), x.size(-1)//2 + 1,
                            dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes1, :self.modes2],
            torch.view_as_complex(self.weights)
        )
        
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


# =============================================================================
# Fusion Gate: Adaptive Weighting
# =============================================================================
class FusionGate(nn.Module):
    """
    Fusion Gate that decides how much to trust each branch.
    
    alpha = sigmoid(GateNet(state))
    - alpha ≈ 1: Trust Surface Branch (short-term)
    - alpha ≈ 0: Trust Deep Ocean Branch (long-term)
    """
    
    def __init__(self, sst_dim: int = 6, z20_dim: int = 64*32):
        super().__init__()
        
        self.gate_net = nn.Sequential(
            nn.Linear(sst_dim + 50, 32),  # SST history + Z20 summary
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Z20 encoder for summary
        self.z20_encoder = nn.Sequential(
            nn.Conv2d(1, 8, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 5))  # Compress to 5x5
        )
    
    def forward(self, sst_history: torch.Tensor, z20_field: torch.Tensor):
        """
        Args:
            sst_history: (batch, history_len)
            z20_field: (batch, 1, ny, nx)
        
        Returns:
            alpha: (batch,) - weight for Surface Branch
        """
        batch_size = sst_history.shape[0]
        
        # Encode Z20 to compact representation
        z20_encoded = self.z20_encoder(z20_field)  # (batch, 16, 5, 5)
        z20_flat = z20_encoded.view(batch_size, -1)  # (batch, 400)
        
        # Use subset of Z20 features
        z20_summary = z20_flat[:, :50]
        
        # Concatenate SST and Z20 summary
        combined = torch.cat([sst_history, z20_summary], dim=1)
        
        # Compute gate value
        alpha = torch.sigmoid(self.gate_net(combined)).squeeze(-1)
        
        return alpha


# =============================================================================
# Axiom Fusion RCLN: The Ultimate ENSO Predictor
# =============================================================================
class AxiomFusionRCLN(nn.Module):
    """
    Axiom-OS Dual-Stream Fusion Architecture.
    
    Combines:
    - Surface Branch: Fast dynamics (SST inertia)
    - Deep Ocean Branch: Slow dynamics (Z20 physics)
    - Fusion Gate: Adaptive weighting based on forecast horizon
    
    Usage:
        model = AxiomFusionRCLN()
        pred, info = model(sst_history, z20_field)
        
        info['alpha'] tells you which branch dominated
    """
    
    def __init__(
        self,
        sst_history_len: int = 6,
        z20_shape: tuple = (32, 64),
        surface_hidden: int = 32,
        deep_hidden: int = 32
    ):
        super().__init__()
        
        self.sst_history_len = sst_history_len
        self.z20_ny, self.z20_nx = z20_shape
        
        # Two branches
        self.surface_branch = SurfaceBranch(sst_history_len, surface_hidden)
        self.deep_ocean_branch = DeepOceanBranch(z20_shape[1], z20_shape[0], 
                                                  modes=12, width=deep_hidden)
        
        # Fusion gate
        self.fusion_gate = FusionGate(sst_history_len, z20_shape[0]*z20_shape[1])
        
    def forward(self, sst_history: torch.Tensor, z20_field: torch.Tensor):
        """
        Args:
            sst_history: (batch, sst_history_len) - past SST values
            z20_field: (batch, 1, ny, nx) - Z20 spatial field
        
        Returns:
            prediction: (batch,) - fused SST forecast
            info: dict with branch outputs and fusion weight
        """
        # Branch predictions
        pred_surface = self.surface_branch(sst_history)
        pred_deep = self.deep_ocean_branch(z20_field)
        
        # Fusion weight
        alpha = self.fusion_gate(sst_history, z20_field)
        
        # Fused prediction
        prediction = alpha * pred_surface + (1 - alpha) * pred_deep
        
        info = {
            'prediction': prediction,
            'pred_surface': pred_surface,
            'pred_deep': pred_deep,
            'alpha': alpha,  # Key: tells us which branch dominated
            'surface_weight': alpha.mean().item(),
            'deep_weight': (1 - alpha).mean().item()
        }
        
        return prediction, info


# =============================================================================
# Baseline Models for Comparison
# =============================================================================
class PersistenceBaseline(nn.Module):
    """Simple persistence: T(t+lead) ≈ T(t)"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, sst_history):
        return sst_history[:, -1], {}


class SSTOnlyModel(nn.Module):
    """Phase 1 model: Only SST history."""
    
    def __init__(self, history_len=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(history_len, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, sst_history, z20_field=None):
        pred = self.net(sst_history).squeeze(-1)
        return pred, {'alpha': torch.ones_like(pred)}


class Z20OnlyModel(nn.Module):
    """Phase 2 model: Only Z20 field."""
    
    def __init__(self, nx=64, ny=32):
        super().__init__()
        self.fno = DeepOceanBranch(nx, ny)
    
    def forward(self, sst_history, z20_field):
        pred = self.fno(z20_field)
        return pred, {'alpha': torch.zeros_like(pred)}


if __name__ == "__main__":
    # Test the model
    print("Testing AxiomFusionRCLN...")
    
    model = AxiomFusionRCLN(sst_history_len=6, z20_shape=(32, 64))
    
    batch_size = 4
    sst_history = torch.randn(batch_size, 6)
    z20_field = torch.randn(batch_size, 1, 32, 64)
    
    pred, info = model(sst_history, z20_field)
    
    print(f"Prediction shape: {pred.shape}")
    print(f"Surface weight: {info['surface_weight']:.3f}")
    print(f"Deep weight: {info['deep_weight']:.3f}")
    print(f"Alpha range: [{info['alpha'].min():.3f}, {info['alpha'].max():.3f}]")
    
    # Test with extreme cases
    print("\nTesting gate behavior...")
    
    # Case 1: Recent SST spike (should use surface)
    sst_spike = torch.zeros(batch_size, 6)
    sst_spike[:, -1] = 2.0  # Recent warm anomaly
    pred1, info1 = model(sst_spike, z20_field)
    print(f"SST spike case: alpha={info1['alpha'].mean():.3f} (should be high)")
    
    # Case 2: Flat SST but strong Z20 signal (should use deep)
    sst_flat = torch.zeros(batch_size, 6)
    z20_strong = torch.ones(batch_size, 1, 32, 64) * 2.0
    pred2, info2 = model(sst_flat, z20_strong)
    print(f"Z20 strong case: alpha={info2['alpha'].mean():.3f} (should be low)")
    
    print("\nModel ready for training!")
