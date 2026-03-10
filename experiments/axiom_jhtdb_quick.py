"""
Axiom-OS + JHTDB Quick Experiment
==================================
Fast demonstration on JHTDB-like turbulence data.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SimpleConv3D(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(3, width, 3, padding=1), nn.GELU(),
            nn.Conv3d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv3d(width, 6, 1)
        )
    def forward(self, x):
        return self.net(x)


def generate_data(n, size=10):
    u = torch.randn(n, 3, size, size, size) * 0.5
    # SGS computation
    tau = torch.zeros(n, 6, size, size, size)
    kernel = torch.ones(1, 1, 3, 3, 3) / 27.0
    for b in range(n):
        for i in range(3):
            ui = u[b, i].unsqueeze(0).unsqueeze(0)
            ui_f = F.conv3d(ui, kernel, padding=1)
            tau[b, i] = (ui**2 - ui_f**2).squeeze()
        for idx, (i1, i2) in enumerate([(0,1), (0,2), (1,2)]):
            ui, uj = u[b, i1].unsqueeze(0).unsqueeze(0), u[b, i2].unsqueeze(0).unsqueeze(0)
            ui_f, uj_f = F.conv3d(ui, kernel, padding=1), F.conv3d(uj, kernel, padding=1)
            tau[b, 3+idx] = (ui * uj - ui_f * uj_f).squeeze()
    return u, tau


def main():
    print("="*60)
    print("Axiom-OS JHTDB Quick Experiment")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Data
    print("Generating data...")
    u_train, tau_train = generate_data(16, 10)
    u_val, tau_val = generate_data(4, 10)
    print(f"Train: {u_train.shape}, Val: {u_val.shape}")
    
    # Model
    model = SimpleConv3D(width=32).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    u_train, tau_train = u_train.to(device), tau_train.to(device)
    u_val, tau_val = u_val.to(device), tau_val.to(device)
    
    # Train
    print("\nTraining...")
    losses = []
    
    for epoch in range(30):
        model.train()
        pred = model(u_train)
        loss = F.mse_loss(pred, tau_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            pred_val = model(u_val)
            val_loss = F.mse_loss(pred_val, tau_val)
            
            r2_list = []
            for c in range(6):
                t, p = tau_val[:, c].flatten(), pred_val[:, c].flatten()
                r2 = 1 - ((t-p)**2).sum() / (((t-t.mean())**2).sum() + 1e-8)
                r2_list.append(r2.item())
        
        losses.append(val_loss.item())
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:2d}: Loss={val_loss:.4f}, R2={np.mean(r2_list):.4f}")
    
    print(f"\nFinal: Loss={losses[-1]:.4f}, R2={np.mean(r2_list):.4f}")
    
    # Save plot
    print("Saving plot...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    axes[0, 0].semilogy(losses)
    axes[0, 0].set_title('Val Loss'); axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].bar(range(6), r2_list)
    axes[0, 1].set_xticks(range(6))
    axes[0, 1].set_xticklabels(['xx', 'yy', 'zz', 'xy', 'xz', 'yz'])
    axes[0, 1].set_title(f'R2 (mean={np.mean(r2_list):.3f})')
    axes[0, 1].axhline(y=0, color='k', linewidth=0.5)
    
    slice_idx = 5
    vmax = max(tau_val[0, 0].abs().max().item(), pred_val[0, 0].abs().max().item())
    axes[1, 0].imshow(pred_val[0, 0, :, :, slice_idx].cpu(), cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1, 0].set_title('Predicted')
    
    axes[1, 1].imshow(tau_val[0, 0, :, :, slice_idx].cpu(), cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1, 1].set_title('True')
    
    plt.suptitle('Axiom-OS JHTDB: SGS Prediction')
    plt.tight_layout()
    plt.savefig('axiom_jhtdb_quick.png', dpi=150)
    print(f"Saved: {os.path.abspath('axiom_jhtdb_quick.png')}")
    
    print("="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
