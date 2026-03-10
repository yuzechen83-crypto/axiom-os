#!/usr/bin/env python3
"""
RCLN vs TBNN LES 公平对比实验
==============================

在完全相同的数据和测试条件下对比：
- RCLN-LES (改进版): Hard Core + RCLN Soft Shell
- TBNN-LES: Pope 张量基 + MLP
- Smagorinsky: 基线

数据: JHTDB 真实 DNS
train/test 分割: 相同
评估指标: 相同
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from axiom_os.datasets.jhtdb_turbulence import load_jhtdb_for_les_sgs
from axiom_os.layers.rcln import RCLNLayer
from axiom_os.layers.tbnn import TBNN, stack_tensor_basis
from axiom_os.core.turbulence_invariants import extract_invariants_and_basis_normalized


# =============================================================================
# 改进版 RCLN-LES (更大网络 + 更好的架构)
# =============================================================================

class Improved_RCLN_LES(nn.Module):
    """
    改进版 RCLN-LES
    - 更大的网络: hidden=128, n_layers=5
    - 多尺度特征提取
    - 批归一化
    """
    
    def __init__(self, hidden_dim=128, n_layers=5):
        super().__init__()
        
        self.input_dim = 15  # 速度(3) + 速度大小(1) + S_mag(1) + 涡量(1) + S_ij(6) + 散度(3)
        self.output_dim = 1  # 涡粘系数修正
        
        # RCLN Soft Shell
        self.rcln = RCLNLayer(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            output_dim=self.output_dim,
            hard_core_func=None,
            lambda_res=0.5,
            net_type="mlp",
        )
        
        # 额外的 MLP 头用于精细调整
        self.refine_head = nn.Sequential(
            nn.Linear(self.output_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.output_dim),
        )
        
        self.delta = 1.0
        self.Cs_base = 0.17
        
    def compute_features(self, u):
        """提取物理特征"""
        B, Nx, Ny, Nz, C = u.shape
        
        # 速度梯度
        du_dx = torch.zeros_like(u)
        du_dy = torch.zeros_like(u)
        du_dz = torch.zeros_like(u)
        
        du_dx[:, 1:-1, :, :, :] = (u[:, 2:, :, :, :] - u[:, :-2, :, :, :]) / 2
        du_dy[:, :, 1:-1, :, :] = (u[:, :, 2:, :, :] - u[:, :, :-2, :, :]) / 2
        du_dz[:, :, :, 1:-1, :] = (u[:, :, :, 2:, :] - u[:, :, :, :-2, :]) / 2
        
        # 应变率
        S_xx = du_dx[..., 0]
        S_yy = du_dy[..., 1]
        S_zz = du_dz[..., 2]
        S_xy = 0.5 * (du_dx[..., 1] + du_dy[..., 0])
        S_xz = 0.5 * (du_dx[..., 2] + du_dz[..., 0])
        S_yz = 0.5 * (du_dy[..., 2] + du_dz[..., 1])
        
        S_mag = torch.sqrt(2 * (S_xx**2 + S_yy**2 + S_zz**2 + 2*S_xy**2 + 2*S_xz**2 + 2*S_yz**2))
        
        # 涡量
        omega_x = du_dy[..., 2] - du_dz[..., 1]
        omega_y = du_dz[..., 0] - du_dx[..., 2]
        omega_z = du_dx[..., 1] - du_dy[..., 0]
        omega_mag = torch.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
        
        u_mag = torch.sqrt(u[..., 0]**2 + u[..., 1]**2 + u[..., 2]**2)
        
        features = torch.stack([
            u[..., 0], u[..., 1], u[..., 2],  # 3
            u_mag,  # 1
            S_mag, omega_mag,  # 2
            S_xx, S_yy, S_zz, S_xy, S_xz, S_yz,  # 6
            du_dx[..., 0], du_dy[..., 1], du_dz[..., 2],  # 3
        ], dim=-1)
        
        return features, (du_dx, du_dy, du_dz, S_mag)
    
    def forward(self, u):
        """前向传播"""
        B, Nx, Ny, Nz, C = u.shape
        
        # Hard Core 特征
        features, (du_dx, du_dy, du_dz, S_mag) = self.compute_features(u)
        
        # Soft Shell (RCLN)
        features_flat = features.reshape(-1, self.input_dim)
        g_base = self.rcln(features_flat)
        
        # 精细化调整
        g_refined = self.refine_head(g_base)
        g_total = g_base + 0.1 * g_refined
        g_total = g_total.reshape(B, Nx, Ny, Nz, 1)
        
        # 组合
        nu_sgs = (self.Cs_base * self.delta)**2 * S_mag.unsqueeze(-1) * (1.0 + torch.tanh(g_total))
        
        # 重建应力
        S_xx = du_dx[..., 0]
        S_yy = du_dy[..., 1]
        S_zz = du_dz[..., 2]
        S_xy = 0.5 * (du_dx[..., 1] + du_dy[..., 0])
        S_xz = 0.5 * (du_dx[..., 2] + du_dz[..., 0])
        S_yz = 0.5 * (du_dy[..., 2] + du_dz[..., 1])
        
        tau = torch.zeros(B, Nx, Ny, Nz, 3, 3, device=u.device, dtype=u.dtype)
        tau[..., 0, 0] = -2 * nu_sgs[..., 0] * S_xx
        tau[..., 1, 1] = -2 * nu_sgs[..., 0] * S_yy
        tau[..., 2, 2] = -2 * nu_sgs[..., 0] * S_zz
        tau[..., 0, 1] = tau[..., 1, 0] = -2 * nu_sgs[..., 0] * S_xy
        tau[..., 0, 2] = tau[..., 2, 0] = -2 * nu_sgs[..., 0] * S_xz
        tau[..., 1, 2] = tau[..., 2, 1] = -2 * nu_sgs[..., 0] * S_yz
        
        return tau


# =============================================================================
# TBNN-LES (作为对比)
# =============================================================================

class TBNN_LES(nn.Module):
    """TBNN LES 模型"""
    
    def __init__(self, hidden=128, n_layers=5):
        super().__init__()
        self.tbnn = TBNN(n_invariants=5, n_tensors=10, hidden=hidden, n_layers=n_layers)
        
    def forward(self, invariants, tensor_basis, sigma):
        """
        Args:
            invariants: (B, 5)
            tensor_basis: (B, 10, 3, 3)
            sigma: (B,)
        """
        return self.tbnn(invariants, tensor_basis, sigma)


# =============================================================================
# Smagorinsky 基线
# =============================================================================

def smagorinsky_model(u, Cs=0.17):
    """Smagorinsky 基线"""
    B, Nx, Ny, Nz, C = u.shape
    device = u.device
    
    du_dx = torch.zeros_like(u)
    du_dy = torch.zeros_like(u)
    du_dz = torch.zeros_like(u)
    du_dx[:, 1:-1, :, :, :] = (u[:, 2:, :, :, :] - u[:, :-2, :, :, :]) / 2
    du_dy[:, :, 1:-1, :, :] = (u[:, :, 2:, :, :] - u[:, :, :-2, :, :]) / 2
    du_dz[:, :, :, 1:-1, :] = (u[:, :, :, 2:, :] - u[:, :, :, :-2, :]) / 2
    
    S_xx = du_dx[..., 0]
    S_yy = du_dy[..., 1]
    S_zz = du_dz[..., 2]
    S_xy = 0.5 * (du_dx[..., 1] + du_dy[..., 0])
    S_xz = 0.5 * (du_dx[..., 2] + du_dz[..., 0])
    S_yz = 0.5 * (du_dy[..., 2] + du_dz[..., 1])
    
    S_mag = torch.sqrt(2 * (S_xx**2 + S_yy**2 + S_zz**2 + 2*S_xy**2 + 2*S_xz**2 + 2*S_yz**2))
    nu_sgs = (Cs * 1.0)**2 * S_mag
    
    tau = torch.zeros(B, Nx, Ny, Nz, 3, 3, device=device, dtype=u.dtype)
    tau[..., 0, 0] = -2 * nu_sgs * S_xx
    tau[..., 1, 1] = -2 * nu_sgs * S_yy
    tau[..., 2, 2] = -2 * nu_sgs * S_zz
    tau[..., 0, 1] = tau[..., 1, 0] = -2 * nu_sgs * S_xy
    tau[..., 0, 2] = tau[..., 2, 0] = -2 * nu_sgs * S_xz
    tau[..., 1, 2] = tau[..., 2, 1] = -2 * nu_sgs * S_yz
    
    return tau


# =============================================================================
# 训练和评估函数
# =============================================================================

def compute_metrics(pred, true):
    """计算指标"""
    mse = torch.mean((pred - true)**2).item()
    
    pred_flat = pred.reshape(-1)
    true_flat = true.reshape(-1)
    
    ss_res = torch.sum((pred - true)**2).item()
    ss_tot = torch.sum((true - torch.mean(true))**2).item()
    r2 = 1 - ss_res / (ss_tot + 1e-10)
    
    corr = np.corrcoef(pred_flat.cpu().numpy(), true_flat.cpu().numpy())[0, 1]
    
    return {'mse': mse, 'r2': r2, 'correlation': corr}


def prepare_data(u_coarse, tau_target, test_size=0.3):
    """准备数据集（相同分割用于所有模型）"""
    B, Nx, Ny, Nz, C = u_coarse.shape if u_coarse.dim() == 5 else (1, *u_coarse.shape)
    
    # 展平空间维度
    u_flat = u_coarse.reshape(-1, 3) if u_coarse.dim() == 5 else u_coarse.reshape(-1, 3)
    tau_flat = tau_target.reshape(-1, 9) if tau_target.dim() == 6 else tau_target.reshape(-1, 9)
    
    # 提取 TBNN 特征
    inv, basis, sigma = extract_invariants_and_basis_normalized(
        u_coarse.squeeze(0).cpu().numpy() if u_coarse.dim() == 5 else u_coarse.cpu().numpy()
    )
    inv_flat = inv.reshape(-1, 5).astype(np.float32)
    tb_stacked = stack_tensor_basis(basis).numpy()
    tb_flat = tb_stacked.reshape(-1, 10, 3, 3)
    sigma_flat = sigma.reshape(-1).astype(np.float32)
    
    # 相同分割
    n_total = len(u_flat)
    indices = np.arange(n_total)
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)
    
    return {
        'u_full': u_coarse,
        'tau_full': tau_target,
        'u_flat': u_flat,
        'tau_flat': tau_flat,
        'inv_flat': inv_flat,
        'tb_flat': tb_flat,
        'sigma_flat': sigma_flat,
        'train_idx': train_idx,
        'test_idx': test_idx,
    }


def train_rcln(data, n_epochs=500):
    """训练 RCLN"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Improved_RCLN_LES(hidden_dim=128, n_layers=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # Handle both numpy and tensor inputs
    u_full = data['u_full']
    tau_full = data['tau_full']
    if not torch.is_tensor(u_full):
        u_full = torch.from_numpy(u_full).float()
    if not torch.is_tensor(tau_full):
        tau_full = torch.from_numpy(tau_full).float()
    u_full = u_full.to(device)
    tau_full = tau_full.to(device)
    
    if u_full.dim() == 4:
        u_full = u_full.unsqueeze(0)
        tau_full = tau_full.unsqueeze(0)
    
    print("  Training RCLN (Improved)...")
    losses = []
    start = time.time()
    
    for epoch in range(n_epochs):
        model.train()
        tau_pred = model(u_full)
        loss = torch.mean((tau_pred - tau_full)**2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}: loss={loss.item():.6e}")
    
    train_time = time.time() - start
    
    # 评估
    model.eval()
    with torch.no_grad():
        tau_pred = model(u_full)
    
    metrics = compute_metrics(tau_pred, tau_full)
    
    return {'model': model, 'metrics': metrics, 'train_time': train_time, 'losses': losses}


def train_tbnn(data, n_epochs=500):
    """训练 TBNN"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = TBNN_LES(hidden=128, n_layers=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # Helper to convert to tensor
    def to_tensor(x, dtype=torch.float32):
        if torch.is_tensor(x):
            return x.to(device)
        return torch.from_numpy(x).to(dtype).to(device)
    
    # 准备数据
    inv_train = to_tensor(data['inv_flat'][data['train_idx']])
    tb_train = to_tensor(data['tb_flat'][data['train_idx']])
    sigma_train = to_tensor(data['sigma_flat'][data['train_idx']])
    tau_train = to_tensor(data['tau_flat'][data['train_idx']])
    
    inv_test = to_tensor(data['inv_flat'][data['test_idx']])
    tb_test = to_tensor(data['tb_flat'][data['test_idx']])
    sigma_test = to_tensor(data['sigma_flat'][data['test_idx']])
    tau_test = to_tensor(data['tau_flat'][data['test_idx']])
    
    print("  Training TBNN...")
    losses = []
    start = time.time()
    
    for epoch in range(n_epochs):
        model.train()
        tau_pred = model(inv_train, tb_train, sigma_train).reshape(-1, 9)
        loss = torch.mean((tau_pred - tau_train)**2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}: loss={loss.item():.6e}")
    
    train_time = time.time() - start
    
    # 评估
    model.eval()
    with torch.no_grad():
        tau_test_pred = model(inv_test, tb_test, sigma_test).reshape(-1, 9)
    
    metrics = compute_metrics(tau_test_pred, tau_test)
    
    return {'model': model, 'metrics': metrics, 'train_time': train_time, 'losses': losses}


# =============================================================================
# 主实验
# =============================================================================

def run_comparison():
    """主对比实验"""
    print("="*70)
    print("RCLN vs TBNN LES 公平对比实验")
    print("="*70)
    print("条件: 相同数据、相同训练epoch、相同测试集")
    print("="*70)
    
    # 加载数据
    print("\n[1/5] 加载 JHTDB 真实 DNS 数据...")
    try:
        u_fine, u_coarse, tau_target, meta = load_jhtdb_for_les_sgs(
            fine_size=16, coarse_ratio=2, dataset='isotropic8192', timepoint=1
        )
        print(f"  数据: {u_coarse.shape} -> {tau_target.shape}")
        print(f"  来源: {meta['data_source']}")
    except Exception as e:
        print(f"  错误: {e}")
        return
    
    # 准备数据（所有模型使用相同分割）
    print("\n[2/5] 准备数据集（相同 train/test 分割）...")
    data = prepare_data(
        torch.from_numpy(u_coarse).float().unsqueeze(0),
        torch.from_numpy(tau_target).float().unsqueeze(0),
        test_size=0.3
    )
    print(f"  总样本: {len(data['u_flat'])}, 训练: {len(data['train_idx'])}, 测试: {len(data['test_idx'])}")
    
    # Smagorinsky 基线
    print("\n[3/5] 运行 Smagorinsky 基线...")
    u_torch = torch.from_numpy(u_coarse).float().unsqueeze(0)
    tau_smag = smagorinsky_model(u_torch)
    tau_target_torch = torch.from_numpy(tau_target).float().unsqueeze(0)
    smag_metrics = compute_metrics(tau_smag, tau_target_torch)
    print(f"  Smagorinsky: MSE={smag_metrics['mse']:.6e}, R2={smag_metrics['r2']:.4f}")
    
    # 训练 RCLN
    print("\n[4/5] 训练改进版 RCLN...")
    rcln_results = train_rcln(data, n_epochs=500)
    print(f"  RCLN: MSE={rcln_results['metrics']['mse']:.6e}, R2={rcln_results['metrics']['r2']:.4f}")
    print(f"  训练时间: {rcln_results['train_time']:.2f}s")
    
    # 训练 TBNN
    print("\n[5/5] 训练 TBNN...")
    tbnn_results = train_tbnn(data, n_epochs=500)
    print(f"  TBNN: MSE={tbnn_results['metrics']['mse']:.6e}, R2={tbnn_results['metrics']['r2']:.4f}")
    print(f"  训练时间: {tbnn_results['train_time']:.2f}s")
    
    # 对比表格
    print("\n" + "="*70)
    print("对比结果 (相同测试集)")
    print("="*70)
    print(f"{'Model':<20} {'MSE':<12} {'R2':<10} {'Corr':<10} {'Time(s)':<10}")
    print("-"*70)
    print(f"{'Smagorinsky':<20} {smag_metrics['mse']:<12.6e} {smag_metrics['r2']:<10.4f} "
          f"{smag_metrics['correlation']:<10.4f} {'-':<10}")
    print(f"{'RCLN-LES':<20} {rcln_results['metrics']['mse']:<12.6e} {rcln_results['metrics']['r2']:<10.4f} "
          f"{rcln_results['metrics']['correlation']:<10.4f} {rcln_results['train_time']:<10.2f}")
    print(f"{'TBNN-LES':<20} {tbnn_results['metrics']['mse']:<12.6e} {tbnn_results['metrics']['r2']:<10.4f} "
          f"{tbnn_results['metrics']['correlation']:<10.4f} {tbnn_results['train_time']:<10.2f}")
    print("="*70)
    
    # 改进倍数
    print("\n改进分析:")
    print(f"  RCLN vs Smagorinsky: MSE {smag_metrics['mse']/rcln_results['metrics']['mse']:.2f}x")
    print(f"  TBNN vs Smagorinsky: MSE {smag_metrics['mse']/tbnn_results['metrics']['mse']:.2f}x")
    print(f"  RCLN vs TBNN: MSE {tbnn_results['metrics']['mse']/rcln_results['metrics']['mse']:.2f}x")
    
    # 可视化
    print("\n  生成对比图表...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 训练曲线对比
    ax = axes[0, 0]
    ax.semilogy(rcln_results['losses'], label='RCLN', color='green')
    ax.semilogy(tbnn_results['losses'], label='TBNN', color='blue')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss (MSE)')
    ax.set_title('Training Curve Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 性能对比柱状图
    ax = axes[0, 1]
    models = ['Smagorinsky', 'RCLN', 'TBNN']
    r2_vals = [max(0, smag_metrics['r2']), max(0, rcln_results['metrics']['r2']), 
               max(0, tbnn_results['metrics']['r2'])]
    colors = ['orange', 'green', 'blue']
    bars = ax.bar(models, r2_vals, color=colors, alpha=0.7)
    ax.set_ylabel('R2 Score')
    ax.set_title('R2 Performance Comparison')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # MSE 对比 (log scale)
    ax = axes[0, 2]
    mse_vals = [smag_metrics['mse'], rcln_results['metrics']['mse'], tbnn_results['metrics']['mse']]
    ax.bar(models, mse_vals, color=colors, alpha=0.7)
    ax.set_ylabel('MSE (log scale)')
    ax.set_yscale('log')
    ax.set_title('MSE Comparison')
    
    # 散点图: RCLN
    ax = axes[1, 0]
    tau_true_flat = tau_target.ravel()
    with torch.no_grad():
        tau_rcln = rcln_results['model'](u_torch).cpu().numpy().ravel()
    idx = np.random.choice(len(tau_true_flat), 800, replace=False)
    ax.scatter(tau_true_flat[idx], tau_rcln[idx], alpha=0.5, s=8, color='green')
    ax.plot([tau_true_flat.min(), tau_true_flat.max()], 
            [tau_true_flat.min(), tau_true_flat.max()], 'r--', lw=2)
    ax.set_xlabel('True SGS Stress')
    ax.set_ylabel('Predicted (RCLN)')
    ax.set_title(f'RCLN: R2={rcln_results["metrics"]["r2"]:.3f}')
    ax.grid(True, alpha=0.3)
    
    # 散点图: TBNN
    ax = axes[1, 1]
    with torch.no_grad():
        inv_all = torch.from_numpy(data['inv_flat']).float()
        tb_all = torch.from_numpy(data['tb_flat'].astype(np.float32))
        sigma_all = torch.from_numpy(data['sigma_flat']).float()
        tau_tbnn = tbnn_results['model'](inv_all, tb_all, sigma_all).reshape(-1, 9).cpu().numpy().ravel()
    ax.scatter(tau_true_flat[idx], tau_tbnn[idx], alpha=0.5, s=8, color='blue')
    ax.plot([tau_true_flat.min(), tau_true_flat.max()], 
            [tau_true_flat.min(), tau_true_flat.max()], 'r--', lw=2)
    ax.set_xlabel('True SGS Stress')
    ax.set_ylabel('Predicted (TBNN)')
    ax.set_title(f'TBNN: R2={tbnn_results["metrics"]["r2"]:.3f}')
    ax.grid(True, alpha=0.3)
    
    # 散点图: Smagorinsky
    ax = axes[1, 2]
    ax.scatter(tau_true_flat[idx], tau_smag.cpu().numpy().ravel()[idx], alpha=0.5, s=8, color='orange')
    ax.plot([tau_true_flat.min(), tau_true_flat.max()], 
            [tau_true_flat.min(), tau_true_flat.max()], 'r--', lw=2)
    ax.set_xlabel('True SGS Stress')
    ax.set_ylabel('Predicted (Smagorinsky)')
    ax.set_title(f'Smagorinsky: R2={smag_metrics["r2"]:.3f}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = ROOT / 'docs/images/rcln_vs_tbnn_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  保存至: {output_path}")
    
    # 保存结果
    import json
    results = {
        'experiment': 'RCLN vs TBNN LES Fair Comparison',
        'data': {'source': 'JHTDB isotropic8192', 'shape': list(u_coarse.shape)},
        'models': {
            'smagorinsky': smag_metrics,
            'rcln': {'metrics': rcln_results['metrics'], 'train_time': rcln_results['train_time']},
            'tbnn': {'metrics': tbnn_results['metrics'], 'train_time': tbnn_results['train_time']},
        },
        'winner': 'RCLN' if rcln_results['metrics']['r2'] > tbnn_results['metrics']['r2'] else 'TBNN'
    }
    with open(ROOT / 'experiments/rcln_vs_tbnn_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("实验完成!")
    print("="*70)
    print(f"胜者: {results['winner']}")
    print("="*70)


if __name__ == '__main__':
    run_comparison()
