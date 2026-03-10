"""
JHTDB FNO-RCLN Mini Test - 最小化测试版本
分辨率: 16^3, 参数量: ~50K
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

print("=" * 60)
print("JHTDB FNO-RCLN Mini Test (16^3, ~50K params)")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

from axiom_os.layers.fno3d import FNO3d


class MiniFNO_RCLN(nn.Module):
    """最小化FNO-RCLN模型"""
    def __init__(self, width=8, modes=4):
        super().__init__()
        self.lambda_res = 0.3
        
        # 简化的Hard Core: 直接输出0，让模型学习
        self.nu = 0.001
        
        # Mini FNO
        self.fno = FNO3d(
            in_channels=3,
            out_channels=6,
            width=width,
            modes1=modes,
            modes2=modes,
            modes3=modes
        )
    
    def forward(self, u):
        # 简化为纯Soft Shell + 小残差
        return self.lambda_res * self.fno(u)


def generate_turbulence(batch=1, res=16):
    """生成合成湍流 - 简化版"""
    u = torch.randn(batch, 3, res, res, res, device=device)
    # 简单归一化即可，无需复杂频谱操作
    return u / (u.std() + 1e-8)


def compute_sgs_simple(u):
    """简化SGS应力计算"""
    B, C, D, H, W = u.shape
    # 简单平均滤波
    kernel = 3
    pad = kernel // 2
    
    # 使用平均池化
    u_avg = torch.zeros_like(u)
    for i in range(D):
        i_s = max(0, i-pad)
        i_e = min(D, i+pad+1)
        for j in range(H):
            j_s = max(0, j-pad)
            j_e = min(H, j+pad+1)
            for k in range(W):
                k_s = max(0, k-pad)
                k_e = min(W, k+pad+1)
                u_avg[:, :, i, j, k] = u[:, :, i_s:i_e, j_s:j_e, k_s:k_e].mean(dim=(2,3,4))
    
    tau = torch.zeros(B, 6, D, H, W, device=u.device)
    tau[:, 0] = u[:, 0]**2 - u_avg[:, 0]**2
    tau[:, 1] = u[:, 1]**2 - u_avg[:, 1]**2
    tau[:, 2] = u[:, 2]**2 - u_avg[:, 2]**2
    tau[:, 3] = u[:, 0]*u[:, 1] - u_avg[:, 0]*u_avg[:, 1]
    tau[:, 4] = u[:, 0]*u[:, 2] - u_avg[:, 0]*u_avg[:, 2]
    tau[:, 5] = u[:, 1]*u[:, 2] - u_avg[:, 1]*u_avg[:, 2]
    return tau


# 超参数
RES = 16
BATCH = 2
WIDTH = 8
MODES = 4
EPOCHS = 3
SAMPLES = 5

print(f"\n[Config]")
print(f"  Resolution: {RES}^3")
print(f"  Batch: {BATCH}")
print(f"  FNO width: {WIDTH}, modes: {MODES}")
print(f"  Epochs: {EPOCHS}, Samples/epoch: {SAMPLES}")

# 创建模型
model = MiniFNO_RCLN(WIDTH, MODES).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"\n[Model] Parameters: {n_params:,}")

# 诊断
print(f"\n[Diagnostics]")
u_test = generate_turbulence(1, RES)
tau_test = compute_sgs_simple(u_test)
tau_pred = model(u_test)

print(f"  Input u: mean={u_test.mean():.4f}, std={u_test.std():.4f}")
print(f"  Target tau: mean={tau_test.mean():.4f}, std={tau_test.std():.4f}")
print(f"  Target range: [{tau_test.min():.4f}, {tau_test.max():.4f}]")
print(f"  Pred tau: mean={tau_pred.mean():.4f}, std={tau_pred.std():.4f}")

# 训练
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

print(f"\n[Training]")
for epoch in range(EPOCHS):
    losses = []
    for i in range(SAMPLES):
        u = generate_turbulence(BATCH, RES)
        target = compute_sgs_simple(u)
        
        pred = model(u)
        loss = criterion(pred, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    avg_loss = np.mean(losses)
    print(f"  Epoch {epoch+1}/{EPOCHS}: Loss = {avg_loss:.6f}")

print("\n" + "=" * 60)
print(f"Done! Final Loss: {avg_loss:.6f}")
print("=" * 60)
