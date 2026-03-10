# -*- coding: utf-8 -*-
"""
测试 RCLN v3 与 Hard Core 3.0 的集成
使用简化的 Hard Core 实现
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn

print("=" * 70)
print("RCLN v3 with Hard Core 3.0 - Simple Test")
print("=" * 70)

from axiom_os.layers.rcln import RCLNLayer
from axiom_os.layers.fno3d import FNO3d


class SimpleTurbulenceHardCore(nn.Module):
    """
    简化的湍流 Hard Core
    直接计算速度梯度作为 SGS 应力的近似
    """
    def __init__(self, nu=0.001, scale=0.01):
        super().__init__()
        self.nu = nn.Parameter(torch.tensor(nu))
        self.scale = scale
    
    def forward(self, velocity):
        """
        velocity: [B, 3, D, H, W]
        return: tau [B, 6, D, H, W]
        """
        B, C, D, H, W = velocity.shape
        dx = 1.0 / D
        
        # 计算 Laplacian（粘性项）
        def laplacian(f):
            f_pad = torch.nn.functional.pad(f, [1,1,1,1,1,1], mode='replicate')
            return (
                f_pad[:, 2:, 1:-1, 1:-1] + f_pad[:, :-2, 1:-1, 1:-1] +
                f_pad[:, 1:-1, 2:, 1:-1] + f_pad[:, 1:-1, :-2, 1:-1] +
                f_pad[:, 1:-1, 1:-1, 2:] + f_pad[:, 1:-1, 1:-1, :-2] -
                6 * f_pad[:, 1:-1, 1:-1, 1:-1]
            ) / (dx**2)
        
        u, v, w = velocity[:, 0], velocity[:, 1], velocity[:, 2]
        
        lap_u = laplacian(u)
        lap_v = laplacian(v)
        lap_w = laplacian(w)
        
        # 构建 SGS 应力张量（6个独立分量）
        tau = torch.zeros(B, 6, D, H, W, device=velocity.device)
        tau[:, 0] = self.nu * lap_u * self.scale  # xx
        tau[:, 1] = self.nu * lap_v * self.scale  # yy
        tau[:, 2] = self.nu * lap_w * self.scale  # zz
        tau[:, 3] = self.nu * (lap_u + lap_v) * self.scale * 0.5  # xy
        tau[:, 4] = self.nu * (lap_u + lap_w) * self.scale * 0.5  # xz
        tau[:, 5] = self.nu * (lap_v + lap_w) * self.scale * 0.5  # yz
        
        return tau


# 创建 RCLN v3 模型
print("\n[1] Creating RCLN v3 model with Hard Core 3.0...")

hard_core = SimpleTurbulenceHardCore(nu=0.001, scale=0.01)

model = RCLNLayer(
    input_dim=3,
    hidden_dim=8,
    output_dim=6,
    net_type='fno3d',
    hard_core_func=hard_core,  # Hard Core 3.0
    lambda_res=0.5,
    fno_modes1=4,
    fno_modes2=4,
    fno_modes3=4,
    fno_layers=2,
)

print(f"[OK] Model created")
print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  - Hard Core parameters: {sum(p.numel() for p in hard_core.parameters()):,}")
print(f"  - nu is learnable: {hard_core.nu.requires_grad}")

# 查看架构信息
info = model.get_architecture_info()
print(f"\n[2] Architecture Info:")
for key, value in info.items():
    print(f"  - {key}: {value}")

# 前向传播测试
print(f"\n[3] Forward pass test...")
x = torch.randn(2, 3, 16, 16, 16)
print(f"  Input: {x.shape}")

y = model(x)
print(f"  Output: {y.shape}")
print(f"[OK] Forward pass successful")

# 梯度测试
print(f"\n[4] Gradient test...")
target = torch.randn_like(y)
loss = nn.MSELoss()(y, target)
print(f"  Loss: {loss.item():.4f}")

loss.backward()
print(f"[OK] Backward pass successful")

# 检查梯度
print(f"\n[5] Gradient statistics:")
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"  - {name}: grad_norm={grad_norm:.6f}")

# 验证 nu 被更新
print(f"\n[6] Parameter update test...")
old_nu = hard_core.nu.item()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer.step()
new_nu = hard_core.nu.item()
print(f"  nu before: {old_nu:.6f}")
print(f"  nu after: {new_nu:.6f}")
print(f"  Changed: {old_nu != new_nu}")

print("\n" + "=" * 70)
print("[SUCCESS] RCLN v3 with Hard Core 3.0 test complete!")
print("=" * 70)
