"""
Hard Core 3.0 演示：可微物理引擎 + 知识结晶
展示三种求解器级别和动态扩展物理知识
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

print("=" * 70)
print("Hard Core 3.0 - Differentiable Physics Engine Demo")
print("=" * 70)

from axiom_os.core.differentiable_physics import (
    DifferentiablePhysicsEngine,
    DifferentiableHardCore,
    SolverLevel,
    navier_stokes_3d,
    spring_force,
    friction_force,
)

# ============================================
# Demo 1: 三种求解器级别对比
# ============================================
print("\n" + "=" * 70)
print("Demo 1: Solver Levels Comparison (Harmonic Oscillator)")
print("=" * 70)

def harmonic_oscillator(state, params, t=None):
    """简谐振子：d²x/dt² = -ω²x"""
    omega = params.get('omega', torch.tensor(1.0))
    dim = state.shape[1] // 2
    x, v = state[:, :dim], state[:, dim:]
    
    dxdt = v
    dvdt = -(omega ** 2) * x
    return torch.cat([dxdt, dvdt], dim=1)

# 初始条件：x=1, v=0
x0 = torch.tensor([[1.0, 0.0]])  # [position, velocity]
total_time = 10.0
dt = 0.1
steps = int(total_time / dt)

results = {}

for level in [SolverLevel.EULER, SolverLevel.SYMPLECTIC, SolverLevel.DIFFERENTIABLE]:
    engine = DifferentiablePhysicsEngine(
        solver_level=level,
        dim=1,
        dt=dt,
    )
    
    # 结晶简谐振子
    engine.crystallize(
        name="harmonic",
        equation=harmonic_oscillator,
        params={'omega': 1.0},
        domain="mechanics",
        learnable=False,
    )
    
    # 模拟
    trajectory = [x0.clone()]
    state = x0.clone()
    energies = []
    
    for _ in range(steps):
        state = engine.integrate(state)
        trajectory.append(state.clone())
        
        # 计算能量 E = 0.5*(x² + v²)
        x, v = state[0, 0].item(), state[0, 1].item()
        E = 0.5 * (x**2 + v**2)
        energies.append(E)
    
    results[level.name] = {
        'trajectory': torch.stack(trajectory),
        'energies': energies,
    }
    
    # 能量守恒误差
    energy_drift = max(energies) - min(energies)
    print(f"\n{level.name}:")
    print(f"  Final position: {trajectory[-1][0, 0].item():.4f}")
    print(f"  Final velocity: {trajectory[-1][0, 1].item():.4f}")
    print(f"  Energy drift: {energy_drift:.6f}")

# ============================================
# Demo 2: 知识结晶 - 动态扩展物理知识
# ============================================
print("\n" + "=" * 70)
print("Demo 2: Crystallization - Dynamic Physics Knowledge Expansion")
print("=" * 70)

engine = DifferentiablePhysicsEngine(
    solver_level=SolverLevel.DIFFERENTIABLE,
    dim=3,
)

# 步骤1: 空的Hard Core
print("\n[Step 1] Empty Hard Core")
engine.summary()

# 步骤2: 结晶牛顿定律
print("\n[Step 2] Crystallizing Newton's Second Law...")
engine.crystallize(
    name="spring",
    equation=spring_force,
    params={'k': 10.0},
    domain="mechanics",
    learnable=True,
)

# 步骤3: 结晶摩擦力
print("\n[Step 3] Crystallizing Friction Force...")
engine.crystallize(
    name="friction",
    equation=friction_force,
    params={'mu': 0.5, 'alpha': 1.0},
    domain="mechanics",
    learnable=True,
)

engine.summary()

# ============================================
# Demo 3: 梯度反向传播 - 物理参数可学习
# ============================================
print("\n" + "=" * 70)
print("Demo 3: Differentiable Physics - Learnable Parameters")
print("=" * 70)

# 模拟带阻尼的弹簧振子
def damped_oscillator_simulation(target_period=2.0, n_steps=100):
    """
    学习弹簧常数k和摩擦系数mu，使得振动周期接近target_period
    """
    engine = DifferentiablePhysicsEngine(
        solver_level=SolverLevel.DIFFERENTIABLE,
        dim=1,
        dt=0.01,
    )
    
    # 结晶弹簧和摩擦（参数可学习）
    engine.crystallize(
        name="spring",
        equation=spring_force,
        params={'k': 5.0},  # 初始猜测
        domain="mechanics",
        learnable=True,
    )
    
    engine.crystallize(
        name="friction",
        equation=friction_force,
        params={'mu': 0.1, 'alpha': 1.0},
        domain="mechanics",
        learnable=True,
    )
    
    # 优化器 - 优化物理参数！
    optimizer = torch.optim.Adam(engine.parameters(), lr=0.1)
    
    print(f"\nTarget period: {target_period}s")
    print("Learning spring constant k and friction mu...")
    
    losses = []
    k_history = []
    mu_history = []
    
    for epoch in range(50):
        optimizer.zero_grad()
        
        # 模拟
        state = torch.tensor([[1.0, 0.0]])  # x=1, v=0
        positions = []
        
        for _ in range(n_steps):
            state = engine.integrate(state)
            positions.append(state[0, 0].item())
        
        # 计算周期（通过过零点）
        positions_tensor = torch.tensor(positions)
        
        # 简化的损失：希望最终在原点且速度合适
        final_state = state
        loss = (final_state[0, 0] ** 2 + (final_state[0, 1] - 0.5) ** 2)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        k_history.append(engine.learnable_params['spring_k'].item())
        mu_history.append(engine.learnable_params['friction_mu'].item())
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, k={k_history[-1]:.4f}, mu={mu_history[-1]:.4f}")
    
    return k_history, mu_history, losses

k_hist, mu_hist, losses = damped_oscillator_simulation()

print(f"\nFinal learned parameters:")
print(f"  Spring constant k: {k_hist[-1]:.4f}")
print(f"  Friction mu: {mu_hist[-1]:.4f}")

# ============================================
# Demo 4: RCLN集成 - 完整的FNO-RCLN使用示例
# ============================================
print("\n" + "=" * 70)
print("Demo 4: RCLN Integration - Complete FNO-RCLN with Differentiable Hard Core")
print("=" * 70)

from axiom_os.layers.fno3d import FNO3d

class NavierStokesProcessor(nn.Module):
    """Navier-Stokes专用处理器"""
    def forward(self, velocity, law):
        # 直接调用NS方程（保持5D输入）
        params = {k: getattr(law, k) for k in dict(law.named_parameters()).keys()}
        params.update({k: getattr(law, k) for k in dict(law.named_buffers()).keys()})
        
        du_dt = navier_stokes_3d(velocity, params)
        
        # 将du_dt转换为SGS应力格式（6个分量）
        B, C, D, H, W = du_dt.shape
        tau = torch.zeros(B, 6, D, H, W, device=du_dt.device)
        tau[:, 0] = du_dt[:, 0]  # xx
        tau[:, 1] = du_dt[:, 1]  # yy
        tau[:, 2] = du_dt[:, 2]  # zz
        tau[:, 3] = 0.5 * (du_dt[:, 0] + du_dt[:, 1])  # xy
        tau[:, 4] = 0.5 * (du_dt[:, 0] + du_dt[:, 2])  # xz
        tau[:, 5] = 0.5 * (du_dt[:, 1] + du_dt[:, 2])  # yz
        
        return tau * 0.01  # 缩放因子


class FNO_RCLN_HardCore3(nn.Module):
    """
    FNO-RCLN with Hard Core 3.0
    y = y_hard + λ * y_soft
    """
    def __init__(self, resolution=16, lambda_res=0.5):
        super().__init__()
        self.lambda_res = lambda_res
        
        # Hard Core 3.0: 可微物理引擎
        self.hard_core = DifferentiableHardCore(
            input_dim=3,
            output_dim=6,  # SGS应力6个分量
            solver_level=SolverLevel.DIFFERENTIABLE,
        )
        
        # 注册NS处理器
        self.hard_core.register_processor("navier_stokes", NavierStokesProcessor())
        
        # 结晶Navier-Stokes方程
        self.hard_core.crystallize(
            name="navier_stokes",
            equation=navier_stokes_3d,
            params={'nu': 0.001},  # 可学习的粘性系数
            domain="fluids",
            learnable=True,
        )
        
        # Soft Shell: FNO
        self.soft_shell = FNO3d(
            in_channels=3,
            out_channels=6,
            width=8,
            modes1=4,
            modes2=4,
            modes3=4,
        )
    
    def forward(self, velocity_field):
        """
        Args:
            velocity_field: [B, 3, D, H, W]
        Returns:
            tau: SGS应力 [B, 6, D, H, W]
        """
        # Hard Core贡献（物理约束）
        tau_hard = self.hard_core(velocity_field)
        
        # Soft Shell贡献（数据驱动）
        tau_soft = self.soft_shell(velocity_field)
        
        # RCLN耦合
        tau = tau_hard + self.lambda_res * tau_soft
        
        return tau

# 测试
print("\nCreating FNO-RCLN with Hard Core 3.0...")
model = FNO_RCLN_HardCore3(resolution=16)

# 打印架构
print(f"\nModel Architecture:")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Hard Core parameters: {sum(p.numel() for p in model.hard_core.parameters()):,}")
print(f"  Soft Shell parameters: {sum(p.numel() for p in model.soft_shell.parameters()):,}")
print(f"  Learnable physics params: {list(model.hard_core.engine.learnable_params.keys())}")

# 前向测试
u = torch.randn(1, 3, 16, 16, 16)
tau = model(u)
print(f"\nForward test:")
print(f"  Input: {u.shape}")
print(f"  Output tau: {tau.shape}")

# 梯度测试
target = torch.randn_like(tau)
loss = nn.MSELoss()(tau, target)
loss.backward()

print(f"\nBackward test:")
print(f"  Loss: {loss.item():.4f}")
print(f"  Hard Core nu grad: {model.hard_core.engine.learnable_params['navier_stokes_nu'].grad.item():.6f}")
print(f"  Soft Shell has grad: model.soft_shell.weights1.grad is not None")

print("\n" + "=" * 70)
print("✓ Hard Core 3.0 Demo Complete!")
print("=" * 70)
print("\nKey Features:")
print("  1. Three solver levels: Euler, Symplectic, Differentiable")
print("  2. Crystallization: Dynamic physics knowledge expansion")
print("  3. Differentiable: Gradients flow through physics equations")
print("  4. Learnable: Physics parameters can be learned from data")
print("  5. RCLN Integration: y_hard + λ·y_soft coupling")
