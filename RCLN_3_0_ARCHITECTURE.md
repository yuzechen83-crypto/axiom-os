# RCLN 3.0 完整架构文档
## Axiom-OS 物理-AI 终极进化

---

## 架构总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RCLN 3.0 - TRIPLE EVOLUTION                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ LAYER 1: SOFT SHELL (Brain Evolution - 4 Paths)                        ││
│  ├─────────────────────────────────────────────────────────────────────────┤│
│  │                                                                         ││
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   ││
│  │   │  FNO-RCLN   │  │ Clifford-   │  │  KAN-RCLN   │  │  Mamba-     │   ││
│  │   │  (Operator) │  │ RCLN        │  │  (Symbolic) │  │  RCLN       │   ││
│  │   │             │  │ (Geometric) │  │             │  │ (Sequential)│   ││
│  │   │ • Spectral  │  │             │  │ • B-splines │  │             │   ││
│  │   │   operators │  │ • Geometric │  │ • Direct    │  │ • State     │   ││
│  │   │ • Global    │  │   algebra   │  │   formula   │  │   Space     │   ││
│  │   │   Fourier   │  │ • O(3)      │  │   extract   │  │   Model     │   ││
│  │   │   modes     │  │   equivariant│  │ • Learnable │  │ • Linear    │   ││
│  │   │             │  │             │  │   activation│  │   O(L)      │   ││
│  │   │ Best:       │  │ Best:       │  │             │  │             │   ││
│  │   │ Fluids,     │  │ Robotics,   │  │ Best:       │  │ Best:       │   ││
│  │   │ Weather     │  │ EM Fields   │  │ Discovery,  │  │ Long seq,   │   ││
│  │   │             │  │             │  │ Control     │  │ Real-time   │   ││
│  │   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   ││
│  │                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                    ↕                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ LAYER 2: HARD CORE (Body Evolution - 3 Levels)                         ││
│  ├─────────────────────────────────────────────────────────────────────────┤│
│  │                                                                         ││
│  │  Level 1: EMPTY                 Level 2: CRYSTALLIZED                   ││
│  │  • No physics                   • Discovered formulas                   ││
│  │  • Pure ML                      • F = -cv, drag laws                    ││
│  │  • Data-driven only             • Battery aging curves                  ││
│  │                                 • Written by KAN discovery              ││
│  │                                                                         ││
│  │  Level 3: DIFFERENTIABLE                                                ││
│  │  • JAX/PyTorch compatible                                               ││
│  │  • Gradients flow through physics                                       ││
│  │  • Learnable parameters (friction, mass)                                ││
│  │  • System identification from data                                      ││
│  │                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                    ↕                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ LAYER 3: COUPLING (Interface Evolution)                                ││
│  ├─────────────────────────────────────────────────────────────────────────┤│
│  │                                                                         ││
│  │  RCLN v1.0 (Classical)    y = y_hard + λ·y_soft                         ││
│  │                           • Simple addition                             ││
│  │                           • No thermodynamic structure                  ││
│  │                                                                         ││
│  │  RCLN v2.0 (Architecture) y = SoftShell(x; architecture)                ││
│  │                           • FNO/Clifford/KAN/Mamba                      ││
│  │                           • Architecture selection                      ││
│  │                                                                         ││
│  │  RCLN v3.0 (Thermodynamic) ż = L(z)∇E(z) + M(z)∇S(z)   ★ NEW ★        ││
│  │                           • GENERIC framework                           ││
│  │                           • Energy conservation                         ││
│  │                           • Entropy production                          ││
│  │                           • Second law guaranteed                       ││
│  │                                                                         ││
│  │  Where:                                                                 ││
│  │    L(z) = Poisson matrix (antisymmetric) → Reversible dynamics          ││
│  │    E(z) = Energy (Hamiltonian) → From Hard Core                        ││
│  │    M(z) = Friction matrix (symmetric PSD) → Irreversible dynamics       ││
│  │    S(z) = Entropy → From Soft Shell                                    ││
│  │                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 核心方程

### RCLN v1.0 (古典)
```
y = F_hard(x) + λ·F_soft(x)

问题：
  - y_hard 和 y_soft 没有物理关系
  - 可能导致能量不守恒
  - 没有热力学一致性
```

### RCLN v3.0 (进化)
```
ż = L(z)∇E(z) + M(z)∇S(z)

其中：
  ż       = 状态变化率 (时间导数)
  L(z)    = 泊松矩阵 (反对称: L^T = -L)
  ∇E(z)   = 能量梯度 (来自硬核心)
  M(z)    = 摩擦矩阵 (对称半正定)
  ∇S(z)   = 熵梯度 (来自软壳)

性质：
  1. 能量守恒: dE/dt = ∇E · L∇E = 0 (因为 L 反对称)
  2. 熵增原理: dS/dt = ∇S · M∇S ≥ 0 (因为 M 半正定)
  3. 退化条件: L·∇S = 0, M·∇E = 0
```

---

## 物理意义

### 可逆部分: L(z)∇E(z)
- **来源**: 硬核心 (Hard Core)
- **物理**: 哈密顿动力学，能量守恒
- **例子**: 行星轨道、理想摆、无摩擦运动
- **数学**: 辛几何，泊松括号

### 不可逆部分: M(z)∇S(z)
- **来源**: 软壳 (Soft Shell)
- **物理**: 耗散过程，熵增
- **例子**: 摩擦、湍流、热传导
- **数学**: 梯度流，最大熵产生原理

### 耦合
两种动力学通过**状态变量 z**耦合：
- 可逆部分驱动**振荡**（能量转换）
- 不可逆部分驱动**弛豫**（趋向平衡）

---

## 代码示例

### 基础使用
```python
from axiom_os.core import RCLN3, create_rcln3

# RCLN v3.0 with GENERIC coupling
rcln = create_rcln3(
    state_dim=4,
    mode="kan",
    use_generic=True,  # Enable GENERIC coupling
)

# Forward pass
z = torch.randn(10, 4)  # State [q1, q2, p1, p2]
z_dot, info = rcln(z, return_thermodynamics=True)

# Check thermodynamic consistency
print(f"Energy: {info['energy'].mean():.4f}")
print(f"dE/dt: {info['dE_dt'].mean():.6f}")  # Should be ~0
print(f"Entropy production: {info['dS_dt'].mean():.6f}")  # Should be > 0
```

### 时间积分
```python
# Rollout trajectory
z0 = torch.randn(1, 4)
trajectory = rcln.rollout(z0, n_steps=1000, dt=0.01)
# trajectory shape: (1, 1001, 4)

# Check energy conservation
energies = [rcln.coupling.compute_energy(z).item() 
            for z in trajectory[0]]
energy_drift = energies[-1] - energies[0]
print(f"Energy drift over 1000 steps: {energy_drift:.6f}")
```

### 软壳发现 + 硬核心结晶
```python
# Train on data
optimizer = torch.optim.Adam(rcln.parameters(), lr=0.01)
for epoch in range(1000):
    optimizer.zero_grad()
    z_dot_pred = rcln(z_train)
    loss = F.mse_loss(z_dot_pred, z_dot_true)
    loss.backward()
    optimizer.step()

# Crystallize discovered formula into Hard Core
rcln.crystallize(formula_name="learned_friction")

# Now Hard Core has Level 2 knowledge
print(rcln.hard_core.get_info())
```

### 系统识别 (Level 3)
```python
# Add learnable physical parameters
rcln.hard_core.add_learnable_parameter(
    name='friction_coef',
    shape=(),
    initial_value=0.1,
    constraint=(0.0, 1.0),
)

# Learn from trajectory data
learned_params = rcln.hard_core.learn_from_data(
    trajectories=observed_data,
    solver=rcln.solver,
    n_epochs=100,
)

print(f"Learned friction: {learned_params['friction_coef']:.4f}")
```

---

## 三层进化的关系

### 类比：人类学习

```
软壳 (Soft Shell) = 大脑皮层
  • 学习新知识
  • 模式识别
  • 直觉/近似

硬核心 (Hard Core) = 小脑/脊髓
  • 自动化技能
  • 本能反应
  • 已固化的知识

耦合 (Coupling) = 神经可塑性机制
  • 短期记忆 → 长期记忆
  •  crystallization 过程
  • 意识 ↔ 潜意识的交互
```

### 训练过程

```
阶段 1: 空白期
  软壳: 随机初始化 (KAN/Mamba)
  硬核心: Level 1 (Empty)
  输出: ż = M(z)∇S(z) (纯软壳)

阶段 2: 学习期
  软壳: 拟合数据，发现模式
  硬核心: Level 2 (Crystallization)
  输出: ż = L∇E + M∇S

阶段 3: 固化期
  软壳: 残差修正 (小修正)
  硬核心: Level 3 (Differentiable)
  输出: ż = L∇E + M∇S (E,S 可学习)

阶段 4: 掌握期
  软壳: 最小活动 (Activity Monitor → 0)
  硬核心: 完整物理引擎
  输出: ż ≈ L∇E (纯硬核心)
```

---

## 文件结构

```
axiom_os/core/
├── rcln_v3.py                   # RCLN 3.0 主实现
├── generic_coupling.py          # GENERIC 耦合框架
├── differentiable_physics.py    # 可微物理求解器
├── hard_core_manager.py         # 硬核心管理器
└── __init__.py                  # 导出 RCLN 3.0
```

---

## 关键组件

### 1. GENERICLayer
```python
from axiom_os.core import GENERICLayer

# Create thermodynamic coupling
generic = GENERICLayer(
    state_dim=4,
    energy_fn=hamiltonian,      # From Hard Core
    poisson_matrix_fn=poisson,  # From Hard Core
    entropy_fn=entropy_net,     # From Soft Shell
    friction_matrix_fn=friction_net,  # From Soft Shell
)

# Forward: ż = L∇E + M∇S
z_dot, info = generic(z, return_thermodynamics=True)
```

### 2. HardCoreManager
```python
from axiom_os.core import HardCoreManager, HardCoreLevel

# Create and evolve
hcm = HardCoreManager(initial_level=HardCoreLevel.BASIC)

# Level 2: Crystallize discovered formula
hcm.crystallize_formula(formula)

# Level 3: Add learnable parameters
hcm.add_learnable_parameter('friction', (), 0.1, (0, 1))
```

### 3. DifferentiablePhysicsSolver
```python
from axiom_os.core import create_solver, SolverLevel

# Level 4 solver
solver = create_solver(
    dynamics_fn=my_physics,
    dt=0.01,
    level='differentiable',  # Full gradient support
)

# Gradients flow through physics!
next_state = solver(state, control, params)
loss = compute_loss(next_state)
loss.backward()  # Gradients through physics!
```

---

## 与传统方法的对比

| 特性 | 传统 PINN | RCLN v2.0 | RCLN v3.0 |
|------|-----------|-----------|-----------|
| **物理嵌入** | 损失函数惩罚 | 加法耦合 | GENERIC结构 |
| **守恒律** | 近似满足 | 可能违反 | 严格保证 |
| **热力学** | 无 | 无 | 内置 |
| **软硬分离** | 无 | 有 | 有+交互 |
| **可解释性** | 低 | 中 (KAN) | 高 (公式+热力学) |
| **发现能力** | 无 | 有 (KAN) | 有+自动结晶 |
| **学习物理** | 无 | 无 | 有 (Level 3) |

---

## 下一步工作

1. **验证热力学一致性**
   - 测试能量守恒误差
   - 验证熵产非负
   - 长时间稳定性测试

2. **扩展物理域**
   - 电磁学 (Maxwell方程)
   - 流体 (Navier-Stokes)
   - 相对论动力学

3. **自动化发现**
   - KAN → Hard Core 自动结晶
   - Activity Monitor 触发结晶
   - 知识库自动扩展

---

## 参考文献

1. GENERIC Framework:
   - Grmela, M., & Öttinger, H. C. (1997). "Dynamics and thermodynamics of complex fluids."

2. KAN Networks:
   - Liu, Z., et al. (2024). "KAN: Kolmogorov-Arnold Networks."

3. Symplectic Integration:
   - Hairer, E., et al. "Geometric Numerical Integration."

4. Differentiable Physics:
   - Hu, Y., et al. "DiffTaichi: Differentiable Programming for Physical Simulation."

---

**RCLN 3.0: The Thermodynamic Evolution of Physics-AI**

From simple addition to GENERIC structure.
From static physics to evolving knowledge.
From black box to thermodynamically consistent white box.
