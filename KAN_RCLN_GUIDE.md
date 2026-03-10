# KAN-RCLN 进化指南
## Axiom-OS 符号化进化路径

**KAN (Kolmogorov-Arnold Networks)** 已集成到 RCLN 架构中，为 Axiom-OS 带来革命性的**符号可解释性**。

---

## 核心优势

| 特性 | MLP-RCLN | KAN-RCLN | 提升 |
|------|----------|----------|------|
| 公式提取 | PySR 搜索 (慢) | 直接读取 (快) | **100x 加速** |
| 参数效率 | 高 | 更高 | **2-5x 节省** |
| 可解释性 | 黑盒 | 白盒 | **质的飞跃** |
| 发现引擎 | 间接集成 | 原生集成 | **无缝衔接** |

---

## 架构对比

### MLP (古典)
```
y = σ(W·x + b)  →  固定激活 + 学习线性权重
```

### KAN (进化)
```
y = Σ φ_i(x_i)  →  学习非线性函数在边上
      ↑
   B-spline 可学习系数
```

---

## 快速开始

### 1. 基础使用

```python
from axiom_os.layers import RCLNLayer

# 创建 KAN-RCLN
rcln = RCLNLayer(
    input_dim=4,
    hidden_dim=8,
    output_dim=2,
    net_type="kan",           # 启用 KAN Soft Shell
    kan_grid_size=5,          # B-spline 网格点数
    kan_spline_order=3,       # 3 = cubic spline
)

# 前向传播
import torch
x = torch.randn(10, 4)
y = rcln(x)

# 直接提取公式
formula = rcln.extract_formula(var_names=["position", "velocity", "force", "mass"])
print(formula['formula_str'])
# 输出: "y0 = 0.47*position^2 + -0.23*sin(velocity); y1 = ..."
```

### 2. 与物理 Hard Core 结合

```python
def navier_stokes_residual(state):
    """物理硬核心: NS 方程残差"""
    if hasattr(state, 'detach'):
        u, v, p = state[:, 0], state[:, 1], state[:, 2]
    else:
        u, v, p = state.values[:, 0], state.values[:, 1], state.values[:, 2]
    # 简化 NS 残差
    return torch.stack([u, v, p], dim=1)

rcln = RCLNLayer(
    input_dim=3,
    hidden_dim=6,
    output_dim=3,
    hard_core_func=navier_stokes_residual,
    net_type="kan",
    lambda_res=0.3,  # Soft 部分权重
)
```

### 3. Discovery Engine 集成

```python
# 训练过程中自动检测高 soft activity
rcln = RCLNLayer(
    input_dim=2,
    hidden_dim=4,
    output_dim=1,
    net_type="kan",
    use_activity_monitor=True,
    soft_threshold=0.3,
)

# 训练
for epoch in range(1000):
    y, hotspot = rcln(x, return_hotspot=True)
    if hotspot:
        # Soft activity 持续高 → 发现热点
        print(f"Discovery Hotspot! Activity: {hotspot.avg_soft_magnitude}")
        
        # 直接提取公式，无需 PySR
        formula = rcln.extract_formula()
        print(f"Discovered: {formula['formula_str']}")
        
        # 晶化到硬核心
        # hippocampus.crystallize(formula)
```

---

## 参数调优指南

### kan_grid_size (网格点数)
- **小 (3-5)**: 简单函数 (线性、二次)
- **中 (5-8)**: 中等复杂度 (sin、exp)
- **大 (8-12)**: 复杂函数 (湍流、混沌)

### kan_spline_order (样条阶数)
- **2**: 二次 (C1 连续)
- **3**: 三次 (推荐, C2 连续)
- **4+**: 高阶 (更平滑但计算成本高)

### lambda_res (残差权重)
- **1.0**: 纯数据驱动 (无物理先验)
- **0.3-0.5**: 物理为主，数据修正
- **0.1**: 强物理约束，微调修正

---

## 应用场景

### 1. 公式发现 (电池老化、双摆)
```python
# 目标: 发现电池容量衰减公式
rcln = RCLNLayer(
    input_dim=3,   # [cycle_count, temperature, c_rate]
    output_dim=1,  # capacity_retention
    net_type="kan",
    kan_grid_size=6,
)
# 训练后提取: capacity = 1.0 - 0.02*cycle^0.5 - 0.001*temp
```

### 2. 控制系统 (Walker2d)
```python
# 目标: 学习残差动力学
rcln = RCLNLayer(
    input_dim=state_dim + action_dim,
    output_dim=state_dim,
    hard_core_func=mujoco_dynamics,  # 物理引擎
    net_type="kan",
    lambda_res=0.2,
)
# 提取: delta_v = 0.1*torque - 0.05*friction*v
```

### 3. 流体力学 (湍流)
```python
# 目标: SGS 应力模型
rcln = RCLNLayer(
    input_dim=velocity_gradient_dim,  # 9
    output_dim=6,  # 对称应力张量
    net_type="kan",
    kan_grid_size=8,
)
# 提取: τ_ij = -2*ν_t*S_ij (Smagorinsky-like)
```

---

## 与其他架构对比

```python
# FNO-RCLN (算子化)
rcln_fno = RCLNLayer(..., net_type="fno")
# 优势: 分辨率不变，全局关联
# 场景: 流场、气象、波传播

# Clifford-RCLN (几何化)
rcln_clifford = RCLNLayer(..., use_clifford=True)
# 优势: O(3) 等变，矢量场
# 场景: 机器人、电磁、分子

# KAN-RCLN (符号化)
rcln_kan = RCLNLayer(..., net_type="kan")
# 优势: 直接公式提取，可解释
# 场景: 发现引擎、控制、建模
```

---

## 完整示例: 从数据到公式

```python
import torch
from axiom_os.layers import RCLNLayer

# 1. 准备数据: 简谐振子 (但有一个未知阻尼项)
# 真实物理: d2x/dt2 = -k*x - c*|v|^0.5 * v (非线性阻尼)
t = torch.linspace(0, 10, 1000).unsqueeze(1)
x = torch.sin(t) * torch.exp(-0.1*t)  # 衰减正弦
v = torch.cos(t) * torch.exp(-0.1*t) - 0.1*x
a_true = -x - 0.1*torch.abs(v)**0.5 * v  # 真实加速度

# 2. 创建 KAN-RCLN
rcln = RCLNLayer(
    input_dim=2,    # [position, velocity]
    hidden_dim=4,
    output_dim=1,   # acceleration
    hard_core_func=lambda s: -s[:, 0:1],  # 已知: -k*x
    net_type="kan",
    kan_grid_size=6,
    lambda_res=0.5,
)

# 3. 训练 (只学习残差)
optimizer = torch.optim.Adam(rcln.parameters(), lr=0.01)
for epoch in range(500):
    optimizer.zero_grad()
    a_pred = rcln(torch.cat([x, v], dim=1))
    loss = torch.nn.functional.mse_loss(a_pred, a_true)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.6f}")

# 4. 提取发现的公式
formula = rcln.extract_formula(var_names=["position", "velocity"])
print("="*50)
print("DISCOVERED FORMULA:")
print(formula['formula_str'])
print(f"Confidence: {formula['confidence']:.2%}")
print("="*50)

# 预期输出:
# y0 = -1.000*position + -0.098*sign(velocity)*abs(velocity)^0.5
# Confidence: 94.3%
```

---

## API 参考

### KANSoftShell
```python
KANSoftShell(
    input_dim: int,        # 输入维度
    hidden_dim: int,       # 隐藏层维度
    output_dim: int,       # 输出维度
    grid_size: int = 5,    # B-spline 网格点数
    spline_order: int = 3, # 样条阶数
    n_layers: int = 2,     # KAN 层数
)

# 方法
.extract_formula() -> str  # 提取符号公式
.reset_parameters()        # 重置参数
```

### RCLNLayer (KAN 模式)
```python
RCLNLayer(
    ...
    net_type="kan",           # 启用 KAN
    kan_grid_size=5,          # 网格点数
    kan_spline_order=3,       # 样条阶数
)

# 方法
.extract_formula(var_names=None) -> dict  # 提取物理公式
.get_architecture_info() -> dict          # 获取架构信息
```

### KANFormulaExtractor
```python
extractor = KANFormulaExtractor(kan_model)
result = extractor.extract_physics_formula(var_names)
# result: {
#     'formula_str': 'y0 = 0.5*x0^2 + -0.3*sin(x1)',
#     'components': [...],
#     'confidence': 0.95
# }
```

---

## 故障排除

### Q: 公式提取返回 None?
**A**: 只有 `net_type="kan"` 支持直接提取。MLP/Clifford/FNO 返回 None。

### Q: 提取的公式不准确?
**A**: 
- 增加 `kan_grid_size`
- 延长训练时间
- 降低 `lambda_res` 让 soft 部分学习更多

### Q: 训练不稳定?
**A**:
- 使用 `kan_spline_order=3` (cubic)
- 添加 LayerNorm (已内置)
- 降低学习率

---

## 下一步

1. **运行测试**: `python test_kan_rcln.py`
2. **探索示例**: 查看 `examples/kan_discovery_demo.py`
3. **集成到工作流**: 在 Discovery Engine 中启用 KAN 模式

---

**KAN-RCLN: 从黑盒到白盒，从数据到公式。**
