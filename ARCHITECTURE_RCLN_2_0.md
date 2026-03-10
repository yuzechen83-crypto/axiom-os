# RCLN 2.0 架构概览
## Axiom-OS 物理-AI 混合引擎

---

## 架构总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RCLN 2.0 - Residual Coupler Linking Neuron         │
│                                                                              │
│     y_total = y_hard + λ·y_soft                                              │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  HARD CORE (Physics)                                                  │  │
│  │  - Analytical equations (Navier-Stokes, Hamiltonian, etc.)            │  │
│  │  - Domain knowledge from physics                                      │  │
│  │  - Zero parameters (frozen)                                           │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    +                                         │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  SOFT SHELL (Neural) - Four Evolution Paths                           │  │
│  │                                                                       │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │  │
│  │  │   FNO-RCLN   │  │  Clifford-   │  │   KAN-RCLN   │  │  Mamba-   │ │  │
│  │  │  (Operator)  │  │   RCLN       │  │  (Symbolic)  │  │   RCLN    │ │  │
│  │  │              │  │  (Geometric) │  │              │  │(Sequential│ │  │
│  │  │  Spectral    │  │              │  │  B-splines   │  │    SSM)   │ │  │
│  │  │  conv in     │  │  Geometric   │  │  on edges    │  │           │ │  │
│  │  │  Fourier     │  │  algebra     │  │              │  │  Discrete │ │  │
│  │  │  space       │  │  multivector │  │  Direct      │  │  ODE      │ │  │
│  │  │              │  │              │  │  formula     │  │  state    │ │  │
│  │  │  Resolution  │  │  O(3)        │  │  extraction  │  │  space    │ │  │
│  │  │  invariant   │  │  equivariant │  │              │  │  model    │ │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └───────────┘ │  │
│  │                                                                       │  │
│  │  Fluids/Weather      Robotics/EM        Discovery/Control   Long      │  │
│  │  Wave propagation    Rotation           Formula extract     Sequences │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    × λ_res                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 核心公式

```
RCLN: y_total = y_hard + λ·y_soft

Where:
  y_hard = Physics equations (analytical, frozen)
  y_soft = Neural approximation (learned, data-driven)
  λ_res  = Residual weight (0 = pure physics, 1 = pure data)
```

---

## 进化方案对比

| 特性 | Spectral | Clifford | KAN | Mamba |
|------|----------|----------|-----|-------|
| **核心机制** | FFT → Weights → IFFT | Geometric Product | B-splines on edges | State Space Model |
| **物理特性** | Frequency domain | O(3) equivariant | Symbolic | Continuous-time |
| **复杂度** | O(n log n) | O(n) | O(n) | O(n) |
| **公式提取** | ❌ | ❌ | ✅ Direct | ❌ |
| **最佳场景** | Periodic physics | Rotational systems | Discovery | Long sequences |

---

## 文件结构

```
axiom_os/layers/
├── rcln.py                  # Main RCLN 2.0 implementation
├── rcln_v2.py              # Source copy
├── rcln_legacy.py          # Backup of old architecture
├── kan_layer.py            # Scheme 3: KAN implementation
├── mamba_layer.py          # Scheme 4: Mamba/SSM implementation
├── spectral.py             # Spectral convolution
├── fno.py                  # Fourier Neural Operator
├── clifford_nn.py          # Geometric algebra layers
├── clifford_transformer.py # Clifford + Attention
└── __init__.py             # Unified exports
```

---

## 使用方式

### 基础使用
```python
from axiom_os.layers import RCLNLayer

# Choose your evolution path explicitly
rcln = RCLNLayer(
    input_dim=4,
    hidden_dim=8, 
    output_dim=2,
    net_type="kan",  # "fno", "clifford", "kan", "mamba", "spectral"
)

y = rcln(x)  # Forward pass
```

### 带物理硬核心
```python
def physics_core(x):
    # Known physics: e.g., Hooke's law F = -kx
    return -0.5 * x

rcln = RCLNLayer(
    input_dim=4,
    hidden_dim=8,
    output_dim=4,
    net_type="kan",
    hard_core_func=physics_core,  # Analytical physics
    lambda_res=0.3,               # Soft contribution weight
)
```

### 工厂函数
```python
from axiom_os.layers import (
    create_fno_rcln,
    create_clifford_rcln, 
    create_kan_rcln,
    create_mamba_rcln,
)

rcln = create_kan_rcln(
    input_dim=4,
    hidden_dim=8,
    output_dim=2,
    grid_size=6,
    spline_order=3,
)
```

---

## 关键特性

### 1. Activity Monitor
```python
rcln = RCLNLayer(
    ..., 
    net_type="kan",
    use_activity_monitor=True,
    soft_threshold=0.5,
)

y, hotspot = rcln(x, return_hotspot=True)
if hotspot:
    print(f"Discovery hotspot detected!")
```

### 2. Formula Extraction (KAN only)
```python
rcln = RCLNLayer(..., net_type="kan")
# Train...
formula = rcln.extract_formula(var_names=["x", "v"])
# Returns: {"formula_str": "y = 0.5*x^2", "confidence": 0.95}
```

### 3. Architecture Info
```python
info = rcln.get_architecture_info()
# {
#     "net_type": "kan",
#     "input_dim": 4,
#     "hidden_dim": 8,
#     "output_dim": 2,
#     "lambda_res": 1.0,
#     "has_hard_core": False,
#     "evolution_path": "Kolmogorov-Arnold Network - Symbolic"
# }
```

---

## 选型指南

```
What is your application?
│
├─► Fluid dynamics / Weather / Waves
│   └── net_type="fno" (Fourier Neural Operator)
│
├─► Robotics / Drones / Electromagnetism / Molecular
│   └── net_type="clifford" (Geometric Algebra)
│
├─► Formula discovery / Control / Interpretability
│   └── net_type="kan" (Kolmogorov-Arnold Network)
│
├─► Long sequences / Real-time control / ODEs
│   └── net_type="mamba" (State Space Model)
│
└─► Periodic physics / Legacy support
    └── net_type="spectral" (Spectral Convolution)
```

---

## 向后兼容性

**BREAKING CHANGE**: MLP fallback removed

Old code:
```python
rcln = RCLNLayer(4, 8, 2)  # ❌ TypeError: net_type required
```

New code:
```python
rcln = RCLNLayer(4, 8, 2, net_type="spectral")  # ✅ Closest to MLP
# or
rcln = RCLNLayer(4, 8, 2, net_type="kan")       # ✅ Recommended
```

---

## 测试结果

```
======================================================================
RCLN 2.0 - Four Evolution Schemes Test
======================================================================

[1] Spectral-RCLN
    torch.Size([5, 4]) -> torch.Size([5, 2]) : OK

[2] Clifford-RCLN  
    torch.Size([5, 4]) -> torch.Size([5, 2]) : OK

[3] KAN-RCLN
    torch.Size([5, 4]) -> torch.Size([5, 2]) : OK
    Formula extraction: OK

[4] Mamba-RCLN
    torch.Size([5, 4]) -> torch.Size([5, 2]) : OK

[5] With Physics Hard Core
    torch.Size([5, 4]) -> torch.Size([5, 4]) : OK
    y = F_hard + 0.5 * F_soft

======================================================================
All Four Evolution Schemes Verified!
======================================================================
```

---

## 总结

RCLN 2.0 完成了从古典 MLP 到现代物理-AI 架构的进化：

1. **FNO-RCLN**: 算子化 - 频域全局操作
2. **Clifford-RCLN**: 几何化 - 旋转等变
3. **KAN-RCLN**: 符号化 - 直接公式提取
4. **Mamba-RCLN**: 序列化 - 线性复杂度

**核心架构保持不变**: `y = y_hard + λ·y_soft`

**移除**: 默认 MLP fallback

**要求**: 显式选择进化路径

---

*Axiom-OS: Physics-AI Hybrid Operating System*
