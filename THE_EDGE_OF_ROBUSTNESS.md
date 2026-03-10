# The Edge of Robustness - 鲁棒性的边界

## 法医报告：Walker2d 敏感性分析

### 原始测试结果

| 工况 | Gravity | Friction | 分数 | 判定 |
|------|---------|----------|------|------|
| 标准 | 1.0 | 1.0 | ~340 | 通过 |
| 轻微扰动 | 1.05 | 0.95 | ~345 | 通过 |
| 中等扰动 | 1.1 | 0.9 | ~345 | 通过 |
| **极端工况** | **1.2** | **0.8** | **~73** | **失败** |

### 失败原因分析

**极端工况**: High Gravity (1.2x) + Low Friction (0.8x)
- 重力增大 20% → 机器人"变重"
- 摩擦减小 20% → 地面"打滑"

**Einstein 模块的"想象" vs 现实**:
```
想象: "施加力 F，根据标准模型，能站住"
现实: 实际加速度 > 预期 (g' > g)
      实际摩擦 < 预期 (μ' < μ)
结果: 脚下一滑，直接跪了
```

**根本原因**: Hard Core 使用**固定旧参数**，没有适应环境变化。

---

## 解决方案：Online Discovery

### 核心洞察

> 如果系统能感知到"实际加速度 > 理论加速度"，就能反推出 g' > g，从而修正 Hard Core。

### 算法：递推最小二乘

```python
# 创新 = 实际 - 预测
delta = a_measured - a_predicted

# 卡尔曼增益
K = P / (P + R)

# 更新重力估计
g_new = g_old + K * delta / sin(theta)
```

---

## 实现

### Online Parameter Identifier

```python
from axiom_os.core.online_identification import (
    OnlineParameterIdentifier, PhysicsParams
)

identifier = OnlineParameterIdentifier(
    initial_params=PhysicsParams(gravity=9.81, friction=1.0)
)

# 实时观测
identifier.observe(
    measured_accel=imu_data['accel'],
    orientation=imu_data['quat'],
    joint_torques=feedback['torque'],
    commanded_torque=action,
)

# 获取修正参数
correction = identifier.get_param_correction(nominal_params)
# {'gravity_scale': 1.18, 'friction_scale': 0.82}
```

### Adaptive Hard Core

```python
from axiom_os.core.online_identification import AdaptiveHardCore

adaptive_core = AdaptiveHardCore(
    base_hard_core=original_hard_core,
    nominal_params=PhysicsParams(gravity=9.81, friction=1.0),
)

# 实时更新
adaptive_core.update(observation)
prediction = adaptive_core.predict(state)
```

---

## 预期改进

| 工况 | Fixed HC | Adaptive HC | 改进 |
|------|----------|-------------|------|
| 极端 (g=1.2, f=0.8) | ~73 | ~300+ | +300% |
| 鲁棒性分数 | 2.0 | 4.0+ | +100% |

---

## 运行测试

```bash
python axiom_os/experiments/walker2d_robustness_with_adaptation.py
```

---

## 结论

**在线参数辨识完成了 Axiom-OS 的最后一步闭环**:

1. **训练**: MuJoCo + Domain Randomization → 软壳策略
2. **部署**: 真实机器人 → 在线辨识物理参数
3. **适应**: 动态更新 Hard Core → Einstein 的"想象"与现实对齐

**这才是真正的物理 AI**: 不仅学习策略，还学习**物理本身**。
