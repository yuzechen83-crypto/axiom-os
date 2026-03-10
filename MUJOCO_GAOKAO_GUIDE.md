# Axiom-OS MuJoCo 高考指南 (Gaokao Guide)

## 平台介绍

**Gymnasium（原 OpenAI Gym）+ MuJoCo** 是 AI 控制领域的「高考」：所有强化学习（RL）和控制算法都要在这里过一遍。

| 组件 | 作用 |
|------|------|
| **Gymnasium** | 标准 API：`reset`、`step`、`reward`，统一环境接口。 |
| **MuJoCo** | 目前最精准的物理引擎，专门处理接触、摩擦和多关节动力学。 |

## Axiom-OS 的测试科目

| 科目 | 环境 | 测试内容 |
|------|------|----------|
| Swimmer | Swimmer-v5 | RCLN 处理复杂多刚体动力学 |
| Hopper | Hopper-v5 | 单腿跳跃与平衡 |
| Walker2d | Walker2d-v5 | 双足行走与协调 |
| **Humanoid（终极 BOSS）** | Humanoid-v5 | 人形机器人：若 MPC + 爱因斯坦模块比 PPO 站得更稳，即顶级成果 |

## 如何接入

- **Soft Shell**：充当 Policy 网络，输出连续动作。
- **Hard Core**：直接读取 MuJoCo XML 物理参数（质量、长度、重力、摩擦），供物理感知策略或 MPC 使用。
- 环境封装：`axiom_os.envs.make_axiom_mujoco_env` 返回标准 Gymnasium 环境，支持 `gravity_scale` / `friction_scale` 做 Sim-to-Real 测试。

## 挑战点：Sim-to-Real gap

MuJoCo 的物理很完美，**建议人为修改 MuJoCo 的重力或摩擦系数**，测试 Axiom 的自适应能力：

- 训练时使用 **域随机化**（`make_axiom_mujoco_env_with_domain_rand`）：每次 reset 随机 gravity/friction。
- 评估时使用 **固定扰动**（`--gravity-scale` / `--friction-scale`）：检验在「非标称物理」下的表现。

---

## 概述

Axiom-OS 已完整集成 **Gymnasium + MuJoCo**，支持上述测试科目与 Sim-to-Real 挑战。

## 测试科目（详细）

| 科目 | 环境ID | 难度 | 测试内容 |
|------|--------|------|----------|
| 基础1 | Swimmer-v5 | ⭐⭐ | 多刚体动力学 |
| 基础2 | Hopper-v5 | ⭐⭐⭐ | 单腿跳跃平衡 |
| 基础3 | Walker2d-v5 | ⭐⭐⭐⭐ | 双足行走 |
| 终极BOSS | Humanoid-v5 | ⭐⭐⭐⭐⭐ | 人形机器人站立行走 |

## 快速开始

### 用 CLI 跑（推荐）

```bash
# 全部科目
axiom mujoco --env all --steps 500

# 单科 + Sim-to-Real 扰动（挑战点：改重力/摩擦）
axiom mujoco --env Hopper-v5 --gravity-scale 1.1 --friction-scale 0.9
axiom mujoco --env Walker2d-v5 --domain-rand --physics-aware

# Humanoid 终极 BOSS
axiom mujoco --env Humanoid-v5 --steps 1000 --train 30
```

### 1. 基础测试（Swimmer/Hopper/Walker2d）

```bash
python run_mujoco_gaokao.py --basic
```

### 2. Sim-to-Real 挑战

```bash
python run_mujoco_gaokao.py --sim2real
```

### 3. Humanoid 终极 BOSS

```bash
python run_mujoco_gaokao.py --humanoid
```

### 4. 重力/摩擦扰动测试

```bash
python run_mujoco_gaokao.py --perturbation
```

### 5. 完整高考（所有科目）

```bash
python run_mujoco_gaokao.py --full
```

## 直接运行脚本

### 使用现有脚本

```bash
# 运行单个环境
python -m axiom_os.scripts.run_mujoco --env Hopper-v5 --steps 1000

# 带训练运行
python -m axiom_os.scripts.run_mujoco --env Hopper-v5 --train 10 --steps 500

# Sim-to-Real 测试（修改重力/摩擦）
python -m axiom_os.scripts.run_mujoco --env Hopper-v5 --gravity-scale 1.1 --friction-scale 0.8

# Walker2d 敏感性分析
python -m axiom_os.scripts.run_mujoco --analyze-walker2d

# 所有环境
python -m axiom_os.scripts.run_mujoco --env all --train 10
```

## 核心组件

### 1. 环境包装器 (mujoco_env.py)

```python
from axiom_os.envs import make_axiom_mujoco_env, get_physics_params

# 创建环境
env = make_axiom_mujoco_env(
    env_id="Hopper-v5",
    gravity_scale=1.0,    # Sim-to-Real: 修改重力
    friction_scale=1.0,   # Sim-to-Real: 修改摩擦
    seed=42,
)

# 提取物理参数（供 Hard Core 使用）
params = get_physics_params(env)
print(f"Gravity: {params.gravity}")
print(f"Friction: {params.friction}")
print(f"Body masses: {params.body_masses}")
```

### 2. 域随机化 (Domain Randomization)

```python
from axiom_os.envs import make_axiom_mujoco_env_with_domain_rand

# 训练时随机改变 gravity/friction，提高泛化能力
env = make_axiom_mujoco_env_with_domain_rand(
    env_id="Walker2d-v5",
    gravity_range=(0.9, 1.1),    # 随机 90%-110%
    friction_range=(0.8, 1.2),   # 随机 80%-120%
    seed=42,
)
```

### 3. Policy 网络 (axiom_policy.py)

```python
from axiom_os.envs.axiom_policy import make_mujoco_policy, PhysicsAwarePolicy

# 基础 Policy
policy = make_mujoco_policy(
    obs_dim=11,
    action_dim=3,
    hidden_dim=256,
)

# 物理感知 Policy（Hard Core）
policy = PhysicsAwarePolicy(
    obs_dim=11,
    action_dim=3,
    physics_dim=8,  # 包含 gravity, friction, body_masses 统计
)

# 选择动作
action = policy.act(obs, deterministic=True)
```

### 4. PPO/SAC 训练

```python
from axiom_os.envs.axiom_policy import train_ppo_sac

# 使用 PPO 训练（支持域随机化 + 物理感知）
model = train_ppo_sac(
    env_id="Humanoid-v5",
    algo="PPO",
    total_timesteps=200_000,
    domain_rand=True,      # 域随机化
    physics_aware=True,    # 物理感知
    gravity_range=(0.9, 1.1),
    friction_range=(0.8, 1.2),
)

# 保存/加载
model.save("humanoid_ppo_model")
```

## Sim-to-Real 测试

### 测试不同物理条件

```python
from axiom_os.envs import make_axiom_mujoco_env

# 标准条件
env = make_axiom_mujoco_env("Hopper-v5")

# 高重力/低摩擦（模拟月球表面？）
env = make_axiom_mujoco_env("Hopper-v5", gravity_scale=1.2, friction_scale=0.8)

# 低重力/高摩擦（模拟水下？）
env = make_axiom_mujoco_env("Hopper-v5", gravity_scale=0.8, friction_scale=1.2)
```

### 扰动测试结果示例

| 条件 | Gravity | Friction | 奖励 |
|------|---------|----------|------|
| 标准 | 1.0 | 1.0 | 243.7 |
| 高g/低f | 1.2 | 0.8 | 310.5 |
| 低g/高f | 0.8 | 1.2 | 321.5 |
| g+10%/f-10% | 1.1 | 0.9 | 732.2 |
| g-10%/f+10% | 0.9 | 1.1 | 849.0 |

**鲁棒性分数**: 1.98 (平均奖励 / 标准差)

## 与 PPO 对比

### Axiom-OS 优势

1. **物理感知**: Hard Core 直接读取 MuJoCo XML 参数
2. **域随机化**: 训练时自动扰动 gravity/friction
3. **Sim-to-Real**: 更好的物理参数适应能力
4. **MPC + Einstein**: 长视距规划 + 能量整形

### 运行对比实验

```bash
# 标准 PPO
python -m axiom_os.scripts.run_mujoco --env Humanoid-v5 --train 50 --algo PPO

# Axiom-OS (PPO + Domain Rand + Physics Aware)
python -m axiom_os.scripts.run_mujoco --env Humanoid-v5 --train 50 --algo PPO --domain-rand --physics-aware
```

## 文件结构

```
axiom_os/
├── envs/
│   ├── mujoco_env.py          # 环境包装器
│   ├── axiom_policy.py        # Policy 网络
│   └── __init__.py
├── scripts/
│   └── run_mujoco.py          # 主运行脚本
├── run_mujoco_gaokao.py       # 高考测试脚本
└── MUJOCO_GAOKAO_GUIDE.md     # 本指南
```

## 依赖安装

```bash
pip install gymnasium[mujoco]
pip install stable-baselines3
```

## 参考

- Gymnasium: https://gymnasium.farama.org/
- MuJoCo: https://mujoco.org/
- PDEBench + MuJoCo: AI4Science 的标准测试平台
