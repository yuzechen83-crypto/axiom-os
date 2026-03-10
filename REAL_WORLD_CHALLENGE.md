# Axiom-OS Real-World Control Challenge

## 概述

从 **MuJoCo 仿真**到**真实机器人**的跨越。这是 AI 控制系统的终极考验！

## 核心挑战

### 1. Sim-to-Real Gap
| 问题 | 仿真 (MuJoCo) | 真实世界 | 解决方案 |
|------|--------------|----------|----------|
| **模型误差** | 完美物理模型 | 质量/摩擦未知 | 域随机化 + 在线适应 |
| **传感器噪声** | 无噪声 | IMU漂移、编码器误差 | 卡尔曼滤波 |
| **执行器延迟** | 即时响应 | 20-100ms延迟 | 延迟补偿 + 动作缓冲区 |
| **通信延迟** | 零延迟 | WiFi/以太网延迟 | 预测控制 |

### 2. 安全约束
- **紧急停止**: 摔倒检测 (< 100ms)
- **关节限位**: 软/硬限位保护
- **力限制**: 防止自碰撞
- **恢复策略**: 摔倒后自动站起

### 3. 实时性要求
| 组件 | 目标频率 | 最大延迟 |
|------|----------|----------|
| 控制循环 | 50-1000Hz | 20ms |
| 传感器读取 | 100-1000Hz | 10ms |
| 策略推理 | 50-200Hz | 20ms |
| 安全监控 | 1000Hz | 1ms |

## 支持平台

### 四足机器人
- **Unitree Go1**: 教育/研究首选
- **Unitree Go2**: 最新一代
- **Unitree A1**: 经典型号
- **MIT Cheetah Mini**: 开源方案

### 人形机器人
- **Unitree H1/H2**: 国内领先
- **Boston Dynamics Atlas**: 工业级
- **Tesla Optimus**: 未来方向

### 通用接口
- **ROS2**: 标准机器人中间件
- **MQTT/WebSocket**: 远程控制

## 快速开始

### 1. 模拟模式测试

```bash
# 测试控制循环（无需真实机器人）
python run_real_world_challenge.py --mock --duration 10

# 基准测试控制频率
python run_real_world_challenge.py --mock --benchmark --duration 5
```

### 2. 加载训练好的 Policy

```bash
# 从 MuJoCo 高考训练的策略
python run_real_world_challenge.py \
  --mock \
  --policy outputs/mujoco_gaokao_model.pt \
  --duration 60
```

### 3. 连接真实机器人 (Unitree Go1)

```bash
# 确保机器人与电脑在同一局域网
python run_real_world_challenge.py \
  --platform unitree_go1 \
  --policy trained_model.pt \
  --duration 30
```

## 架构

```
MuJoCo 训练好的 Policy
         ↓
[Sim-to-Real Adapter]
   - 动作缩放/偏移
   - 延迟补偿
         ↓
[Real-Time Controller] @ 50Hz
   ├─ [Sensor Interface] ← IMU + Encoders
   ├─ [State Estimator] ← EKF 融合
   ├─ [Safety Layer] ← 限位/急停
   └─ [Action Filter] ← 平滑输出
         ↓
[Robot Interface]
   ├─ Unitree SDK
   ├─ ROS2
   └─ Custom Hardware
         ↓
    Real Robot!
```

## 核心组件

### 1. Robot Interface (`robot_interface.py`)

```python
from axiom_os.real_world import RealRobotInterface, RobotConfig, RobotPlatform

# 配置机器人
config = RobotConfig(
    platform=RobotPlatform.UNITREE_GO1,
    ip_address="192.168.123.161",
    control_freq=50.0,
)

# 连接并控制
with RealRobotInterface(config) as robot:
    # 读取传感器
    sensors = robot.read_sensors()
    print(f"IMU: {sensors['imu_quat']}")
    print(f"Joint pos: {sensors['joint_pos']}")
    
    # 发送动作
    action = np.zeros(12)  # 12个关节
    robot.send_action(action)
```

### 2. State Estimator (`state_estimator.py`)

```python
from axiom_os.real_world import StateEstimator, IMUData, JointData

estimator = StateEstimator(num_joints=12, use_ekf=True)

# 更新传感器数据
estimator.update_imu(IMUData(
    quaternion=[1, 0, 0, 0],
    angular_velocity=[0, 0, 0],
    linear_acceleration=[0, 0, -9.81],
    timestamp=time.time()
))

estimator.update_joint(JointData(
    positions=joint_pos,
    velocities=joint_vel,
    torques=joint_torque,
    timestamp=time.time()
))

# 获取估计状态
state = estimator.estimate()
print(f"Base height: {state.base_position[2]}")
```

### 3. Safety Layer (`safety_layer.py`)

```python
from axiom_os.real_world import SafetyLayer, SafetyConstraint

constraints = SafetyConstraint(
    joint_pos_limits=(-3.14, 3.14),
    joint_vel_limits=(-10, 10),
    max_roll=0.8,  # rad
    max_pitch=0.8,
)

safety = SafetyLayer(constraints, num_joints=12)

# 检查状态
level = safety.check_state(
    joint_pos=pos,
    joint_vel=vel,
    base_quat=quat,
    base_ang_vel=ang_vel,
    base_height=height,
)

if level == SafetyLevel.EMERGENCY:
    print("FALL DETECTED! Emergency stop!")

# 过滤动作
safe_action = safety.filter_action(action, joint_pos)
```

### 4. Deployment Controller (`deployment.py`)

```python
from axiom_os.real_world import DeployController

# 一键部署
deploy = DeployController(
    robot_platform=RobotPlatform.UNITREE_GO1,
    policy_path="trained_model.pt"
)

# 运行 60 秒
deploy.run_episode(duration=60.0)
```

## Sim-to-Real 适配指南

### 1. 动作缩放

真实机器人通常需要更保守的动作：

```python
# 在仿真中训练的动作
sim_action = policy(obs)  # [-1, 1]

# 部署时缩放
real_action = sim_action * 0.5  # 更保守
```

### 2. 延迟补偿

```python
# 记录动作历史
action_buffer.append(action)

# 预测当前状态（考虑 20ms 延迟）
predicted_state = current_state + velocity * 0.02
```

### 3. 域随机化（训练时）

```python
from axiom_os.envs import make_axiom_mujoco_env_with_domain_rand

# 训练时随机化物理参数
env = make_axiom_mujoco_env_with_domain_rand(
    "Hopper-v5",
    gravity_range=(0.9, 1.1),
    friction_range=(0.8, 1.2),
)
```

## 安全清单

### 部署前检查

- [ ] 机器人在安全环境中（无障碍物、柔软地面）
- [ ] 紧急停止按钮可用
- [ ] 电池电量 > 50%
- [ ] 所有关节正常，无松动
- [ ] 网络连接稳定
- [ ] 从模拟模式开始测试

### 运行时监控

- [ ] 控制频率达标 (> 50Hz)
- [ ] 无通信超时
- [ ] 姿态正常（roll/pitch < 45度）
- [ ] 关节温度正常
- [ ] 准备随时按下急停

## 调试技巧

### 1. 控制频率不够

```python
# 简化观测，减少计算
obs = obs[:36]  # 只保留关键观测

# 使用更小的网络
policy = make_mujoco_policy(obs_dim=36, hidden_dim=128)
```

### 2. 动作不稳定

```python
# 增加滤波
deployment_config.action_filter_alpha = 0.1  # 更平滑

# 降低控制频率
control_freq = 50  # 而不是 100Hz
```

### 3. 摔倒频繁

```python
# 收紧安全约束
safety_constraints.max_roll = 0.5  # 更严格
safety_constraints.max_pitch = 0.5

# 降低动作幅度
action_scale = 0.3
```

## 进阶挑战

### 1. 自适应控制
- 在线识别地面摩擦
- 根据负载调整策略
- 故障检测与恢复

### 2. 多机器人协同
- 分布式控制
- 通信延迟处理
- 编队控制

### 3. 人机交互
- 语音/手势控制
- 安全协作
- 意图预测

## 参考

- Unitree SDK: https://github.com/unitreerobotics
- ROS2: https://docs.ros.org/en/humble/
- Sim-to-Real Survey: https://arxiv.org/abs/2011.06235
