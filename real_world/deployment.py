"""
Deployment Controller - 部署控制器
Copyright (c) 2026 yuzechen83-crypto. All Rights Reserved.

将训练好的 Axiom-OS Policy 部署到真实机器人：
1. 加载训练好的模型（来自 MuJoCo 训练）
2. 连接真实机器人
3. 实时控制循环（50-1000Hz）
4. 监控和安全保护
"""

from __future__ import annotations

from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
import time
import threading
from collections import deque

import numpy as np
import torch

from .robot_interface import RealRobotInterface, RobotConfig, RobotPlatform
from .state_estimator import StateEstimator, RobotState
from .safety_layer import SafetyLayer, SafetyConstraint, SafetyLevel


@dataclass
class DeploymentConfig:
    """部署配置"""
    
    # 控制频率
    control_freq: float = 50.0  # Hz
    
    # 观测历史（用于延迟补偿）
    obs_history_length: int = 5
    
    # 动作滤波（平滑输出）
    action_filter_alpha: float = 0.2  # 指数移动平均
    
    # Sim-to-Real 适配
    action_scale: float = 0.5  # 通常真实机器人需要更保守的动作
    action_offset: Optional[np.ndarray] = None
    
    # 安全设置
    enable_safety_layer: bool = True
    max_episode_steps: int = 1000
    
    # 调试
    verbose: bool = True
    log_interval: int = 100  # 每 N 步打印日志


class RealTimeController:
    """
    实时控制器
    
    核心控制循环：
    1. 读取传感器 -> 2. 状态估计 -> 3. Policy 推理 -> 4. 安全检查 -> 5. 发送指令
    
    目标延迟 < 20ms (50Hz)
    """
    
    def __init__(
        self,
        robot_config: RobotConfig,
        deployment_config: DeploymentConfig,
        policy: Optional[torch.nn.Module] = None,
        safety_constraints: Optional[SafetyConstraint] = None,
    ):
        self.robot_config = robot_config
        self.deployment_config = deployment_config
        self.policy = policy
        
        # 初始化组件
        self.robot = RealRobotInterface(robot_config)
        self.state_estimator = StateEstimator(
            num_joints=robot_config.action_dim or 12,
            use_ekf=True
        )
        
        if deployment_config.enable_safety_layer and safety_constraints:
            self.safety_layer = SafetyLayer(
                constraints=safety_constraints,
                num_joints=robot_config.action_dim or 12
            )
        else:
            self.safety_layer = None
        
        # 观测历史（用于时序策略）
        self._obs_history: deque = deque(
            maxlen=deployment_config.obs_history_length
        )
        
        # 动作滤波
        self._last_action: Optional[np.ndarray] = None
        
        # 运行状态
        self._is_running: bool = False
        self._step_count: int = 0
        self._episode_reward: float = 0.0
        
        # 性能监控
        self._loop_times: deque = deque(maxlen=100)
        
    def load_policy(self, policy_path: str):
        """加载训练好的策略"""
        checkpoint = torch.load(policy_path, map_location="cpu")
        if self.policy is not None:
            self.policy.load_state_dict(checkpoint.get("policy_state_dict", checkpoint))
            self.policy.eval()
            print(f"[Deployment] Policy loaded from {policy_path}")
        else:
            print("[Warning] Policy model not initialized")
    
    def start(self):
        """启动控制循环"""
        if self._is_running:
            print("[Warning] Controller already running")
            return
        
        # 连接机器人
        if not self.robot.connect():
            print("[Error] Failed to connect to robot")
            return
        
        self._is_running = True
        self._control_thread = threading.Thread(target=self._control_loop)
        self._control_thread.start()
        
        print(f"[Deployment] Controller started at {self.deployment_config.control_freq}Hz")
    
    def stop(self):
        """停止控制循环"""
        self._is_running = False
        if hasattr(self, "_control_thread"):
            self._control_thread.join(timeout=2.0)
        self.robot.disconnect()
        print("[Deployment] Controller stopped")
    
    def _control_loop(self):
        """主控制循环（在独立线程中运行）"""
        dt = 1.0 / self.deployment_config.control_freq
        
        while self._is_running:
            loop_start = time.time()
            
            try:
                self._step()
            except Exception as e:
                print(f"[Error] Control step failed: {e}")
                if self.safety_layer:
                    self.safety_layer._trigger_emergency_stop(f"Exception: {e}")
            
            # 维持恒定频率
            elapsed = time.time() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # 记录循环时间
            self._loop_times.append(time.time() - loop_start)
    
    def _step(self):
        """单步控制"""
        # 1. 读取传感器
        sensors = self.robot.read_sensors()
        
        # 2. 更新状态估计器
        from .state_estimator import IMUData, JointData
        
        imu_data = IMUData(
            quaternion=sensors["imu_quat"],
            angular_velocity=sensors["imu_ang_vel"],
            linear_acceleration=sensors["imu_lin_acc"],
            timestamp=time.time()
        )
        joint_data = JointData(
            positions=sensors["joint_pos"],
            velocities=sensors["joint_vel"],
            torques=sensors["joint_torque"],
            timestamp=time.time()
        )
        
        self.state_estimator.update_imu(imu_data)
        self.state_estimator.update_joint(joint_data)
        
        # 3. 获取估计状态
        state = self.state_estimator.estimate()
        
        # 4. 构建观测（与 MuJoCo 对齐）
        obs = self._build_observation(state)
        
        # 5. Policy 推理
        if self.policy is not None:
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
                action_tensor = self.policy(obs_tensor)
                action = action_tensor.cpu().numpy()[0]
        else:
            # 默认：零动作
            action = np.zeros(self.robot_config.action_dim or 12)
        
        # 6. 动作滤波（平滑）
        action = self._filter_action(action)
        
        # 7. 安全检查
        if self.safety_layer:
            safety_level = self.safety_layer.check_state(
                joint_pos=state.joint_positions,
                joint_vel=state.joint_velocities,
                base_quat=state.base_orientation,
                base_ang_vel=state.base_ang_vel,
                base_height=state.base_position[2],
            )
            
            if safety_level == SafetyLevel.EMERGENCY:
                print("[Emergency] Stopping controller")
                self._is_running = False
                return
            elif safety_level == SafetyLevel.WARNING:
                action = self.safety_layer.filter_action(
                    action, state.joint_positions
                )
        
        # 8. Sim-to-Real 适配
        action = action * self.deployment_config.action_scale
        if self.deployment_config.action_offset is not None:
            action = action + self.deployment_config.action_offset
        
        # 9. 发送指令
        self.robot.send_action(action)
        
        # 10. 记录
        self._step_count += 1
        
        if self.deployment_config.verbose and self._step_count % self.deployment_config.log_interval == 0:
            avg_loop_time = np.mean(list(self._loop_times)) if self._loop_times else 0
            print(f"[Step {self._step_count}] "
                  f"Height: {state.base_position[2]:.3f}m, "
                  f"Loop: {avg_loop_time*1000:.1f}ms")
    
    def _build_observation(self, state: RobotState) -> np.ndarray:
        """构建观测向量（与 MuJoCo 训练对齐）"""
        # 正弦/余弦编码
        joint_pos_sin = np.sin(state.joint_positions)
        joint_pos_cos = np.cos(state.joint_positions)
        
        # 观测历史
        obs = np.concatenate([
            joint_pos_sin,
            joint_pos_cos,
            state.joint_velocities,
            state.base_orientation,
            state.base_ang_vel,
        ])
        
        # 添加到历史
        self._obs_history.append(obs)
        
        # 填充历史（如果不够）
        while len(self._obs_history) < self._obs_history.maxlen:
            self._obs_history.append(obs)
        
        # 返回拼接的历史观测
        return np.concatenate(list(self._obs_history))
    
    def _filter_action(self, action: np.ndarray) -> np.ndarray:
        """动作滤波（指数移动平均）"""
        alpha = self.deployment_config.action_filter_alpha
        
        if self._last_action is None:
            filtered = action
        else:
            filtered = alpha * action + (1 - alpha) * self._last_action
        
        self._last_action = filtered
        return filtered
    
    def get_stats(self) -> dict:
        """获取部署统计"""
        return {
            "step_count": self._step_count,
            "avg_loop_time_ms": np.mean(list(self._loop_times)) * 1000 if self._loop_times else 0,
            "current_freq_hz": 1.0 / np.mean(list(self._loop_times)) if self._loop_times else 0,
        }


class DeployController:
    """
    部署控制器（高级接口）
    
    一键部署训练好的 MuJoCo Policy 到真实机器人：
    
    ```python
    deploy = DeployController()
    deploy.load_policy("trained_model.pt")
    deploy.run_episode(duration=10.0)
    ```
    """
    
    def __init__(
        self,
        robot_platform: RobotPlatform = RobotPlatform.MOCK,
        policy_path: Optional[str] = None,
    ):
        self.robot_platform = robot_platform
        
        # 配置
        robot_config = RobotConfig(
            platform=robot_platform,
            control_freq=50.0,
        )
        
        deployment_config = DeploymentConfig(
            control_freq=50.0,
            verbose=True,
        )
        
        safety_constraints = SafetyConstraint(
            joint_pos_limits=(-np.pi, np.pi),
            joint_vel_limits=(-10, 10),
            joint_torque_limits=(-20, 20),
        )
        
        # 创建控制器
        self.controller = RealTimeController(
            robot_config=robot_config,
            deployment_config=deployment_config,
            safety_constraints=safety_constraints,
        )
        
        # 加载策略
        if policy_path:
            self.controller.load_policy(policy_path)
    
    def run_episode(self, duration: float = 10.0):
        """运行一个 episode"""
        print(f"[Deploy] Starting episode for {duration}s")
        
        self.controller.start()
        
        try:
            time.sleep(duration)
        except KeyboardInterrupt:
            print("[Deploy] Interrupted by user")
        
        self.controller.stop()
        
        # 打印统计
        stats = self.controller.get_stats()
        print(f"[Deploy] Episode completed:")
        print(f"  Steps: {stats['step_count']}")
        print(f"  Avg loop time: {stats['avg_loop_time_ms']:.1f}ms")
        print(f"  Effective freq: {stats['current_freq_hz']:.1f}Hz")
    
    def benchmark(self, duration: float = 5.0):
        """基准测试控制频率"""
        print(f"[Deploy] Benchmarking for {duration}s...")
        
        self.controller.start()
        time.sleep(duration)
        self.controller.stop()
        
        stats = self.controller.get_stats()
        print(f"[Benchmark] Results:")
        print(f"  Target freq: {self.controller.deployment_config.control_freq}Hz")
        print(f"  Actual freq: {stats['current_freq_hz']:.1f}Hz")
        print(f"  Latency: {stats['avg_loop_time_ms']:.1f}ms")
