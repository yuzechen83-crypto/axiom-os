"""
Online Parameter Identification - 在线物理参数辨识
Copyright (c) 2026 yuzechen83-crypto. All Rights Reserved.

核心思想：
- 如果 实际加速度 > 理论加速度 → 实际重力 g' > 假设重力 g
- 如果 实际摩擦 < 理论摩擦 → 摩擦系数 μ' < 假设 μ

实现贝叶斯在线学习，实时更新 Hard Core 参数。
"""

from __future__ import annotations

from typing import Optional, Dict, Tuple, Callable, List
from dataclasses import dataclass, field
from collections import deque
import time

import numpy as np


@dataclass
class PhysicsParams:
    """可辨识的物理参数"""
    gravity: float = 9.81  # m/s^2
    friction: float = 1.0   # 摩擦系数
    mass: float = 1.0       # 质量缩放
    damping: float = 0.0    # 阻尼系数
    
    # 置信度 (方差倒数，越高越可信)
    gravity_confidence: float = 1.0
    friction_confidence: float = 1.0


class OnlineParameterIdentifier:
    """
    在线参数辨识器
    
    算法：递推最小二乘 (Recursive Least Squares) + 贝叶斯更新
    
    观测方程：
        a_measured = g' * sin(θ) + noise
        
    其中 g' 是真实重力，θ 是倾角
    """
    
    def __init__(
        self,
        initial_params: Optional[PhysicsParams] = None,
        window_size: int = 50,
        learning_rate: float = 0.1,
    ):
        self.params = initial_params or PhysicsParams()
        self.window_size = window_size
        self.learning_rate = learning_rate
        
        # 观测缓冲区
        self._accel_buffer: deque = deque(maxlen=window_size)
        self._orientation_buffer: deque = deque(maxlen=window_size)
        self._torque_buffer: deque = deque(maxlen=window_size)
        self._velocity_buffer: deque = deque(maxlen=window_size)
        
        # 参数协方差 (不确定性)
        self._P_gravity = 1.0  # 重力估计的协方差
        self._P_friction = 1.0  # 摩擦估计的协方差
        
        # 创新序列 (实际 - 预测)
        self._innovation_history: deque = deque(maxlen=100)
        
    def observe(
        self,
        measured_accel: np.ndarray,
        orientation: np.ndarray,  # 四元数 [w, x, y, z]
        joint_torques: np.ndarray,
        joint_velocities: np.ndarray,
        commanded_torque: np.ndarray,
    ):
        """
        观测新数据
        
        Args:
            measured_accel: IMU 测量的线加速度 (世界坐标系)
            orientation: 机身姿态四元数
            joint_torques: 关节力矩反馈
            joint_velocities: 关节速度
            commanded_torque: 指令力矩
        """
        # 存储到缓冲区
        self._accel_buffer.append(measured_accel)
        self._orientation_buffer.append(orientation)
        self._torque_buffer.append(joint_torques)
        self._velocity_buffer.append(joint_velocities)
        
        # 如果缓冲区满，进行参数更新
        if len(self._accel_buffer) >= self.window_size:
            self._update_gravity_estimate()
            self._update_friction_estimate(commanded_torque, joint_torques)
    
    def _update_gravity_estimate(self):
        """
        更新重力估计
        
        原理：当机身倾斜时，加速度计应该测量到 g*sin(θ)
        如果测量值 ≠ 预期值 → 实际 g' ≠ 假设 g
        """
        if len(self._accel_buffer) < 10:
            return
        
        # 计算机身倾角 (从四元数)
        orientations = np.array(self._orientation_buffer)
        thetas = np.array([self._quat_to_tilt_angle(q) for q in orientations])
        
        # 预期加速度 (基于当前重力假设)
        expected_accel = self.params.gravity * np.sin(thetas)
        
        # 实际测量 (z轴分量)
        measured_accels = np.array([a[2] for a in self._accel_buffer])
        
        # 创新：实际 - 预测
        innovation = np.mean(measured_accels) - np.mean(expected_accel)
        self._innovation_history.append(innovation)
        
        # 如果创新显著 (> 0.5 m/s²)，更新重力估计
        if abs(innovation) > 0.5:
            # 递推最小二乘更新
            # g_new = g_old + K * (z - h(g_old))
            # 其中 K = P / (P + R) 是卡尔曼增益
            R = 0.5  # 观测噪声
            K = self._P_gravity / (self._P_gravity + R)
            
            # 修正重力估计
            g_correction = K * innovation / np.mean(np.abs(np.sin(thetas)) + 1e-6)
            new_gravity = self.params.gravity + g_correction
            
            # 限制在合理范围
            new_gravity = np.clip(new_gravity, 5.0, 15.0)
            
            # 更新
            self.params.gravity = new_gravity
            self.params.gravity_confidence = 1.0 / (self._P_gravity + 1e-6)
            
            # 更新协方差
            self._P_gravity = (1 - K) * self._P_gravity + 0.01  # 加一点过程噪声
            
            print(f"[OnlineID] Gravity updated: {self.params.gravity:.2f} m/s² "
                  f"(innovation: {innovation:.2f}, confidence: {self.params.gravity_confidence:.1f})")
    
    def _update_friction_estimate(
        self,
        commanded_torque: np.ndarray,
        measured_torque: np.ndarray,
    ):
        """
        更新摩擦估计
        
        原理：如果 实际关节力矩 < 指令力矩，可能存在滑移（摩擦不足）
        或者：实际速度响应慢于预期 → 摩擦/阻尼增大
        """
        if len(self._velocity_buffer) < 10:
            return
        
        # 计算力矩误差率
        torque_ratio = np.mean(measured_torque) / (np.mean(np.abs(commanded_torque)) + 1e-6)
        
        # 如果力矩传递效率低 (< 0.8)，可能存在滑移
        if torque_ratio < 0.8:
            # 摩擦可能降低
            friction_correction = -0.1 * (1.0 - torque_ratio)
        elif torque_ratio > 1.2:
            # 摩擦可能增加
            friction_correction = 0.05 * (torque_ratio - 1.0)
        else:
            return
        
        # 更新摩擦估计
        R = 0.2
        K = self._P_friction / (self._P_friction + R)
        
        new_friction = self.params.friction + K * friction_correction
        new_friction = np.clip(new_friction, 0.1, 2.0)
        
        self.params.friction = new_friction
        self.params.friction_confidence = 1.0 / (self._P_friction + 1e-6)
        self._P_friction = (1 - K) * self._P_friction + 0.01
        
        print(f"[OnlineID] Friction updated: {self.params.friction:.2f} "
              f"(torque_ratio: {torque_ratio:.2f})")
    
    def _quat_to_tilt_angle(self, q: np.ndarray) -> float:
        """四元数转换为倾角（简化版，假设绕 x 轴倾斜）"""
        # w, x, y, z
        # 计算重力方向与机身 z 轴的夹角
        return 2 * np.arccos(np.clip(q[0], -1, 1))
    
    def get_param_correction(self, nominal_params: PhysicsParams) -> Dict[str, float]:
        """
        获取参数修正量
        
        Returns:
            {'gravity_scale': 1.1, 'friction_scale': 0.9, ...}
        """
        return {
            'gravity_scale': self.params.gravity / (nominal_params.gravity + 1e-6),
            'friction_scale': self.params.friction / (nominal_params.friction + 1e-6),
            'mass_scale': self.params.mass / (nominal_params.mass + 1e-6),
        }
    
    def is_confident(self, threshold: float = 5.0) -> bool:
        """参数估计是否足够自信"""
        return (self.params.gravity_confidence > threshold and 
                self.params.friction_confidence > threshold)
    
    def get_innovation_stats(self) -> Dict[str, float]:
        """获取创新序列统计（用于检测异常）"""
        if not self._innovation_history:
            return {'mean': 0.0, 'std': 0.0, 'max_abs': 0.0}
        
        innov = np.array(self._innovation_history)
        return {
            'mean': float(np.mean(innov)),
            'std': float(np.std(innov)),
            'max_abs': float(np.max(np.abs(innov))),
        }


class AdaptiveHardCore:
    """
    自适应 Hard Core
    
    将在线辨识的参数实时注入 Hard Core
    实现 Einstein 模块与现实的闭环
    """
    
    def __init__(
        self,
        base_hard_core: Callable,
        nominal_params: PhysicsParams,
        update_threshold: float = 0.1,  # 参数变化超过 10% 才更新
    ):
        self.base_hard_core = base_hard_core
        self.nominal_params = nominal_params
        self.update_threshold = update_threshold
        
        # 在线辨识器
        self.identifier = OnlineParameterIdentifier(nominal_params)
        
        # 当前有效参数（可能被修正）
        self.current_params = PhysicsParams(
            gravity=nominal_params.gravity,
            friction=nominal_params.friction,
            mass=nominal_params.mass,
        )
        
        # 历史记录
        self._param_history: deque = deque(maxlen=100)
        
    def update(self, observation: Dict[str, np.ndarray]):
        """
        根据新观测更新 Hard Core 参数
        
        Args:
            observation: 包含 'accel', 'orientation', 'torque', 'velocity' 等
        """
        # 喂数据给辨识器
        self.identifier.observe(
            measured_accel=observation.get('accel', np.zeros(3)),
            orientation=observation.get('orientation', np.array([1, 0, 0, 0])),
            joint_torques=observation.get('torque', np.zeros(12)),
            joint_velocities=observation.get('velocity', np.zeros(12)),
            commanded_torque=observation.get('commanded_torque', np.zeros(12)),
        )
        
        # 获取参数修正
        correction = self.identifier.get_param_correction(self.nominal_params)
        
        # 检查是否需要更新 Hard Core
        if abs(correction['gravity_scale'] - 1.0) > self.update_threshold:
            self._update_hard_core_params(correction)
        
        if abs(correction['friction_scale'] - 1.0) > self.update_threshold:
            self._update_hard_core_params(correction)
    
    def _update_hard_core_params(self, correction: Dict[str, float]):
        """更新 Hard Core 的参数"""
        old_g = self.current_params.gravity
        old_f = self.current_params.friction
        
        # 应用修正
        self.current_params.gravity = self.nominal_params.gravity * correction['gravity_scale']
        self.current_params.friction = self.nominal_params.friction * correction['friction_scale']
        
        # 记录
        self._param_history.append({
            'timestamp': time.time(),
            'gravity': self.current_params.gravity,
            'friction': self.current_params.friction,
        })
        
        print(f"[AdaptiveHardCore] Params updated: "
              f"g: {old_g:.2f} → {self.current_params.gravity:.2f}, "
              f"f: {old_f:.2f} → {self.current_params.friction:.2f}")
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        使用更新后的参数进行预测
        
        这就是 Einstein 模块的"想象"，但现在基于真实物理参数
        """
        # 使用修正后的参数调用基础 Hard Core
        # 实际实现取决于 Hard Core 的具体形式
        return self.base_hard_core(state, self.current_params)
    
    def get_diagnostics(self) -> Dict[str, any]:
        """获取诊断信息"""
        return {
            'current_params': {
                'gravity': self.current_params.gravity,
                'friction': self.current_params.friction,
                'mass': self.current_params.mass,
            },
            'nominal_params': {
                'gravity': self.nominal_params.gravity,
                'friction': self.nominal_params.friction,
            },
            'innovation_stats': self.identifier.get_innovation_stats(),
            'is_confident': self.identifier.is_confident(),
            'num_updates': len(self._param_history),
        }


# =============================================================================
# Discovery Engine: (s, a, s') buffer + fit g, μ to minimize prediction error
# =============================================================================

def _default_predict_from_transitions(
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    dt: float = 0.002,
    vel_dim_hint: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Heuristic g_scale and friction_scale from (s, a, s') when no predictor is available.
    
    Improved algorithm for Walker2d/Humanoid:
    - Uses robust statistics (median) instead of mean to handle outliers
    - Gravity estimation: from vertical velocity trend over multiple steps
    - Friction estimation: from velocity decay when joints are not actively driving
    """
    n = len(states)
    if n < 10:  # Need more samples for robust estimation
        return 1.0, 1.0
    
    dim = states.shape[1]
    # Auto-detect velocity start index if not provided
    if vel_dim_hint is not None:
        vel_start = vel_dim_hint
    else:
        # Common MuJoCo envs: Walker2d(17)->9, Humanoid(376)->?, Hopper(11)->6
        # Use a simple heuristic: look for where values become more dynamic
        vel_start = dim // 2
        # Adjust for Walker2d (17 dim -> 9) and Hopper (11 dim -> 6)
        if dim == 17:  # Walker2d-v5
            vel_start = 9
        elif dim == 11:  # Hopper-v5
            vel_start = 6
    
    if vel_start >= dim:
        return 1.0, 1.0
    
    vel_dim = dim - vel_start
    
    # Extract velocities
    vels = states[:, vel_start:]  # (n, vel_dim)
    next_vels = next_states[:, vel_start:]  # (n, vel_dim)
    
    # Acceleration from velocity change
    acc = (next_vels - vels) / (dt + 1e-8)  # (n, vel_dim)
    
    # === Gravity Estimation ===
    # For Walker2d/Humanoid: qvel[0] is root z-velocity (vertical)
    # When robot is standing/walking, vertical acceleration should average to 0
    # BUT if gravity changes, the controller will produce different vertical forces
    # We detect this from the trend of vertical velocity
    g_scale = 1.0
    if vel_dim >= 1:
        v_z = vels[:, 0]  # vertical velocity
        # Compute average vertical velocity over the buffer
        mean_vz = np.mean(v_z)
        
        # If mean_vz is significantly negative, robot is falling -> gravity might be higher
        # If mean_vz is significantly positive, robot is jumping/bouncing -> gravity might be lower
        # Use a robust estimator based on the spread of vertical velocity
        vz_std = np.std(v_z)
        vz_range = np.max(v_z) - np.min(v_z)
        
        # Heuristic: higher gravity -> more constrained motion -> smaller vz_range
        # Normal vz_range for Walker2d is typically 0.5-2.0
        expected_range = 1.0  # nominal
        if vz_range > 0.1:  # Avoid division by zero
            # If range is smaller, gravity might be higher (more constrained)
            # If range is larger, gravity might be lower
            g_scale = expected_range / (vz_range + 0.1)
            g_scale = np.clip(g_scale, 0.5, 1.5)  # Conservative estimate
    
    # === Friction Estimation ===
    # Friction manifests as velocity decay when not actively driven
    # Look at velocity magnitude changes
    v_norm = np.linalg.norm(vels, axis=1)
    next_v_norm = np.linalg.norm(next_vels, axis=1)
    
    # Filter out steps with large acceleration (actively driven)
    acc_norm = np.linalg.norm(acc, axis=1)
    passive_mask = acc_norm < np.percentile(acc_norm, 30)  # Bottom 30% = more passive
    
    if np.sum(passive_mask) > 5:
        # During passive motion, v' should be slightly less than v due to friction
        ratios = next_v_norm[passive_mask] / (v_norm[passive_mask] + 1e-8)
        median_ratio = np.median(ratios)
        
        # ratio < 1 => decay => higher friction
        # ratio > 1 => growth => lower friction
        # Heuristic: friction_scale = 1 / median_ratio (approximately)
        friction_scale = 1.0 / (median_ratio + 0.1)
        friction_scale = np.clip(friction_scale, 0.5, 1.5)
    else:
        friction_scale = 1.0
    
    # Final clipping to bounds
    return float(np.clip(g_scale, 0.2, 2.0)), float(np.clip(friction_scale, 0.1, 2.0))


class DiscoveryEngine:
    """
    Discovery Engine: update physics parameters (g, friction) in real time from (s, a, s').
    Buffers last N steps, compares HardCore.predict(s, a) vs Real(s'), and adjusts
    g_scale and friction_scale to minimize prediction error (least squares or gradient descent).
    Constraint: g_scale in (g_min, g_max), friction_scale in (f_min, f_max).
    """

    def __init__(
        self,
        buffer_size: int = 100,
        predict_fn: Optional[Callable[[np.ndarray, np.ndarray, float, float], np.ndarray]] = None,
        dt: float = 0.002,
        g_nominal: float = 1.0,
        friction_nominal: float = 1.0,
        g_bounds: Tuple[float, float] = (0.2, 2.0),
        friction_bounds: Tuple[float, float] = (0.1, 2.0),
        update_interval: int = 20,
        lr: float = 0.05,
        vel_dim_hint: Optional[int] = None,
    ):
        self.buffer_size = buffer_size
        self.predict_fn = predict_fn
        self.dt = dt
        self.g_nominal = g_nominal
        self.friction_nominal = friction_nominal
        self.g_bounds = g_bounds
        self.friction_bounds = friction_bounds
        self.update_interval = update_interval
        self.lr = lr
        self.vel_dim_hint = vel_dim_hint

        self._buffer_s: List[np.ndarray] = []
        self._buffer_a: List[np.ndarray] = []
        self._buffer_s_next: List[np.ndarray] = []
        self._g_scale = g_nominal
        self._friction_scale = friction_nominal
        self._step_count = 0
        self._update_count = 0

    def observe(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> None:
        """Append one transition (s, a, s')."""
        self._buffer_s.append(np.asarray(state, dtype=np.float64).ravel())
        self._buffer_a.append(np.asarray(action, dtype=np.float64).ravel())
        self._buffer_s_next.append(np.asarray(next_state, dtype=np.float64).ravel())
        if len(self._buffer_s) > self.buffer_size:
            self._buffer_s.pop(0)
            self._buffer_a.pop(0)
            self._buffer_s_next.pop(0)
        self._step_count += 1

    def _get_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._buffer_s:
            return np.zeros((0, 1)), np.zeros((0, 1)), np.zeros((0, 1))
        S = np.array(self._buffer_s)
        A = np.array(self._buffer_a)
        S_next = np.array(self._buffer_s_next)
        return S, A, S_next

    def update(self) -> Tuple[float, float]:
        """
        Run parameter update: minimize prediction error over buffer.
        Returns (g_scale, friction_scale) clipped to bounds.
        """
        S, A, S_next = self._get_arrays()
        if len(S) < max(10, self.update_interval):
            return self._g_scale, self._friction_scale

        if self.predict_fn is None:
            g_scale, friction_scale = _default_predict_from_transitions(
                S, A, S_next, dt=self.dt, vel_dim_hint=self.vel_dim_hint
            )
            self._g_scale = np.clip(g_scale, self.g_bounds[0], self.g_bounds[1])
            self._friction_scale = np.clip(
                friction_scale, self.friction_bounds[0], self.friction_bounds[1]
            )
            self._update_count += 1
            return self._g_scale, self._friction_scale

        # Gradient descent: minimize sum ||s' - predict(s, a, g, f)||^2
        g, f = self._g_scale, self._friction_scale
        for _ in range(25):
            err_sum = 0.0
            dg, df = 0.0, 0.0
            eps = 1e-5
            for i in range(len(S)):
                pred = self.predict_fn(S[i], A[i], g, f)
                err = S_next[i] - pred
                err_sum += float(np.sum(err ** 2))
                # Finite differences for gradient
                pred_g = self.predict_fn(S[i], A[i], g + eps, f)
                pred_f = self.predict_fn(S[i], A[i], g, f + eps)
                # grad of sum ||s' - pred||^2: d/dg = -2 * err . d_pred/dg
                dg += float(np.sum(err * (pred_g - pred) / (eps + 1e-10)))
                df += float(np.sum(err * (pred_f - pred) / (eps + 1e-10)))
            if len(S) > 0:
                dg = -2.0 * dg / len(S)
                df = -2.0 * df / len(S)
            g = g + self.lr * dg
            f = f + self.lr * df
            g = np.clip(g, self.g_bounds[0], self.g_bounds[1])
            f = np.clip(f, self.friction_bounds[0], self.friction_bounds[1])
        self._g_scale = g
        self._friction_scale = f
        self._update_count += 1
        return self._g_scale, self._friction_scale

    def get_physics_scale(self) -> Tuple[float, float]:
        """Current (g_scale, friction_scale) for the MPC/Policy."""
        return self._g_scale, self._friction_scale

    def maybe_update(self) -> Tuple[float, float]:
        """Call update every update_interval steps; else return current scale."""
        if self._step_count > 0 and self._step_count % self.update_interval == 0:
            return self.update()
        return self._g_scale, self._friction_scale
