"""
State Estimator - 状态估计器
Copyright (c) 2026 yuzechen83-crypto. All Rights Reserved.

真实世界的传感器融合：
- IMU: 高频率 (1000Hz)，但有漂移
- 编码器: 精确，但仅限关节
- 视觉/激光雷达: 低频 (30Hz)，但全局定位

使用卡尔曼滤波/扩展卡尔曼滤波 (EKF) 融合多传感器数据。
"""

from __future__ import annotations

from typing import Optional, Tuple, List
from dataclasses import dataclass
import time

import numpy as np


@dataclass
class IMUData:
    """IMU 数据"""
    quaternion: np.ndarray  # [w, x, y, z]
    angular_velocity: np.ndarray  # rad/s
    linear_acceleration: np.ndarray  # m/s^2
    timestamp: float


@dataclass
class JointData:
    """关节编码器数据"""
    positions: np.ndarray  # rad
    velocities: np.ndarray  # rad/s
    torques: np.ndarray  # Nm
    timestamp: float


@dataclass
class RobotState:
    """估计的机器人状态"""
    base_position: np.ndarray  # [x, y, z] in world frame
    base_velocity: np.ndarray  # [vx, vy, vz]
    base_orientation: np.ndarray  # quaternion [w, x, y, z]
    base_ang_vel: np.ndarray  # angular velocity
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    foot_positions: Optional[np.ndarray] = None  # 足端位置（四足）
    contact_states: Optional[np.ndarray] = None  # 接触状态


class KalmanFilter:
    """简化版卡尔曼滤波器（用于传感器融合）"""
    
    def __init__(self, state_dim: int, measurement_dim: int):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        
        # 状态向量
        self.x = np.zeros(state_dim)
        
        # 状态协方差
        self.P = np.eye(state_dim)
        
        # 过程噪声
        self.Q = np.eye(state_dim) * 0.01
        
        # 观测噪声
        self.R = np.eye(measurement_dim) * 0.1
        
        # 状态转移矩阵（假设恒速模型）
        self.F = np.eye(state_dim)
        
        # 观测矩阵
        self.H = np.zeros((measurement_dim, state_dim))
        self.H[:measurement_dim, :measurement_dim] = np.eye(measurement_dim)
    
    def predict(self, dt: float):
        """预测步骤"""
        # 更新状态转移矩阵（考虑时间步长）
        for i in range(self.state_dim // 2):
            self.F[i, i + self.state_dim // 2] = dt
        
        # 状态预测
        self.x = self.F @ self.x
        
        # 协方差预测
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, z: np.ndarray):
        """更新步骤"""
        # 卡尔曼增益
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 状态更新
        y = z - self.H @ self.x  # 观测残差
        self.x = self.x + K @ y
        
        # 协方差更新
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P


class StateEstimator:
    """
    机器人状态估计器
    
    融合 IMU、编码器、足端接触传感器，估计：
    - 机身位置和速度
    - 机身姿态和角速度
    - 足端位置（四足机器人）
    """
    
    def __init__(self, num_joints: int = 12, use_ekf: bool = True):
        self.num_joints = num_joints
        self.use_ekf = use_ekf
        
        # 卡尔曼滤波器（估计机身位置和速度）
        if use_ekf:
            self.kf = KalmanFilter(state_dim=6, measurement_dim=3)
            self.kf.x[:3] = np.array([0.0, 0.0, 0.3])  # 初始高度 0.3m
        
        # 状态缓存
        self._last_imu: Optional[IMUData] = None
        self._last_joint: Optional[JointData] = None
        self._last_update_time: float = time.time()
        
        # 速度积分（IMU）
        self._velocity_integral: np.ndarray = np.zeros(3)
        self._position_integral: np.ndarray = np.array([0.0, 0.0, 0.3])
        
        # 零速更新阈值
        self._zero_velocity_threshold: float = 0.1
        
    def update_imu(self, imu_data: IMUData):
        """更新 IMU 数据"""
        self._last_imu = imu_data
        
        # 积分计算速度（简化的惯性导航）
        if self._last_imu is not None:
            dt = imu_data.timestamp - self._last_imu.timestamp
            if dt > 0:
                # 去除重力
                gravity_world = np.array([0, 0, -9.81])
                
                # 转换加速度到世界坐标系
                R = self._quat_to_rotation_matrix(imu_data.quaternion)
                acc_world = R @ imu_data.linear_acceleration + gravity_world
                
                # 积分
                self._velocity_integral += acc_world * dt
                self._position_integral += self._velocity_integral * dt
    
    def update_joint(self, joint_data: JointData):
        """更新关节数据"""
        self._last_joint = joint_data
        self._last_update_time = time.time()
    
    def update_contact(self, contact_states: np.ndarray):
        """
        更新足端接触状态
        
        Args:
            contact_states: [num_legs] 0=swing, 1=stance
        """
        # 零速更新：如果所有足端都接触地面，假设速度为 0
        if np.all(contact_states == 1):
            self._velocity_integral *= 0.9  # 衰减而不是直接置零
    
    def estimate(self) -> RobotState:
        """
        估计当前机器人状态
        
        使用传感器融合算法：
        - IMU：高频姿态和角速度
        - 编码器：关节状态
        - 运动学：计算机身位置
        """
        if self._last_imu is None or self._last_joint is None:
            return RobotState(
                base_position=np.array([0.0, 0.0, 0.3]),
                base_velocity=np.zeros(3),
                base_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                base_ang_vel=np.zeros(3),
                joint_positions=np.zeros(self.num_joints),
                joint_velocities=np.zeros(self.num_joints),
            )
        
        # 直接使用 IMU 的姿态
        base_orientation = self._last_imu.quaternion
        base_ang_vel = self._last_imu.angular_velocity
        
        # 位置和速度估计（卡尔曼滤波或积分）
        if self.use_ekf:
            # 使用卡尔曼滤波器
            dt = time.time() - self._last_update_time
            self.kf.predict(dt)
            
            # 观测更新（如果有位置观测，如视觉或足端位置）
            # self.kf.update(measurement)
            
            base_position = self.kf.x[:3]
            base_velocity = self.kf.x[3:]
        else:
            # 直接使用积分结果
            base_position = self._position_integral
            base_velocity = self._velocity_integral
        
        return RobotState(
            base_position=base_position,
            base_velocity=base_velocity,
            base_orientation=base_orientation,
            base_ang_vel=base_ang_vel,
            joint_positions=self._last_joint.positions,
            joint_velocities=self._last_joint.velocities,
        )
    
    def _quat_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """四元数转旋转矩阵"""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])


class SensorFusion:
    """
    多传感器融合
    
    结合：
    - IMU (1000Hz): 姿态、角速度、线加速度
    - 编码器 (100Hz): 关节位置/速度
    - 足端传感器 (100Hz): 接触力/状态
    - 视觉/激光雷达 (30Hz): 全局位置
    """
    
    def __init__(self):
        self.imu_buffer: List[IMUData] = []
        self.joint_buffer: List[JointData] = []
        self.buffer_size = 100
        
        # 时间同步容差 (秒)
        self.sync_tolerance = 0.01
    
    def add_imu(self, imu: IMUData):
        """添加 IMU 数据"""
        self.imu_buffer.append(imu)
        if len(self.imu_buffer) > self.buffer_size:
            self.imu_buffer.pop(0)
    
    def add_joint(self, joint: JointData):
        """添加关节数据"""
        self.joint_buffer.append(joint)
        if len(self.joint_buffer) > self.buffer_size:
            self.joint_buffer.pop(0)
    
    def get_synced_data(self) -> Tuple[Optional[IMUData], Optional[JointData]]:
        """
        获取时间同步的传感器数据
        
        Returns:
            (imu_data, joint_data) 或 (None, None) 如果找不到同步数据
        """
        if not self.imu_buffer or not self.joint_buffer:
            return None, None
        
        # 找到最近的同步对
        latest_joint = self.joint_buffer[-1]
        
        # 在 IMU 缓冲区中找到最接近的
        best_imu = None
        best_dt = float('inf')
        
        for imu in self.imu_buffer:
            dt = abs(imu.timestamp - latest_joint.timestamp)
            if dt < best_dt:
                best_dt = dt
                best_imu = imu
        
        if best_dt < self.sync_tolerance:
            return best_imu, latest_joint
        
        return None, None
