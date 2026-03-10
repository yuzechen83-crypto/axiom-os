"""
Real Robot Interface - 真实机器人接口层
Copyright (c) 2026 yuzechen83-crypto. All Rights Reserved.

支持平台：
- Unitree Go1/Go2/A1 (四足机器人)
- Unitree H1/H2 (人形机器人)
- MIT Cheetah (迷你猎豹)
- ROS2 通用接口
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
from collections import deque

import numpy as np


class RobotPlatform(Enum):
    """支持的真实机器人平台"""
    UNITREE_GO1 = "unitree_go1"      # 宇树 Go1 四足
    UNITREE_GO2 = "unitree_go2"      # 宇树 Go2 四足
    UNITREE_H1 = "unitree_h1"        # 宇树 H1 人形
    UNITREE_H2 = "unitree_h2"        # 宇树 H2 人形
    MIT_CHEETAH = "mit_cheetah"      # MIT 猎豹
    ROS2_GENERIC = "ros2_generic"    # ROS2 通用接口
    MOCK = "mock"                     # 模拟模式（测试用）


@dataclass
class RobotConfig:
    """机器人配置"""
    platform: RobotPlatform = RobotPlatform.MOCK
    
    # 控制频率 (Hz)
    control_freq: float = 50.0  # 50Hz 是常见的机器人控制频率
    
    # 观测维度（与 MuJoCo 对齐）
    obs_dim: int = 0  # 自动检测
    action_dim: int = 0  # 自动检测
    
    # 硬件连接参数
    ip_address: str = "192.168.123.161"  # 默认 Unitree 局域网 IP
    port: int = 8080
    
    # ROS2 参数
    ros2_namespace: str = "/robot"
    ros2_joint_states_topic: str = "/joint_states"
    ros2_cmd_vel_topic: str = "/cmd_vel"
    
    # 安全参数
    max_joint_velocity: float = 10.0  # rad/s
    max_joint_torque: float = 20.0    # Nm
    emergency_stop_threshold: float = 0.5  # 摔倒检测阈值
    
    # Sim-to-Real 适配参数
    action_scale: float = 1.0  # 动作缩放
    action_offset: np.ndarray = field(default_factory=lambda: np.zeros(12))
    
    # 延迟补偿 (秒)
    actuator_delay: float = 0.02  # 20ms 典型延迟
    sensor_delay: float = 0.01    # 10ms 典型延迟


class RealRobotInterface:
    """
    真实机器人统一接口
    
    职责：
    1. 连接真实机器人硬件
    2. 读取传感器数据（关节位置、速度、IMU）
    3. 发送控制指令（关节位置/力矩）
    4. 处理通信延迟和丢包
    """
    
    def __init__(self, config: RobotConfig):
        self.config = config
        self.platform = config.platform
        
        # 状态变量
        self._joint_positions: np.ndarray = np.zeros(config.action_dim or 12)
        self._joint_velocities: np.ndarray = np.zeros(config.action_dim or 12)
        self._joint_torques: np.ndarray = np.zeros(config.action_dim or 12)
        self._imu_quat: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        self._imu_ang_vel: np.ndarray = np.zeros(3)
        self._imu_lin_acc: np.ndarray = np.zeros(3)
        
        # 控制指令缓冲区（延迟补偿）
        self._action_buffer: deque = deque(maxlen=10)
        
        # 通信状态
        self._is_connected: bool = False
        self._last_update_time: float = 0.0
        self._communication_latency: float = 0.0
        
        # 硬件接口（延迟初始化）
        self._hardware_interface: Optional[Any] = None
        
    def connect(self) -> bool:
        """连接真实机器人"""
        if self.platform == RobotPlatform.MOCK:
            self._is_connected = True
            print("[Mock Robot] Connected (simulation mode)")
            return True
        
        elif self.platform == RobotPlatform.UNITREE_GO1:
            return self._connect_unitree_go1()
        
        elif self.platform == RobotPlatform.ROS2_GENERIC:
            return self._connect_ros2()
        
        else:
            raise NotImplementedError(f"Platform {self.platform} not implemented")
    
    def _connect_unitree_go1(self) -> bool:
        """连接宇树 Go1 四足机器人"""
        try:
            # 尝试导入宇树 SDK
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
            
            ChannelFactoryInitialize(0, self.config.ip_address)
            
            # 初始化发布者和订阅者
            self._hardware_interface = {
                "low_cmd": None,  # Publisher
                "low_state": None,  # Subscriber
            }
            
            self._is_connected = True
            print(f"[Unitree Go1] Connected to {self.config.ip_address}")
            return True
            
        except ImportError:
            print("[Error] unitree_sdk2py not installed. Using MOCK mode.")
            self.platform = RobotPlatform.MOCK
            self._is_connected = True
            return True
    
    def _connect_ros2(self) -> bool:
        """连接 ROS2 机器人"""
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import JointState
            from geometry_msgs.msg import Twist
            
            rclpy.init()
            self._hardware_interface = Node("axiom_os_controller")
            
            # 创建订阅者和发布者
            self._hardware_interface.create_subscription(
                JointState,
                self.config.ros2_joint_states_topic,
                self._ros2_joint_callback,
                10
            )
            
            self._ros2_cmd_pub = self._hardware_interface.create_publisher(
                Twist,
                self.config.ros2_cmd_vel_topic,
                10
            )
            
            self._is_connected = True
            print(f"[ROS2] Connected to namespace {self.config.ros2_namespace}")
            return True
            
        except ImportError:
            print("[Error] ROS2 not installed. Using MOCK mode.")
            self.platform = RobotPlatform.MOCK
            self._is_connected = True
            return True
    
    def _ros2_joint_callback(self, msg: Any):
        """ROS2 关节状态回调"""
        self._joint_positions = np.array(msg.position)
        self._joint_velocities = np.array(msg.velocity)
        self._last_update_time = time.time()
    
    def read_sensors(self) -> Dict[str, np.ndarray]:
        """
        读取传感器数据
        
        Returns:
            dict: {
                "joint_pos": ndarray,
                "joint_vel": ndarray,
                "imu_quat": ndarray,
                "imu_ang_vel": ndarray,
                "imu_lin_acc": ndarray,
            }
        """
        if self.platform == RobotPlatform.MOCK:
            # 模拟模式：添加噪声
            self._joint_positions += np.random.normal(0, 0.01, size=self._joint_positions.shape)
            self._joint_velocities += np.random.normal(0, 0.1, size=self._joint_velocities.shape)
        
        # 检查通信超时
        if time.time() - self._last_update_time > 0.5:
            print("[Warning] Sensor data timeout!")
        
        return {
            "joint_pos": self._joint_positions.copy(),
            "joint_vel": self._joint_velocities.copy(),
            "joint_torque": self._joint_torques.copy(),
            "imu_quat": self._imu_quat.copy(),
            "imu_ang_vel": self._imu_ang_vel.copy(),
            "imu_lin_acc": self._imu_lin_acc.copy(),
        }
    
    def send_action(self, action: np.ndarray) -> bool:
        """
        发送控制指令
        
        Args:
            action: 归一化的动作 [-1, 1]，会被映射到关节位置/力矩
        
        Returns:
            bool: 是否发送成功
        """
        # 动作缩放和偏移（Sim-to-Real 适配）
        action_scaled = action * self.config.action_scale + self.config.action_offset
        
        # 裁剪到安全范围
        action_clipped = np.clip(action_scaled, -1.0, 1.0)
        
        # 添加到缓冲区（延迟补偿）
        self._action_buffer.append({
            "action": action_clipped,
            "timestamp": time.time()
        })
        
        if self.platform == RobotPlatform.MOCK:
            # 模拟模式：直接应用动作
            self._joint_positions += action_clipped * 0.1
            return True
        
        elif self.platform == RobotPlatform.UNITREE_GO1:
            return self._send_unitree_action(action_clipped)
        
        elif self.platform == RobotPlatform.ROS2_GENERIC:
            return self._send_ros2_action(action_clipped)
        
        return False
    
    def _send_unitree_action(self, action: np.ndarray) -> bool:
        """发送宇树控制指令"""
        # 实际实现需要使用宇树 SDK
        # 这里只是一个框架
        return True
    
    def _send_ros2_action(self, action: np.ndarray) -> bool:
        """发送 ROS2 控制指令"""
        if self._hardware_interface is None:
            return False
        
        # 创建 Twist 消息
        from geometry_msgs.msg import Twist
        cmd = Twist()
        cmd.linear.x = float(action[0])
        cmd.angular.z = float(action[1])
        
        self._ros2_cmd_pub.publish(cmd)
        return True
    
    def get_observation(self) -> np.ndarray:
        """
        获取与 MuJoCo 对齐的观测向量
        
        包括：
        - 关节位置 (正弦/余弦编码)
        - 关节速度
        - IMU 四元数
        - IMU 角速度
        - 上一时刻动作（延迟补偿）
        """
        sensors = self.read_sensors()
        
        # 构建观测（与 MuJoCo 环境对齐）
        joint_pos = sensors["joint_pos"]
        joint_vel = sensors["joint_vel"]
        imu_quat = sensors["imu_quat"]
        imu_ang_vel = sensors["imu_ang_vel"]
        
        # 正弦/余弦编码（周期性关节）
        joint_pos_sin = np.sin(joint_pos)
        joint_pos_cos = np.cos(joint_pos)
        
        # 上一时刻动作（用于延迟补偿）
        last_action = (self._action_buffer[-1]["action"] 
                      if self._action_buffer else np.zeros(self.config.action_dim))
        
        obs = np.concatenate([
            joint_pos_sin,
            joint_pos_cos,
            joint_vel,
            imu_quat,
            imu_ang_vel,
            last_action,
        ])
        
        return obs.astype(np.float32)
    
    def disconnect(self):
        """断开连接"""
        if self.platform == RobotPlatform.ROS2_GENERIC and self._hardware_interface:
            import rclpy
            self._hardware_interface.destroy_node()
            rclpy.shutdown()
        
        self._is_connected = False
        print(f"[{self.platform.value}] Disconnected")
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False
