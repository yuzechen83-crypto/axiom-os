"""
Safety Layer - 安全保护层
Copyright (c) 2026 yuzechen83-crypto. All Rights Reserved.

真实机器人的安全至关重要！
- 紧急停止 (Emergency Stop)
- 关节限位检查
- 摔倒检测与恢复
- 碰撞避免
"""

from __future__ import annotations

from typing import Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
import time

import numpy as np


class SafetyLevel(Enum):
    """安全等级"""
    NORMAL = 0      # 正常运行
    WARNING = 1     # 警告（接近限位）
    CRITICAL = 2    # 危险（需要干预）
    EMERGENCY = 3   # 紧急停止


@dataclass
class SafetyConstraint:
    """安全约束配置"""
    
    # 关节限位
    joint_pos_limits: tuple  # (min, max) in rad
    joint_vel_limits: tuple  # (min, max) in rad/s
    joint_torque_limits: tuple  # (min, max) in Nm
    
    # 机身姿态限制
    max_roll: float = 0.8  # rad (~45度)
    max_pitch: float = 0.8  # rad
    
    # 高度限制
    min_height: float = 0.15  # m
    max_height: float = 1.0   # m
    
    # 速度限制
    max_linear_vel: float = 2.0  # m/s
    max_angular_vel: float = 3.0  # rad/s
    
    # 力限制
    max_ground_reaction_force: float = 500.0  # N


class SafetyLayer:
    """
    安全保护层
    
    职责：
    1. 实时监控机器人状态
    2. 检测危险情况
    3. 触发安全响应（限幅、恢复、急停）
    4. 记录安全事件
    """
    
    def __init__(self, constraints: SafetyConstraint, num_joints: int = 12):
        self.constraints = constraints
        self.num_joints = num_joints
        
        # 当前安全状态
        self.safety_level = SafetyLevel.NORMAL
        self.last_violation: Optional[str] = None
        self.violation_count = 0
        
        # 紧急停止回调
        self._emergency_stop_callbacks: List[Callable] = []
        
        # 安全事件日志
        self._safety_log: List[dict] = []
        
        # 恢复策略
        self._recovery_mode = False
        self._recovery_start_time: Optional[float] = None
    
    def register_emergency_callback(self, callback: Callable):
        """注册紧急停止回调函数"""
        self._emergency_stop_callbacks.append(callback)
    
    def check_state(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        base_quat: np.ndarray,
        base_ang_vel: np.ndarray,
        base_height: float,
    ) -> SafetyLevel:
        """
        检查当前状态是否安全
        
        Returns:
            SafetyLevel: 当前安全等级
        """
        violations = []
        
        # 1. 检查关节位置限位
        pos_min, pos_max = self.constraints.joint_pos_limits
        if np.any(joint_pos < pos_min) or np.any(joint_pos > pos_max):
            violations.append(f"Joint position out of limits: {joint_pos}")
        
        # 2. 检查关节速度
        vel_min, vel_max = self.constraints.joint_vel_limits
        if np.any(np.abs(joint_vel) > vel_max):
            violations.append(f"Joint velocity too high: {np.max(np.abs(joint_vel)):.2f}")
        
        # 3. 检查机身姿态（从四元数提取 roll/pitch）
        roll, pitch = self._quat_to_rpy(base_quat)[:2]
        if abs(roll) > self.constraints.max_roll:
            violations.append(f"Roll too large: {roll:.2f}")
        if abs(pitch) > self.constraints.max_pitch:
            violations.append(f"Pitch too large: {pitch:.2f}")
        
        # 4. 检查高度
        if base_height < self.constraints.min_height:
            violations.append(f"Height too low: {base_height:.3f}")
        if base_height > self.constraints.max_height:
            violations.append(f"Height too high: {base_height:.3f}")
        
        # 5. 检查角速度
        if np.linalg.norm(base_ang_vel) > self.constraints.max_angular_vel:
            violations.append(f"Angular velocity too high: {np.linalg.norm(base_ang_vel):.2f}")
        
        # 确定安全等级
        if violations:
            self.violation_count += 1
            self.last_violation = violations[0]
            
            # 记录安全事件
            self._safety_log.append({
                "timestamp": time.time(),
                "level": "CRITICAL",
                "violations": violations,
            })
            
            # 判断严重程度
            if base_height < 0.1 or abs(roll) > 1.2 or abs(pitch) > 1.2:
                # 摔倒检测
                self.safety_level = SafetyLevel.EMERGENCY
                self._trigger_emergency_stop("FALL DETECTED")
            else:
                self.safety_level = SafetyLevel.WARNING
        else:
            self.safety_level = SafetyLevel.NORMAL
            self.last_violation = None
        
        return self.safety_level
    
    def filter_action(
        self,
        action: np.ndarray,
        joint_pos: np.ndarray,
        safety_margin: float = 0.9
    ) -> np.ndarray:
        """
        过滤/限幅动作，确保不超过安全约束
        
        Args:
            action: 原始动作
            joint_pos: 当前关节位置
            safety_margin: 安全裕度 (0-1)
        
        Returns:
            过滤后的安全动作
        """
        pos_min, pos_max = self.constraints.joint_pos_limits
        
        # 计算目标位置
        target_pos = joint_pos + action
        
        # 应用限位（带安全裕度）
        margin = (pos_max - pos_min) * (1 - safety_margin) / 2
        safe_min = pos_min + margin
        safe_max = pos_max - margin
        
        target_pos_clipped = np.clip(target_pos, safe_min, safe_max)
        
        # 转换回动作
        safe_action = target_pos_clipped - joint_pos
        
        return safe_action
    
    def get_recovery_action(self, current_state: dict) -> Optional[np.ndarray]:
        """
        生成恢复动作（当机器人摔倒或不稳定时）
        
        Returns:
            恢复动作或 None（如果无法恢复）
        """
        if not self._recovery_mode:
            self._recovery_mode = True
            self._recovery_start_time = time.time()
        
        recovery_time = time.time() - self._recovery_start_time
        
        # 简单的恢复策略：
        # 1. 0-1s: 保持当前姿势，稳定
        # 2. 1-3s: 缓慢回到初始位置
        # 3. 3s+: 退出恢复模式
        
        if recovery_time < 1.0:
            # 阶段1：保持稳定
            return np.zeros(self.num_joints)
        elif recovery_time < 3.0:
            # 阶段2：回到初始位置（假设初始为0）
            current_pos = current_state.get("joint_pos", np.zeros(self.num_joints))
            recovery_action = -current_pos * 0.5  # 按比例回到0
            return recovery_action
        else:
            # 恢复完成
            self._recovery_mode = False
            self.safety_level = SafetyLevel.NORMAL
            return None
    
    def _trigger_emergency_stop(self, reason: str):
        """触发紧急停止"""
        print(f"[EMERGENCY STOP] {reason}")
        
        # 执行所有注册的回调
        for callback in self._emergency_stop_callbacks:
            try:
                callback(reason)
            except Exception as e:
                print(f"[Error] Emergency callback failed: {e}")
    
    def _quat_to_rpy(self, q: np.ndarray) -> np.ndarray:
        """四元数转欧拉角 (roll, pitch, yaw)"""
        w, x, y, z = q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    
    def get_safety_report(self) -> dict:
        """获取安全状态报告"""
        return {
            "current_level": self.safety_level.name,
            "violation_count": self.violation_count,
            "last_violation": self.last_violation,
            "recovery_mode": self._recovery_mode,
            "log_entries": len(self._safety_log),
        }
