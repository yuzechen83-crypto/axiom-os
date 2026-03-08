"""
Axiom-OS Envs: Gymnasium + MuJoCo 集成

AI 控制领域的「高考」：Swimmer / Hopper / Walker2d / Humanoid
- Gymnasium: 标准 API (reset, step, reward)
- MuJoCo: 高精度物理引擎（接触、摩擦、多关节动力学）

Axiom 接入：
- Soft Shell 充当 Policy 网络
- Hard Core 可读取 MuJoCo XML 物理参数（质量、长度）
- Sim-to-Real: 人为修改重力/摩擦系数，测试 Axiom 自适应能力
"""

from .mujoco_env import (
    MuJoCoPhysicsParams,
    SimToRealWrapper,
    DomainRandomizationWrapper,
    PhysicsAwareObsWrapper,
    make_axiom_mujoco_env,
    make_axiom_mujoco_env_with_domain_rand,
    get_physics_params,
    MUJOCO_ENV_IDS,
)

__all__ = [
    "MuJoCoPhysicsParams",
    "SimToRealWrapper",
    "DomainRandomizationWrapper",
    "PhysicsAwareObsWrapper",
    "make_axiom_mujoco_env",
    "make_axiom_mujoco_env_with_domain_rand",
    "get_physics_params",
    "MUJOCO_ENV_IDS",
]
