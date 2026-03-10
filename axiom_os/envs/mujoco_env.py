"""
Gymnasium + MuJoCo 集成：Axiom-OS 控制测试平台

测试科目：
- Swimmer / Hopper / Walker2d：RCLN 处理复杂多刚体动力学
- Humanoid：终极 BOSS，MPC + 爱因斯坦模块 vs PPO

Hard Core：从 MuJoCo XML 读取质量、长度等物理参数
Sim-to-Real：修改重力/摩擦，测试 Axiom 自适应能力
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

# 可选依赖
try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


MUJOCO_ENV_IDS = [
    "Swimmer-v5",
    "Hopper-v5",
    "Walker2d-v5",
    "Humanoid-v5",
]


@dataclass
class MuJoCoPhysicsParams:
    """
    从 MuJoCo XML 提取的物理参数，供 Hard Core 使用。
    支持 Swimmer/Hopper/Walker2d/Humanoid 等。
    """

    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81]))
    body_masses: np.ndarray = field(default_factory=lambda: np.array([]))
    body_lengths: np.ndarray = field(default_factory=lambda: np.array([]))
    friction: float = 1.0
    damping: float = 0.0
    xml_path: Optional[str] = None
    model_name: str = ""

    @classmethod
    def from_env(cls, env: Any) -> "MuJoCoPhysicsParams":
        """从 Gymnasium MuJoCo 环境提取物理参数"""
        if not HAS_MUJOCO:
            return cls()
        e = env
        while hasattr(e, "env"):
            e = e.env
        if hasattr(e, "unwrapped"):
            e = e.unwrapped
        model = getattr(e, "model", None)
        if model is None:
            return cls()
        out = cls()
        out.gravity = np.array(model.opt.gravity, dtype=np.float64)
        # 提取 body 质量
        if hasattr(model, "body_mass"):
            out.body_masses = np.array(model.body_mass, dtype=np.float64)
        # 摩擦（geom 或 actuator）
        if hasattr(model, "geom_friction"):
            out.friction = float(np.mean(model.geom_friction))
        out.model_name = getattr(model, "names", "unknown")
        return out

    @classmethod
    def from_xml(cls, xml_path: str) -> "MuJoCoPhysicsParams":
        """从 MuJoCo XML 文件直接加载参数"""
        if not HAS_MUJOCO:
            return cls(xml_path=xml_path)
        model = mujoco.MjModel.from_xml_path(xml_path)
        out = cls()
        out.gravity = np.array(model.opt.gravity, dtype=np.float64)
        if hasattr(model, "body_mass"):
            out.body_masses = np.array(model.body_mass, dtype=np.float64)
        if hasattr(model, "geom_friction"):
            out.friction = float(np.mean(model.geom_friction))
        out.xml_path = xml_path
        return out

    def to_hard_core_dict(self) -> Dict[str, Any]:
        """转为 Hard Core 可用的参数字典"""
        return {
            "gravity": self.gravity.tolist(),
            "body_masses": self.body_masses.tolist(),
            "friction": self.friction,
            "damping": self.damping,
        }


class DomainRandomizationWrapper(gym.Wrapper):
    """
    域随机化：每次 reset 时随机采样 gravity/friction，提高对 Sim-to-Real 扰动的泛化。
    训练时使用，使策略对物理参数变化更不敏感。
    """

    def __init__(
        self,
        env: Any,
        gravity_range: Tuple[float, float] = (0.9, 1.1),
        friction_range: Tuple[float, float] = (0.8, 1.2),
        seed: Optional[int] = None,
    ):
        if not HAS_GYM:
            raise ImportError("请安装 gymnasium: pip install gymnasium")
        super().__init__(env)
        self.gravity_range = gravity_range
        self.friction_range = friction_range
        self._rng = np.random.default_rng(seed)
        self._original_gravity: Optional[np.ndarray] = None
        self._original_friction: Optional[float] = None
        self._current_gravity_scale = 1.0
        self._current_friction_scale = 1.0

    def _sample_and_apply(self) -> None:
        """每次 reset 随机采样并应用"""
        if not HAS_MUJOCO:
            return
        env = self.env
        while hasattr(env, "env"):
            env = env.env
        model = getattr(env, "model", None)
        if model is None:
            return
        if self._original_gravity is None:
            self._original_gravity = np.array(model.opt.gravity, dtype=np.float64)
        if self._original_friction is None and hasattr(model, "geom_friction"):
            self._original_friction = float(np.mean(model.geom_friction))

        self._current_gravity_scale = float(
            self._rng.uniform(self.gravity_range[0], self.gravity_range[1])
        )
        self._current_friction_scale = float(
            self._rng.uniform(self.friction_range[0], self.friction_range[1])
        )
        model.opt.gravity[:] = self._original_gravity * self._current_gravity_scale
        if hasattr(model, "geom_friction") and self._original_friction is not None:
            model.geom_friction[:] = (
                self._original_friction * self._current_friction_scale
            )

    def reset(self, *args, **kwargs) -> Tuple[Any, Dict]:
        self._sample_and_apply()
        obs, info = self.env.reset(*args, **kwargs)
        info = dict(info) if info else {}
        info["gravity_scale"] = self._current_gravity_scale
        info["friction_scale"] = self._current_friction_scale
        return obs, info

    def step(self, action: Any):
        return self.env.step(action)

    def get_current_physics_scale(self) -> Tuple[float, float]:
        """返回当前 episode 的 gravity_scale, friction_scale（供 Physics-aware policy）"""
        return self._current_gravity_scale, self._current_friction_scale


class SimToRealWrapper(gym.Wrapper):
    """
    Sim-to-Real 扰动包装器：人为修改重力或摩擦，测试 Axiom 自适应能力。

    用法：
        env = gym.make("Hopper-v5")
        env = SimToRealWrapper(env, gravity_scale=1.1, friction_scale=0.8)
    """

    def __init__(
        self,
        env: Any,
        gravity_scale: float = 1.0,
        gravity_offset: Optional[np.ndarray] = None,
        friction_scale: float = 1.0,
        friction_offset: float = 0.0,
    ):
        if not HAS_GYM:
            raise ImportError("请安装 gymnasium: pip install gymnasium")
        super().__init__(env)
        self.gravity_scale = gravity_scale
        self.gravity_offset = gravity_offset or np.zeros(3)
        self.friction_scale = friction_scale
        self.friction_offset = friction_offset
        self._original_gravity: Optional[np.ndarray] = None
        self._original_friction: Optional[float] = None

    def _apply_perturbation(self) -> None:
        """应用重力/摩擦扰动"""
        if not HAS_MUJOCO:
            return
        env = self.env
        while hasattr(env, "env"):
            env = env.env
        model = getattr(env, "model", None)
        if model is None:
            return
        if self._original_gravity is None:
            self._original_gravity = np.array(model.opt.gravity, dtype=np.float64)
        if self._original_friction is None and hasattr(model, "geom_friction"):
            self._original_friction = float(np.mean(model.geom_friction))

        model.opt.gravity[:] = self._original_gravity * self.gravity_scale + self.gravity_offset
        if hasattr(model, "geom_friction") and self._original_friction is not None:
            new_f = self._original_friction * self.friction_scale + self.friction_offset
            model.geom_friction[:] = new_f

    def reset(self, *args, **kwargs) -> Tuple[Any, Dict]:
        self._apply_perturbation()
        return self.env.reset(*args, **kwargs)

    def step(self, action: Any):
        return self.env.step(action)


def make_axiom_mujoco_env_with_domain_rand(
    env_id: str = "Hopper-v5",
    gravity_range: Tuple[float, float] = (0.9, 1.1),
    friction_range: Tuple[float, float] = (0.8, 1.2),
    seed: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """创建带域随机化的环境（用于训练，提高 Sim-to-Real 鲁棒性）"""
    if not HAS_GYM:
        raise ImportError("请安装 gymnasium 和 mujoco")
    env = gym.make(env_id, **kwargs)
    env = DomainRandomizationWrapper(
        env,
        gravity_range=gravity_range,
        friction_range=friction_range,
        seed=seed,
    )
    return env


def make_axiom_mujoco_env(
    env_id: str = "Hopper-v5",
    gravity_scale: float = 1.0,
    friction_scale: float = 1.0,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """
    创建 Axiom 可用的 MuJoCo 环境。

    Args:
        env_id: Swimmer-v5 | Hopper-v5 | Walker2d-v5 | Humanoid-v5
        gravity_scale: 重力缩放（Sim-to-Real 测试）
        friction_scale: 摩擦缩放（Sim-to-Real 测试）
        seed: 随机种子
        **kwargs: 传给 gymnasium.make 的额外参数

    Returns:
        Gymnasium 环境（可选 SimToRealWrapper）
    """
    if not HAS_GYM:
        raise ImportError(
            "请安装 gymnasium 和 mujoco: pip install gymnasium[mujoco] 或 pip install gymnasium mujoco"
        )
    env = gym.make(env_id, **kwargs)
    if seed is not None:
        env.reset(seed=seed)
    if abs(gravity_scale - 1.0) > 1e-6 or abs(friction_scale - 1.0) > 1e-6:
        env = SimToRealWrapper(env, gravity_scale=gravity_scale, friction_scale=friction_scale)
    return env


class PhysicsAwareObsWrapper(gym.Wrapper):
    """
    将物理参数追加到观测，供 PPO/SAC 等算法使用物理感知策略。
    需与 DomainRandomizationWrapper 配合：info 中的 gravity_scale、friction_scale 会传入。
    """

    def __init__(
        self,
        env: Any,
        physics_dim: int = 8,
        eval_gravity_scale: Optional[float] = None,
        eval_friction_scale: Optional[float] = None,
    ):
        if not HAS_GYM:
            raise ImportError("请安装 gymnasium")
        super().__init__(env)
        self.physics_dim = physics_dim
        self._eval_g = eval_gravity_scale
        self._eval_f = eval_friction_scale
        self._physics: np.ndarray = np.zeros(physics_dim, dtype=np.float32)
        self._mass_stats: Optional[np.ndarray] = None
        self._init_mass_stats()
        orig = self.env.observation_space
        if hasattr(orig, "shape"):
            new_shape = (orig.shape[0] + self.physics_dim,)
            self.observation_space = gym.spaces.Box(
                low=np.concatenate([np.asarray(orig.low), np.full(self.physics_dim, -np.inf)]),
                high=np.concatenate([np.asarray(orig.high), np.full(self.physics_dim, np.inf)]),
                shape=new_shape,
                dtype=orig.dtype if hasattr(orig, "dtype") else np.float32,
            )
        else:
            self.observation_space = orig

    def _init_mass_stats(self) -> None:
        params = get_physics_params(self.env)
        if params.body_masses is not None and len(params.body_masses) > 0:
            m = np.asarray(params.body_masses, dtype=np.float64)
            mean_m = np.mean(m)
            std_m = np.std(m) if len(m) > 1 else 0.0
            self._mass_stats = np.array([
                np.log1p(mean_m), np.log1p(std_m + 1e-8),
                np.log1p(np.max(m)), np.log1p(np.sum(m)),
            ], dtype=np.float32)
        else:
            self._mass_stats = np.zeros(4, dtype=np.float32)

    def _build_physics(self, g_scale: float = 1.0, f_scale: float = 1.0) -> np.ndarray:
        base = np.array([g_scale, f_scale, 9.81 * abs(g_scale), 1.0 * f_scale], dtype=np.float32)
        return np.concatenate([base, self._mass_stats])

    def reset(self, *args, **kwargs) -> Tuple[Any, Dict]:
        obs, info = self.env.reset(*args, **kwargs)
        g = self._eval_g if self._eval_g is not None else info.get("gravity_scale", 1.0)
        f = self._eval_f if self._eval_f is not None else info.get("friction_scale", 1.0)
        self._physics = self._build_physics(g, f)
        obs_ext = np.concatenate([np.asarray(obs, dtype=np.float32), self._physics])
        return obs_ext, info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        obs, reward, term, trunc, info = self.env.step(action)
        obs_ext = np.concatenate([np.asarray(obs, dtype=np.float32), self._physics])
        return obs_ext, reward, term, trunc, info


def get_physics_params(env: Any) -> MuJoCoPhysicsParams:
    """从环境中提取物理参数（用于 Hard Core）"""
    # 若被 SimToRealWrapper 包装，需要 unwrap 到原始 env
    e = env
    while hasattr(e, "env"):
        e = e.env
    return MuJoCoPhysicsParams.from_env(e)
