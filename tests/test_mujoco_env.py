"""
MuJoCo 测试科目：Swimmer / Hopper / Walker2d / Humanoid

需安装: pip install gymnasium mujoco 或 pip install axiom-os[mujoco]
"""

import pytest

try:
    import gymnasium
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

if HAS_MUJOCO:
    from axiom_os.envs import (
        make_axiom_mujoco_env,
        get_physics_params,
        MuJoCoPhysicsParams,
        SimToRealWrapper,
        MUJOCO_ENV_IDS,
    )
    from axiom_os.envs.axiom_policy import make_mujoco_policy, MuJoCoPolicy

pytestmark = pytest.mark.skipif(not HAS_MUJOCO, reason="需要 gymnasium 和 mujoco")


@pytest.mark.parametrize("env_id", MUJOCO_ENV_IDS)
def test_env_reset_step(env_id: str):
    """测试科目：环境 reset/step 正常"""
    env = make_axiom_mujoco_env(env_id=env_id, seed=42)
    obs, info = env.reset(seed=42)
    assert obs is not None
    assert obs.ndim == 1
    action = env.action_space.sample()
    obs2, reward, term, trunc, info2 = env.step(action)
    assert obs2 is not None
    assert isinstance(reward, (int, float))
    env.close()


@pytest.mark.parametrize("env_id", MUJOCO_ENV_IDS)
def test_physics_params(env_id: str):
    """测试科目：物理参数提取（供 Hard Core）"""
    env = make_axiom_mujoco_env(env_id=env_id, seed=42)
    params = get_physics_params(env)
    assert isinstance(params, MuJoCoPhysicsParams)
    assert params.gravity.shape == (3,)
    assert len(params.body_masses) >= 1
    d = params.to_hard_core_dict()
    assert "gravity" in d and "friction" in d
    env.close()


def test_sim_to_real_wrapper():
    """Sim-to-Real：重力/摩擦扰动"""
    env = make_axiom_mujoco_env("Hopper-v5", gravity_scale=1.1, friction_scale=0.9, seed=42)
    obs, _ = env.reset(seed=42)
    action = env.action_space.sample()
    obs2, reward, _, _, _ = env.step(action)
    assert obs2 is not None
    env.close()


@pytest.mark.parametrize("env_id", MUJOCO_ENV_IDS)
def test_policy_rollout(env_id: str):
    """测试科目：Soft Shell Policy  rollout"""
    env = make_axiom_mujoco_env(env_id=env_id, seed=42)
    obs, _ = env.reset(seed=42)
    policy = make_mujoco_policy(obs_dim=obs.shape[0], action_dim=env.action_space.shape[0])
    for _ in range(10):
        action = policy.act(obs, deterministic=True)
        obs, reward, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset()
    env.close()
