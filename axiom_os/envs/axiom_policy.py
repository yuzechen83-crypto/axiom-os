"""
Axiom Policy：Soft Shell 作为 MuJoCo 环境的 Policy 网络

将 RCLN Soft Shell 或 StudentPolicy 适配为 Gymnasium 的连续动作策略。
支持 Swimmer / Hopper / Walker2d / Humanoid。
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn


class MuJoCoPolicy(nn.Module):
    """
    Soft Shell 风格的 Policy：obs -> action。
    适配 MuJoCo 的连续动作空间，输出 tanh 缩放到 env.action_space。
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        layers = []
        d = obs_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(d, hidden_dim), nn.ReLU()])
            d = hidden_dim
        layers.append(nn.Linear(d, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))

    def act(
        self,
        obs: np.ndarray,
        device: Optional[Any] = None,
        deterministic: bool = True,
    ) -> np.ndarray:
        """选择动作（用于 rollout）"""
        if device is None:
            device = next(self.parameters()).device
        x = torch.as_tensor(obs, dtype=torch.float32, device=device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            a = self.forward(x)
        out = a.cpu().numpy()
        return out[0] if obs.ndim == 1 else out


def make_mujoco_policy(
    obs_dim: int,
    action_dim: int,
    hidden_dim: int = 256,
) -> MuJoCoPolicy:
    """创建 MuJoCo Policy（Soft Shell 风格 MLP）"""
    return MuJoCoPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
    )


def _body_masses_stats(masses: np.ndarray) -> np.ndarray:
    """从 body_masses 提取统计量，归一化后供 Policy 使用"""
    if masses is None or len(masses) == 0:
        return np.zeros(4, dtype=np.float32)
    m = np.asarray(masses, dtype=np.float64)
    mean_m = np.mean(m)
    std_m = np.std(m) if len(m) > 1 else 0.0
    max_m = np.max(m)
    total_m = np.sum(m)
    return np.array([
        np.log1p(mean_m),
        np.log1p(std_m + 1e-8),
        np.log1p(max_m),
        np.log1p(total_m),
    ], dtype=np.float32)


class PhysicsAwarePolicy(nn.Module):
    """
    Hard Core 物理感知 Policy：将 MuJoCo 物理参数作为额外输入。
    输入: obs + [gravity_scale, friction_scale, |g|, friction] + [mean_m, std_m, max_m, total_m](body_masses)
    physics_dim=8 含 body_masses 统计量
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        physics_dim: int = 8,
        hidden_dim: int = 256,
        n_layers: int = 2,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.physics_dim = physics_dim
        input_dim = obs_dim + physics_dim
        layers = []
        d = input_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(d, hidden_dim), nn.ReLU()])
            d = hidden_dim
        layers.append(nn.Linear(d, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, physics: Optional[torch.Tensor] = None) -> torch.Tensor:
        if physics is not None and physics.numel() > 0:
            x = torch.cat([obs, physics], dim=-1)
        else:
            pad = torch.zeros(obs.shape[0], self.physics_dim, device=obs.device, dtype=obs.dtype)
            pad[:, 0] = 1.0
            pad[:, 1] = 1.0
            x = torch.cat([obs, pad], dim=-1)
        return torch.tanh(self.net(x))

    def act(
        self,
        obs: np.ndarray,
        device: Optional[Any] = None,
        deterministic: bool = True,
        gravity_scale: float = 1.0,
        friction_scale: float = 1.0,
        body_masses: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if device is None:
            device = next(self.parameters()).device
        x = torch.as_tensor(obs, dtype=torch.float32, device=device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        base = [gravity_scale, friction_scale, 9.81 * abs(gravity_scale), 1.0 * friction_scale]
        mass_stats = _body_masses_stats(body_masses)
        physics = torch.tensor(
            [base + mass_stats.tolist()],
            dtype=torch.float32,
            device=device,
        ).expand(x.shape[0], -1)
        with torch.no_grad():
            a = self.forward(x, physics)
        out = a.cpu().numpy()
        return out[0] if obs.ndim == 1 else out


def train_policy_reinforce(
    policy: MuJoCoPolicy,
    env: Any,
    episodes: int = 50,
    steps_per_episode: int = 200,
    lr: float = 1e-3,
    gamma: float = 0.99,
    seed: Optional[int] = None,
) -> list:
    """
    简单 REINFORCE 训练，改进 Policy。
    若 policy 为 PhysicsAwarePolicy，传入标准 physics (g=1, f=1, body_masses)。
    返回每 episode 的奖励列表。
    """
    from axiom_os.envs import get_physics_params

    use_physics = isinstance(policy, PhysicsAwarePolicy)
    body_masses = get_physics_params(env).body_masses if use_physics else None
    mass_stats = _body_masses_stats(body_masses) if use_physics else None
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    rewards_history = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep if seed is not None else None)
        log_probs = []
        rewards = []
        for _ in range(steps_per_episode):
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            if use_physics:
                base = [1.0, 1.0, 9.81, 1.0]
                phys = torch.tensor([base + mass_stats.tolist()], dtype=torch.float32).expand(x.shape[0], -1)
                a_raw = policy.net(torch.cat([x, phys], dim=-1))
            else:
                a_raw = policy.net(x)
            a = torch.tanh(a_raw)
            std = torch.ones_like(a) * 0.2
            dist = torch.distributions.Normal(a, std)
            a_samp = dist.sample().clamp(-1.0, 1.0)
            log_prob = dist.log_prob(a_samp).sum(dim=-1)
            action = a_samp[0].detach().numpy()
            obs, reward, term, trunc, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            if term or trunc:
                obs, _ = env.reset()
        R = sum(rewards)
        rewards_history.append(float(R))
        if len(log_probs) > 0 and R > -1000:
            discounts = [gamma ** i for i in range(len(rewards))]
            returns = [sum(r * d for r, d in zip(rewards[i:], discounts[: len(rewards) - i]))
                       for i in range(len(rewards))]
            loss = -sum(lp * ret for lp, ret in zip(log_probs, returns)) / len(log_probs)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
    return rewards_history


def train_with_domain_rand(
    policy: Any,
    env_id: str,
    episodes: int = 50,
    steps_per_episode: int = 200,
    gravity_range: Tuple[float, float] = (0.9, 1.1),
    friction_range: Tuple[float, float] = (0.8, 1.2),
    lr: float = 1e-3,
    gamma: float = 0.99,
    seed: Optional[int] = None,
) -> list:
    """
    域随机化训练：每次 reset 随机 gravity/friction，提高对 Sim-to-Real 的泛化。
    若 policy 为 PhysicsAwarePolicy，将当前 physics 传入。
    """
    from axiom_os.envs import make_axiom_mujoco_env_with_domain_rand, get_physics_params

    env = make_axiom_mujoco_env_with_domain_rand(
        env_id=env_id,
        gravity_range=gravity_range,
        friction_range=friction_range,
        seed=seed,
    )
    use_physics = isinstance(policy, PhysicsAwarePolicy)
    body_masses = get_physics_params(env).body_masses if use_physics else None
    mass_stats = _body_masses_stats(body_masses)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    rewards_history = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep if seed is not None else None)
        g_scale = info.get("gravity_scale", 1.0) if use_physics else 1.0
        f_scale = info.get("friction_scale", 1.0) if use_physics else 1.0
        log_probs, rewards = [], []
        for _ in range(steps_per_episode):
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            if use_physics:
                base = [g_scale, f_scale, 9.81 * abs(g_scale), 1.0 * f_scale]
                phys = torch.tensor([base + mass_stats.tolist()], dtype=torch.float32).expand(x.shape[0], -1)
                a_raw = policy.net(torch.cat([x, phys], dim=-1))
            else:
                a_raw = policy.net(x)
            a = torch.tanh(a_raw)
            std = torch.ones_like(a) * 0.2
            dist = torch.distributions.Normal(a, std)
            a_samp = dist.sample().clamp(-1.0, 1.0)
            log_prob = dist.log_prob(a_samp).sum(dim=-1)
            action = a_samp[0].detach().numpy()
            obs, reward, term, trunc, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            if term or trunc:
                obs, info = env.reset()
                if use_physics:
                    g_scale, f_scale = info.get("gravity_scale", 1.0), info.get("friction_scale", 1.0)
        R = sum(rewards)
        rewards_history.append(float(R))
        if len(log_probs) > 0 and R > -1000:
            discounts = [gamma ** i for i in range(len(rewards))]
            returns = [sum(r * d for r, d in zip(rewards[i:], discounts[: len(rewards) - i])) for i in range(len(rewards))]
            loss = -sum(lp * ret for lp, ret in zip(log_probs, returns)) / len(log_probs)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
    return rewards_history


def train_ppo_sac(
    env_id: str,
    algo: str = "PPO",
    total_timesteps: int = 200_000,
    domain_rand: bool = True,
    physics_aware: bool = True,
    gravity_range: Tuple[float, float] = (0.9, 1.1),
    friction_range: Tuple[float, float] = (0.8, 1.2),
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    policy_kwargs: Optional[dict] = None,
    seed: Optional[int] = None,
) -> Any:
    """
    使用 PPO 或 SAC 训练，支持域随机化与物理感知。
    返回训练好的 SB3 模型。
    """
    from axiom_os.envs import (
        make_axiom_mujoco_env,
        make_axiom_mujoco_env_with_domain_rand,
        PhysicsAwareObsWrapper,
    )

    if domain_rand:
        env = make_axiom_mujoco_env_with_domain_rand(
            env_id=env_id,
            gravity_range=gravity_range,
            friction_range=friction_range,
            seed=seed,
        )
    else:
        env = make_axiom_mujoco_env(env_id=env_id, seed=seed)

    if physics_aware:
        env = PhysicsAwareObsWrapper(env)

    pk = dict(policy_kwargs) if policy_kwargs else {}
    net_arch = pk.pop("net_arch", None)
    if net_arch is None:
        net_arch = dict(pi=[256, 256, 128], vf=[256, 256, 128]) if algo.upper() == "PPO" else [256, 256, 128]
    pk["net_arch"] = net_arch

    if algo.upper() == "PPO":
        from stable_baselines3 import PPO
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=pk,
            seed=seed,
            verbose=0,
        )
    else:
        from stable_baselines3 import SAC
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=100_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            policy_kwargs=pk,
            seed=seed,
            verbose=0,
        )

    model.learn(total_timesteps=total_timesteps)
    env.close()
    return model
