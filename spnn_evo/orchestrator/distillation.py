"""
Policy Distillation - "Muscle Memory" Upgrade
Distill the expensive MPC (Teacher) into a fast MLP (Student).
Target: <1ms inference for 1kHz robotics (vs >20ms for 1000 rollouts).
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Callable, Any
from collections import deque
import time

from ..core.imagination import double_pendulum_H, step_controlled
from .mpc import ParallelImaginationController, angle_normalize

PI = np.pi


def _grad_H(H: Callable, qp: np.ndarray, state_dim: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Finite-diff gradient for numpy H."""
    eps = 1e-6
    qp = np.asarray(qp, dtype=np.float64)
    g = np.zeros_like(qp)
    for i in range(len(qp)):
        qp_plus = qp.copy()
        qp_plus[i] += eps
        qp_minus = qp.copy()
        qp_minus[i] -= eps
        g[i] = (H(qp_plus) - H(qp_minus)) / (2 * eps)
    return g[:state_dim], g[state_dim:]


# -----------------------------------------------------------------------------
# Environment: Double Pendulum Simulator
# -----------------------------------------------------------------------------


class DoublePendulumEnv:
    """
    Minimal double pendulum environment for data collection.
    State: (q1, q2, p1, p2) - 4D
    Action: tau (scalar torque on first joint)
    """

    def __init__(
        self,
        H: Optional[Callable] = None,
        dt: float = 0.02,
        friction: float = 0.1,
        noise_std: float = 0.02,
        max_steps: int = 500,
        seed: Optional[int] = None,
    ):
        self.H = H or double_pendulum_H(g_over_L=10.0, L1=1.0, L2=1.0)
        self.dt = dt
        self.friction = friction
        self.noise_std = noise_std
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._q = np.zeros(2)
        self._p = np.zeros(2)

    def reset(
        self,
        q: Optional[np.ndarray] = None,
        p: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Reset to initial state. Returns obs (4D)."""
        if q is not None and p is not None:
            self._q = np.asarray(q, dtype=np.float64).ravel()[:2]
            self._p = np.asarray(p, dtype=np.float64).ravel()[:2]
        else:
            # Near-upright with small perturbation
            self._q = np.array([PI - 0.1 * self._rng.random(), PI - 0.1 * self._rng.random()])
            self._p = 0.1 * self._rng.standard_normal(2)
        self._step_count = 0
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self._q, self._p]).astype(np.float32)

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, dict]:
        """
        action: tau (scalar)
        Returns: obs, reward, done, info
        """
        def grad_H(qp):
            return _grad_H(self.H, qp, state_dim=2)

        q_new, p_new = step_controlled(
            (self._q, self._p),
            self.dt,
            float(action),
            self.H,
            grad_H,
            torque_idx=0,
            friction=self.friction,
        )
        if self.noise_std > 0:
            q_new = q_new + self._rng.standard_normal(2).astype(np.float64) * self.noise_std
            p_new = p_new + self._rng.standard_normal(2).astype(np.float64) * self.noise_std

        self._q = q_new
        self._p = p_new
        self._step_count += 1

        # Simple reward: penalize deviation from upright
        err1 = angle_normalize(self._q[0] - PI)
        err2 = angle_normalize(self._q[1] - PI)
        reward = -float(err1**2 + err2**2 + 0.1 * (self._p[0]**2 + self._p[1]**2))

        done = self._step_count >= self.max_steps
        info = {}
        return self._get_obs(), reward, done, info


# -----------------------------------------------------------------------------
# Student Policy: Small MLP
# -----------------------------------------------------------------------------


class StudentPolicy(nn.Module):
    """
    Lightweight MLP: State (4D) -> 64 -> 64 -> Action (1D).
    Target: <1ms inference.
    """

    def __init__(
        self,
        state_dim: int = 4,
        hidden_dim: int = 64,
        action_dim: int = 1,
        action_bounds: Tuple[float, float] = (-25.0, 25.0),
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bounds = action_bounds

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return torch.clamp(out, self.action_bounds[0], self.action_bounds[1])

    def act(self, obs: np.ndarray, device: Optional[torch.device] = None) -> float:
        """Inference: obs -> action (numpy)."""
        self.eval()
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            if device is not None:
                x = x.to(device)
            out = self.forward(x)
            return float(out.cpu().numpy().ravel()[0])


# -----------------------------------------------------------------------------
# Replay Buffer
# -----------------------------------------------------------------------------


class ReplayBuffer:
    """Store (observation, mpc_action) pairs for distillation."""

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self._obs: List[np.ndarray] = []
        self._actions: List[float] = []

    def add(self, obs: np.ndarray, action: float) -> None:
        if len(self._obs) >= self.capacity:
            self._obs.pop(0)
            self._actions.pop(0)
        self._obs.append(np.asarray(obs, dtype=np.float32))
        self._actions.append(float(action))

    def __len__(self) -> int:
        return len(self._obs)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample batch for training."""
        n = len(self._obs)
        if n == 0:
            raise ValueError("Buffer empty")
        indices = np.random.choice(n, size=min(batch_size, n), replace=True)
        obs_batch = torch.from_numpy(np.stack([self._obs[i] for i in indices])).float()
        act_batch = torch.from_numpy(np.array([self._actions[i] for i in indices])).float().unsqueeze(-1)
        return obs_batch, act_batch

    def get_all(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all data (for full-batch training)."""
        if not self._obs:
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)
        return np.stack(self._obs), np.array(self._actions, dtype=np.float32)


# -----------------------------------------------------------------------------
# Policy Trainer: Data Collection + Training + DAgger
# -----------------------------------------------------------------------------


class PolicyTrainer:
    """
    Orchestrates MPC -> Student distillation.
    - Data Collection: Run MPC on env, store (obs, mpc_action)
    - Training: MSE(Student(obs), mpc_action)
    - DAgger: Student drives, MPC labels, aggregate & retrain
    """

    def __init__(
        self,
        mpc: ParallelImaginationController,
        env: DoublePendulumEnv,
        student: Optional[StudentPolicy] = None,
        buffer_capacity: int = 100_000,
        device: Optional[torch.device] = None,
    ):
        self.mpc = mpc
        self.env = env
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.student = student or StudentPolicy(
            state_dim=4,
            hidden_dim=64,
            action_dim=1,
            action_bounds=getattr(mpc, "action_bounds", (-25.0, 25.0)),
        ).to(self.device)
        self.buffer = ReplayBuffer(capacity=buffer_capacity)

    def collect_mpc_data(
        self,
        n_episodes: int = 50,
        steps_per_episode: Optional[int] = None,
        verbose: bool = True,
    ) -> int:
        """
        Run MPC on environment for N episodes.
        Store (Observation, MPC_Action) in replay buffer.
        Returns: number of transitions collected.
        """
        steps_per_episode = steps_per_episode or self.env.max_steps
        total = 0

        for ep in range(n_episodes):
            obs = self.env.reset()
            done = False
            step = 0

            while not done and step < steps_per_episode:
                q, p = obs[:2], obs[2:4]
                mpc_action = self.mpc.plan(q, p)
                self.buffer.add(obs, mpc_action)

                obs, _, done, _ = self.env.step(mpc_action)
                step += 1
                total += 1

            if verbose and (ep + 1) % 10 == 0:
                print(f"  [Distillation] Collected episode {ep+1}/{n_episodes}, buffer size={len(self.buffer)}")

        return total

    def train_student(
        self,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        verbose: bool = True,
    ) -> List[float]:
        """
        Train Student MLP to minimize MSE(Student(obs), MPC_Action).
        Returns: list of epoch losses.
        """
        if len(self.buffer) == 0:
            raise ValueError("Buffer empty. Run collect_mpc_data first.")

        optimizer = torch.optim.Adam(self.student.parameters(), lr=lr)
        losses = []

        for epoch in range(epochs):
            obs_batch, act_batch = self.buffer.sample(batch_size)
            obs_batch = obs_batch.to(self.device)
            act_batch = act_batch.to(self.device)

            pred = self.student(obs_batch)
            loss = nn.functional.mse_loss(pred, act_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())
            if verbose and (epoch + 1) % 20 == 0:
                print(f"  [Distillation] Epoch {epoch+1}/{epochs}, MSE={loss.item():.6f}")

        return losses

    def run_dagger_iteration(
        self,
        n_episodes: int = 10,
        steps_per_episode: Optional[int] = None,
        train_epochs: int = 50,
        batch_size: int = 256,
        verbose: bool = True,
    ) -> Tuple[int, List[float]]:
        """
        DAgger: Let Student drive, ask MPC "what would YOU do here?", add to buffer, retrain.
        Returns: (new_transitions, epoch_losses).
        """
        steps_per_episode = steps_per_episode or self.env.max_steps
        new_count = 0

        for ep in range(n_episodes):
            obs = self.env.reset()
            done = False
            step = 0

            while not done and step < steps_per_episode:
                # Student acts
                student_action = self.student.act(obs, self.device)
                # MPC labels: "What would I have done?"
                q, p = obs[:2], obs[2:4]
                mpc_action = self.mpc.plan(q, p)
                self.buffer.add(obs, mpc_action)
                new_count += 1

                obs, _, done, _ = self.env.step(student_action)
                step += 1

        losses = self.train_student(epochs=train_epochs, batch_size=batch_size, verbose=verbose)
        if verbose:
            print(f"  [DAgger] Added {new_count} transitions, buffer={len(self.buffer)}")
        return new_count, losses

    def run_dagger(
        self,
        n_iterations: int = 5,
        n_episodes_per_iter: int = 10,
        train_epochs: int = 50,
        verbose: bool = True,
    ) -> List[List[float]]:
        """Run full DAgger loop."""
        all_losses = []
        for i in range(n_iterations):
            if verbose:
                print(f"\n[DAgger] Iteration {i+1}/{n_iterations}")
            _, losses = self.run_dagger_iteration(
                n_episodes=n_episodes_per_iter,
                train_epochs=train_epochs,
                verbose=verbose,
            )
            all_losses.append(losses)
        return all_losses

    def compute_policy_error(
        self,
        obs: np.ndarray,
        mpc_action: float,
    ) -> float:
        """MSE between Student(obs) and MPC action. Used for anomaly detection."""
        self.student.eval()
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            pred = self.student(x)
            return float((pred.cpu().numpy().ravel()[0] - mpc_action) ** 2)

    def infer_latency_ms(self, n_trials: int = 1000) -> float:
        """Measure Student inference latency in milliseconds."""
        obs = np.random.randn(4).astype(np.float32)
        self.student.eval()
        with torch.no_grad():
            x = torch.as_tensor(obs).unsqueeze(0).to(self.device)
            # Warmup
            for _ in range(10):
                _ = self.student(x)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_trials):
                _ = self.student(x)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
        return (t1 - t0) / n_trials * 1000.0
