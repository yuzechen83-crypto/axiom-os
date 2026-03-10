"""
Axiom-OS Gymnasium Adapter: Physics-Informed Control via ImaginationMPC
=======================================================================

使用 Axiom-OS 的 ImaginationMPC 控制器解决 Acrobot-v1 / Pendulum-v1，替代强化学习。

具体要求与实现对应
------------------
1. Wrapper (Gym Agent)
   - AxiomAgent.act(observation): 接收 Gym state，包装成 UPIState，再交给 MPC。
   - 实现位置: AxiomAgent 类、GymUPIWrapper.wrap()。

2. Physics Injection
   - Hard Core: 手动硬编码 Acrobot 运动学（拉格朗日/哈密顿），见 double_pendulum_H、
     acrobot_hard_core、GymUPIWrapper._compute_canonical_momenta。
   - Soft Shell: 在线学习 Sim-to-Real 误差（Gym 物理与预设不一致），见 SimToRealAdapter。

3. Loop
   - 标准 Gym 循环: Reset -> Step -> Render；见 run_episode()。
   - Reward 曲线: run_benchmark() 记录每 episode 的 total_reward，plot_reward_curve() 绘制并保存。

Usage:
------
    python benchmarks/gym_adapter.py --env Acrobot-v1 --episodes 10 --render
    python benchmarks/gym_adapter.py --env Pendulum-v1 --episodes 5 --plot --save results.json
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple, Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from axiom_os.core.upi import UPIState, Units
from axiom_os.orchestrator.mpc import (
    ImaginationMPC,
    double_pendulum_H,
    angle_normalize,
)
from axiom_os.layers.rcln import RCLNLayer

PI = np.pi


# =============================================================================
# Physical Constants & Unit Definitions
# =============================================================================

@dataclass(frozen=True)
class AcrobotPhysics:
    """Physical parameters for Acrobot (double pendulum).
    
    Default values match Gymnasium's Acrobot-v1 internal physics.
    These are used for Hard Core (analytical model).
    """
    # Link lengths (m)
    L1: float = 1.0
    L2: float = 1.0
    # Link masses (kg)
    m1: float = 1.0
    m2: float = 1.0
    # Center of mass positions (m) - relative to joint
    lc1: float = 0.5
    lc2: float = 0.5
    # Moments of inertia (kg·m²)
    I1: float = 1.0
    I2: float = 1.0
    # Gravity (m/s²)
    g: float = 9.8
    # Friction coefficients (N·m·s/rad) - estimated
    friction1: float = 0.1
    friction2: float = 0.1
    # Simulation timestep (s)
    dt: float = 0.02
    
    @property
    def g_over_L(self) -> float:
        """Normalized gravity for Hamiltonian formulation."""
        return self.g / self.L1


# =============================================================================
# UPIState Wrapper for Gym Observations
# =============================================================================

class GymUPIWrapper:
    """Wraps Gymnasium observations into UPIState with proper physical units.
    
    Acrobot-v1 observation space:
        [cos(theta1), sin(theta1), cos(theta2), sin(theta2), theta1_dot, theta2_dot]
    
    UPIState encapsulates:
        - values: Physical quantities
        - units: [M, L, T, Q, Θ] dimensional signature
        - spacetime: [t, x, y, z] timestamp and position
    """
    
    # Unit definitions for Acrobot state components
    # State: [q1, q2, dq1/dt, dq2/dt, p1, p2] (after conversion)
    # q: angles [rad] = [L/L] = dimensionless
    # dq/dt: angular velocity [rad/s] = [T^-1]
    # p: angular momentum [kg·m²/s] = [M·L²·T^-1]
    
    ANGLE_UNITS = [0, 0, 0, 0, 0]  # Dimensionless (rad)
    ANGVEL_UNITS = [0, 0, -1, 0, 0]  # T^-1
    ANGMOM_UNITS = [1, 2, -1, 0, 0]  # M·L²·T^-1
    
    def __init__(self, env_name: str, physics: Optional[AcrobotPhysics] = None):
        self.env_name = env_name
        self.physics = physics or AcrobotPhysics()
        self._step_count = 0
        
    def _acrobot_gym_to_state(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert Gymnasium Acrobot observation to (q, p) canonical coordinates.
        
        Gym obs: [cos(q1), sin(q1), cos(q2), sin(q2), dq1, dq2]
        Returns: q=[q1, q2], p=[p1, p2] (generalized momenta)
        """
        cos_q1, sin_q1, cos_q2, sin_q2, dq1, dq2 = obs
        
        # Extract angles from trigonometric encoding
        q1 = np.arctan2(sin_q1, cos_q1)
        q2 = np.arctan2(sin_q2, cos_q2)
        q = np.array([q1, q2], dtype=np.float64)
        
        # Convert angular velocities to canonical momenta
        # For double pendulum: p = ∂L/∂(dq/dt)
        # This requires the mass matrix (inertia tensor)
        p = self._compute_canonical_momenta(q, np.array([dq1, dq2], dtype=np.float64))
        
        return q, p
    
    def _compute_canonical_momenta(
        self, 
        q: np.ndarray, 
        dq: np.ndarray
    ) -> np.ndarray:
        """Compute canonical momenta p = M(q) * dq from generalized velocities.
        
        Mass matrix M(q) for double pendulum:
        M11 = m1*lc1² + m2*(L1² + lc2² + 2*L1*lc2*cos(q2)) + I1 + I2
        M12 = m2*(lc2² + L1*lc2*cos(q2)) + I2
        M22 = m2*lc2² + I2
        """
        phy = self.physics
        c2 = np.cos(q[1])
        
        M11 = (phy.m1 * phy.lc1**2 + 
               phy.m2 * (phy.L1**2 + phy.lc2**2 + 2*phy.L1*phy.lc2*c2) + 
               phy.I1 + phy.I2)
        M12 = phy.m2 * (phy.lc2**2 + phy.L1*phy.lc2*c2) + phy.I2
        M22 = phy.m2 * phy.lc2**2 + phy.I2
        
        # Mass matrix
        M = np.array([[M11, M12], [M12, M22]], dtype=np.float64)
        p = M @ dq
        
        return p
    
    def _pendulum_gym_to_state(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert Gymnasium Pendulum-v1 observation to (q, p) for MPC.
        
        Pendulum obs: [cos(theta), sin(theta), theta_dot]
        Returns: q=[theta, 0], p=[p1, 0] with p1 = I*theta_dot (I=1), for compatibility with double-pendulum MPC.
        """
        cos_t, sin_t, theta_dot = obs[0], obs[1], obs[2]
        theta = np.arctan2(sin_t, cos_t)
        q = np.array([theta, 0.0], dtype=np.float64)
        # Canonical momentum for single pendulum: p = I * theta_dot (I=1)
        p = np.array([float(theta_dot), 0.0], dtype=np.float64)
        return q, p

    def wrap(self, observation: np.ndarray, timestamp: Optional[float] = None) -> UPIState:
        """Convert Gym observation to UPIState with physical units.
        
        Args:
            observation: Raw Gym observation
            timestamp: Optional simulation time (s)
            
        Returns:
            UPIState with values [q1, q2, p1, p2] and proper units
        """
        if self.env_name == "Acrobot-v1":
            q, p = self._acrobot_gym_to_state(observation)
        elif self.env_name == "Pendulum-v1":
            q, p = self._pendulum_gym_to_state(observation)
        else:
            raise ValueError(f"Unsupported environment: {self.env_name}")
        
        # Construct state vector [q1, q2, p1, p2]
        values = np.concatenate([q, p])
        
        # Units: [dimensionless, dimensionless, M·L²·T^-1, M·L²·T^-1]
        # But UPIState expects single unit vector per value, so we use dimensionless
        # and track semantics separately
        units = [0, 0, 0, 0, 0]  # Store as dimensionless, semantics carry meaning
        
        # Spacetime: [t, x, y, z]
        t = timestamp if timestamp is not None else self._step_count * self.physics.dt
        spacetime = [t, 0.0, 0.0, 0.0]
        
        self._step_count += 1
        
        return UPIState(
            values=values,
            units=units,
            spacetime=spacetime,
            semantics=f"AcrobotState[q1,q2,p1,p2]",
        )
    
    def unwrap(self, upi_state: UPIState) -> Tuple[np.ndarray, np.ndarray]:
        """Extract (q, p) from UPIState for MPC controller.
        
        Returns:
            (q, p) as separate numpy arrays
        """
        vals = upi_state.values.detach().cpu().numpy()
        q = vals[:2]
        p = vals[2:4]
        return q, p


# =============================================================================
# Hard Core: Analytical Acrobot Dynamics
# =============================================================================

def acrobot_hard_core(physics: AcrobotPhysics) -> Callable:
    """Create Hard Core function for Acrobot dynamics.
    
    Returns a function that computes the analytical Hamiltonian flow.
    This is the 'known physics' component: F_hard(x)
    
    The Hamiltonian H(q,p) = T(q,p) + V(q) encodes total energy:
    - T: Kinetic energy (function of q and p via inverse mass matrix)
    - V: Potential energy (function of q only)
    """
    H_analytical = double_pendulum_H(
        g_over_L=physics.g_over_L,
        L1=physics.L1,
        L2=physics.L2,
    )
    
    def hard_core_fn(x: Union[UPIState, torch.Tensor]) -> torch.Tensor:
        """Compute analytical dynamics (Hamiltonian gradient flow).
        
        For control purposes, this returns the expected next state
        under pure Hamiltonian evolution (no control, no friction).
        """
        if isinstance(x, UPIState):
            vals = x.values
        else:
            vals = x
        
        if vals.dim() == 1:
            vals = vals.unsqueeze(0)
        
        # Extract canonical coordinates
        q = vals[:, :2].detach().cpu().numpy()
        p = vals[:, 2:4].detach().cpu().numpy()
        
        # Compute Hamiltonian at current state
        # This is the 'energy surface' we're on
        batch_size = q.shape[0]
        H_vals = np.zeros(batch_size, dtype=np.float64)
        
        for i in range(batch_size):
            qp = np.concatenate([q[i], p[i]])
            H_vals[i] = H_analytical(qp)
        
        # Return as tensor (energy values)
        return torch.tensor(H_vals, dtype=torch.float32, device=vals.device)
    
    return hard_core_fn


# =============================================================================
# Soft Shell: Online Sim-to-Real Adaptation
# =============================================================================

class SimToRealAdapter(nn.Module):
    """Soft Shell for online learning of Sim-to-Real discrepancies.
    
    Learns the residual between predicted (Hard Core) and observed dynamics:
        residual = observed_next_state - predicted_next_state
        
    This captures:
    - Unmodeled friction/damping
    - Parameter mismatch (masses, lengths)
    - Motor dynamics and delays
    - Flexibility and backlash
    
    Architecture: RCLN with MLP soft shell
        y_total = y_hard + λ_res * y_soft
    """
    
    def __init__(
        self,
        state_dim: int = 4,
        hidden_dim: int = 64,
        lambda_res: float = 1.0,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.lambda_res = lambda_res
        
        # RCLN: No hard core (pure residual learning)
        # Input: current state [q1, q2, p1, p2, action]
        # Output: predicted state residual [dq1, dq2, dp1, dp2]
        self.rcln = RCLNLayer(
            input_dim=state_dim + 1,  # state + action
            hidden_dim=hidden_dim,
            output_dim=state_dim,
            hard_core_func=None,  # Pure soft learning
            lambda_res=lambda_res,
            net_type="mlp",
            use_activity_monitor=True,
            soft_threshold=0.1,
            monitor_window=50,
        )
        
        self.optimizer = optim.Adam(self.rcln.parameters(), lr=learning_rate)
        self._residual_history: List[float] = []
        
    def forward(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[object]]:
        """Predict state residual given current state and action.
        
        Args:
            state: [q1, q2, p1, p2] current canonical state
            action: scalar torque applied to first joint
            
        Returns:
            residual: predicted change in state
            hotspot: DiscoveryHotspot if soft shell is highly active
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 0:
            action = action.unsqueeze(0).unsqueeze(1)
        elif action.dim() == 1:
            action = action.unsqueeze(1)
        
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        
        # Forward through RCLN with hotspot detection
        result = self.rcln(x, return_hotspot=True)
        
        if isinstance(result, tuple):
            residual, hotspot = result
        else:
            residual, hotspot = result, None
        
        return residual, hotspot
    
    def update(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        predicted_next: torch.Tensor,
    ) -> float:
        """Online learning step: update Soft Shell to minimize prediction error.
        
        Args:
            state: Current state
            action: Applied action
            next_state: Observed next state (from Gym environment)
            predicted_next: Predicted next state (from Hard Core physics)
            
        Returns:
            loss: MSE between predicted and observed
        """
        self.optimizer.zero_grad()
        
        # Predict residual
        residual_pred, _ = self.forward(state, action)
        
        # Actual residual
        residual_actual = next_state - predicted_next
        
        # Loss: MSE on residual prediction
        loss = nn.functional.mse_loss(residual_pred, residual_actual)
        
        loss.backward()
        self.optimizer.step()
        
        loss_val = loss.item()
        self._residual_history.append(loss_val)
        
        return loss_val
    
    def get_residual_magnitude(self) -> float:
        """Return average magnitude of learned residuals."""
        if not self._residual_history:
            return 0.0
        return np.mean(self._residual_history[-100:])


# =============================================================================
# AxiomAgent: Gym-Compatible Agent Interface
# =============================================================================

class AxiomAgent:
    """Axiom-OS Agent for Gymnasium environments.
    
    Replaces traditional RL policy with Physics-Informed MPC + Online Adaptation.
    
    Interface (Gym-compatible):
        - act(observation) -> action
        - reset() -> None
        
    Internal Architecture:
        1. UPI Wrapper: Gym obs -> UPIState
        2. Hard Core: Analytical Hamiltonian dynamics
        3. Soft Shell: RCLN-based residual learning
        4. ImaginationMPC: Sample-based MPC planning
    """
    
    def __init__(
        self,
        env_name: str = "Acrobot-v1",
        physics: Optional[AcrobotPhysics] = None,
        horizon_steps: int = 80,
        n_samples: int = 2000,
        use_adaptation: bool = True,
        adaptation_lr: float = 1e-3,
        device: str = "cpu",
    ):
        """Initialize AxiomAgent.
        
        Args:
            env_name: Gymnasium environment name
            physics: Physical parameters (uses defaults if None)
            horizon_steps: MPC prediction horizon
            n_samples: Number of trajectory samples for MPC
            use_adaptation: Enable online Sim-to-Real adaptation
            adaptation_lr: Learning rate for Soft Shell
            device: torch device
        """
        self.env_name = env_name
        self.physics = physics or AcrobotPhysics()
        self.device = torch.device(device)
        self.use_adaptation = use_adaptation
        self._discrete_action = env_name == "Acrobot-v1"  # Pendulum-v1 has continuous action
        
        # UPI Wrapper for observation conversion
        self.upi_wrapper = GymUPIWrapper(env_name, self.physics)
        
        # Hard Core: Analytical Hamiltonian
        self.H = double_pendulum_H(
            g_over_L=self.physics.g_over_L,
            L1=self.physics.L1,
            L2=self.physics.L2,
        )
        
        # Soft Shell: Online adaptation (if enabled)
        self.adapter: Optional[SimToRealAdapter] = None
        if use_adaptation:
            self.adapter = SimToRealAdapter(
                state_dim=4,
                hidden_dim=64,
                lambda_res=1.0,
                learning_rate=adaptation_lr,
            ).to(self.device)
        
        # ImaginationMPC: The "mind's eye" controller
        self.mpc = ImaginationMPC(
            H=self.H,
            horizon_steps=horizon_steps,
            n_samples=n_samples,
            dt=self.physics.dt,
            friction=self.physics.friction1,
            action_std=2.0,
            action_bounds=(-1.0, 1.0),  # Gymnasium Acrobot action bounds
            state_dim=2,
            target_state=np.array([PI, PI]),  # Upright position
            target_energy=None,
            distance_threshold=0.5,
        )
        
        # State tracking for adaptation
        self._last_obs: Optional[np.ndarray] = None
        self._last_upi: Optional[UPIState] = None
        self._adaptation_loss = 0.0
        self._step_count = 0
        
    def reset(self) -> None:
        """Reset agent state (call at episode start)."""
        self._last_obs = None
        self._last_upi = None
        self._adaptation_loss = 0.0
        self._step_count = 0
        self.upi_wrapper._step_count = 0
        
        # Reset MPC internal state
        self.mpc._last_tau = 0.0
        
    def act(self, observation: np.ndarray) -> Union[int, np.ndarray]:
        """Select action given Gym observation.
        
        This is the main interface matching Gym's agent pattern.
        
        Args:
            observation: Raw Gym observation (Acrobot: [cos,sin,cos,sin,dq1,dq2]; Pendulum: [cos,sin,dtheta])
            
        Returns:
            action: For Acrobot-v1 discrete {0,1,2}; for Pendulum-v1 continuous np.array([tau]) in [-2,2]
        """
        # 1. Wrap observation in UPIState (Physical Constitution)
        upi_state = self.upi_wrapper.wrap(observation)
        q, p = self.upi_wrapper.unwrap(upi_state)
        
        # 2. Online adaptation update (if we have previous state)
        if self.use_adaptation and self._last_obs is not None and self.adapter is not None:
            self._update_adapter(observation, upi_state)
        
        # 3. ImaginationMPC planning (Hard Core + sampling)
        continuous_action = self.mpc.plan(q, p)
        
        if self._discrete_action:
            # 4a. Acrobot-v1: Discretize to {0, 1, 2} -> {-1.0, 0.0, +1.0}
            if continuous_action < -0.5:
                discrete_action = 0
            elif continuous_action > 0.5:
                discrete_action = 2
            else:
                discrete_action = 1
            torque_applied = {0: -1.0, 1: 0.0, 2: 1.0}[discrete_action]
            self.mpc._last_tau = torque_applied
            self._last_obs = observation.copy()
            self._last_upi = upi_state
            self._step_count += 1
            return discrete_action
        
        # 4b. Pendulum-v1: Continuous action in [-2, 2]
        torque_applied = np.clip(float(continuous_action), -2.0, 2.0)
        self.mpc._last_tau = torque_applied
        self._last_obs = observation.copy()
        self._last_upi = upi_state
        self._step_count += 1
        return np.array([torque_applied], dtype=np.float32)
    
    def _update_adapter(
        self, 
        current_obs: np.ndarray, 
        current_upi: UPIState
    ) -> None:
        """Update Soft Shell with observed transition."""
        assert self.adapter is not None
        assert self._last_upi is not None
        
        # Convert to tensors (float32 for neural network)
        last_state = self._last_upi.values.unsqueeze(0).float().to(self.device)
        current_state = current_upi.values.unsqueeze(0).float().to(self.device)
        
        # Use last action (stored in MPC)
        last_action = torch.tensor([self.mpc._last_tau], dtype=torch.float32, device=self.device)
        
        # Predict next state using Hard Core only (simple Euler step)
        # In practice, we'd use the symplectic integrator from MPC
        predicted_next = self._predict_with_physics(last_state, last_action)
        
        # Update adapter
        loss = self.adapter.update(last_state, last_action, current_state, predicted_next)
        self._adaptation_loss = loss
    
    def _predict_with_physics(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> torch.Tensor:
        """Simple physics prediction for adaptation baseline."""
        # Extract q, p
        q = state[:, :2].detach().cpu().numpy()[0]
        p = state[:, 2:4].detach().cpu().numpy()[0]
        
        # Single symplectic step (simplified)
        from axiom_os.orchestrator.mpc import step_env
        q_next, p_next = step_env(
            q, p, float(action[0]), self.H,
            dt=self.physics.dt,
            friction=self.physics.friction1,
            state_dim=2,
        )
        
        return torch.tensor(
            np.concatenate([q_next, p_next]), 
            dtype=torch.float32, 
            device=state.device
        ).unsqueeze(0)
    
    def get_stats(self) -> dict:
        """Return agent statistics."""
        stats = {
            "steps": self._step_count,
            "mpc_samples": self.mpc.n_samples,
            "horizon": self.mpc.horizon_steps,
        }
        
        if self.adapter is not None:
            stats["adaptation_loss"] = self._adaptation_loss
            stats["residual_magnitude"] = self.adapter.get_residual_magnitude()
        
        return stats


# =============================================================================
# Gym Environment Runner
# =============================================================================

def run_episode(
    env,
    agent: AxiomAgent,
    max_steps: int = 500,
    render: bool = False,
    verbose: bool = False,
) -> dict:
    """Run a single episode with AxiomAgent.
    
    Args:
        env: Gymnasium environment
        agent: AxiomAgent instance
        max_steps: Maximum episode length
        render: Whether to render environment
        verbose: Print step-by-step info
        
    Returns:
        episode_data: Dictionary with trajectory and metrics
    """
    obs, info = env.reset()
    agent.reset()
    
    trajectory = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "states_q": [],
        "states_p": [],
    }
    
    total_reward = 0.0
    done = False
    truncated = False
    step = 0
    
    for step in range(max_steps):
        # Agent selects action
        action = agent.act(obs)
        
        # Environment step
        next_obs, reward, done, truncated, info = env.step(action)  # Acrobot expects int
        
        # Store trajectory
        trajectory["observations"].append(obs)
        trajectory["actions"].append(action)
        trajectory["rewards"].append(reward)
        
        # Extract canonical coordinates for logging (env-agnostic via wrap/unwrap)
        upi = agent.upi_wrapper.wrap(obs)
        q, p = agent.upi_wrapper.unwrap(upi)
        trajectory["states_q"].append(q)
        trajectory["states_p"].append(p)
        
        total_reward += reward
        obs = next_obs
        
        if render:
            env.render()
        
        if verbose and step % 50 == 0:
            stats = agent.get_stats()
            err = np.sqrt(np.sum(angle_normalize(q - PI) ** 2))
            print(f"  Step {step:4d}: err={err:.3f}, reward={reward:.2f}, "
                  f"adapt_loss={stats.get('adaptation_loss', 0):.4f}")
        
        if done or truncated:
            break
    
    trajectory["total_reward"] = total_reward
    trajectory["length"] = step + 1
    trajectory["success"] = check_success(trajectory, env_name=agent.env_name)

    return trajectory


def check_success(trajectory: dict, threshold: float = 0.5, env_name: str = "Acrobot-v1") -> bool:
    """Check if episode succeeded (stabilized near upright).
    
    Acrobot-v1: Both angles within threshold of PI for last 50 steps.
    Pendulum-v1: Single angle |theta| within threshold for last 50 steps.
    """
    if len(trajectory["states_q"]) < 50:
        return False
    
    last_q = np.array(trajectory["states_q"][-50:])
    if env_name == "Pendulum-v1":
        # Upright is theta=0
        err = np.abs(angle_normalize(last_q[:, 0]))
        max_err = np.max(err)
    else:
        err = angle_normalize(last_q - PI)
        max_err = np.max(np.abs(err))
    return max_err < threshold


def run_benchmark(
    env_name: str = "Acrobot-v1",
    n_episodes: int = 10,
    max_steps: int = 500,
    render: bool = False,
    fast: bool = False,
    save_path: Optional[str] = None,
    use_adaptation: bool = True,
) -> dict:
    """Run full benchmark on Gymnasium environment.

    Args:
        env_name: Environment name (Acrobot-v1 or Pendulum-v1)
        n_episodes: Number of episodes to run
        max_steps: Max steps per episode
        render: Enable rendering
        fast: Use fewer MPC samples for quick testing
        save_path: Optional path to save results
        use_adaptation: Enable Soft Shell Sim-to-Real adaptation

    Returns:
        benchmark_results: Aggregated statistics
    """
    try:
        import gymnasium as gym
    except ImportError:
        print("Error: gymnasium not installed. Run: pip install gymnasium")
        sys.exit(1)
    
    print("=" * 70)
    print(f"Axiom-OS Gymnasium Benchmark: {env_name}")
    print("=" * 70)
    print(f"Controller: Physics-Informed MPC + RCLN Adaptation")
    print(f"Episodes: {n_episodes}, Max Steps: {max_steps}")
    print("-" * 70)
    
    # Create environment
    env = gym.make(env_name, render_mode="human" if render else None)
    
    # Create agent
    horizon = 30 if fast else 80
    n_samples = 200 if fast else 2000
    
    agent = AxiomAgent(
        env_name=env_name,
        horizon_steps=horizon,
        n_samples=n_samples,
        use_adaptation=use_adaptation,
        adaptation_lr=1e-3,
    )
    
    # Run episodes
    all_episodes = []
    success_count = 0
    
    for episode in range(n_episodes):
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        
        traj = run_episode(
            env, agent, max_steps=max_steps, 
            render=render, verbose=(episode == 0)
        )
        
        all_episodes.append(traj)
        if traj["success"]:
            success_count += 1
        
        print(f"  Length: {traj['length']}, Reward: {traj['total_reward']:.2f}, "
              f"Success: {traj['success']}")
    
    # Aggregate statistics
    rewards = [t["total_reward"] for t in all_episodes]
    lengths = [t["length"] for t in all_episodes]
    
    results = {
        "env_name": env_name,
        "n_episodes": n_episodes,
        "success_rate": success_count / n_episodes,
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_length": np.mean(lengths),
        "all_rewards": rewards,
        "all_lengths": lengths,
        "episodes": all_episodes,
    }
    
    # Print summary
    print("\n" + "=" * 70)
    print("Benchmark Summary")
    print("=" * 70)
    print(f"Success Rate: {results['success_rate']*100:.1f}% ({success_count}/{n_episodes})")
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Length: {results['mean_length']:.1f}")
    print("=" * 70)
    
    # Save results
    if save_path:
        save_results(results, save_path)
    
    env.close()
    return results


def save_results(results: dict, path: str) -> None:
    """Save benchmark results to JSON."""
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    json_safe = {
        k: v if not isinstance(v, np.ndarray) else v.tolist()
        for k, v in results.items()
        if k != "episodes"  # Don't save full trajectories
    }
    
    with open(path, "w") as f:
        json.dump(json_safe, f, indent=2)
    
    print(f"\nResults saved to: {path}")


def plot_reward_curve(
    results: dict,
    save_path: Optional[str] = None,
    show_cumulative: bool = False,
) -> None:
    """Plot per-episode reward curve (and optionally cumulative).
    
    Args:
        results: Dict from run_benchmark with "all_rewards", "env_name".
        save_path: If set, save figure to this path.
        show_cumulative: If True, add a second subplot for cumulative reward.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping reward curve")
        return
    
    rewards = results.get("all_rewards", [])
    if not rewards:
        print("No rewards in results, skipping reward curve")
        return
    
    env_name = results.get("env_name", "Unknown")
    n = len(rewards)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(np.arange(1, n + 1), rewards, "b-o", markersize=4, label="Episode reward")
    ax.axhline(np.mean(rewards), color="green", linestyle="--", alpha=0.7, label=f"Mean = {np.mean(rewards):.2f}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title(f"Reward curve: {env_name} (Axiom-OS ImaginationMPC)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if show_cumulative:
        ax2 = ax.twinx()
        cum = np.cumsum(rewards)
        ax2.plot(np.arange(1, n + 1), cum, "gray", linestyle="-", alpha=0.6, label="Cumulative")
        ax2.set_ylabel("Cumulative reward")
        ax2.legend(loc="upper right")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Reward curve saved to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_episode(trajectory: dict, save_path: Optional[str] = None) -> None:
    """Plot episode trajectory."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return
    
    q_traj = np.array(trajectory["states_q"])
    p_traj = np.array(trajectory["states_p"])
    actions = np.asarray(trajectory["actions"]).squeeze()  # (T,) or (T,1) for Pendulum
    rewards = np.array(trajectory["rewards"])
    t = np.arange(len(actions)) * 0.02  # Assuming dt=0.02
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # Joint angles
    ax = axes[0]
    ax.plot(t, q_traj[:, 0], "b-", label=r"$\theta_1$")
    ax.plot(t, q_traj[:, 1], "g-", label=r"$\theta_2$")
    ax.axhline(PI, color="k", linestyle="--", alpha=0.3, label="Target")
    ax.set_ylabel("Angle (rad)")
    ax.set_title("Axiom-OS Acrobot Control: Joint Angles")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Angular velocities
    ax = axes[1]
    ax.plot(t, p_traj[:, 0], "b-", label=r"$p_1$")
    ax.plot(t, p_traj[:, 1], "g-", label=r"$p_2$")
    ax.set_ylabel("Angular Momentum")
    ax.set_title("Canonical Momenta")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Control action
    ax = axes[2]
    ax.plot(t, actions, "r-", label="Torque")
    ax.set_ylabel("Torque (N·m)")
    ax.set_title("Control Action")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Reward
    ax = axes[3]
    ax.plot(t, rewards, "purple", label="Reward")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Reward")
    ax.set_title("Instantaneous Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Axiom-OS Gymnasium Benchmark: Physics-Informed Control"
    )
    parser.add_argument(
        "--env", 
        type=str, 
        default="Acrobot-v1",
        help="Gymnasium environment name (default: Acrobot-v1)"
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=10,
        help="Number of episodes (default: 10)"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=500,
        help="Max steps per episode (default: 500)"
    )
    parser.add_argument(
        "--render", 
        action="store_true",
        help="Enable rendering"
    )
    parser.add_argument(
        "--fast", 
        action="store_true",
        help="Fast mode: fewer MPC samples"
    )
    parser.add_argument(
        "--no-adapt", 
        action="store_true",
        help="Disable online adaptation"
    )
    parser.add_argument(
        "--save", 
        type=str,
        default=None,
        help="Path to save results JSON"
    )
    parser.add_argument(
        "--plot", 
        type=str,
        default=None,
        help="Path to save trajectory plot"
    )
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_benchmark(
        env_name=args.env,
        n_episodes=args.episodes,
        max_steps=args.steps,
        render=args.render,
        fast=args.fast,
        save_path=args.save,
        use_adaptation=not args.no_adapt,
    )
    
    # Optional: plot first episode trajectory and reward curve
    if args.plot:
        if results["episodes"]:
            plot_episode(results["episodes"][0], args.plot)
        # Reward curve: same path with _reward_curve before extension
        import os
        base, ext = os.path.splitext(args.plot)
        reward_curve_path = base + "_reward_curve" + (ext or ".png")
        plot_reward_curve(results, save_path=reward_curve_path)
    
    # Return success code
    return 0 if results["success_rate"] > 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
