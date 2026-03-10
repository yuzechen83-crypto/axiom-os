"""
MPC v2 - Model Predictive Path Integral Controller (Optimized)
===============================================================

Imagination-Augmented Controller with:
- PyTorch Tensor Parallelization (batch rollouts)
- JIT-compiled core operations
- Full type hints and physical semantics

Physical Conventions:
---------------------
- q: Generalized coordinates (angles in radians) [dim: state_dim]
- p: Generalized momenta (conjugate to q) [dim: state_dim]  
- H: Hamiltonian H(q,p) = T(q,p) + V(q) [units: Energy]
- tau: Control torque applied to system [units: N·m]
- dt: Symplectic integration timestep [units: s]

Symplectic Structure:
---------------------
The system evolves under Hamilton's equations:
    dq/dt = ∂H/∂p  (velocity)
    dp/dt = -∂H/∂q + tau  (force + control)

We use Stormer-Verlet (leapfrog) integration to preserve
phase space volume and energy approximately.
"""

from __future__ import annotations

from typing import Tuple, Optional, Union, List
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

PI = 3.14159265358979323846  # JIT-compatible constant

# Type aliases for physical semantics
GeneralizedCoordinate = Tensor  # Shape: (..., state_dim), units: rad
GeneralizedMomentum = Tensor    # Shape: (..., state_dim), units: kg·m²/s
Action = Tensor                 # Shape: (...) or (..., 1), units: N·m


# =============================================================================
# JIT-Compiled Core Physics (Numerical Gradients for Compatibility)
# =============================================================================

@torch.jit.script
def hamiltonian_double_pendulum(qp: Tensor, g_over_L: float, L1: float, L2: float) -> Tensor:
    """
    JIT-compiled Hamiltonian for double pendulum (acrobot).
    
    H(q,p) = T(q,p) + V(q)
    
    Args:
        qp: Canonical state [q1, q2, p1, p2] with shape (..., 4)
        g_over_L: Gravitational acceleration normalized by length
        L1, L2: Link lengths
        
    Returns:
        H: Total energy (scalar or batched)
        
    Physical Units:
        - q: angles [rad]
        - p: angular momenta [kg·m²/s]
        - H: energy [Joules]
    """
    q1 = qp[..., 0]
    q2 = qp[..., 1]
    p1 = qp[..., 2]
    p2 = qp[..., 3]
    
    dq = q1 - q2
    
    # Kinetic energy: T = (p1^2 + 2*p2^2 - 2*p1*p2*cos(dq)) / (2*(1+sin^2(dq)))
    denom = 2.0 * (1.0 + torch.sin(dq) ** 2)
    T = (p1 ** 2 + 2.0 * p2 ** 2 - 2.0 * p1 * p2 * torch.cos(dq)) / denom
    
    # Potential energy
    V = g_over_L * (L1 * (1.0 - torch.cos(q1)) + L2 * (1.0 - torch.cos(q2)))
    
    return T + V


@torch.jit.script  
def hamiltonian_gradients_numerical(
    qp: Tensor, 
    g_over_L: float, 
    L1: float, 
    L2: float,
    eps: float = 1e-5
) -> Tuple[Tensor, Tensor]:
    """
    Numerical gradients of Hamiltonian (JIT-compatible).
    
    Computes ∂H/∂q and ∂H/∂p using central differences:
        ∂H/∂x ≈ (H(x+ε) - H(x-ε)) / (2ε)
    
    Args:
        qp: State [q, p] with shape (..., 4)
        g_over_L, L1, L2: Physical parameters
        eps: Finite difference step size
        
    Returns:
        dH_dq: ∂H/∂q [..., 2] (generalized forces)
        dH_dp: ∂H/∂p [..., 2] (generalized velocities)
    """
    state_dim = qp.shape[-1] // 2
    n_dims = qp.shape[-1]
    
    # Compute H at qp
    H_center = hamiltonian_double_pendulum(qp, g_over_L, L1, L2)
    
    # Initialize gradients
    dH = torch.zeros_like(qp)
    
    # Central differences for each dimension
    for i in range(n_dims):
        qp_plus = qp.clone()
        qp_minus = qp.clone()
        qp_plus[..., i] = qp_plus[..., i] + eps
        qp_minus[..., i] = qp_minus[..., i] - eps
        
        H_plus = hamiltonian_double_pendulum(qp_plus, g_over_L, L1, L2)
        H_minus = hamiltonian_double_pendulum(qp_minus, g_over_L, L1, L2)
        
        dH[..., i] = (H_plus - H_minus) / (2.0 * eps)
    
    dH_dq = dH[..., :state_dim]
    dH_dp = dH[..., state_dim:]
    
    return dH_dq, dH_dp


@torch.jit.script
def symplectic_step_jit(
    q: Tensor,
    p: Tensor,
    action: Tensor,
    g_over_L: float,
    L1: float,
    L2: float,
    dt: float,
    friction: float,
    torque_idx: int,
) -> Tuple[Tensor, Tensor]:
    """
    Single symplectic integration step (JIT-compiled).
    
    Stormer-Verlet leapfrog integration:
        1. p_{n+1/2} = p_n - (dt/2) * (∂H/∂q + friction*∂H/∂p)
        2. q_{n+1} = q_n + dt * ∂H/∂p(p_{n+1/2})  
        3. p_{n+1} = p_{n+1/2} - (dt/2) * (∂H/∂q + friction*∂H/∂p) + action*dt
    
    Args:
        q: Coordinates [..., state_dim] [rad]
        p: Momenta [..., state_dim] [kg·m²/s]
        action: Control torque [..., ] [N·m]
        g_over_L, L1, L2: Physical parameters
        dt: Time step [s]
        friction: Damping [1/s]
        torque_idx: Joint index for torque (0 or 1)
        
    Returns:
        q_new, p_new: Updated state
    """
    # Concatenate for gradient computation
    qp = torch.cat([q, p], dim=-1)
    
    # Step 1: Half momentum update
    dH_dq, dH_dp = hamiltonian_gradients_numerical(qp, g_over_L, L1, L2)
    p_half = p - 0.5 * dt * dH_dq - 0.5 * dt * friction * dH_dp
    
    # Step 2: Full position update
    qp_mid = torch.cat([q, p_half], dim=-1)
    _, dH_dp_mid = hamiltonian_gradients_numerical(qp_mid, g_over_L, L1, L2)
    q_new = q + dt * dH_dp_mid
    
    # Step 3: Final momentum update
    qp_end = torch.cat([q_new, p_half], dim=-1)
    dH_dq_end, dH_dp_end = hamiltonian_gradients_numerical(qp_end, g_over_L, L1, L2)
    p_new = p_half - 0.5 * dt * dH_dq_end - 0.5 * dt * friction * dH_dp_end
    
    # Apply control torque
    if action.numel() == 1:
        # Scalar action, broadcast
        p_new_new = p_new.clone()
        p_new_new[..., torque_idx] += action.item() * dt
        return q_new, p_new_new
    else:
        # Batched action
        p_new_new = p_new.clone()
        if action.dim() == 1:
            p_new_new[..., torque_idx] += action * dt
        else:
            p_new_new[..., torque_idx] += action.squeeze(-1) * dt
        return q_new, p_new_new


@torch.jit.script
def rollout_trajectories_batched(
    q0: Tensor,
    p0: Tensor,
    actions: Tensor,
    g_over_L: float,
    L1: float,
    L2: float,
    dt: float,
    friction: float,
    torque_idx: int,
) -> Tuple[Tensor, Tensor]:
    """
    Batched trajectory rollout for MPPI (JIT-compiled).
    
    Parallelizes n_samples trajectories of horizon_steps each.
    All rollouts execute in parallel via tensor operations.
    
    Args:
        q0: Initial coordinates [n_samples, state_dim] [rad]
        p0: Initial momenta [n_samples, state_dim] [kg·m²/s]
        actions: Action sequences [n_samples, horizon_steps] [N·m]
        g_over_L, L1, L2: Physical parameters
        dt: Time step [s]
        friction: Damping [1/s]
        torque_idx: Torque application joint
        
    Returns:
        q_traj: Coordinate trajectories [n_samples, horizon_steps+1, state_dim]
        p_traj: Momentum trajectories [n_samples, horizon_steps+1, state_dim]
    """
    n_samples = q0.shape[0]
    horizon_steps = actions.shape[1]
    state_dim = q0.shape[1]
    
    # Preallocate trajectory tensors
    q_traj = torch.zeros(n_samples, horizon_steps + 1, state_dim, dtype=q0.dtype, device=q0.device)
    p_traj = torch.zeros(n_samples, horizon_steps + 1, state_dim, dtype=p0.dtype, device=p0.device)
    
    q_traj[:, 0] = q0
    p_traj[:, 0] = p0
    
    q = q0
    p = p0
    
    # Unroll all trajectories in parallel (batched operations)
    for t in range(horizon_steps):
        action_t = actions[:, t]  # [n_samples]
        q, p = symplectic_step_jit(q, p, action_t, g_over_L, L1, L2, dt, friction, torque_idx)
        q_traj[:, t + 1] = q
        p_traj[:, t + 1] = p
    
    return q_traj, p_traj


@torch.jit.script
def angle_normalize_torch(theta: Tensor) -> Tensor:
    """
    Wrap angles to [-PI, PI] range.
    
    Args:
        theta: Angles [rad] of any shape
        
    Returns:
        Normalized angles in [-PI, PI]
    """
    PI_VAL = 3.14159265358979323846
    TWO_PI = 6.28318530717958647692
    return ((theta + PI_VAL) % TWO_PI) - PI_VAL


@torch.jit.script
def compute_trajectory_cost_jit(
    q_traj: Tensor,
    p_traj: Tensor,
    actions: Tensor,
    target_state: Tensor,
    w_pos_1: float,
    w_pos_2: float,
    w_vel_1: float,
    w_vel_2: float,
    w_u: float,
    w_term_pos: float,
    w_term_vel: float,
) -> Tensor:
    """
    JIT-compiled trajectory cost computation.
    
    Cost = Position error + Velocity penalty + Control effort + Terminal cost
    
    Args:
        q_traj: Coordinates [n_samples, T+1, state_dim] [rad]
        p_traj: Momenta [n_samples, T+1, state_dim] [kg·m²/s]
        actions: Controls [n_samples, T] [N·m]
        target_state: Target q [state_dim] [rad]
        w_pos_1, w_pos_2: Position weights for joint 1 and 2
        w_vel_1, w_vel_2: Velocity weights
        w_u: Control effort weight
        w_term_pos, w_term_vel: Terminal cost weights
        
    Returns:
        costs: Total cost per trajectory [n_samples]
    """
    n_samples = q_traj.shape[0]
    T = actions.shape[1]
    
    # Expand target: [state_dim] -> [n_samples, state_dim]
    target = target_state.unsqueeze(0).expand(n_samples, -1)
    
    # Position errors at each timestep
    err = angle_normalize_torch(q_traj[:, :T, :] - target.unsqueeze(1))
    
    err1 = err[..., 0]  # [n_samples, T]
    err2 = err[..., 1]
    vel1 = p_traj[:, :T, 0]
    vel2 = p_traj[:, :T, 1]
    
    # Running costs
    pos_cost = w_pos_1 * (err1 ** 2) + w_pos_2 * (err2 ** 2)
    vel_cost = w_vel_1 * (vel1 ** 2) + w_vel_2 * (vel2 ** 2)
    control_cost = w_u * (actions ** 2)
    
    running_cost = pos_cost + vel_cost + control_cost
    
    # Terminal cost (final state)
    err_term = angle_normalize_torch(q_traj[:, -1, :] - target)
    vel_term = p_traj[:, -1, :]
    
    terminal_cost = w_term_pos * (err_term ** 2).sum(dim=-1) + w_term_vel * (vel_term ** 2).sum(dim=-1)
    
    total_cost = running_cost.sum(dim=-1) + terminal_cost
    
    return total_cost


# =============================================================================
# Main Controller Class
# =============================================================================

class ImaginationMPCV2(nn.Module):
    """
    Model Predictive Path Integral (MPPI) Controller - Optimized V2.
    
    "The Mind's Eye": Simulates N parallel futures, picks best action.
    
    Key Improvements over V1:
    - **Parallelization**: All n_samples rollouts execute simultaneously via 
      PyTorch tensor operations (no Python for-loop over samples)
    - **JIT Compilation**: Core physics (symplectic_step_jit, 
      rollout_trajectories_batched) compiled with @torch.jit.script
    - **Type Safety**: Full type hints with physical unit annotations
    - **GPU Support**: Can run on CUDA for massive speedup
    
    Physical Model:
        - Double pendulum (acrobot) with Hamiltonian dynamics
        - Stormer-Verlet symplectic integration (preserves energy approximately)
        - Friction modeled as velocity-dependent damping
    
    Args:
        g_over_L: Normalized gravity (g/L) [1/s²]
        L1, L2: Link lengths [m]
        horizon_steps: Planning horizon T [steps]
        n_samples: Number of parallel rollouts N [samples]
        dt: Integration timestep [s]
        friction: Damping coefficient [1/s]
        action_std: Action sampling std dev [N·m]
        action_bounds: (min, max) torque limits [N·m]
        state_dim: System dimension (2 for acrobot)
        target_state: Target configuration q_target [rad]
        device: 'cpu' or 'cuda'
    """
    
    def __init__(
        self,
        g_over_L: float = 10.0,
        L1: float = 1.0,
        L2: float = 1.0,
        horizon_steps: int = 80,
        n_samples: int = 2000,
        dt: float = 0.02,
        friction: float = 0.1,
        action_std: float = 2.0,
        action_bounds: Tuple[float, float] = (-25.0, 25.0),
        state_dim: int = 2,
        target_state: Optional[np.ndarray] = None,
        target_energy: Optional[float] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        
        # Physical parameters (stored as buffers for JIT compatibility)
        self.register_buffer("_g_over_L", torch.tensor(g_over_L, dtype=torch.float64))
        self.register_buffer("_L1", torch.tensor(L1, dtype=torch.float64))
        self.register_buffer("_L2", torch.tensor(L2, dtype=torch.float64))
        
        # Control parameters
        self.horizon_steps = horizon_steps
        self.n_samples = n_samples
        self.dt = dt
        self.friction = friction
        self.action_std = action_std
        self.action_bounds = action_bounds
        self.state_dim = state_dim
        self.target_energy = target_energy
        self.device = torch.device(device)
        
        # Target state (upright: [π, π] for acrobot)
        if target_state is not None:
            target_t = torch.tensor(target_state, dtype=torch.float64)
        else:
            target_t = torch.full((state_dim,), 3.14159265358979323846, dtype=torch.float64)
        self.register_buffer("target_state", target_t)
        
        # Action smoothing memory
        self.register_buffer("_last_tau", torch.tensor(0.0, dtype=torch.float64))
        
    def to(self, device: Union[str, torch.device]) -> "ImaginationMPCV2":
        """Move model to device."""
        super().to(device)
        self.device = torch.device(device)
        return self
    
    def plan(self, q: Union[Tensor, np.ndarray], p: Union[Tensor, np.ndarray]) -> float:
        """
        MPPI planning: Sample N action sequences, evaluate in parallel, return best.
        
        Algorithm:
            1. Sample N random action sequences from N(0, action_std)
            2. Parallel rollout: Simulate N trajectories simultaneously
            3. Compute cost for each trajectory
            4. Select action with minimum cost
            5. Apply temporal smoothing (80% new + 20% previous)
        
        Args:
            q: Current generalized coordinates [state_dim] [rad]
               Shape: [state_dim] for single state or [batch, state_dim]
            p: Current generalized momenta [state_dim] [kg·m²/s]
               Shape: [state_dim] for single state or [batch, state_dim]
            
        Returns:
            action: Optimal control torque tau* [float] [N·m]
        """
        # Convert inputs to tensors
        if not isinstance(q, Tensor):
            q = torch.tensor(q, dtype=torch.float64, device=self.device)
        if not isinstance(p, Tensor):
            p = torch.tensor(p, dtype=torch.float64, device=self.device)
        
        # Ensure 2D: [n_samples, state_dim] for batch rollout
        if q.dim() == 1:
            q = q.unsqueeze(0).expand(self.n_samples, -1)
        if p.dim() == 1:
            p = p.unsqueeze(0).expand(self.n_samples, -1)
        
        # Move to device
        q = q.to(self.device)
        p = p.to(self.device)
        
        # Sample action sequences: [n_samples, horizon_steps]
        actions = torch.randn(
            self.n_samples, self.horizon_steps, 
            dtype=torch.float64, device=self.device
        )
        actions = actions * self.action_std
        actions = torch.clamp(
            actions, 
            self.action_bounds[0], 
            self.action_bounds[1]
        )
        
        # Parallel rollout (JIT-compiled): [n_samples, T+1, state_dim]
        q_traj, p_traj = rollout_trajectories_batched(
            q, p, actions,
            float(self._g_over_L), 
            float(self._L1), 
            float(self._L2),
            self.dt, 
            self.friction, 
            0
        )
        
        # Compute costs (JIT-compiled): [n_samples]
        costs = compute_trajectory_cost_jit(
            q_traj, p_traj, actions, self.target_state,
            w_pos_1=5000.0, w_pos_2=1000.0,
            w_vel_1=500.0, w_vel_2=100.0,
            w_u=0.0001,
            w_term_pos=100000.0, w_term_vel=1000.0
        )
        
        # Select best action
        best_idx = torch.argmin(costs)
        best_tau = actions[best_idx, 0]
        
        # Smoothing: 80% new + 20% previous (reduces jitter)
        tau_smooth = 0.8 * best_tau + 0.2 * self._last_tau
        self._last_tau = tau_smooth
        
        return float(tau_smooth)
    
    def forward(self, state: Tensor) -> Tensor:
        """
        PyTorch nn.Module interface: state -> action.
        
        Args:
            state: Concatenated [q, p] with shape [..., 2*state_dim]
            
        Returns:
            action: Scalar action [...]
        """
        state_dim = state.shape[-1] // 2
        q = state[..., :state_dim]
        p = state[..., state_dim:]
        
        if state.dim() == 1:
            return torch.tensor(self.plan(q, p), dtype=torch.float32)
        else:
            actions = [self.plan(q[i], p[i]) for i in range(state.shape[0])]
            return torch.tensor(actions, dtype=torch.float32)


# =============================================================================
# Backward Compatibility Wrapper
# =============================================================================

class ImaginationMPC:
    """
    Legacy wrapper maintaining backward compatibility with V1 API.
    
    Internally uses ImaginationMPCV2 but exposes numpy-compatible interface.
    """
    
    def __init__(
        self,
        H: Optional[callable] = None,
        horizon_steps: int = 80,
        n_samples: int = 2000,
        dt: float = 0.02,
        friction: float = 0.1,
        action_std: float = 2.0,
        action_bounds: Tuple[float, float] = (-25.0, 25.0),
        state_dim: int = 2,
        target_state: Optional[np.ndarray] = None,
        target_energy: Optional[float] = None,
        distance_threshold: float = 0.5,
    ):
        """Initialize with V1-compatible numpy interface."""
        self._v2 = ImaginationMPCV2(
            g_over_L=10.0, L1=1.0, L2=1.0,
            horizon_steps=horizon_steps,
            n_samples=n_samples,
            dt=dt, friction=friction,
            action_std=action_std,
            action_bounds=action_bounds,
            state_dim=state_dim,
            target_state=target_state,
            target_energy=target_energy,
        )
        self.H = H
        self.state_dim = state_dim
        self.distance_threshold = distance_threshold
        self._last_tau = 0.0  # For compatibility
        
    def plan(self, q: np.ndarray, p: np.ndarray) -> float:
        """Numpy-compatible plan method."""
        return self._v2.plan(q, p)
    
    @staticmethod
    def energy_shaping_cost(state: np.ndarray, target_energy: float, H_func: callable) -> float:
        """Legacy energy shaping cost."""
        state = np.asarray(state)
        if state.ndim == 1:
            E = float(np.asarray(H_func(state)).ravel()[0])
            return (E - target_energy) ** 2
        return float(np.mean([
            (float(np.asarray(H_func(state[t])).ravel()[0]) - target_energy) ** 2
            for t in range(state.shape[0])
        ]))
    
    @staticmethod
    def stabilization_cost(
        state: Tuple[np.ndarray, np.ndarray],
        target_state: np.ndarray,
        w_pos: float = 100.0,
        w_vel: float = 10.0,
    ) -> float:
        """Legacy stabilization cost."""
        q_traj, p_traj = state
        q_traj = np.asarray(q_traj)
        p_traj = np.asarray(p_traj)
        target = np.asarray(target_state).ravel()
        err = ((q_traj - target + PI) % (2 * PI)) - PI
        return float(np.mean(
            w_pos * np.sum(err**2, axis=-1) + w_vel * np.sum(p_traj**2, axis=-1)
        ))


# Legacy function exports
def double_pendulum_H(g_over_L: float = 10.0, L1: float = 1.0, L2: float = 1.0) -> callable:
    """Legacy numpy Hamiltonian (for backward compatibility)."""
    def H(qp):
        qp = np.atleast_1d(np.asarray(qp, dtype=np.float64))
        if len(qp) < 4:
            return 0.0
        q1, q2, p1, p2 = qp[0], qp[1], qp[2], qp[3]
        dq = q1 - q2
        denom = 2.0 * (1.0 + np.sin(dq) ** 2)
        if abs(denom) < 1e-12:
            denom = 1e-12
        T = (p1**2 + 2*p2**2 - 2*p1*p2*np.cos(dq)) / denom
        V = g_over_L * (L1 * (1 - np.cos(q1)) + L2 * (1 - np.cos(q2)))
        return T + V
    return H


def step_env(q, p, action, H_func, dt=0.02, friction=0.1, state_dim=2):
    """Legacy numpy step function (for backward compatibility)."""
    from ..core.einstein import _grad_H_numpy
    
    q = np.asarray(q, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    qp = np.concatenate([q, p])
    
    dH_dq, dH_dp = _grad_H_numpy(H_func, qp, state_dim)
    
    p_half = p - 0.5 * dt * dH_dq - 0.5 * dt * friction * dH_dp
    qp_mid = np.concatenate([q, p_half])
    _, dH_dp_mid = _grad_H_numpy(H_func, qp_mid, state_dim)
    
    q_new = q + dt * dH_dp_mid
    qp_end = np.concatenate([q_new, p_half])
    dH_dq_end, dH_dp_end = _grad_H_numpy(H_func, qp_end, state_dim)
    
    p_new = p_half - 0.5 * dt * dH_dq_end - 0.5 * dt * friction * dH_dp_end
    p_new[0] += action * dt
    
    return q_new, p_new


def angle_normalize(theta):
    """Legacy angle normalization."""
    return ((np.asarray(theta, dtype=np.float64) + PI) % (2 * PI)) - PI
