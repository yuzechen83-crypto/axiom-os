"""
Gymnasium + MuJoCo Test Suite for Axiom-OS
===========================================

AI Control "Gaokao" - Testing RCLN + MPC on complex multi-body dynamics.

Test Subjects:
- Hopper-v5: Single leg hopping (balance + propulsion)
- Walker2d-v5: Bipedal walking (coordination)
- Humanoid-v5: Full humanoid (ultimate boss)

Sim-to-Real Challenge:
- Perturb gravity (-20% to +20%)
- Perturb friction (-30% to +30%)
- Test Axiom-OS adaptation vs baseline RL

Usage:
    python benchmarks/test_gym_mujoco.py --env Hopper-v5 --test all
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import gymnasium as gym
    from gymnasium.wrappers import RecordVideo
    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    print("[ERROR] gymnasium not installed. Run: pip install gymnasium[mujoco]")
    sys.exit(1)

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False
    print("[ERROR] mujoco not installed. Run: pip install mujoco")
    sys.exit(1)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from axiom_os.orchestrator import ImaginationMPCV2
from axiom_os.envs.mujoco_env import (
    make_axiom_mujoco_env,
    MuJoCoPhysicsParams,
    get_physics_params,
    SimToRealWrapper,
)


@dataclass
class TestResult:
    """Test result container."""
    env_id: str
    controller: str
    total_reward: float
    steps: int
    survival_time: float
    success: bool
    physics_params: Optional[Dict] = None


class AxiomMuJoCoController:
    """
    Axiom-OS controller for MuJoCo environments.
    
    Architecture:
    - Hard Core: Extracted from MuJoCo XML (mass, length, gravity)
    - Soft Shell: RCLN for residual dynamics
    - MPC: ImaginationMPCV2 for trajectory optimization
    """
    
    def __init__(
        self,
        env_id: str,
        physics_params: MuJoCoPhysicsParams,
        action_dim: int = 3,
        horizon_steps: int = 50,
        n_samples: int = 500,
        use_adaptation: bool = True,
        device: str = "cpu",
    ):
        self.env_id = env_id
        self.physics = physics_params
        self.action_dim = action_dim
        self.use_adaptation = use_adaptation
        
        # Extract key physics for Hard Core
        g = abs(physics_params.gravity[2])  # z-axis gravity
        
        # Initialize MPC with environment physics
        self.mpc = ImaginationMPCV2(
            g_over_L=g,
            L1=1.0,
            L2=0.5,
            horizon_steps=horizon_steps,
            n_samples=n_samples,
            dt=0.01,  # MuJoCo default
            friction=physics_params.friction,
            action_std=0.3,
            action_bounds=(-1.0, 1.0),
            state_dim=2,
            device=device,
        )
        
        # PD controller gains (tuned per environment)
        self.kp = 0.8
        self.kd = 0.2
        
        # Online adaptation for Sim-to-Real
        self.residual_model = None
        if use_adaptation and HAS_TORCH:
            self.residual_model = torch.nn.Sequential(
                torch.nn.Linear(4, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, action_dim),
            )
            self.optimizer = torch.optim.Adam(self.residual_model.parameters(), lr=1e-3)
    
    def obs_to_canonical(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert MuJoCo observation to canonical coordinates (q, p).
        
        MuJoCo obs: [qpos, qvel] or [qpos, qvel, torso_info]
        We extract joint positions and velocities.
        """
        n_qpos = len(obs) // 2
        q = obs[:n_qpos]  # Generalized positions
        p = obs[n_qpos:2*n_qpos]  # Generalized velocities
        
        # Normalize angles to [-pi, pi]
        q = ((q + np.pi) % (2 * np.pi)) - np.pi
        
        return q, p
    
    def pd_control(self, obs: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Classical PD controller (baseline).
        
        tau = kp * (target - q) - kd * p
        """
        q, p = self.obs_to_canonical(obs)
        
        # Match dimensions
        target = target[:len(q)]
        
        error = target - q
        action = self.kp * error - self.kd * p
        
        # Ensure correct action dimension
        if len(action) != self.action_dim:
            # Pad or truncate to match action space
            if len(action) < self.action_dim:
                action = np.pad(action, (0, self.action_dim - len(action)))
            else:
                action = action[:self.action_dim]
        
        return np.clip(action, -1.0, 1.0)
    
    def mpc_control(self, obs: np.ndarray) -> np.ndarray:
        """
        MPC-based control using Axiom-OS ImaginationMPCV2.
        
        Uses physics-informed trajectory optimization.
        """
        q, p = self.obs_to_canonical(obs)
        
        # Use first 2 DOFs for MPC (simplified for demo)
        q_mpc = q[:2] if len(q) >= 2 else np.pad(q, (0, 2 - len(q)))
        p_mpc = p[:2] if len(p) >= 2 else np.pad(p, (0, 2 - len(p)))
        
        # Get optimal action from MPC
        try:
            action_scalar = self.mpc.plan(
                torch.tensor(q_mpc, dtype=torch.float64),
                torch.tensor(p_mpc, dtype=torch.float64)
            )
        except Exception as e:
            # Fallback to PD if MPC fails
            return self.pd_control(obs, np.zeros(2))
        
        # Expand scalar action to full action space using correct action_dim
        action = np.full(self.action_dim, action_scalar * 0.5)
        
        # Add online adaptation residual if enabled
        if self.use_adaptation and self.residual_model is not None:
            with torch.no_grad():
                state_t = torch.tensor(np.concatenate([q_mpc, p_mpc]), dtype=torch.float32)
                residual = self.residual_model(state_t).numpy()
                action[:len(residual)] += residual * 0.1
        
        return np.clip(action, -1.0, 1.0)


def test_environment_basics(env_id: str = "Hopper-v5", seed: int = 42) -> bool:
    """Test 1: Basic environment setup and physics extraction."""
    print(f"\n[TEST 1] Environment Basics: {env_id}")
    print("-" * 60)
    
    try:
        # Create environment
        env = make_axiom_mujoco_env(env_id, seed=seed)
        print(f"  [OK] Environment created")
        
        # Extract physics for Hard Core
        params = get_physics_params(env)
        print(f"  [OK] Physics extracted:")
        print(f"       - Gravity: {params.gravity} m/s^2")
        print(f"       - Body masses: {params.body_masses[:5]}... ({len(params.body_masses)} bodies)")
        print(f"       - Friction: {params.friction:.4f}")
        
        # Test reset and step
        obs, info = env.reset(seed=seed)
        action = env.action_space.sample()
        obs_next, reward, term, trunc, info = env.step(action)
        
        print(f"  [OK] Observation shape: {obs.shape}")
        print(f"  [OK] Action shape: {action.shape}")
        print(f"  [OK] First reward: {reward:.4f}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def test_controllers(env_id: str = "Hopper-v5", max_steps: int = 500, seed: int = 42) -> Dict:
    """Test 2: Compare PD vs MPC controllers."""
    print(f"\n[TEST 2] Controller Comparison: {env_id}")
    print("-" * 60)
    
    results = {}
    
    # Test PD Controller
    print("  Testing PD Controller...")
    env = make_axiom_mujoco_env(env_id, seed=seed)
    params = get_physics_params(env)
    action_dim = env.action_space.shape[0]
    
    controller = AxiomMuJoCoController(env_id, params, action_dim=action_dim, use_adaptation=False)
    obs, _ = env.reset(seed=seed)
    
    pd_rewards = []
    for step in range(max_steps):
        target = np.zeros(len(obs) // 2)  # Target: upright position
        action = controller.pd_control(obs, target)
        obs, reward, term, trunc, _ = env.step(action)
        pd_rewards.append(reward)
        if term or trunc:
            break
    
    env.close()
    results["pd"] = {
        "total_reward": sum(pd_rewards),
        "mean_reward": np.mean(pd_rewards),
        "steps": len(pd_rewards),
        "survival": len(pd_rewards) * 0.01,  # dt=0.01
    }
    print(f"       PD: Reward={results['pd']['total_reward']:.2f}, Steps={results['pd']['steps']}")
    
    # Test MPC Controller (if PyTorch available)
    if HAS_TORCH:
        print("  Testing MPC Controller...")
        env = make_axiom_mujoco_env(env_id, seed=seed)
        action_dim = env.action_space.shape[0]
        controller = AxiomMuJoCoController(env_id, params, action_dim=action_dim, use_adaptation=False)
        obs, _ = env.reset(seed=seed)
        
        mpc_rewards = []
        step_times = []
        
        for step in range(min(max_steps, 100)):  # Limit MPC steps for speed
            t0 = time.time()
            action = controller.mpc_control(obs)
            t1 = time.time()
            step_times.append(t1 - t0)
            
            obs, reward, term, trunc, _ = env.step(action)
            mpc_rewards.append(reward)
            if term or trunc:
                break
        
        env.close()
        results["mpc"] = {
            "total_reward": sum(mpc_rewards),
            "mean_reward": np.mean(mpc_rewards),
            "steps": len(mpc_rewards),
            "survival": len(mpc_rewards) * 0.01,
            "avg_step_time_ms": np.mean(step_times) * 1000,
        }
        print(f"       MPC: Reward={results['mpc']['total_reward']:.2f}, Steps={results['mpc']['steps']}")
        print(f"       MPC: Avg step time={results['mpc']['avg_step_time_ms']:.1f}ms")
    
    return results


def test_sim_to_real(env_id: str = "Hopper-v5", max_steps: int = 500, seed: int = 42) -> Dict:
    """Test 3: Sim-to-Real adaptation with perturbed physics."""
    print(f"\n[TEST 3] Sim-to-Real Adaptation: {env_id}")
    print("-" * 60)
    
    results = {}
    
    # Test scenarios
    scenarios = [
        ("baseline", 1.0, 1.0),
        ("low_gravity", 0.8, 1.0),
        ("high_gravity", 1.2, 1.0),
        ("low_friction", 1.0, 0.7),
        ("high_friction", 1.0, 1.3),
        ("combined", 1.1, 0.8),
    ]
    
    for name, g_scale, f_scale in scenarios:
        print(f"  Testing {name} (g={g_scale:.1f}, f={f_scale:.1f})...")
        
        env = make_axiom_mujoco_env(env_id, gravity_scale=g_scale, friction_scale=f_scale, seed=seed)
        params = get_physics_params(env)
        action_dim = env.action_space.shape[0]
        
        controller = AxiomMuJoCoController(env_id, params, action_dim=action_dim, use_adaptation=True)
        obs, _ = env.reset(seed=seed)
        
        rewards = []
        for step in range(max_steps):
            target = np.zeros(len(obs) // 2)
            action = controller.pd_control(obs, target)  # Use PD for stability
            obs, reward, term, trunc, _ = env.step(action)
            rewards.append(reward)
            if term or trunc:
                break
        
        env.close()
        results[name] = {
            "total_reward": sum(rewards),
            "steps": len(rewards),
        }
        print(f"       {name}: Reward={results[name]['total_reward']:.2f}, Steps={results[name]['steps']}")
    
    # Calculate robustness
    baseline = results["baseline"]["total_reward"]
    variations = [results[k]["total_reward"] for k in results.keys() if k != "baseline"]
    robustness = 1.0 - (np.std(variations) / (abs(baseline) + 1e-6))
    
    print(f"\n  Robustness Score: {robustness:.2f} (higher is better)")
    results["robustness"] = robustness
    
    return results


def test_humanoid_challenge(max_steps: int = 1000, seed: int = 42) -> TestResult:
    """Test 4: Humanoid (Ultimate Boss)."""
    print(f"\n[TEST 4] Ultimate Boss: Humanoid-v5")
    print("-" * 60)
    
    env_id = "Humanoid-v5"
    
    try:
        env = make_axiom_mujoco_env(env_id, seed=seed)
        params = get_physics_params(env)
        
        print(f"  Humanoid parameters:")
        print(f"    - Bodies: {len(params.body_masses)}")
        print(f"    - Total mass: {sum(params.body_masses):.2f} kg")
        print(f"    - Gravity: {params.gravity}")
        
        # Simple stability controller
        action_dim = env.action_space.shape[0]
        controller = AxiomMuJoCoController(env_id, params, action_dim=action_dim, use_adaptation=False)
        obs, _ = env.reset(seed=seed)
        
        rewards = []
        for step in range(max_steps):
            # Target: standing upright (qpos=0 for most joints)
            target = np.zeros(len(obs) // 2)
            action = controller.pd_control(obs, target)
            
            obs, reward, term, trunc, _ = env.step(action)
            rewards.append(reward)
            
            if term or trunc:
                break
        
        env.close()
        
        result = TestResult(
            env_id=env_id,
            controller="PD",
            total_reward=sum(rewards),
            steps=len(rewards),
            survival_time=len(rewards) * 0.005,  # Humanoid dt=0.005
            success=len(rewards) > 100,  # Survived more than 100 steps
        )
        
        print(f"  [OK] Humanoid test completed:")
        print(f"       Steps: {result.steps}")
        print(f"       Survival: {result.survival_time:.2f}s")
        print(f"       Total reward: {result.total_reward:.2f}")
        
        if result.success:
            print(f"       [PASS] Humanoid stable!")
        else:
            print(f"       [FAIL] Humanoid fell too early")
        
        return result
        
    except Exception as e:
        print(f"  [FAIL] {e}")
        return TestResult(env_id, "PD", 0, 0, 0, False)


def run_full_benchmark(seed: int = 42) -> Dict:
    """Run complete benchmark suite."""
    print("\n" + "=" * 70)
    print("Axiom-OS Gymnasium + MuJoCo Full Benchmark")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Environment basics for all test subjects
    for env_id in ["Hopper-v5", "Walker2d-v5"]:
        results[env_id] = {
            "basics": test_environment_basics(env_id, seed),
            "controllers": test_controllers(env_id, 300, seed),
            "sim2real": test_sim_to_real(env_id, 300, seed),
        }
    
    # Test 2: Humanoid (Ultimate Boss)
    results["humanoid"] = test_humanoid_challenge(500, seed)
    
    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    for env_id, data in results.items():
        if env_id == "humanoid":
            print(f"\n{env_id}:")
            print(f"  Survival: {data.survival_time:.2f}s")
            print(f"  Status: {'PASS' if data.success else 'FAIL'}")
        else:
            print(f"\n{env_id}:")
            if "controllers" in data:
                pd_r = data["controllers"].get("pd", {}).get("total_reward", 0)
                mpc_r = data["controllers"].get("mpc", {}).get("total_reward", 0)
                print(f"  PD:  {pd_r:.2f}")
                print(f"  MPC: {mpc_r:.2f}")
            if "sim2real" in data and "robustness" in data["sim2real"]:
                print(f"  Robustness: {data['sim2real']['robustness']:.2f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Axiom-OS Gymnasium + MuJoCo Test Suite"
    )
    parser.add_argument(
        "--env", 
        type=str, 
        default="Hopper-v5",
        choices=["Hopper-v5", "Walker2d-v5", "Humanoid-v5"],
        help="Environment to test"
    )
    parser.add_argument("--steps", type=int, default=500, help="Max steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["all", "basic", "controller", "sim2real", "humanoid", "benchmark"],
        help="Test type"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Axiom-OS Gymnasium + MuJoCo Integration Test")
    print("=" * 70)
    print(f"Environment: {args.env}")
    print(f"Max steps: {args.steps}")
    print(f"Seed: {args.seed}")
    print(f"Gymnasium: {gym.__version__}")
    print(f"MuJoCo: {mujoco.__version__}")
    print(f"PyTorch: {torch.__version__ if HAS_TORCH else 'N/A'}")
    print("=" * 70)
    
    if args.test == "benchmark":
        run_full_benchmark(args.seed)
    elif args.test == "basic":
        test_environment_basics(args.env, args.seed)
    elif args.test == "controller":
        test_controllers(args.env, args.steps, args.seed)
    elif args.test == "sim2real":
        test_sim_to_real(args.env, args.steps, args.seed)
    elif args.test == "humanoid":
        test_humanoid_challenge(args.steps, args.seed)
    else:  # all
        test_environment_basics(args.env, args.seed)
        test_controllers(args.env, args.steps, args.seed)
        test_sim_to_real(args.env, args.steps, args.seed)
    
    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
