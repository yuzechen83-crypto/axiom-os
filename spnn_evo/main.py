"""
SPNN-Evo 3.0 (Axiom-OS) - The Life Cycle
Unifies Perception (RCLN), Cognition (Einstein/Hippocampus), and Action (MPC).

Life Cycle:
  1. Initialize: Load Hard Core from Hippocampus
  2. Observe: Receive UPI data → RCLN predicts y_pred
  3. Control: MPC plans action using Einstein Core
  4. Learn (Fast): Update RCLN via backprop on prediction error
  5. Discover (Slow): If Soft Shell active → Discovery → Update Hippocampus → Reset RCLN

Policy Distillation (Muscle Memory):
  - Boot: MPC control + collect (obs, mpc_action) data
  - Sleep: Train Student MLP on gathered data
  - Run: Switch to Policy for <1ms inference (1kHz robotics)
  - Anomaly: If Policy error spikes → fail-over to MPC (System 2 takes over)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field

from .core.upi import UPIState, Units, VELOCITY, UNITLESS
from .core.hippocampus import HippocampusLibrary, init_default_library, EinsteinCore
from .core.imagination import double_pendulum_H, rollout_controlled
from .core.einstein import SymplecticIntegrator
from .layers.rcln import RCLNLayer, DiscoveryHotspot
from .engine.discovery import DiscoveryEngine
from .orchestrator.mpc import ParallelImaginationController, angle_normalize
from .orchestrator.distillation import (
    PolicyTrainer,
    StudentPolicy,
    DoublePendulumEnv,
)


@dataclass
class AxiomConfig:
    """Axiom-OS configuration."""
    latent_dim: int = 32
    lambda_soft: float = 1.0
    soft_threshold: float = 0.5
    discovery_interval: int = 10
    mpc_horizon: int = 50
    mpc_samples: int = 1000
    cost_mode: str = "stabilization"
    # Policy Distillation
    control_mode: str = "MPC"  # "MPC" (Teacher/Slow) or "Policy" (Student/Fast)
    anomaly_error_threshold: float = 10.0  # MSE above this → fail-over to MPC
    anomaly_check_interval: int = 5  # Check every N steps (0=disable, saves MPC cost)


class AxiomOS:
    """
    Axiom-OS v3.0: Self-Evolving AI
    System 1 (Fast): RCLN neural reflexes
    System 2 (Slow): Einstein symplectic reasoning
    Dual Loop: Engineering (Prediction/Control) + Scientific (Discovery/Crystallization)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        config: Optional[AxiomConfig] = None,
        library: Optional[HippocampusLibrary] = None,
    ):
        self.config = config or AxiomConfig()
        self.library = library or init_default_library()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # RCLN: y = F_hard(x) + λ·F_soft(x)
        self.rcln = RCLNLayer(
            dim=in_dim,
            library=self.library,
            lambda_soft=self.config.lambda_soft,
            soft_threshold=self.config.soft_threshold,
        )
        self.decoder = nn.Linear(in_dim, out_dim)
        self.discovery = DiscoveryEngine(self.library)
        self._hotspots: List[DiscoveryHotspot] = []

        # MPC: Imagination-augmented control (Teacher)
        H = double_pendulum_H(g_over_L=10.0, L1=1.0, L2=1.0)
        self.mpc = ParallelImaginationController(
            H=H,
            horizon_steps=self.config.mpc_horizon,
            n_samples=self.config.mpc_samples,
            cost_mode=self.config.cost_mode,
        )

        # Policy Distillation: Student MLP + Trainer
        self._env = DoublePendulumEnv(H=H, dt=0.02, friction=0.1, noise_std=0.02, max_steps=500)
        self._policy_trainer: Optional[PolicyTrainer] = None
        self._phase: str = "boot"  # boot | sleep | run
        self._last_mpc_action: float = 0.0
        self._anomaly_count: int = 0
        self._use_mpc_until_step: int = -1  # When anomaly: use MPC for next N steps
        self._control_step: int = 0  # For anomaly check interval

        self._step = 0

    def initialize(self) -> None:
        """Load Hard Core from Hippocampus."""
        self.library = self.library or init_default_library()
        self._step = 0

    def observe(
        self,
        x: torch.Tensor,
        return_hotspot: bool = True,
    ) -> Tuple[torch.Tensor, Optional[DiscoveryHotspot]]:
        """
        Receive UPI-compatible input, RCLN predicts y_pred.
        x: (batch, in_dim) input. Can wrap in UPIState if needed.
        """
        y, hotspot = self.rcln(x, return_hotspot=return_hotspot)
        y_pred = self.decoder(y)
        if hotspot is not None:
            self._hotspots.append(hotspot)
        return y_pred, hotspot

    def control(
        self,
        q: np.ndarray,
        p: np.ndarray,
    ) -> float:
        """
        Plan action: MPC (Teacher) or Policy (Student).
        Mode switch via config.control_mode.
        Anomaly: If Policy error spikes (periodic check) → instant fail-over to MPC.
        """
        self._control_step += 1
        q = np.asarray(q).ravel()
        p = np.asarray(p).ravel()
        obs = np.concatenate([q[:2], p[:2]]).astype(np.float32)

        if self.config.control_mode == "MPC":
            tau = self.mpc.plan(q, p)
            self._last_mpc_action = tau
            return tau

        # Policy mode: use Student, with optional anomaly check
        if self._policy_trainer is None:
            tau = self.mpc.plan(q, p)
            self._last_mpc_action = tau
            return tau

        # Fail-over window: use MPC for N steps after anomaly
        if self._control_step <= self._use_mpc_until_step:
            tau = self.mpc.plan(q, p)
            self._last_mpc_action = tau
            return tau

        policy_action = self._policy_trainer.student.act(obs, self._policy_trainer.device)

        # Periodic anomaly check (avoids running MPC every step)
        check_interval = self.config.anomaly_check_interval
        if check_interval > 0 and self._control_step % check_interval == 0:
            mpc_action = self.mpc.plan(q, p)
            self._last_mpc_action = mpc_action
            err = self._policy_trainer.compute_policy_error(obs, mpc_action)
            if err > self.config.anomaly_error_threshold:
                self._anomaly_count += 1
                self._use_mpc_until_step = self._control_step + 10  # MPC for next 10 steps
                return mpc_action

        return policy_action

    def boot_phase(
        self,
        n_episodes: int = 30,
        verbose: bool = True,
    ) -> int:
        """
        Boot: Use MPC to control safely while collecting (obs, mpc_action) data.
        Returns: number of transitions collected.
        """
        if self._policy_trainer is None:
            self._policy_trainer = PolicyTrainer(
                mpc=self.mpc,
                env=self._env,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
        if verbose:
            print("[Boot] MPC control + data collection...")
        n = self._policy_trainer.collect_mpc_data(n_episodes=n_episodes, verbose=verbose)
        self._phase = "boot"
        return n

    def sleep_phase(
        self,
        epochs: int = 100,
        batch_size: int = 256,
        dagger_iterations: int = 0,
        verbose: bool = True,
    ) -> List[float]:
        """
        Sleep: Train Policy Network on gathered data.
        Optionally run DAgger iterations.
        Returns: final epoch losses.
        """
        if self._policy_trainer is None or len(self._policy_trainer.buffer) == 0:
            raise RuntimeError("Run boot_phase first to collect data.")
        if verbose:
            print("[Sleep] Training Student policy...")
        losses = self._policy_trainer.train_student(
            epochs=epochs, batch_size=batch_size, verbose=verbose
        )
        for _ in range(dagger_iterations):
            self._policy_trainer.run_dagger_iteration(
                n_episodes=10, train_epochs=50, verbose=verbose
            )
        self._phase = "sleep"
        return losses

    def run_phase(self) -> None:
        """Run: Switch to Policy Network for high-speed operation."""
        self.config.control_mode = "Policy"
        self._phase = "run"

    def get_phase(self) -> str:
        return self._phase

    def get_anomaly_count(self) -> int:
        return self._anomaly_count

    def get_policy_trainer(self) -> Optional[PolicyTrainer]:
        return self._policy_trainer

    def learn(
        self,
        x: torch.Tensor,
        y_target: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Fast: Update RCLN via backprop on prediction error."""
        y_pred, _ = self.observe(x, return_hotspot=False)
        loss = nn.functional.mse_loss(y_pred, y_target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()
        return loss.item()

    def discover(self) -> List[str]:
        """Slow: If Soft Shell active, run Discovery → Crystallize → Reset RCLN."""
        formulas = []
        for hotspot in self._hotspots:
            f = self.discovery.process_hotspot(hotspot, self.rcln, "Discovered")
            if f:
                formulas.append(f)
        self._hotspots.clear()
        return formulas

    def parameters(self):
        return list(self.rcln.parameters()) + list(self.decoder.parameters())

    def life_cycle_step(
        self,
        x: torch.Tensor,
        y_target: Optional[torch.Tensor] = None,
        q: Optional[np.ndarray] = None,
        p: Optional[np.ndarray] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """
        One step of the full life cycle.
        Observe → Control (if q,p) → Learn (if y_target, optimizer) → Discover (if interval).
        """
        metrics: Dict[str, Any] = {}

        # Observe
        y_pred, hotspot = self.observe(x)
        metrics["hotspot"] = hotspot is not None

        # Control
        if q is not None and p is not None:
            tau = self.control(q, p)
            metrics["action"] = tau

        # Learn
        if y_target is not None and optimizer is not None:
            loss = self.learn(x, y_target, optimizer)
            metrics["loss"] = loss

        # Discover (slow loop)
        if (self._step + 1) % self.config.discovery_interval == 0 and self._hotspots:
            discovered = self.discover()
            if discovered:
                metrics["discovered"] = discovered

        self._step += 1
        return metrics


def demo():
    """Demo: Axiom-OS v3.0 Life Cycle + Policy Distillation"""
    print("=" * 60)
    print("SPNN-Evo 3.0 (Axiom-OS) - Life Cycle + Muscle Memory Demo")
    print("=" * 60)

    config = AxiomConfig(
        latent_dim=16,
        discovery_interval=5,
        mpc_samples=200,
        control_mode="MPC",
    )
    axiom = AxiomOS(in_dim=4, out_dim=2, config=config)
    optimizer = torch.optim.Adam(axiom.parameters(), lr=1e-3)

    axiom.initialize()

    print("\n1. Life Cycle: Observe → Control → Learn → Discover")
    n = 64
    for step in range(30):
        x = torch.randn(n, 4) * 0.5
        y_target = torch.randn(n, 2) * 0.3
        q = np.array([np.pi - 0.5, np.pi - 0.5])
        p = np.array([0.1, 0.1])

        m = axiom.life_cycle_step(x, y_target, q, p, optimizer)

        if (step + 1) % 10 == 0:
            loss = m.get("loss", 0)
            act = m.get("action", 0)
            h = "HOTSPOT" if m.get("hotspot") else ""
            d = m.get("discovered", [])
            print(f"  Step {step+1}: loss={loss:.6f} action={act:.3f} {h} discoveries={len(d)}")
            if d:
                for f in d:
                    print(f"    -> {f}")

    print("\n2. Policy Distillation: Boot → Sleep → Run")
    n_boot = axiom.boot_phase(n_episodes=5, verbose=True)  # Reduced for demo speed
    print(f"   Collected {n_boot} (obs, mpc_action) pairs")
    axiom.sleep_phase(epochs=30, batch_size=128, dagger_iterations=0, verbose=True)  # Quick train
    axiom.run_phase()
    print(f"   Switched to Policy mode. Phase={axiom.get_phase()}")

    # Quick control test with Policy
    q = np.array([np.pi - 0.1, np.pi - 0.1])
    p = np.array([0.05, 0.05])
    tau_policy = axiom.control(q, p)
    print(f"   Policy action at test state: tau={tau_policy:.3f}")

    pt = axiom.get_policy_trainer()
    if pt is not None:
        latency_ms = pt.infer_latency_ms(n_trials=1000)
        print(f"   Student inference: {latency_ms:.3f} ms (target <1ms for 1kHz)")

    print("\nLibrary:", axiom.library.list_ids())
    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
