"""
Axiom-OS - Physics-AI Hybrid Operating System
System Entry Point & Main Lifecycle Loop

Life Cycle:
  1. Setup: Hippocampus, RCLN (Acrobot Hard Core + Empty Soft Shell), MPC
  2. Observe: UPIState from environment
  3. Plan: MPC imagination-augmented action (or Policy when in Run phase)
  4. Act: Apply action to environment
  5. Learn (Fast): Train RCLN Soft Shell on prediction error
  6. Discover (Slow): Every 100 steps, check soft activity → distill → crystallize

Policy Distillation (Boot/Sleep/Run):
  - Boot: MPC control + collect (obs, mpc_action) for Student
  - Sleep: Train Student MLP on gathered data (optional DAgger)
  - Run: Switch to Policy for <1ms inference; anomaly → fail-over to MPC

Probabilistic mode (uncertainty_mode=True):
  Use diffusion to sample p(x_{t+1} | x_t). Returns mean_prediction and uncertainty_bound.
  Requires pre-trained ScoreNet; see tests/test_probabilistic_pendulum.py.
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional

from axiom_os.core import UPIState, Hippocampus
from axiom_os.layers import RCLNLayer
from axiom_os.engine import DiscoveryEngine
from axiom_os.orchestrator import ImaginationMPC, AIIntelligence
from axiom_os.orchestrator.mpc import double_pendulum_H, step_env
from axiom_os.orchestrator.distillation import (
    PolicyTrainer,
    StudentPolicy,
    DoublePendulumEnv,
)

PI = np.pi


@dataclass
class AxiomConfig:
    """Axiom-OS configuration including Policy Distillation and AI."""
    T_max: int = 500  # 主循环步数，增加以观察收敛与 Discovery 稳定性
    discover_interval: int = 50   # 更频繁检查，便于触发 Discovery
    soft_activity_threshold: float = 0.02  # 降低阈值，提高 Discovery 触发率
    buffer_capacity: int = 500
    # Policy Distillation
    control_mode: str = "MPC"  # "MPC" (Teacher) or "Policy" (Student)
    anomaly_error_threshold: float = 10.0
    anomaly_check_interval: int = 5  # 0 = disable
    # AI 智能化
    use_ai: bool = False  # 启用 AIIntelligence 编排
    ai_auto_discovery: bool = True  # 高活动时自动建议/执行 Discovery


def make_acrobot_hard_core(H_func, dt=0.02, friction=0.1):
    """
    Acrobot Physics Hard Core: one-step prediction (q,p) -> (q_next, p_next) with zero action.
    """
    from axiom_os.orchestrator.mpc import _step_controlled

    def hard_core(x):
        if isinstance(x, torch.Tensor):
            vals = x
        elif hasattr(x, "values") and not isinstance(x, torch.Tensor):
            vals = x.values
        else:
            vals = x
        vals = torch.as_tensor(vals, dtype=torch.float32) if not isinstance(vals, torch.Tensor) else vals.float()
        if vals.dim() == 1:
            vals = vals.unsqueeze(0)
        v = vals.cpu().numpy() if vals.is_cuda else vals.numpy()
        q, p = v[:, :2], v[:, 2:4]
        out_list = []
        for i in range(v.shape[0]):
            qn, pn = _step_controlled(q[i], p[i], 0.0, H_func, 2, dt, friction)
            out_list.append(np.concatenate([qn, pn]))
        out = np.stack(out_list).astype(np.float32).copy()
        return torch.from_numpy(out)

    return hard_core


def _control(
    mpc: ImaginationMPC,
    policy_trainer: Optional[PolicyTrainer],
    config: AxiomConfig,
    q: np.ndarray,
    p: np.ndarray,
    control_step: int,
    use_mpc_until_step: int,
) -> tuple:
    """
    Plan action: MPC or Policy. Anomaly check triggers fail-over to MPC.
    Returns: (action, control_step, use_mpc_until_step).
    """
    obs = np.concatenate([q, p]).astype(np.float32)
    control_step_new = control_step + 1

    if config.control_mode == "MPC":
        return mpc.plan(q, p), control_step_new, use_mpc_until_step

    if policy_trainer is None:
        return mpc.plan(q, p), control_step_new, use_mpc_until_step

    if control_step_new <= use_mpc_until_step:
        return mpc.plan(q, p), control_step_new, use_mpc_until_step

    policy_action = policy_trainer.student.act(obs, policy_trainer.device)

    check_interval = config.anomaly_check_interval
    if check_interval > 0 and control_step_new % check_interval == 0:
        mpc_action = mpc.plan(q, p)
        err = policy_trainer.compute_policy_error(obs, mpc_action)
        if err > config.anomaly_error_threshold:
            use_mpc_until_step = control_step_new + 10
            return mpc_action, control_step_new, use_mpc_until_step

    return policy_action, control_step_new, use_mpc_until_step


def main(
    uncertainty_mode: bool = False,
    hippocampus=None,
    discovery_engine=None,
    config: Optional[AxiomConfig] = None,
    use_distillation: bool = False,
    use_ai: bool = False,
):
    """Axiom-OS main lifecycle.
    uncertainty_mode: Reserved for probabilistic mode.
    hippocampus: Optional external Hippocampus; if None, creates one.
    discovery_engine: Optional external DiscoveryEngine; if None, creates one.
    config: Optional AxiomConfig; if None, uses defaults.
    use_distillation: If True, run Boot phase first to collect MPC data for Policy.
    use_ai: If True, enable AIIntelligence for smart discovery orchestration.
    """
    cfg = config or AxiomConfig()
    cfg.use_ai = cfg.use_ai or use_ai
    print("=" * 60)
    print("Axiom-OS - Physics-AI Hybrid Operating System")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Setup
    # -------------------------------------------------------------------------
    H = double_pendulum_H(g_over_L=10.0, L1=1.0, L2=1.0)

    if hippocampus is None:
        hippocampus = Hippocampus(dim=32, capacity=5000)
    if discovery_engine is None:
        discovery_engine = DiscoveryEngine(use_pysr=False)
    rcln = RCLNLayer(
        input_dim=4,
        hidden_dim=64,
        output_dim=4,
        hard_core_func=make_acrobot_hard_core(H),
        lambda_res=0.5,
    )
    mpc = ImaginationMPC(
        H=H,
        horizon_steps=20,
        n_samples=50,
        dt=0.02,
        friction=0.1,
    )

    optimizer = torch.optim.Adam(rcln.soft_shell.parameters(), lr=1e-3)
    data_buffer = []
    buffer_capacity = cfg.buffer_capacity

    # AI 智能化
    ai: Optional[AIIntelligence] = None
    if cfg.use_ai:
        ai = AIIntelligence(
            rcln=rcln,
            hippocampus=hippocampus,
            discovery=discovery_engine,
            activity_threshold=cfg.soft_activity_threshold,
            use_mock_llm=True,
        )
        ai.set_data_buffer(data_buffer)

    # Policy Distillation
    policy_trainer: Optional[PolicyTrainer] = None
    env = DoublePendulumEnv(H=H, dt=0.02, friction=0.1, noise_std=0.02, max_steps=500)
    control_step = 0
    use_mpc_until_step = -1

    if use_distillation:
        policy_trainer = PolicyTrainer(mpc=mpc, env=env, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        n_boot = policy_trainer.collect_mpc_data(n_episodes=10, verbose=True)
        print(f"  [Boot] Collected {n_boot} (obs, mpc_action) pairs")
        policy_trainer.train_student(epochs=50, batch_size=128, verbose=True)
        cfg.control_mode = "Policy"
        print("  [Run] Switched to Policy mode")

    # Environment state
    q = np.array([PI - 0.2, PI - 0.2])
    p = np.array([0.1, 0.1])
    dt_env = 0.02

    # Logging
    log = {"t": [], "q": [], "p": [], "action": [], "loss": [], "discovered": []}
    T_max = cfg.T_max
    discover_interval = cfg.discover_interval
    soft_activity_threshold = cfg.soft_activity_threshold

    print("\nSetup: Hippocampus, RCLN (Acrobot Hard Core + Soft Shell), MPC, Discovery")
    print(f"T_max={T_max}, discover_interval={discover_interval}, control_mode={cfg.control_mode}, use_ai={cfg.use_ai}")
    print("-" * 60)

    # -------------------------------------------------------------------------
    # 2. Simulation Loop
    # -------------------------------------------------------------------------
    for t in range(T_max):
        obs = np.concatenate([q, p]).astype(np.float32)
        upi_state = UPIState(values=obs, units=[0, 0, 0, 0, 0], semantics="AcrobotState")

        # Plan: MPC or Policy (with anomaly fail-over)
        action, control_step, use_mpc_until_step = _control(
            mpc, policy_trainer, cfg, q, p, control_step, use_mpc_until_step
        )

        # Act
        q_next, p_next = step_env(q, p, action, H, dt=dt_env, friction=0.1, state_dim=2)
        target = np.concatenate([q_next, p_next]).astype(np.float32)

        # Learn (Fast)
        x_t = torch.from_numpy(obs).float().unsqueeze(0)
        y_target = torch.from_numpy(target).float().unsqueeze(0)
        y_pred = rcln(x_t)
        loss = torch.nn.functional.mse_loss(y_pred, y_target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rcln.parameters(), 1.0)
        optimizer.step()

        # Buffer for discovery
        y_soft_np = rcln._last_y_soft.cpu().numpy().copy() if rcln._last_y_soft is not None else np.zeros(4)
        data_buffer.append((obs.copy(), y_soft_np))
        if len(data_buffer) > buffer_capacity:
            data_buffer.pop(0)

        # Discover (Slow) with validation loop
        discovered = None
        if ai is not None:
            ai.set_data_buffer(data_buffer)
        if (t + 1) % discover_interval == 0:
            activity = rcln.get_soft_activity()
            formula = None
            buf = data_buffer[-100:] if len(data_buffer) >= 50 else data_buffer
            # Discovery 事件日志
            log.setdefault("discovery_events", []).append((t + 1, activity, None, False))
            # AI 智能化：优先使用 AI 编排的 Discovery 建议
            if ai is not None and cfg.ai_auto_discovery and ai.recommend_discovery() and len(buf) >= 10:
                recommended, formula_or_msg = ai.auto_discovery(data_buffer=buf, execute=True)
                if formula_or_msg and "Error" not in str(formula_or_msg) and "未发现" not in str(formula_or_msg) and "数据不足" not in str(formula_or_msg):
                    formula = formula_or_msg
            elif activity > soft_activity_threshold:
                formula = discovery_engine.distill(
                    rcln_layer=rcln,
                    data_buffer=buf,
                    input_units=[[0, 0, 0, 0, 0]] * 4,
                )
            if formula is not None and len(str(formula)) > 2:
                X = np.stack([np.asarray(p[0]).ravel() for p in buf])
                y = np.stack([np.asarray(p[1]).ravel() for p in buf])
                valid, mse = discovery_engine.validate_formula(formula, X, y, output_dim=4)
                if valid:
                    hippocampus.crystallize(formula, rcln, formula_id=f"law_{t}")
                    discovered = formula
                    if log.get("discovery_events"):
                        log["discovery_events"][-1] = (t + 1, activity, formula, True)
                    optimizer = torch.optim.Adam(rcln.soft_shell.parameters(), lr=1e-3)
                    if ai is not None:
                        ai.reset_activity_counter()

        q, p = q_next, p_next

        log["t"].append(t * dt_env)
        log["q"].append(q.copy())
        log["p"].append(p.copy())
        log["action"].append(action)
        log["loss"].append(loss.item())
        log["discovered"].append(discovered)

        if (t + 1) % 100 == 0:
            err = np.sqrt(np.mean((np.array([PI, PI]) - q) ** 2))
            mode = "Policy" if cfg.control_mode == "Policy" else "MPC"
            print(f"  t={t+1:4d}  loss={loss.item():.6f}  activity={rcln.get_soft_activity():.4f}  err={err:.3f}  tau={action:.2f} [{mode}]" + (f"  DISCOVERED: {discovered}" if discovered else ""))

    # -------------------------------------------------------------------------
    # 3. Summary
    # -------------------------------------------------------------------------
    print("-" * 60)
    print("Lifecycle complete.")
    n_cryst = sum(1 for e in log.get("discovery_events", []) if len(e) >= 4 and e[3])
    print(f"Discovery events: {len(log.get('discovery_events', []))} checks, {n_cryst} crystallized")
    print(f"Knowledge base: {list(hippocampus.knowledge_base.keys())}")
    for fid, meta in hippocampus.knowledge_base.items():
        formula_str = meta.get("formula", str(meta)) if isinstance(meta, dict) else str(meta)
        print(f"  [{fid}] {formula_str[:120]}{'...' if len(str(formula_str)) > 120 else ''}")
    if policy_trainer is not None:
        latency_ms = policy_trainer.infer_latency_ms(n_trials=1000)
        print(f"Policy inference: {latency_ms:.3f} ms")
    print("=" * 60)


def boot_phase(mpc: ImaginationMPC, env: DoublePendulumEnv, n_episodes: int = 30) -> PolicyTrainer:
    """Boot: MPC control + collect (obs, mpc_action). Returns PolicyTrainer."""
    pt = PolicyTrainer(mpc=mpc, env=env, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    pt.collect_mpc_data(n_episodes=n_episodes, verbose=True)
    return pt


def sleep_phase(pt: PolicyTrainer, epochs: int = 100, dagger_iterations: int = 0) -> list:
    """Sleep: Train Student on gathered data. Optional DAgger. Returns epoch losses."""
    losses = pt.train_student(epochs=epochs, batch_size=256, verbose=True)
    for _ in range(dagger_iterations):
        pt.run_dagger_iteration(n_episodes=10, train_epochs=50, verbose=True)
    return losses


def run_phase(config: AxiomConfig) -> None:
    """Run: Switch to Policy mode."""
    config.control_mode = "Policy"


if __name__ == "__main__":
    main(use_distillation=False, use_ai=True)  # use_ai=True 启用智能化编排
