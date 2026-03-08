"""
Axiom-OS v1.0 Mission Control Dashboard (钢铁侠仪表盘)
Live Acrobot • Energy Monitor • Brain Map • Discovery Console.

Features:
- Live Simulation: Double pendulum with torque color-coding
- Kick Button: Apply random impulse, watch it recover
- Brain Map: RCLN activation heat (red=learning, green=crystallized)
- Energy Monitor: Real vs Einstein Ideal
- Discovery Console: Log laws, Crystallize Knowledge
- Control Override: Sliders for MPC params

Run: streamlit run axiom_os/mission_control.py
"""

import sys
import logging
from pathlib import Path

class _ScriptRunContextFilter(logging.Filter):
    def filter(self, record):
        msg = record.msg % record.args if record.args else str(record.msg)
        return "ScriptRunContext" not in msg
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").addFilter(_ScriptRunContextFilter())

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from axiom_os.core.upi import UPIState, Units
from axiom_os.layers.rcln import RCLNLayer
from axiom_os.engine.discovery import DiscoveryEngine
from axiom_os.core.hippocampus import Hippocampus
from axiom_os.orchestrator.mpc import (
    ImaginationMPC,
    double_pendulum_H,
    step_env,
    angle_normalize,
)

PI = np.pi


@st.cache_resource
def get_discovery_components():
    """RCLN, DiscoveryEngine, Hippocampus for Discovery Console and Chief Scientist."""
    def hard_core_zero(x):
        t = torch.as_tensor(x.values if hasattr(x, "values") else x, dtype=torch.float32)
        return torch.zeros_like(t)
    rcln = RCLNLayer(1, 32, 1, hard_core_func=hard_core_zero, lambda_res=1.0)
    engine = DiscoveryEngine(use_pysr=False)
    hippo = Hippocampus()
    return rcln, engine, hippo


st.set_page_config(page_title="Axiom-OS Mission Control", layout="wide", initial_sidebar_state="expanded")

st.title("Axiom-OS v1.0 Mission Control")
st.caption("Live Acrobot • Energy Monitor • Discovery Console • Self-Evolving Physics")


# ============== Session State ==============
if "acrobot_history" not in st.session_state:
    st.session_state.acrobot_history = {"t": [], "q1": [], "q2": [], "p1": [], "p2": [], "tau": [], "E_real": [], "E_ideal": [], "soft_activity": []}
if "discovery_log" not in st.session_state:
    st.session_state.discovery_log = []
if "discovery_data_buffer" not in st.session_state:
    st.session_state.discovery_data_buffer = []
if "last_state" not in st.session_state:
    st.session_state.last_state = None  # (q, p) for Kick


# ============== Sidebar: Control Override ==============
st.sidebar.header("Control Override")
st.sidebar.markdown("**MPC Parameters**")
horizon = st.sidebar.slider("Horizon steps", 30, 100, 80)
n_samples = st.sidebar.slider("Samples", 500, 3000, 2000)
action_std = st.sidebar.slider("Action std (noise)", 0.5, 5.0, 2.0)
friction = st.sidebar.slider("Friction", 0.01, 0.3, 0.1)
dt = st.sidebar.slider("dt", 0.01, 0.05, 0.02)

st.sidebar.divider()
st.sidebar.markdown("**Simulation**")
n_steps_run = st.sidebar.slider("Steps per Run", 10, 100, 50)
q1_init = st.sidebar.slider("θ1 init (rad)", 2.5, 3.5, PI - 0.02)
q2_init = st.sidebar.slider("θ2 init (rad)", 2.5, 3.5, PI - 0.02)

st.sidebar.divider()
st.sidebar.markdown("**CAD 建模**")
with st.sidebar.expander("3D 建模", expanded=False):
    cad_shape_mc = st.selectbox("形状", ["l_bracket", "box", "cylinder", "sphere", "simple_gear"], key="mc_cad_shape")
    if st.button("生成 STL", key="mc_cad_btn"):
        try:
            from axiom_os.agent.tools import run_cad_model
            r = run_cad_model(shape=cad_shape_mc)
            if r.get("ok"):
                st.success(f"已保存: {r.get('path', '')}")
            else:
                st.error(r.get("error", "失败"))
        except Exception as e:
            st.error(str(e))

st.sidebar.divider()
st.sidebar.markdown("**🤖 AI 智能化**")
use_real_llm = st.sidebar.checkbox("使用真实 LLM (Ollama)", value=False, help="需先运行 ollama pull llama3.2")
with st.sidebar.expander("Chief Scientist 对话", expanded=False):
    if "chief_chat" not in st.session_state:
        st.session_state.chief_chat = []
    rcln_d, engine_d, hippo_d = get_discovery_components()
    ai_inst = None
    try:
        from axiom_os.orchestrator import AIIntelligence
        ai_inst = AIIntelligence(
            rcln=rcln_d, hippocampus=hippo_d, discovery=engine_d,
            use_mock_llm=not use_real_llm,
            backend="ollama",
            model="llama3.2",
        )
        suggestion = ai_inst.get_proactive_suggestion()
        if suggestion:
            st.caption(f"💡 **主动建议:** {suggestion[:150]}{'...' if len(suggestion) > 150 else ''}")
    except Exception:
        ai_inst = None
    st.caption("**目标驱动** (超级智能化)")
    goal_text = st.text_input("输入目标", placeholder="e.g. 研究双摆摩擦", key="goal_input")
    if st.button("执行目标", key="run_goal_btn"):
        if goal_text.strip() and ai_inst is not None:
            try:
                buf = st.session_state.get("discovery_data_buffer", [])
                ai_inst.chief.set_data_buffer(buf if len(buf) >= 10 else None)
                ans = ai_inst.chief._tool_run_goal(goal_text.strip())
                st.session_state.chief_chat.append((f"[目标] {goal_text.strip()}", ans))
                st.rerun()
            except Exception as ex:
                st.session_state.chief_chat.append((f"[目标] {goal_text.strip()}", f"[Error: {ex}]"))
                st.rerun()
    st.divider()
    q_ai = st.text_input("向 Chief Scientist 提问", placeholder="e.g. 当前知识库有哪些定律？")
    if st.button("Ask", key="ask_chief"):
        if q_ai.strip() and ai_inst is not None:
            try:
                ans = ai_inst.ask(q_ai.strip())
                st.session_state.chief_chat.append((q_ai.strip(), ans))
                st.rerun()
            except Exception as ex:
                st.session_state.chief_chat.append((q_ai.strip(), f"[Error: {ex}]"))
                st.rerun()
        elif q_ai.strip() and ai_inst is None:
            st.session_state.chief_chat.append((q_ai.strip(), "[Error: AI 初始化失败]"))
            st.rerun()
    for q, a in st.session_state.chief_chat[-5:]:
        st.markdown(f"**Q:** {q}")
        st.caption(a[:300] + ("..." if len(a) > 300 else ""))


# ============== Build MPC & Env ==============
@st.cache_resource
def get_mpc(_horizon, _n_samples, _action_std, _friction, _dt):
    H = double_pendulum_H(g_over_L=10.0, L1=1.0, L2=1.0)
    return ImaginationMPC(
        H=H,
        horizon_steps=_horizon,
        n_samples=_n_samples,
        dt=_dt,
        friction=_friction,
        action_std=_action_std,
        target_state=np.array([PI, PI]),
    ), H

mpc, H = get_mpc(horizon, n_samples, action_std, friction, dt)
E_ideal = float(np.asarray(H(np.concatenate([[PI, PI], [0.0, 0.0]]))).ravel()[0])


@st.cache_resource
def get_brain_rcln():
    def hard_core_zero(x):
        if isinstance(x, torch.Tensor):
            vals = x
        elif hasattr(x, "values") and not isinstance(x, torch.Tensor):
            vals = x.values
        else:
            vals = x
        t = torch.as_tensor(vals, dtype=torch.float32)
        return torch.zeros_like(t)
    return RCLNLayer(1, 32, 1, hard_core_func=hard_core_zero, lambda_res=1.0)

def get_soft_activity_for_state(q, p):
    """RCLN soft shell activation: feed velocity magnitude. High = red (learning), Low = green (crystallized)."""
    brain = get_brain_rcln()
    v_mag = float(np.sqrt(np.sum(np.array(p)**2)))
    x = torch.tensor([[v_mag]], dtype=torch.float32)
    with torch.no_grad():
        _ = brain(x)
    return brain.get_soft_activity()


# ============== Run Simulation ==============
def run_acrobot_steps(n_steps, q0, p0):
    q, p = np.array(q0, dtype=np.float64), np.array(p0, dtype=np.float64)
    hist = st.session_state.acrobot_history
    for _ in range(n_steps):
        tau = mpc.plan(q, p)
        q, p = step_env(q, p, tau, H, dt=dt, friction=friction, state_dim=2)
        E_real = float(np.asarray(H(np.concatenate([q, p]))).ravel()[0])
        soft_a = get_soft_activity_for_state(q, p)
        t = len(hist["t"]) * dt
        hist["t"].append(t)
        hist["q1"].append(q[0])
        hist["q2"].append(q[1])
        hist["p1"].append(p[0])
        hist["p2"].append(p[1])
        hist["tau"].append(tau)
        hist["E_real"].append(E_real)
        hist["E_ideal"].append(E_ideal)
        hist["soft_activity"].append(soft_a)
    st.session_state.last_state = (q.copy(), p.copy())


col_run, col_reset, col_kick, _ = st.columns([1, 1, 1, 3])
with col_run:
    if st.button("▶ Run Simulation", type="primary"):
        q0 = np.array([q1_init, q2_init])
        p0 = np.array([0.0, 0.0])
        run_acrobot_steps(n_steps_run, q0, p0)
        st.rerun()
with col_reset:
    if st.button("↺ Reset"):
        st.session_state.acrobot_history = {"t": [], "q1": [], "q2": [], "p1": [], "p2": [], "tau": [], "E_real": [], "E_ideal": [], "soft_activity": []}
        st.session_state.last_state = None
        st.rerun()
with col_kick:
    kick_enabled = st.session_state.last_state is not None or bool(st.session_state.acrobot_history["t"])
    if st.button("⚡ Kick!", type="secondary", disabled=not kick_enabled, help="Apply random impulse - watch it recover"):
        if st.session_state.last_state is not None:
            q, p = st.session_state.last_state
        else:
            hist = st.session_state.acrobot_history
            q = np.array([hist["q1"][-1], hist["q2"][-1]])
            p = np.array([hist["p1"][-1], hist["p2"][-1]])
        # Apply large random impulse
        rng = np.random.default_rng()
        p = p + rng.uniform(-8, 8, 2)
        st.session_state.last_state = (q, p)
        run_acrobot_steps(n_steps_run, q, p)
        st.rerun()


# ============== Live Simulation: Double Pendulum ==============
st.subheader("Live Simulation: Double Pendulum (Acrobot)")
hist = st.session_state.acrobot_history

if hist["t"]:
    # Pendulum visualization (last state)
    L1, L2 = 1.0, 1.0
    q1, q2 = hist["q1"][-1], hist["q2"][-1]
    tau_abs = abs(hist["tau"][-1])
    x0, y0 = 0.0, 0.0
    x1 = L1 * np.sin(q1)
    y1 = -L1 * np.cos(q1)
    x2 = x1 + L2 * np.sin(q2)
    y2 = y1 - L2 * np.cos(q2)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.plot([x0, x1], [y0, y1], "o-", color="blue", linewidth=4, markersize=10, label="Link 1")
    ax.plot([x1, x2], [y1, y2], "o-", color="green", linewidth=4, markersize=10, label="Link 2")
    # Color by torque stress (red tint for high torque)
    stress = min(1.0, tau_abs / 15.0)
    ax.set_facecolor((1, 1 - stress * 0.3, 1 - stress * 0.3))
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_title(f"θ1={q1:.3f} θ2={q2:.3f} | τ={hist['tau'][-1]:.2f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()
else:
    st.info("Click **Run Simulation** to start.")


# ============== Telemetry Panels ==============
col_energy, col_neural = st.columns(2)

with col_energy:
    st.subheader("Energy Monitor")
    if hist["t"]:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(hist["t"], hist["E_real"], "b-", label="Real Energy")
        ax.axhline(E_ideal, color="green", linestyle="--", label="Einstein Ideal (upright)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Energy H")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    else:
        st.caption("—")

with col_neural:
    st.subheader("Brain Map: RCLN Activation Heat")
    if hist["t"] and hist.get("soft_activity"):
        # Red = learning (high), Green = crystallized (low)
        act = np.array(hist["soft_activity"])
        act_norm = (act - act.min()) / (act.max() - act.min() + 1e-8)
        fig, ax = plt.subplots(figsize=(6, 2))
        for i in range(len(act_norm) - 1):
            r = 1 - act_norm[i]
            g = act_norm[i]
            ax.axvspan(hist["t"][i], hist["t"][i + 1], color=(r, g, 0.3), alpha=0.8)
        ax.set_xlim(hist["t"][0], hist["t"][-1])
        ax.set_ylim(0, 1)
        ax.set_xlabel("Time")
        ax.set_ylabel("Heat")
        ax.set_title("🔴 Learning (perturbed)  →  🟢 Crystallized (stable)")
        st.pyplot(fig)
        plt.close()
        st.caption(f"Current: {act[-1]:.4f} — Red=high activity, Green=low")
    else:
        st.metric("Soft Shell Magnitude", "—")
        st.caption("Run simulation to see activation heat over time.")


# ============== Discovery Console ==============
st.divider()
st.subheader("Discovery Console")

rcln_disc, engine_disc, hippo_disc = get_discovery_components()

# Run discovery on friction data
friction_slider = st.slider("Friction c (for discovery)", 0.1, 1.0, 0.5)
v_train = torch.linspace(-3, 3, 80).reshape(-1, 1).float()
F_true = -friction_slider * v_train
inp_upi = UPIState(v_train, units=Units.VELOCITY, semantics="velocity")
opt = torch.optim.Adam(rcln_disc.parameters(), lr=0.01)
for _ in range(200):
    opt.zero_grad()
    F_pred = rcln_disc(inp_upi)
    loss = torch.mean((F_pred.float() - F_true.float()) ** 2)
    loss.backward()
    opt.step()
with torch.no_grad():
    _ = rcln_disc(inp_upi)
y_soft = rcln_disc._last_y_soft.cpu().numpy()
x_np = v_train.numpy()
st.session_state.discovery_data_buffer = list(zip(x_np.tolist(), y_soft.tolist()))
formula = engine_disc.discover(x_np, y_soft)
formula_str = formula if formula else "—"

st.markdown("**Latest formula:**")
st.code(formula_str, language="text")

if st.button("🔬 Run Discovery"):
    st.session_state.discovery_log.append(f"New term found: {formula_str}")
    st.rerun()

st.markdown("**Discovery Log:**")
st.text_area("Log", value="\n".join(st.session_state.discovery_log[-10:]) if st.session_state.discovery_log else "(empty)", height=80, disabled=True)

if st.button("💎 Crystallize Knowledge"):
    if formula_str and formula_str != "—":
        try:
            fid = hippo_disc.crystallize(formula_str, rcln_disc)
            st.session_state.discovery_log.append(f"[CRYSTALLIZED] {formula_str} → {fid}")
            st.success(f"Crystallized as {fid}")
        except Exception as e:
            st.error(str(e))
    else:
        st.warning("No formula to crystallize.")
    st.rerun()


# ============== Trajectory Plots ==============
if hist["t"]:
    st.divider()
    st.subheader("Trajectory")
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(hist["t"], hist["q1"], "b-", label="θ1")
    axes[0].axhline(PI, color="k", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("θ1")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(hist["t"], hist["q2"], "g-", label="θ2")
    axes[1].axhline(PI, color="k", linestyle="--", alpha=0.5)
    axes[1].set_ylabel("θ2")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(hist["t"], hist["tau"], "r-", label="τ")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Torque")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()
