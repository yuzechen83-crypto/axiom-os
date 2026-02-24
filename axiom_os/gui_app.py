"""
Axiom-OS 可视化窗口 - 统一仪表盘

整合：主控台 | RAR 发现 | 实验 | 知识库 | 设置

运行: streamlit run axiom_os/gui_app.py
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

# ============== Page Config ==============
st.set_page_config(
    page_title="Axiom-OS 可视化",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============== Session State ==============
if "page" not in st.session_state:
    st.session_state.page = "主控台"
if "rar_result" not in st.session_state:
    st.session_state.rar_result = None
if "hippocampus" not in st.session_state:
    st.session_state.hippocampus = None

# ============== Sidebar Navigation ==============
st.sidebar.title("🔬 Axiom-OS")
st.sidebar.caption("Physics-AI 混合操作系统")
st.sidebar.divider()

page = st.sidebar.radio(
    "导航",
    ["主控台", "RAR 发现", "实验", "知识库", "设置"],
    index=["主控台", "RAR 发现", "实验", "知识库", "设置"].index(st.session_state.page),
    label_visibility="collapsed",
)
st.session_state.page = page

st.sidebar.divider()
st.sidebar.caption("v1.0 • 钢铁侠仪表盘")

# ============== Main Content Router ==============
def render_main_control():
    """主控台：双摆仿真、能量监控、Brain Map"""
    st.header("主控台")
    st.caption("Live Acrobot • 能量监控 • RCLN 激活热图")

    try:
        from axiom_os.orchestrator.mpc import ImaginationMPC, double_pendulum_H, step_env
    except ImportError:
        st.error("无法加载 MPC 模块")
        return

    PI = np.pi
    if "acrobot_history" not in st.session_state:
        st.session_state.acrobot_history = {
            "t": [], "q1": [], "q2": [], "p1": [], "p2": [], "tau": [],
            "E_real": [], "E_ideal": [], "soft_activity": []
        }
    if "last_state" not in st.session_state:
        st.session_state.last_state = None

    # Sidebar params
    with st.sidebar.expander("仿真参数", expanded=False):
        horizon = st.slider("Horizon", 30, 100, 80)
        n_samples = st.slider("Samples", 500, 3000, 2000)
        friction = st.slider("Friction", 0.01, 0.3, 0.1)
        dt = st.slider("dt", 0.01, 0.05, 0.02)
        n_steps = st.slider("每步运行", 10, 100, 50)
        q1_init = st.slider("θ1 init", 2.5, 3.5, PI - 0.02)
        q2_init = st.slider("θ2 init", 2.5, 3.5, PI - 0.02)

    H = double_pendulum_H(g_over_L=10.0, L1=1.0, L2=1.0)
    mpc = ImaginationMPC(H=H, horizon_steps=horizon, n_samples=n_samples, dt=dt, friction=friction,
                         target_state=np.array([PI, PI]))
    E_ideal = float(np.asarray(H(np.concatenate([[PI, PI], [0.0, 0.0]]))).ravel()[0])

    def get_soft_activity(q, p):
        try:
            from axiom_os.layers.rcln import RCLNLayer
            def hc(x):
                t = torch.as_tensor(x.values if hasattr(x, "values") else x, dtype=torch.float32)
                return torch.zeros_like(t)
            rcln = RCLNLayer(1, 32, 1, hard_core_func=hc, lambda_res=1.0)
            v = np.sqrt(np.sum(np.array(p)**2))
            with torch.no_grad():
                _ = rcln(torch.tensor([[v]], dtype=torch.float32))
            return rcln.get_soft_activity()
        except Exception:
            return 0.5

    def run_steps(n, q0, p0):
        q, p = np.array(q0, dtype=np.float64), np.array(p0, dtype=np.float64)
        hist = st.session_state.acrobot_history
        for _ in range(n):
            tau = mpc.plan(q, p)
            q, p = step_env(q, p, tau, H, dt=dt, friction=friction, state_dim=2)
            E = float(np.asarray(H(np.concatenate([q, p]))).ravel()[0])
            sa = get_soft_activity(q, p)
            t = len(hist["t"]) * dt
            hist["t"].append(t)
            hist["q1"].append(q[0]); hist["q2"].append(q[1])
            hist["p1"].append(p[0]); hist["p2"].append(p[1])
            hist["tau"].append(tau)
            hist["E_real"].append(E)
            hist["E_ideal"].append(E_ideal)
            hist["soft_activity"].append(sa)
        st.session_state.last_state = (q.copy(), p.copy())

    col1, col2, col3, _ = st.columns([1, 1, 1, 3])
    with col1:
        if st.button("▶ 运行", type="primary"):
            run_steps(n_steps, [q1_init, q2_init], [0.0, 0.0])
            st.rerun()
    with col2:
        if st.button("↺ 重置"):
            st.session_state.acrobot_history = {
                "t": [], "q1": [], "q2": [], "p1": [], "p2": [], "tau": [],
                "E_real": [], "E_ideal": [], "soft_activity": []
            }
            st.session_state.last_state = None
            st.rerun()
    with col3:
        kick_ok = st.session_state.last_state is not None or bool(st.session_state.acrobot_history["t"])
        if st.button("⚡ 扰动", disabled=not kick_ok, help="施加随机冲量"):
            if st.session_state.last_state:
                q, p = st.session_state.last_state
            else:
                h = st.session_state.acrobot_history
                q, p = [h["q1"][-1], h["q2"][-1]], [h["p1"][-1], h["p2"][-1]]
            p = np.array(p) + np.random.uniform(-8, 8, 2)
            run_steps(n_steps, q, p)
            st.rerun()

    hist = st.session_state.acrobot_history
    if hist["t"]:
        L1, L2 = 1.0, 1.0
        q1, q2 = hist["q1"][-1], hist["q2"][-1]
        x0, y0 = 0, 0
        x1 = L1 * np.sin(q1); y1 = -L1 * np.cos(q1)
        x2 = x1 + L2 * np.sin(q2); y2 = y1 - L2 * np.cos(q2)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_aspect("equal")
        ax.plot([x0, x1], [y0, y1], "o-", color="blue", lw=3, markersize=8)
        ax.plot([x1, x2], [y1, y2], "o-", color="green", lw=3, markersize=8)
        ax.axhline(0, color="gray", ls="--", alpha=0.3)
        ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
        ax.set_title(f"θ1={q1:.3f} θ2={q2:.3f} | τ={hist['tau'][-1]:.2f}")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

        c1, c2 = st.columns(2)
        with c1:
            fig2, ax2 = plt.subplots(figsize=(5, 2.5))
            ax2.plot(hist["t"], hist["E_real"], "b-", label="Real")
            ax2.axhline(E_ideal, color="g", ls="--", label="Ideal")
            ax2.set_xlabel("Time"); ax2.set_ylabel("Energy")
            ax2.legend(); ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            plt.close()
        with c2:
            if hist.get("soft_activity"):
                act = np.array(hist["soft_activity"])
                act_n = (act - act.min()) / (act.max() - act.min() + 1e-8)
                fig3, ax3 = plt.subplots(figsize=(5, 2))
                for i in range(len(act_n) - 1):
                    r, g = 1 - act_n[i], act_n[i]
                    ax3.axvspan(hist["t"][i], hist["t"][i+1], color=(r, g, 0.3), alpha=0.8)
                ax3.set_xlim(hist["t"][0], hist["t"][-1])
                ax3.set_title("🔴 Learning → 🟢 Crystallized")
                st.pyplot(fig3)
                plt.close()
    else:
        st.info("点击 **运行** 开始仿真")


def render_rar_discovery():
    """RAR 发现：运行 RAR 实验，显示 RAR 图"""
    st.header("RAR 发现")
    st.caption("径向加速度关系 (Radial Acceleration Relation) • SPARC 星系数据")

    n_galaxies = st.sidebar.slider("星系数量", 10, 175, 50)

    if st.button("🔬 运行 RAR 发现", type="primary"):
        with st.spinner("运行 RAR Discovery..."):
            try:
                from axiom_os.experiments.discovery_rar import run_rar_discovery, _plot_rar
                res = run_rar_discovery(n_galaxies=n_galaxies, epochs=400)
                _plot_rar(res, ROOT / "axiom_os" / "discovery_rar_plot.png")
                st.session_state.rar_result = res
                st.success("RAR 发现完成")
                st.rerun()
            except Exception as e:
                st.error(f"运行失败: {e}")
                st.exception(e)

    res = st.session_state.rar_result
    if res and "error" not in res:
        st.subheader("结果")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("R²", f"{res.get('r2', 0):.4f}")
        with c2:
            st.metric("MSE", f"{res.get('mse', 0):.2e}")
        with c3:
            st.metric("样本数", res.get("n_samples", 0))

        cry = res.get("crystallized", {})
        if cry:
            st.markdown("**结晶定律**")
            for i, (name, formula, r2, _) in enumerate(cry.get("best3", [])[:3], 1):
                st.code(f"#{i} {name} (R²={r2:.4f}): {formula}")

        plot_path = ROOT / "axiom_os" / "discovery_rar_plot.png"
        if plot_path.exists():
            st.subheader("RAR 图")
            st.image(str(plot_path), use_container_width=True)
        else:
            st.caption("RAR 图未生成，请运行 discovery_rar.main() 生成")
    else:
        st.info("点击 **运行 RAR 发现** 开始")


def render_experiments():
    """实验：启动各类实验"""
    st.header("实验")
    st.caption("Discovery • Battery • Turbulence • 诊断")

    experiments = [
        ("RAR Discovery", "axiom_os.experiments.discovery_rar", "main", "RAR 径向加速度关系"),
        ("RAR 诊断", "axiom_os.experiments.diagnose_rar_g0", "main", "g0 残差分析"),
        ("Battery RUL", "axiom_os.main_battery", "main", "电池剩余寿命"),
    ]

    for name, mod, func, desc in experiments:
        with st.expander(f"**{name}** — {desc}"):
            if st.button(f"运行 {name}", key=f"run_{name}"):
                with st.spinner(f"运行 {name}..."):
                    try:
                        import importlib
                        m = importlib.import_module(mod)
                        getattr(m, func)()
                        st.success(f"{name} 完成")
                    except Exception as e:
                        st.error(str(e))

    st.divider()
    st.subheader("输出图像")
    img_dir = ROOT / "axiom_os"
    patterns = ["discovery_rar_plot.png", "rar_g0_diagnostic_residuals.png", "battery_rul_plot.png", "paper_rar_trophy_plot.png"]
    for p in patterns:
        fp = img_dir / p
        if fp.exists():
            st.image(str(fp), caption=p, use_container_width=True)


def render_knowledge_base():
    """知识库：Hippocampus 内容"""
    st.header("知识库")
    st.caption("Hippocampus • 结晶定律")

    try:
        from axiom_os.core.hippocampus import Hippocampus
        if st.session_state.hippocampus is None:
            st.session_state.hippocampus = Hippocampus(dim=32, capacity=5000)
        hippo = st.session_state.hippocampus
        kb = hippo.knowledge_base
        if kb:
            for fid, entry in list(kb.items())[:20]:
                st.code(f"{fid}: {entry.get('formula', 'N/A')}")
        else:
            st.info("知识库为空。运行 Discovery 并 Crystallize 后会有内容。")
    except Exception as e:
        st.error(f"加载失败: {e}")


def render_settings():
    """设置"""
    st.header("设置")
    st.caption("Axiom-OS 配置")

    st.markdown("""
    - **主控台**: 双摆仿真、MPC 控制、能量监控
    - **RAR 发现**: SPARC 星系数据，McGaugh 公式拟合
    - **实验**: 各类 Discovery 实验入口
    - **知识库**: Hippocampus 结晶定律

    运行主循环: `python -m axiom_os.main`
    运行 RAR: `python axiom_os/experiments/discovery_rar.py`
    """)


# ============== Route ==============
if page == "主控台":
    render_main_control()
elif page == "RAR 发现":
    render_rar_discovery()
elif page == "实验":
    render_experiments()
elif page == "知识库":
    render_knowledge_base()
else:
    render_settings()
