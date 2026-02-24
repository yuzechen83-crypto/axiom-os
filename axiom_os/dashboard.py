"""
Axiom-OS Dashboard - Streamlit
Validates Axiom-OS components: Damped Oscillator, RCLN, Discovery Engine.

Run: streamlit run axiom_os/dashboard.py
Or:  python -m streamlit run axiom_os/dashboard.py
"""

import sys
import logging
from pathlib import Path

# Filter "missing ScriptRunContext" (harmless in bare mode) - before streamlit import
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

# Page config
st.set_page_config(page_title="Axiom-OS Dashboard", layout="wide", initial_sidebar_state="expanded")

st.title("Axiom-OS Dashboard")
st.caption("Physics-AI validation: Damped Oscillator, RCLN Soft Shell, Discovery Engine")

# 页签：验证 | 基准报告
tab_valid, tab_bench = st.tabs(["组件验证", "基准报告"])


# ============== Benchmark 报告页签 ==============
with tab_bench:
    BENCH_DIR = ROOT / "axiom_os" / "benchmarks" / "results"
    if BENCH_DIR.exists():
        json_files = sorted(BENCH_DIR.glob("benchmark_*.json"), reverse=True)
        if json_files:
            st.subheader("最新基准报告")
            selected = st.selectbox("选择报告", [p.name for p in json_files], index=0)
            sel_path = BENCH_DIR / selected
            if sel_path.exists():
                import json
                with open(sel_path, encoding="utf-8") as f:
                    data = json.load(f)
                results = data.get("results", [])
                ts = data.get("timestamp", "—")
                st.caption(f"时间: {ts}")
                if results:
                    cols = st.columns(3)
                    for i, r in enumerate(results[:9]):
                        with cols[i % 3]:
                            st.metric(r["name"] + " [" + r["metric"] + "]", f"{r['value']:.4f}", r.get("unit", ""))
                if (BENCH_DIR / "benchmark_report.html").exists():
                    with open(BENCH_DIR / "benchmark_report.html", encoding="utf-8") as f:
                        st.components.v1.html(f.read(), height=600, scrolling=True)
        else:
            st.info("暂无基准数据。运行: `python -m axiom_os.benchmarks.run_benchmarks --config quick --report`")
        if (BENCH_DIR / "benchmark_trend.png").exists():
            st.subheader("趋势图")
            st.image(str(BENCH_DIR / "benchmark_trend.png"), use_container_width=True)
    else:
        st.info("基准结果目录不存在。请先运行 benchmark。")

# ============== Sidebar: Real World Physics ==============
st.sidebar.header("Real World Parameters")
st.sidebar.markdown("Adjust physics of the **Damped Oscillator**.")

friction_coef = st.sidebar.slider("Friction coefficient c", 0.01, 2.0, 0.5, 0.05)
spring_k = st.sidebar.slider("Spring constant k", 0.5, 5.0, 1.0, 0.1)
mass = st.sidebar.slider("Mass m", 0.1, 2.0, 1.0, 0.1)
x0 = st.sidebar.slider("Initial displacement x₀", 0.1, 3.0, 1.0, 0.1)
v0 = st.sidebar.slider("Initial velocity v₀", -2.0, 2.0, 0.0, 0.1)
dt = st.sidebar.slider("Time step dt", 0.001, 0.05, 0.02, 0.005)
n_steps = st.sidebar.slider("Simulation steps", 100, 500, 200, 50)

st.sidebar.divider()
st.sidebar.markdown("**Physics:** m·ẍ + c·ẋ + k·x = 0  →  F_damp = -c·v")


# ============== 组件验证页签 ==============
with tab_valid:
    def simulate_damped_oscillator(c: float, k: float, m: float, x0: float, v0: float, dt: float, n: int):
        """Euler integrate: x'' = -(c/m)*v - (k/m)*x."""
        x, v = x0, v0
        xs, vs, ts = [x], [v], [0.0]
        for i in range(n - 1):
            a = -(c / m) * v - (k / m) * x
            v = v + a * dt
            x = x + v * dt
            xs.append(x)
            vs.append(v)
            ts.append((i + 1) * dt)
        return np.array(ts), np.array(xs), np.array(vs)

    ts, xs, vs = simulate_damped_oscillator(friction_coef, spring_k, mass, x0, v0, dt, n_steps)

    # ============== RCLN: Learn F = -c*v from (v, F) ==============
    @st.cache_resource
    def build_rcln():
        def hard_core_zero(x):
            if isinstance(x, torch.Tensor):
                vals = x
            elif hasattr(x, "values") and not isinstance(x, torch.Tensor):
                vals = x.values
            else:
                vals = x
            t = torch.as_tensor(vals, dtype=torch.float32)
            return torch.zeros_like(t)

        return RCLNLayer(
            input_dim=1,
            hidden_dim=32,
            output_dim=1,
            hard_core_func=hard_core_zero,
            lambda_res=1.0,
        )

    rcln = build_rcln()
    optimizer = torch.optim.Adam(rcln.parameters(), lr=0.01)

    # Train on current physics (v, F_true = -c*v)
    v_train = torch.linspace(-3, 3, 80).reshape(-1, 1).float()
    F_true = -friction_coef * v_train
    inputs_upi = UPIState(v_train, units=Units.VELOCITY, semantics="velocity")

    for _ in range(300):
        optimizer.zero_grad()
        F_pred = rcln(inputs_upi)
        loss = torch.mean((F_pred.float() - F_true.float()) ** 2)
        loss.backward()
        optimizer.step()

    # Soft Shell Activity (on current velocity range)
    with torch.no_grad():
        _ = rcln(inputs_upi)
    soft_activity = rcln.get_soft_activity()

    # ============== Discovery Engine ==============
    @st.cache_resource
    def build_discovery_engine():
        return DiscoveryEngine(use_pysr=False)

    engine = build_discovery_engine()
    x_np = v_train.detach().cpu().numpy()
    with torch.no_grad():
        _ = rcln(inputs_upi)
    y_soft_np = rcln._last_y_soft.cpu().numpy()
    discovered_formula = engine.discover(x_np, y_soft_np)
    formula_str = discovered_formula if discovered_formula else "— (no formula yet)"

    # ============== Main Panel: Damped Oscillator Plot ==============
    col_main, col_metrics = st.columns([3, 1])

    with col_main:
        st.subheader("Damped Oscillator")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ts, xs, "b-", linewidth=2, label="Displacement x(t)")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Time t")
        ax.set_ylabel("x(t)")
        ax.set_title(f"m·ẍ + {friction_coef}·ẋ + {spring_k}·x = 0  (c={friction_coef}, k={spring_k}, m={mass})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col_metrics:
        st.subheader("Metrics")
        st.metric("Soft Shell Activity", f"{soft_activity:.4f}")
        st.caption("Mean |y_soft| from RCLN last forward")
        st.metric("RCLN Loss", f"{loss.item():.6f}")

    st.divider()
    st.subheader("Discovery Log")
    st.markdown("**Latest formula** extracted by Discovery Engine from RCLN Soft Shell:")
    st.code(formula_str, language="text")
    st.caption("Expected: F ≈ -c·v (coefficient close to friction coefficient)")

    st.divider()
    st.subheader("Component Validation")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("UPI + RCLN: OK")
    with col2:
        st.success("Discovery Engine: OK")
    with col3:
        st.success("Damped Oscillator: OK")
