"""
Axiom-Agent: Streamlit Chat UI
User -> LLM generates -> Axiom runs -> Agent interprets result.

用法: streamlit run axiom_os/agent/chat_ui.py
或:   python -m axiom_os.agent.chat --ui
"""

import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# 若用 python 直接运行，自动改用 streamlit run 启动（避免 ScriptRunContext 警告）
if __name__ == "__main__" and os.environ.get("AXIOM_STREAMLIT_RUN") != "1":
    os.environ["AXIOM_STREAMLIT_RUN"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())] + sys.argv[1:],
        cwd=str(ROOT),
        env=os.environ.copy(),
    )
    sys.exit(result.returncode)

import json
import streamlit as st

st.set_page_config(page_title="Axiom-Agent Chat", page_icon="🔬", layout="centered")
st.title("🔬 Axiom-Agent: Text-to-Physics")
st.caption("输入物理问题，AI 生成代码并运行 Axiom 仿真 | 或选择多领域学习 (MLL)")

# 对话持久化：JSON 文件
CHAT_HISTORY_PATH = ROOT / "agent_output" / "chat_history.json"


def load_chat_history():
    """从文件加载对话历史"""
    if CHAT_HISTORY_PATH.exists():
        try:
            with open(CHAT_HISTORY_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_chat_history(messages):
    """保存对话历史到文件"""
    CHAT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHAT_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)


# 模式选择：Chat 或 MLL
mode = st.sidebar.radio("模式", ["Chat (Text-to-Physics)", "MLL (多领域学习)"], index=0)

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# 清空历史按钮
if st.sidebar.button("清空对话历史"):
    st.session_state.messages = []
    save_chat_history([])
    st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# MLL 模式：多领域学习入口
if mode == "MLL (多领域学习)":
    st.subheader("多领域学习 (MLL)")
    domains = st.multiselect(
        "选择领域",
        ["rar", "battery", "turbulence"],
        default=["rar", "battery"],
        help="RAR=星系旋转, Battery=电池寿命, Turbulence=湍流",
    )
    epochs = st.slider("每领域 Epochs", 100, 1000, 200)
    do_crystallize = st.checkbox("结晶到 Hippocampus", value=False)
    if st.button("运行 MLL"):
        with st.spinner("MLL 训练中..."):
            try:
                from axiom_os.core import Hippocampus
                from axiom_os.mll import MLLOrchestrator
                from axiom_os.mll.domain_protocols import PROTOCOL_REGISTRY
                from axiom_os.mll.orchestrator import CouplingConfig

                hippocampus = Hippocampus(dim=64, capacity=5000)
                protocols = {k: v for k, v in PROTOCOL_REGISTRY.items() if k in domains}
                config = CouplingConfig(order=domains, hippocampus_shared=True)
                orchestrator = MLLOrchestrator(protocols=protocols, hippocampus=hippocampus, config=config)
                results = orchestrator.run_all(
                    epochs_per_domain={d: epochs for d in protocols},
                    do_discover=True,
                    do_crystallize=do_crystallize,
                )
                ok_count = sum(1 for r in results.values() if r.ok)
                summary = "\n".join(
                    f"- **{k}**: {'✅' if r.ok else '❌'} " + (", ".join(f"{mk}={mv:.4f}" for mk, mv in (r.metrics or {}).items()) or "—")
                    for k, r in results.items()
                )
                st.success(f"MLL 完成: {ok_count}/{len(results)} 通过")
                st.markdown(summary)
            except Exception as e:
                st.error(f"MLL 异常: {e}")
    st.stop()

if prompt := st.chat_input("输入物理问题，如：双摆带摩擦平衡..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("生成中..."):
            try:
                import os
                from axiom_os.agent.coder import AxiomCoder
                from axiom_os.agent.runner import AxiomRunner

                api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    try:
                        from run_agent import _mock_generate
                    except ImportError:
                        def _mock_generate(_):
                            return {
                                "physics.py": "# 无 API Key，使用模板",
                                "config.yaml": "state_dim: 2",
                                "objective.py": "# mock",
                            }
                    artifacts = _mock_generate(prompt)
                else:
                    coder = AxiomCoder(api_key=api_key, provider="openai")
                    artifacts = coder.generate(prompt)

                output_dir = ROOT / "agent_output"
                output_dir.mkdir(parents=True, exist_ok=True)
                for name, content in artifacts.items():
                    (output_dir / name).write_text(content, encoding="utf-8")

                runner = AxiomRunner(output_dir=output_dir)
                stdout, result, err = runner.run(
                    artifacts["physics.py"],
                    artifacts["config.yaml"],
                    artifacts["objective.py"],
                    max_steps=50,
                )

                if err:
                    reply = f"❌ 错误: {err}"
                else:
                    discovered = result.get("discovered", [])
                    if discovered:
                        reply = f"✅ 仿真完成。发现公式: {discovered[0]}"
                    else:
                        reply = "✅ 仿真完成，未发现新物理公式。"
            except Exception as e:
                reply = f"❌ 异常: {e}"

        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    save_chat_history(st.session_state.messages)
