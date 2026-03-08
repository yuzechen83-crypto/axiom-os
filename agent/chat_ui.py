"""
Axiom-Agent: Streamlit Chat UI - 交互式物理仿真与 AI 扩展界面

本模块提供 Axiom-OS 的 Streamlit 聊天界面，支持两种运行模式：

1. Chat 模式（Text-to-Physics）：
   - 本地意图识别：解析用户输入，调度 Axiom 模块执行物理仿真
   - DeepSeek 扩展：接入 DeepSeek LLM，通过工具调用运行基准、RAR、Discovery 等
   
2. MLL 模式（多领域学习）：
   - 选择多个领域（RAR、Battery、Turbulence）进行联合训练
   - 支持结晶到 Hippocampus 共享知识库

DeepSeek 扩展功能：
- 运行基准测试和获取报告
- 执行 RAR 星系旋转曲线发现
- 运行 Discovery 演示
- 列出和应用领域扩展
- 生成优化建议和扩展代码

技术实现：
- UI 框架：Streamlit
- LLM 集成：DeepSeek API（通过 axiom_os.agent.deepseek_agent）
- 对话持久化：JSON 文件存储（agent_output/chat_history.json）
- 环境变量：DEEPSEEK_API_KEY（DeepSeek 扩展模式）

使用方法：
    # 方法 1：使用 Streamlit 直接运行
    streamlit run axiom_os/agent/chat_ui.py
    
    # 方法 2：使用 Python 模块运行（自动调用 Streamlit）
    python -m axiom_os.agent.chat --ui
    
    # 方法 3：直接运行脚本（自动重启为 Streamlit）
    python axiom_os/agent/chat_ui.py

配置要求：
- DeepSeek 扩展：设置环境变量 DEEPSEEK_API_KEY
- 本地意图模式：设置 OPENAI_API_KEY 或 ANTHROPIC_API_KEY（可选）

界面功能：
- 侧边栏：模式选择（Chat/MLL）、DeepSeek 扩展开关、清空历史
- 主界面：对话历史显示、用户输入框、AI 回复展示
- MLL 模式：领域选择、训练参数配置、结晶选项

注意事项：
- 对话历史自动保存到 agent_output/chat_history.json
- DeepSeek 扩展需要有效的 API 密钥
- MLL 模式需要相应的领域协议已注册
"""

import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# 加载 LLM 配置（axiom-os 二）
_env_path = ROOT / "axiom_os" / "config" / "axiom_os_llm.env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except ImportError:
        pass

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

# CAD 快捷入口（集成到系统）
with st.sidebar.expander("CAD 建模", expanded=False):
    cad_shape = st.selectbox("形状", ["l_bracket", "box", "cylinder", "sphere", "simple_gear"], key="cad_shape")
    if st.button("生成 STL", key="cad_btn"):
        try:
            from axiom_os.agent.tools import run_cad_model
            r = run_cad_model(shape=cad_shape)
            if r.get("ok"):
                st.success(f"已保存: {r.get('path', '')}")
                st.caption(r.get("message", ""))
            else:
                st.error(r.get("error", "失败"))
        except Exception as e:
            st.error(str(e))

# Chat 引擎：本地意图 或 DeepSeek 扩展（大模型收集数据/扩展优化 Axiom）
use_deepseek = st.sidebar.checkbox(
    "使用 DeepSeek 扩展",
    value=False,
    help="接入 DeepSeek，可运行基准、RAR、Discovery 等工具并扩展/优化 Axiom。需设置 DEEPSEEK_API_KEY。",
)

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
        ["rar", "battery", "turbulence", "engine", "chip"],
        default=["rar", "battery"],
        help="RAR=星系旋转, Battery=电池寿命, Turbulence=湍流, Engine=发动机(占位), Chip=芯片(占位)",
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

if prompt := st.chat_input("输入物理问题，如：跑一下 RAR、跑基准、双摆平衡..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("理解意图并执行..." if not use_deepseek else "DeepSeek 扩展运行中..."):
            try:
                # DeepSeek 扩展模式：大模型调用工具收集数据/扩展优化
                if use_deepseek:
                    from axiom_os.agent.deepseek_agent import run_agent_loop
                    
                    # 读取 DeepSeek API 密钥
                    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
                    
                    # 调用 DeepSeek Agent（密钥验证在 agent 内部处理）
                    reply = run_agent_loop(
                        user_message=prompt,
                        api_key=deepseek_api_key,
                        max_tool_rounds=5
                    )
                    if not reply:
                        reply = "（无回复）"
                else:
                    from axiom_os.agent.llm_layer import AxiomLLMLayer, INTENT_MAIN_SIM
                    from axiom_os.agent.dispatcher import run_intent

                    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
                    llm = AxiomLLMLayer(api_key=api_key, provider="openai")
                    plan = llm.plan(prompt)
                    intent, params = plan["intent"], plan.get("params") or {}

                    # 双摆/控制仿真：走原有 Coder + Runner 流程
                    if intent == INTENT_MAIN_SIM:
                        from axiom_os.agent.coder import AxiomCoder
                        from axiom_os.agent.runner import AxiomRunner
                        if not api_key:
                            try:
                                from run_agent import _mock_generate
                            except ImportError:
                                def _mock_generate(_):
                                    return {"physics.py": "# 无 API Key", "config.yaml": "state_dim: 2", "objective.py": "# mock"}
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
                            artifacts["physics.py"], artifacts["config.yaml"], artifacts["objective.py"], max_steps=50,
                        )
                        reply = llm.format_reply(intent, result if not err else {}, error=err)
                    else:
                        # 其他意图：调度器直接跑 Axiom 模块
                        out = run_intent(intent, params)
                        result = out.get("result") or {}
                        err = out.get("error")
                        reply = llm.format_reply(intent, result, error=err)
                        if out.get("elapsed"):
                            reply += f" （耗时 {out['elapsed']:.1f}s）"
            except Exception as e:
                reply = f"❌ 异常: {e}"

        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    save_chat_history(st.session_state.messages)


def run_ui():
    """Entry point for axiom-ui console script. Launches Streamlit chat UI."""
    from pathlib import Path
    import subprocess
    import sys
    ui_path = Path(__file__).resolve()
    result = subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(ui_path), "--server.headless", "true"],
        cwd=str(ui_path.parents[2]),
        env=os.environ.copy(),
    )
    sys.exit(result.returncode)
