# SPNN-Opt-Rev5 (Axiom-OS)

物理不变尺度锚定 + 规范耦合神经元的集成框架，实现「物理公理兜底 + 未知边界探索」双重目标。

> **核心技术保护**：部分算法为专有技术，详见 [PROPRIETARY_COMPONENTS.md](PROPRIETARY_COMPONENTS.md)。

## 框架定义

```
SPNN = ⟨A, N, C, I, M, O, H, B⟩
```

| 符号 | 组件 | 说明 |
|------|------|------|
| 𝒜 | Physical Anchoring | 物理锚定系统 |
| 𝒩 | Neuron Network | RCLN 残差耦合链接神经元 |
| 𝒞 | Coupling Coordinator | 耦合协调器 |
| ℐ | UPI Interface | 统一物理接口 |
| ℳ | Hippocampus | 海马体记忆系统 |
| 𝒪 | Axiom-OS | 主脑调度器 |
| ℋ | HAL | 硬件抽象层 |
| ℬ | Physical Scale | 物理标尺系统 |

## 六阶段认知学习链路

1. **物理锚定** - 普朗克尺度归一化
2. **结构蒸馏编码** - Encoder + 海马体检索增强
3. **主神经元计算** - 物质项传播
4. **RCLN 残差耦合** - F_soft + λ_res·P[F_hard]
5. **自适应筛选** - 主脑协调激活神经元
6. **尺度感知反推** - 标尺恢复输出

## 快速开始

```bash
git clone <your-repo-url>
cd PythonProject1
pip install -r requirements.txt
python run_spnn_example.py
```

**可运行性**：无 `axiom_core_proprietary` 时，基准与公式发现 Demo 使用内置多项式回退，仍可完整运行。

## 基准测试 (Benchmark)

```bash
# 快速验证 (~30s)
python -m axiom_os.benchmarks.run_benchmarks --config quick --report --trend

# 标准配置 (~1min，含 E2E Quick + Memory + Discovery 鲁棒性)
python -m axiom_os.benchmarks.run_benchmarks --config standard --report

# 完整配置 (含全 E2E)
python -m axiom_os.benchmarks.run_benchmarks --config full --report --trend -o full_report.json

# 可选：与 PySR/SINDy 对比（需 pip install pysr pysindy）
python -m axiom_os.benchmarks.run_benchmarks --config quick --compare-pysr --report

# 高难度 Discovery 套件（稀疏/外推/小样本/Feynman/Lorenz 混沌）
python -m axiom_os.benchmarks.run_benchmarks --hard --report
```

报告输出：`axiom_os/benchmarks/results/`（JSON、趋势图、Markdown、HTML、**Discovery vs Baseline 对比图**）

报告含**亮点摘要**：RCLN 吞吐、Discovery R² 提升 vs 线性回归、RAR 符号发现 R²。

**发布到 GitHub**：push 到 `main` 时 CI 会自动跑基准并发布到 GitHub Pages。在仓库 **Settings → Pages** 中来源选 **GitHub Actions**，即可在 `https://<org>.github.io/<repo>/` 查看最新报告（`report.html`、`latest.json`）。

## 公式发现 Demo

```bash
python -m axiom_os.demos.discovery_demo
```

从带噪数据 `y = x0² + 0.5*x1` 恢复符号公式，对比 Axiom Discovery 与线性回归的 R²。

## Agent 与 MLL

```bash
# Chat UI：文本→物理仿真
streamlit run axiom_os/agent/chat_ui.py

# 多领域学习 (MLL)
python run_mll.py --domains rar battery turbulence --epochs 200
```

Chat 侧边栏可选 **MLL 模式**，一键运行 rar/battery/turbulence 多领域训练。

## Dashboard

```bash
streamlit run axiom_os/dashboard.py
```

页签：**组件验证**（阻尼振子、RCLN、Discovery）| **基准报告**（历史 JSON、趋势图）

## 全流程验证

```bash
python -m axiom_os.validate_all
```

## 项目结构

```
spnn/
├── core/           # 物理标尺 B、UPI 接口
├── hal/            # 硬件抽象层 ℋ
├── neurons/        # RCLN (F_hard, F_soft, 物理映射)
├── memory/         # 海马体 M_H
├── orchestrator/   # Axiom-OS 主脑
├── report/         # 可解释报告
├── training/       # 多目标损失、三阶段训练
├── safety/         # 双路径验证、SSC
├── distributed/    # 物理 MoE、光锥协调
├── model.py        # SPNN 完整模型
└── physical_ai.py  # PhysicalAI 完整工作流
```

## 系统工作流 (PhysicalAI)

```
输入物理问题
    ↓
[物理标尺系统] → 尺度归一化 → 量纲检查
    ↓
[海马体] → 检索类似问题经验
    ↓
[主脑] → 制定解决协议链
    ↓
For each protocol step:
    ├─ [UPI接口] → 契约验证
    ├─ [物理标尺] → 尺度适配
    ├─ [RCLN细胞] → 计算耦合
    ├─ [海马体] → 记忆检索/存储
    └─ [主脑] → 动态路由决策
    ↓
[物理标尺系统] → 反归一化输出
    ↓
[双路径验证] → 异常检测与处理
    ↓
输出物理解 + 置信度 + 可解释报告
```

```python
from spnn import PhysicalAI

ai = PhysicalAI(in_dim=4, hidden_dim=32, out_dim=1)
res = ai.solve(x, l_e="harmonic_oscillator")
print(res.result, res.confidence, res.report)
```

## 主要创新

- **可微公理势阱**：物理约束架构级引力场
- **弹性守恒协议**：可协商网络共识
- **光锥时间协调**：相对论因果结构
- **结构化海马体**：物理记忆主动维护
- **智能主脑调度**：资源-物理-风险联合优化
- **物理标尺系统**：普朗克→工程尺度统一
- **双路径验证**：安全与探索平衡
