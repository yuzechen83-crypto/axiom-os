# Axiom-OS (SPNN-Opt-Rev5)

[![PyPI Version](https://img.shields.io/pypi/v/axiom-os)](https://pypi.org/project/axiom-os/)
[![Python Version](https://img.shields.io/pypi/pyversions/axiom-os)](https://pypi.org/project/axiom-os/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/code%20style-Black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/yuzechen83-crypto/axiom-os/actions/workflows/benchmark.yml/badge.svg)](https://github.com/yuzechen83-crypto/axiom-os/actions)
[![Downloads](https://pepy.tech/badge/axiom-os)](https://pepy.tech/project/axiom-os)

> **物理不变尺度锚定 + 规范耦合神经元的集成框架**，实现「物理公理兜底 + 未知边界探索」双重目标。

**核心创新**: RCLN 残差耦合链接神经元 | 物理锚定系统 | 海马体记忆 | 电网 Pulse | 公式发现

---

## 🚀 快速开始

```bash
# 安装（含 UI + API）
pip install axiom-os[full]

# 或开发安装
git clone https://github.com/yuzechen83-crypto/axiom-os.git
cd axiom-os
pip install -e ".[full]"

# OpenClaw 风格统一 CLI
axiom init                    # 初始化 ~/.axiom_os 配置与 workspace
axiom agent -m "跑基准"       # 单轮 agent（DeepSeek + 工具）
axiom agent -m "跑 Grid Pulse MPC"
axiom agent -m "建一个 L 形支架"  # CAD 建模（需 DeepSeek）
axiom demo acrobot [--fast]   # 演示：双摆踢一脚
axiom demo eureka [--fast]    # 演示：顿悟时刻公式发现
axiom demo cad                # 演示：3D 建模过程
axiom api                     # 启动 REST API (:8000)
axiom ui                      # 启动 Streamlit 交互界面
axiom download-elia [--sample] # 下载 Elia 电网数据

# 快捷方式（仍可用）
axiom-ui           # = axiom ui
axiom-api          # = axiom api
axiom-main         # 主循环（双摆控制）
```

---

## 📐 框架定义

```
SPNN = ⟨A, N, C, I, M, O, H, B⟩
```

| 符号 | 组件 | 说明 |
|:----:|------|------|
| 𝒜 | Physical Anchoring | 物理锚定系统 |
| 𝒩 | Neuron Network | RCLN 残差耦合链接神经元 |
| 𝒞 | Coupling Coordinator | 耦合协调器 |
| ℐ | UPI Interface | 统一物理接口 |
| ℳ | Hippocampus | 海马体记忆系统 |
| 𝒪 | Axiom-OS | 主脑调度器 |
| ℋ | HAL | 硬件抽象层 |
| ℬ | Physical Scale | 物理标尺系统 |

---

## 🔬 六阶段认知学习链路

1. **物理锚定** - 普朗克尺度归一化
2. **结构蒸馏编码** - Encoder + 海马体检索增强
3. **主神经元计算** - 物质项传播
4. **RCLN 残差耦合** - F_soft + λ_res·P[F_hard]
5. **自适应筛选** - 主脑协调激活神经元
6. **尺度感知反推** - 标尺恢复输出

---

## ⚙️ 配置（OpenClaw 风格）

配置路径：`~/.axiom_os/axiom_os.json`（首次运行 `axiom init` 自动创建）

```json
{
  "identity": { "name": "Axiom", "theme": "Physics-Aware AI", "emoji": "🔬" },
  "agent": { "workspace": "~/.axiom_os/workspace", "model": { "primary": "deepseek-chat" } },
  "gateway": { "port": 8000, "host": "0.0.0.0" }
}
```

Workspace 结构：`~/.axiom_os/workspace/` 含 `IDENTITY.md`、`SOUL.md`、`PROJECTS.md`、`ENGINE_DESIGN.md`、`CHIP_DEV.md`。Agent 启动时自动注入 IDENTITY+PROJECTS 到 system prompt，实现「先学习再使用」。

---

## 🌐 交互与服务

### 前端 UI（Streamlit）
```bash
axiom-ui
# 或 streamlit run axiom_os/agent/chat_ui.py
```
- 自然语言输入：跑基准、跑 RAR、跑 Grid Pulse MPC、CAD 建模等
- 侧边栏 CAD 快捷入口：选择形状一键生成 STL
- DeepSeek 扩展：需配置 `axiom_os/config/axiom_os_llm.env` 中 `DEEPSEEK_API_KEY`

### REST API
```bash
axiom-api
# 或 uvicorn axiom_os.api.server:app --host 0.0.0.0 --port 8000
```
- `GET /` - 服务信息
- `POST /api/v1/grid-mpc` - Grid Pulse MPC
- `POST /api/v1/chat` - 自然语言对话
- `POST /api/v1/cad-model` - CAD 建模（shape, width, height, depth, radius）
- `GET /api/v1/cad-shapes` - 列出 CAD 形状
- `POST /api/v1/workspace/read` - 读取 workspace 文档
- `GET /api/v1/workspace/docs` - 列出 workspace 文档
- `POST /api/v1/hippocampus/retrieve` - 检索 Hippocampus 物理定律
- `POST /api/v1/fetch-url` - 抓取 URL 正文（可选保存到 workspace）
- `POST /api/v1/web-search` - 网页搜索
- `GET /api/v1/benchmark/report` - 基准报告
- `POST /api/v1/benchmark/run` - 运行基准
- `POST /api/v1/rar` - RAR 星系发现
- API 文档：`http://localhost:8000/docs`

---

## 📊 基准测试

```bash
# 快速验证 (~30s)
python -m axiom_os.benchmarks.run_benchmarks --config quick --report --trend

# 标准配置 (~1min)
python -m axiom_os.benchmarks.run_benchmarks --config standard --report

# 完整配置
python -m axiom_os.benchmarks.run_benchmarks --config full --report --trend -o full_report.json
```

---

## 🎯 主要功能

### 物理AI核心
- **RCLN 神经元**: 残差耦合链接，物理映射
- **物理锚定**: 普朗克→工程尺度统一
- **UPI 接口**: 统一物理契约验证

### 电网分析 (Grid Pulse)
```bash
python main_grid.py
```

### 湍流建模 (JHTDB)
```bash
python -m axiom_os.experiments.jhtdb_les_sgs
```

### 公式发现 (Discovery)
```bash
python -m axiom_os.demos.discovery_demo
python -m axiom_os.experiments.crystallize_formulas --rar-only
```

### Agent / Chat UI

Axiom-OS 提供交互式聊天界面，支持 DeepSeek LLM 扩展功能。

#### 配置 DeepSeek API

1. **获取 API 密钥**：访问 [DeepSeek Platform](https://platform.deepseek.com) 注册并获取 API 密钥

2. **设置环境变量**：
   ```bash
   # Linux/Mac
   export DEEPSEEK_API_KEY=your_api_key_here
   
   # Windows (CMD)
   set DEEPSEEK_API_KEY=your_api_key_here
   
   # Windows (PowerShell)
   $env:DEEPSEEK_API_KEY="your_api_key_here"
   ```

3. **运行 Chat UI**：
   ```bash
   streamlit run axiom_os/agent/chat_ui.py
   ```

#### DeepSeek 扩展功能

启用 DeepSeek 扩展后，可以通过自然语言：
- 运行基准测试和获取报告
- 执行 RAR 星系旋转曲线发现
- 运行 Discovery 演示
- 列出和应用领域扩展
- 获取优化建议

**API 配置**：
- 端点：`https://api.deepseek.com`
- 默认模型：`deepseek-chat`
- 接口格式：OpenAI 兼容 API

---

## 📁 项目结构

```
axiom_os/
├── core/           # 物理标尺 B、UPI 接口
├── hal/            # 硬件抽象层 ℋ
├── layers/         # RCLN, FNO, TBNN, PINN-LSTM
├── neurons/        # RCLN (F_hard, F_soft, 物理映射)
├── memory/         # 海马体 M_H
├── orchestrator/   # Axiom-OS 主脑
├── datasets/       # 数据加载器
├── experiments/    # 实验脚本
├── benchmarks/     # 基准测试
├── agent/          # Agent / Chat UI
└── tests/          # 测试套件
```

---

## 🔧 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 代码格式化
black .
ruff check --fix .

# 类型检查
mypy axiom_os
```

---

## 📝 许可证

MIT License - 详见 [LICENSE](LICENSE)

---

## 🙏 致谢

- PyTorch, NumPy, SciPy
- JHTDB (Johns Hopkins Turbulence Database)
- 物理AI研究社区

---

> **核心技术声明**: 详见 [PROPRIETARY_COMPONENTS.md](PROPRIETARY_COMPONENTS.md)
