# Axiom 生态网设计：开放神经元与 UPI 协议

> 愿景：所有人都可以设计神经元，通过 UPI 协议接入系统，参与到系统的自进化和物理发现中来。

---

## 0. SPNN-Evo 整合 + 智能化升级

### SPNN-Evo 整合

| 模块 | 位置 | 作用 |
|------|------|------|
| **Policy Distillation** | `orchestrator/distillation.py` | MPC→Student 蒸馏，Boot/Sleep/Run，DAgger，异常回退 |
| **ActivityMonitor** | `layers/rcln.py` | 滑动窗口 \|F_soft\| → DiscoveryHotspot |
| **Theory Coupling** | `core/imagination.py` | couple_theories, simulate_coupled (H_new = H_s + H_p + ε·H_int) |
| **PDE 特征库** | `core/features.py`, `core/features_2d.py` | build_pde_library, vorticity_from_velocity, weak_form_patch_average |

### 智能化升级（通用 AI 标准）

| 模块 | 位置 | 作用 |
|------|------|------|
| **ChiefScientist** | `orchestrator/llm_brain.py` | Ollama/OpenAI，max_tool_calls 熔断，RAG（Hippocampus 检索），retry，run_rar_discovery |
| **AIIntelligence** | `orchestrator/ai_orchestrator.py` | 活动感知 + 自动发现建议 + ChiefScientist 编排 |
| **BaseNeuron + NeuronRegistry** | `neurons/` | 按物理域（mechanics, fluids, control）划分的可插拔神经元 |
| **Discovery 验证闭环** | `engine/discovery.py` | validate_formula()，验证通过才结晶 |
| **uncertainty_gate** | `layers/rcln.py` | get_uncertainty_status()，return_uncertainty=True 时标注不可靠预测 |
| **Hippocampus** | `core/hippocampus.py` | retrieve_by_query() 语义/关键词 RAG |

**AI 智能化特性**：
- 活动感知提示：soft_activity > 0.05 时自动注入「建议 run_discovery」提示
- write_hippocampus 解析：从 LLM 响应中解析 formula 并执行结晶
- run_rar_discovery 工具：RAR/Meta-Axis discovery（SPARC 星系，McGaugh 拟合）
- Mission Control：侧边栏「Chief Scientist 对话」可向 AI 提问

使用 Policy 模式：`main(use_distillation=True)` 或调用 `boot_phase()` → `sleep_phase()` → `run_phase()`。

使用 Ollama：`ChiefScientist(backend="ollama", model="llama3.2")`，需先 `ollama pull llama3.2`。

### 超级智能化（T1-T2）

| 模块 | 位置 | 作用 |
|------|------|------|
| **Task / Goal** | `core/task.py` | 目标树、子任务、依赖、状态 |
| **TaskDecomposer** | `orchestrator/task_decomposer.py` | 自然语言目标 → 子任务 DAG |
| **TaskExecutor** | `orchestrator/task_executor.py` | 按依赖执行 Discovery/结晶/读库 |
| **SymplecticCausal** | `core/symplectic_causal.py` | 辛结构因果约束（dq/dt 依赖 p，dp/dt 依赖 q,p） |
| **LightConeFilter** | `core/light_cone_filter.py` | 光锥时空因果过滤 |
| **run_goal** | ChiefScientist 工具 | 目标驱动：分解并执行 |

**因果约束**：Discovery.distill(use_symplectic_causal=True, state_dim=4) 可启用辛因果约束。

**Mission Control**：侧边栏「输入目标」→「执行目标」可运行目标驱动流程。

---

## 1. 核心概念

### 1.1 生态参与者

| 角色 | 职责 |
|------|------|
| **神经元设计者** | 实现符合 UPI 的神经元，贡献物理先验或数据驱动能力 |
| **系统** | 提供 UPI 协议、发现引擎、结晶机制、知识库 |
| **贡献者** | 神经元产生的 soft activity 参与发现，结晶后的定律回馈社区 |

### 1.2 数据流

```
[环境/数据] → UPIState → [神经元网络] → UPIState → [下游]
                              ↓
                    soft_activity → Discovery → 结晶 → Hippocampus
```

---

## 2. UPI 协议规范（扩展版）

### 2.1 当前 UPI 能力

- **单位向量** `[M, L, T, Q, Θ]`：质量、长度、时间、电荷、温度
- **UPIState**：values + units + spacetime + semantics
- **量纲校验**：加减同量纲、乘除自动推导
- **因果性**：`assert_causality` 光锥约束

### 2.2 生态扩展：UPI Protocol v1

为支持第三方神经元接入，需定义：

#### 2.2.1 输入/输出契约

```python
# 神经元必须声明
input_units: List[List[int]]   # 每个输入通道的 [M,L,T,Q,Θ]
output_units: List[List[int]]  # 每个输出通道的 [M,L,T,Q,Θ]
input_semantics: List[str]     # 可选，如 ["q1","q2","p1","p2"]
output_semantics: List[str]    # 可选
```

#### 2.2.2 接口规范

```python
class UPICompatible(Protocol):
    """第三方神经元必须实现的协议"""
    
    def forward(self, x: UPIState) -> UPIState: ...
    
    def get_soft_activity(self) -> float:
        """用于发现触发：soft 部分活跃度"""
        ...
    
    @property
    def input_units(self) -> List[List[int]]: ...
    
    @property
    def output_units(self) -> List[List[int]]: ...
```

#### 2.2.3 量纲链校验

系统在接入时校验：上游 output_units 与下游 input_units 是否兼容（可广播或显式匹配）。

---

## 3. 神经元接口设计

### 3.1 BaseNeuron 抽象基类

```python
from abc import ABC, abstractmethod
from typing import List, Optional
import torch

class BaseNeuron(ABC, nn.Module):
    """
    生态网神经元基类。
    设计者继承此类，实现 forward，声明 input/output units。
    """
    
    # 子类必须定义
    INPUT_UNITS: List[List[int]]   # 每个输入维度的 [M,L,T,Q,Θ]
    OUTPUT_UNITS: List[List[int]]  # 每个输出维度的 [M,L,T,Q,Θ]
    
    @abstractmethod
    def forward(self, x: UPIState) -> UPIState:
        """输入输出均为 UPIState，保证量纲正确"""
        pass
    
    def get_soft_activity(self) -> float:
        """默认 0；若有 soft 部分，返回其活跃度"""
        return 0.0
    
    def get_contribution_buffer(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        可选：贡献 (X, y_soft) 给 Discovery。
        返回 None 表示不参与发现。
        """
        return None
```

### 3.2 神经元类型

| 类型 | 说明 | 参与发现 |
|------|------|----------|
| **HardOnly** | 纯物理公式，无 soft | 否 |
| **RCLN** | 物理 + 残差 MLP | 是（y_soft） |
| **Custom** | 用户自定义混合 | 可选 |

### 3.3 示例：社区贡献的湍流神经元

```python
class TurbulenceResidualNeuron(BaseNeuron):
    INPUT_UNITS = [[0,1,-1,0,0], [0,1,-1,0,0], [0,0,0,0,1]]  # u,v,T
    OUTPUT_UNITS = [[0,1,-2,0,0]]  # 加速度量纲
    
    def __init__(self):
        super().__init__()
        self.soft = nn.Sequential(nn.Linear(3,32), nn.SiLU(), nn.Linear(32,1))
        self._last_soft = None
    
    def forward(self, x: UPIState) -> UPIState:
        v = x.values
        y_soft = self.soft(v.float())
        self._last_soft = y_soft.detach()
        # 假设 hard_core 为 0，或调用已知物理
        return UPIState(y_soft, units=self.OUTPUT_UNITS[0], semantics="a_res")
    
    def get_soft_activity(self) -> float:
        return float(self._last_soft.abs().mean()) if self._last_soft is not None else 0.0
```

---

## 4. 神经元注册与发现

### 4.1 NeuronRegistry

```python
class NeuronRegistry:
    """生态神经元注册表"""
    
    def register(
        self,
        neuron_id: str,
        neuron_class: Type[BaseNeuron],
        author: str,
        domain: str,           # "battery" | "turbulence" | "control" | ...
        description: str = "",
        version: str = "0.1.0",
    ) -> None: ...
    
    def get(self, neuron_id: str, version: Optional[str] = None) -> BaseNeuron: ...
    
    def list_by_domain(self, domain: str) -> List[Dict]: ...
    
    def validate(self, neuron: BaseNeuron) -> bool:
        """校验 UPI 兼容性、量纲声明正确性"""
        ...
```

### 4.2 发现参与流程

```
1. 系统定期收集所有已接入神经元的 get_contribution_buffer()
2. 合并 (X, y_soft) 到 DiscoveryEngine
3. DiscoveryEngine 用 Buckingham Pi + 符号回归提取公式
4. 通过验证的公式 → Hippocampus.crystallize()
5. 结晶结果可选：回馈给原神经元作者、发布到知识库
```

### 4.3 结晶回馈

- **本地结晶**：更新该任务/域下的 RCLN hard_core
- **全局知识库**：Hippocampus 存储，可被其他神经元引用
- **社区公示**：公式 + 贡献者 ID，形成可追溯的物理发现链

---

## 5. 接入流程（SDK）

### 5.1 开发者接入步骤

1. **安装**：`pip install axiom-os`（未来）
2. **实现**：继承 `BaseNeuron`，声明 `INPUT_UNITS` / `OUTPUT_UNITS`
3. **测试**：`axiom validate my_neuron.py` 本地校验
4. **注册**：`axiom register my_neuron.py --domain battery`
5. **贡献**：接入主网/沙盒，参与发现与结晶

### 5.2 验证工具

```bash
axiom validate ./neurons/turbulence_v2.py
# 检查：UPI 兼容、量纲声明、forward 签名
```

### 5.3 沙盒模式

- 新神经元先在沙盒运行，不写入主知识库
- 沙盒发现结果可提交审核，通过后合并到主网

---

## 6. 自进化闭环

```
[观测] → [神经元1] → [神经元2] → ... → [输出]
              ↓           ↓
         soft_activity  soft_activity
              ↓           ↓
         ┌─────────────────────────┐
         │   Discovery Engine      │
         │   (符号回归 + Pi 约束)   │
         └─────────────────────────┘
                      ↓
         ┌─────────────────────────┐
         │   Hippocampus           │
         │   crystallize → 知识库   │
         └─────────────────────────┘
                      ↓
         [新定律] → 更新 hard_core / 发布
```

### 6.1 进化策略

- **渐进结晶**：soft 活跃度高且公式稳定时结晶
- **多源融合**：多个神经元发现相似公式时，投票或 AIC 择优
- **版本管理**：知识库支持 law_v1, law_v2，可回滚

---

## 7. 实现路线图

| 阶段 | 内容 | 优先级 |
|------|------|--------|
| **P0** | 定义 `BaseNeuron`、`UPICompatible` Protocol | 高 |
| **P0** | 扩展 UPIState 支持多通道 units 声明 | 高 |
| **P1** | 实现 `NeuronRegistry`，支持本地注册 | 高 |
| **P1** | RCLN 实现 `get_contribution_buffer()` | 高 |
| **P2** | Discovery 支持多神经元贡献合并 | 中 |
| **P2** | 结晶时记录贡献者、公式溯源 | 中 |
| **P3** | `axiom validate` CLI 工具 | 中 |
| **P3** | 沙盒模式、审核流程 | 中 |
| **P4** | 远程注册中心、版本管理 | 低 |
| **P4** | 知识库 API、社区公示 | 低 |

---

## 8. 与现有架构的衔接

| 现有组件 | 生态扩展 |
|----------|----------|
| `UPIState` | 增加 `input_units`/`output_units` 元数据传递 |
| `RCLNLayer` | 继承 `BaseNeuron`，实现 `get_contribution_buffer` |
| `DiscoveryEngine` | 支持多源 (X,y) 输入，标注来源 |
| `Hippocampus` | 结晶时写入 `contributor_id`，支持查询 |
| `SymplecticLawRegistry` | 与 Hippocampus 打通，可注册为 hard_core 来源 |

---

## 9. 安全与治理

- **公式执行**：`_formula_to_callable` 使用受限 namespace，禁止 `exec`/`import`
- **量纲强制**：接入时校验，防止量纲错误传播
- **资源限制**：沙盒内限制算力、内存，防止恶意神经元
- **贡献归属**：结晶公式记录 author，支持引用与溯源

---

## 10. Meta-Axis 元轴理论

元轴模块（`layers/meta_kernel.py`）实现修正引力/暗物质替代方案，统一几何核与 RAR 投影。

### 10.1 核心关系

| 符号 | 含义 | 关系 |
|------|------|------|
| \(L\) | 元轴尺度（宇宙学曲率半径） | \(L \approx 80\) Gly |
| \(a_0\) | 加速度尺度（MOND） | \(a_0 = c^2/L \approx 1.2\times 10^{-10}\) m/s² |
| \(g_0\) | 数据单位下的 \(a_0\) | \(g_0 \approx 3700\) (km/s)²/kpc |

### 10.2 投影算子

- **几何核** \(K(z) = k_0/\sqrt{1-(z/L)^2}\)：沿元轴 \(z\) 的投影，\(\Delta V^2 \propto \int_{-L}^{L} K(z)\cdot\rho(r,z)\,dz\)
- **McGaugh 投影** \(\nu(g) = 1/(1 - e^{-\sqrt{g/a_0}})\)：加速度空间 RAR，\(g_{\text{obs}} = g_{\text{bar}} \cdot \nu(g_{\text{bar}})\)

### 10.3 统一性

在相同 \(L\) / \(a_0\) 下，\(K(z)\) 与 \(\nu(g)\) 对应同一物理：元轴投影在加速度空间的等效表现为 McGaugh 形式。Discovery 实验（`experiments/discovery_rar.py`）拟合得到 \(g_0\)，可经 `compute_meta_length(a0_si)` 得到 \(L\)。

### 10.4 实现

- `nu_mcgaugh(g, a0)` / `nu_mcgaugh_torch(g, a0)`：McGaugh 投影
- `compute_meta_length(a0_si)`：\(L = c^2/a_0\)
- `MetaProjectionLayer(projection_mode="mcgaugh", a0_init=3700)`：RAR 投影层
- `MetaProjectionLayer(projection_mode="geometric", a0_init=1.2e-10)`：几何核层，\(L\) 由 \(a_0\) 初始化

---

## 11. 总结

通过 **UPI 协议** + **BaseNeuron 接口** + **NeuronRegistry** + **发现-结晶闭环** + **Meta-Axis 元轴理论**，可构建一个开放、可扩展的物理-AI 生态网。任何人设计的神经元只要符合 UPI，即可接入并参与系统的自进化与物理发现，形成持续演化的集体智能。
