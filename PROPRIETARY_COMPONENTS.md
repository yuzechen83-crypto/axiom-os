# 核心技术组件声明 (Proprietary Components)

本仓库中以下模块的**实现**已全部开源，可直接在 `axiom_os` 中调取使用。

## 架构说明

- **axiom_os**：公开仓库，含完整实现，全部开源可调取。
- **axiom_core_proprietary**：已废弃，实现已合并至 axiom_os。

## 核心组件（实现在 axiom_os，全部开源）

| 模块 | 路径 | 说明 |
|------|--------|------|
| RCLN | `axiom_os/layers/rcln.py` | 残差耦合链接神经元，Hard Core + Soft Shell 混合架构 |
| Discovery Engine | `axiom_os/engine/discovery.py` | 符号回归与量纲分析，Buckingham Pi 约束下的公式发现 |
| Hippocampus | `axiom_os/core/hippocampus.py` | 知识注册与检索，物理定律结晶存储 |
| Wind Hard Core | `axiom_os/core/wind_hard_core.py` | 湍流物理 Hard Core 实现 |
| SPNN-Evo Coach | `axiom_os/coach/spnn_evo_coach.py` | 物理约束教练，量纲与范围检查 |
| 元轴丛场 | `axiom_os/core/bundle_field/` | 分区 × 域 × 体制的纤维丛结构 |
| Adaptive Hard Core | `axiom_os/core/adaptive_hard_core.py` | 极端情况下的自适应物理约束 |
| 智能分区 | `axiom_os/core/partition.py` | 分区定义与课程学习 |

## 安装与运行

1. 克隆本仓库后，在项目根目录执行：
   ```bash
   pip install -r requirements.txt
   ```
2. 直接运行 axiom_os 应用，无需额外安装。

## 使用条款

- **开源部分**：本仓库采用 MIT License，允许使用、修改与分发。
- **专有部分**：上述组件的**算法实现、架构设计、核心逻辑**为专有技术。
- **禁止行为**：未经授权对上述模块进行商业复制、二次分发、逆向工程或用于竞争性产品。
- **允许行为**：学习、研究、非商业用途的参考与集成。

## 联系方式

如有商业合作或授权需求，请联系：yuzechen83-crypto

---

*Copyright (c) 2026 yuzechen83-crypto. All Rights Reserved.*
