# Requirements Document

## Introduction

本文档定义了将 Axiom-OS 系统的 LLM 接口从 Gemini API 迁移到 DeepSeek API 的需求。当前系统在 `axiom_os/agent/gemini_agent.py` 和 `axiom_os/agent/chat_ui.py` 中使用 Google Generative AI SDK 调用 Gemini 模型。迁移目标是使用 DeepSeek 的 OpenAI 兼容接口替换 Gemini，同时保持现有功能（包括工具调用）完整性。

系统还包含另一个 LLM 集成（`axiom_os/orchestrator/llm_brain.py`），已经使用 OpenAI 兼容接口，可作为实现参考。

## Glossary

- **Gemini_Agent**: 当前使用 Google Generative AI SDK 的代理模块，位于 `axiom_os/agent/gemini_agent.py`
- **Chat_UI**: Streamlit 聊天界面，位于 `axiom_os/agent/chat_ui.py`
- **DeepSeek_API**: DeepSeek 提供的 OpenAI 兼容 API 服务，端点为 https://api.deepseek.com
- **Tool_Calling**: 函数调用功能，允许 LLM 调用预定义的工具函数
- **LLM_Brain**: 现有的 OpenAI 兼容 LLM 集成，位于 `axiom_os/orchestrator/llm_brain.py`
- **TOOL_DEFS**: 工具定义列表，包含可用工具的名称、参数和描述
- **System**: Axiom-OS 系统整体

## Requirements

### Requirement 1: API 客户端迁移

**User Story:** 作为开发者，我希望将 Gemini API 客户端替换为 DeepSeek API 客户端，以便使用 DeepSeek 的语言模型服务。

#### Acceptance Criteria

1. THE System SHALL 使用 OpenAI Python SDK 连接到 DeepSeek API 端点
2. WHEN 初始化 API 客户端时，THE System SHALL 使用 base_url 参数指向 https://api.deepseek.com
3. WHEN 进行 API 认证时，THE System SHALL 使用环境变量 DEEPSEEK_API_KEY 中的 API 密钥
4. THE System SHALL 移除对 google-generativeai 包的依赖
5. THE System SHALL 在 requirements.txt 中添加或确认 openai 包依赖

### Requirement 2: 模型调用接口适配

**User Story:** 作为开发者，我希望适配 DeepSeek 的 API 调用格式，以便正确生成响应。

#### Acceptance Criteria

1. WHEN 调用 LLM 生成内容时，THE System SHALL 使用 OpenAI Chat Completions API 格式
2. WHEN 构造请求消息时，THE System SHALL 使用标准的 messages 数组格式（包含 role 和 content 字段）
3. WHEN 接收 API 响应时，THE System SHALL 从 response.choices[0].message.content 提取生成的文本
4. THE System SHALL 支持 system、user 和 assistant 三种消息角色
5. WHEN API 调用失败时，THE System SHALL 捕获异常并返回错误信息

### Requirement 3: 工具调用功能保持

**User Story:** 作为用户，我希望工具调用功能继续正常工作，以便 LLM 能够执行 Axiom 系统操作。

#### Acceptance Criteria

1. WHEN LLM 需要调用工具时，THE System SHALL 支持 OpenAI 函数调用格式
2. THE System SHALL 将现有 TOOL_DEFS 转换为 OpenAI tools 参数格式
3. WHEN LLM 返回工具调用请求时，THE System SHALL 解析 tool_calls 字段
4. WHEN 执行工具后，THE System SHALL 将工具结果以 tool 角色消息返回给 LLM
5. THE System SHALL 支持多轮工具调用对话流程
6. THE System SHALL 保持现有工具的功能不变（run_benchmark_quick、get_benchmark_report、run_rar、run_discovery_demo、list_domains、apply_domain_extension）

### Requirement 4: 环境变量配置

**User Story:** 作为系统管理员，我希望通过环境变量配置 API 密钥，以便安全管理凭证。

#### Acceptance Criteria

1. THE System SHALL 从环境变量 DEEPSEEK_API_KEY 读取 API 密钥
2. WHEN DEEPSEEK_API_KEY 未设置时，THE System SHALL 返回明确的错误提示
3. THE System SHALL 移除对 GEMINI_API_KEY 和 GOOGLE_API_KEY 环境变量的依赖
4. WHEN Chat_UI 检测到缺少 API 密钥时，THE System SHALL 显示用户友好的错误消息

### Requirement 5: Chat UI 集成更新

**User Story:** 作为用户，我希望 Chat UI 能够使用 DeepSeek 后端，以便通过界面与新的 LLM 交互。

#### Acceptance Criteria

1. WHEN 用户启用扩展模式时，THE Chat_UI SHALL 调用更新后的 DeepSeek 代理
2. THE Chat_UI SHALL 将 "使用 Gemini 扩展" 选项更新为 "使用 DeepSeek 扩展"
3. THE Chat_UI SHALL 在帮助文本中引用 DEEPSEEK_API_KEY 而非 GEMINI_API_KEY
4. WHEN 调用 DeepSeek 代理时，THE Chat_UI SHALL 传递正确的 API 密钥和配置参数
5. THE Chat_UI SHALL 保持现有的用户界面布局和交互流程

### Requirement 6: 模型参数配置

**User Story:** 作为开发者，我希望配置 DeepSeek 模型参数，以便优化生成质量和性能。

#### Acceptance Criteria

1. THE System SHALL 使用 deepseek-chat 作为默认模型名称
2. THE System SHALL 支持通过参数配置 temperature、max_tokens 等生成参数
3. WHEN 调用 API 时，THE System SHALL 设置合理的默认值（temperature=0.7, max_tokens=2000）
4. THE System SHALL 允许调用者覆盖默认模型名称和生成参数

### Requirement 7: 错误处理和重试机制

**User Story:** 作为开发者，我希望系统能够优雅处理 API 错误，以便提高可靠性。

#### Acceptance Criteria

1. WHEN API 调用超时时，THE System SHALL 返回超时错误信息
2. WHEN API 返回速率限制错误时，THE System SHALL 记录错误并通知用户
3. WHEN API 返回认证错误时，THE System SHALL 提示检查 API 密钥配置
4. WHEN 网络连接失败时，THE System SHALL 返回连接错误信息
5. THE System SHALL 在错误消息中包含足够的上下文信息以便调试

### Requirement 8: 代码重构和模块重命名

**User Story:** 作为开发者，我希望代码结构清晰反映使用的 LLM 提供商，以便维护和理解。

#### Acceptance Criteria

1. THE System SHALL 将 gemini_agent.py 重命名为 deepseek_agent.py
2. THE System SHALL 更新所有导入语句以引用新的模块名称
3. THE System SHALL 更新函数和类名称以反映 DeepSeek 而非 Gemini
4. THE System SHALL 保持向后兼容的导入路径（如需要）
5. THE System SHALL 更新模块文档字符串以反映 DeepSeek 集成

### Requirement 9: 文档更新

**User Story:** 作为用户和开发者，我希望文档反映最新的 API 集成，以便正确配置和使用系统。

#### Acceptance Criteria

1. THE System SHALL 更新 README.md 中的 API 配置说明
2. THE System SHALL 在文档中说明如何获取和设置 DEEPSEEK_API_KEY
3. THE System SHALL 更新代码注释以反映 DeepSeek API 的使用
4. THE System SHALL 提供 DeepSeek API 端点和模型名称的文档
5. THE System SHALL 在 CHANGELOG.md 中记录此次迁移

### Requirement 10: 测试和验证

**User Story:** 作为开发者，我希望验证迁移后的功能正确性，以便确保系统稳定运行。

#### Acceptance Criteria

1. WHEN 运行基本对话测试时，THE System SHALL 成功生成响应
2. WHEN 测试工具调用功能时，THE System SHALL 正确执行工具并返回结果
3. WHEN 测试多轮对话时，THE System SHALL 保持上下文连贯性
4. WHEN 测试错误场景时，THE System SHALL 返回适当的错误消息
5. WHEN 通过 Chat UI 测试时，THE System SHALL 正常显示对话和结果
