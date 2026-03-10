# Implementation Plan: LLM DeepSeek Migration

## Overview

本实施计划将 Axiom-OS 系统从 Gemini API 迁移到 DeepSeek API。实施策略采用增量方式：首先创建新的 DeepSeek Agent 模块，然后更新 Chat UI 集成，最后进行测试验证和文档更新。每个步骤都包含相应的测试任务以确保功能正确性。

## Tasks

- [x] 1. 创建 DeepSeek Agent 核心模块
  - 创建 `axiom_os/agent/deepseek_agent.py` 文件
  - 实现 OpenAI 客户端初始化逻辑
  - 实现基本的 API 调用函数 `_invoke_deepseek`
  - 配置 DeepSeek API 端点 (https://api.deepseek.com)
  - 实现环境变量 DEEPSEEK_API_KEY 读取
  - _Requirements: 1.1, 1.2, 1.3, 4.1_

- [x] 1.1 编写 API 客户端配置测试
  - **Property 1: API 客户端配置正确性**
  - **Validates: Requirements 1.2, 1.3, 4.1**

- [x] 2. 实现工具定义格式转换
  - [x] 2.1 实现 `_convert_tools_to_openai_format` 函数
    - 将 Axiom TOOL_DEFS 格式转换为 OpenAI tools 格式
    - 处理参数类型映射（int → integer, str → string）
    - 生成正确的 JSON Schema 结构
    - _Requirements: 3.1, 3.2_

  - [ ]* 2.2 编写工具格式转换属性测试
    - **Property 4: 工具定义格式转换正确性**
    - **Validates: Requirements 3.1, 3.2**

- [x] 3. 实现消息构造和格式化
  - [x] 3.1 实现消息数组构造逻辑
    - 支持 system、user、assistant 三种角色
    - 确保每个消息包含 role 和 content 字段
    - 实现消息历史管理
    - _Requirements: 2.1, 2.2, 2.4_

  - [ ]* 3.2 编写消息格式一致性属性测试
    - **Property 2: OpenAI 消息格式一致性**
    - **Validates: Requirements 2.1, 2.2, 2.4**

  - [x] 3.3 实现 API 响应解析逻辑
    - 从 response.choices[0].message.content 提取文本
    - 处理空响应和异常情况
    - _Requirements: 2.3_

  - [ ]* 3.4 编写响应解析属性测试
    - **Property 3: API 响应解析正确性**
    - **Validates: Requirements 2.3**

- [x] 4. 实现工具调用功能
  - [x] 4.1 实现工具调用请求处理
    - 解析 LLM 返回的 tool_calls 字段
    - 提取工具名称和参数
    - 调用对应的工具函数
    - _Requirements: 3.3_

  - [x] 4.2 实现工具结果消息构造
    - 将工具执行结果格式化为 tool 角色消息
    - 正确设置 tool_call_id 字段
    - _Requirements: 3.4_

  - [ ]* 4.3 编写工具调用解析和执行属性测试
    - **Property 5: 工具调用解析和执行正确性**
    - **Validates: Requirements 3.3, 3.4**

- [x] 5. 实现多轮对话循环
  - [x] 5.1 实现 `run_agent_loop` 函数
    - 实现用户输入 → LLM 响应 → 工具调用 → 工具结果 → 最终回复的完整流程
    - 维护消息历史状态
    - 支持最大轮数限制（max_tool_rounds）
    - _Requirements: 3.5_

  - [x] 5.2 实现 `run_agent_single` 函数
    - 实现单轮对话（不执行工具）
    - 用于简单问答场景
    - _Requirements: 2.1, 2.3_

  - [ ]* 5.3 编写多轮对话状态一致性属性测试
    - **Property 6: 多轮工具调用对话状态一致性**
    - **Validates: Requirements 3.5, 10.3**

- [x] 6. 实现错误处理机制
  - [x] 6.1 实现 API 异常捕获和处理
    - 捕获认证错误（401）
    - 捕获速率限制错误（429）
    - 捕获超时错误
    - 捕获网络连接错误
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [x] 6.2 实现错误消息格式化
    - 生成用户友好的错误消息
    - 包含错误类型和上下文信息
    - 提供解决建议
    - _Requirements: 7.5_

  - [x] 6.3 实现 API 密钥缺失检查
    - 在初始化时检查 DEEPSEEK_API_KEY
    - 返回明确的错误提示
    - _Requirements: 4.2_

  - [ ]* 6.4 编写异常处理健壮性属性测试
    - **Property 8: 异常处理健壮性**
    - **Validates: Requirements 2.5, 7.5, 10.4**

  - [ ]* 6.5 编写边缘情况单元测试
    - 测试 API 密钥缺失场景
    - 测试各种 API 错误响应
    - 测试超时场景
    - _Requirements: 4.2, 7.1, 7.2, 7.3, 7.4_

- [x] 7. 实现生成参数配置
  - [x] 7.1 实现参数配置逻辑
    - 设置默认值（temperature=0.7, max_tokens=2000, model="deepseek-chat"）
    - 支持参数覆盖
    - 将参数传递给 API 调用
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [ ]* 7.2 编写生成参数配置属性测试
    - **Property 9: 生成参数配置灵活性**
    - **Validates: Requirements 6.2, 6.3, 6.4**

  - [ ]* 7.3 编写默认参数单元测试
    - 测试默认模型名称
    - 测试默认 temperature 和 max_tokens
    - _Requirements: 6.1, 6.3_

- [x] 8. Checkpoint - 核心功能验证
  - 确保所有核心功能测试通过
  - 验证工具调用流程正常工作
  - 如有问题请向用户报告

- [x] 9. 更新 Chat UI 集成
  - [x] 9.1 更新导入语句
    - 将 `from axiom_os.agent.gemini_agent` 改为 `from axiom_os.agent.deepseek_agent`
    - 更新函数调用引用
    - _Requirements: 5.1, 8.2_

  - [x] 9.2 更新 UI 文本和标签
    - 将 "使用 Gemini 扩展" 改为 "使用 DeepSeek 扩展"
    - 更新帮助文本中的环境变量引用（GEMINI_API_KEY → DEEPSEEK_API_KEY）
    - _Requirements: 5.2, 5.3_

  - [x] 9.3 更新环境变量读取逻辑
    - 从 DEEPSEEK_API_KEY 读取密钥
    - 移除对 GEMINI_API_KEY 和 GOOGLE_API_KEY 的引用
    - 更新错误提示消息
    - _Requirements: 4.1, 4.4_

  - [x] 9.4 更新 API 调用参数传递
    - 确保正确传递 api_key 参数
    - 确保正确传递 model 和其他配置参数
    - _Requirements: 5.4_

  - [ ]* 9.5 编写 Chat UI 集成单元测试
    - 测试 UI 启动和配置
    - 测试环境变量检查
    - 测试错误消息显示
    - _Requirements: 4.4, 5.1, 5.4_

- [x] 10. 更新依赖和配置
  - [x] 10.1 更新 requirements.txt
    - 确认 openai 包已包含
    - 移除 google-generativeai 包（如果不再需要）
    - _Requirements: 1.4, 1.5_

  - [x] 10.2 更新模块文档字符串
    - 更新 deepseek_agent.py 的文档说明
    - 更新 chat_ui.py 的文档说明
    - _Requirements: 8.5_

- [x] 11. 工具功能回归测试
  - [ ]* 11.1 编写工具功能回归属性测试
    - **Property 7: 工具功能回归一致性**
    - **Validates: Requirements 3.6, 10.2**

  - [ ]* 11.2 编写端到端集成测试
    - 测试完整对话流程
    - 测试工具调用流程
    - 测试多轮对话
    - _Requirements: 10.1, 10.2, 10.3_

- [x] 12. Checkpoint - 集成测试验证
  - 运行所有测试确保通过
  - 手动测试 Chat UI 功能
  - 验证所有工具正常工作
  - 如有问题请向用户报告

- [x] 13. 更新文档
  - [x] 13.1 更新 README.md
    - 添加 DeepSeek API 配置说明
    - 说明如何获取和设置 DEEPSEEK_API_KEY
    - 更新 API 端点和模型名称信息
    - _Requirements: 9.1, 9.2, 9.4_

  - [x] 13.2 更新代码注释
    - 更新所有相关文件的注释
    - 确保注释反映 DeepSeek API 的使用
    - _Requirements: 9.3_

  - [x] 13.3 更新 CHANGELOG.md
    - 记录此次迁移的详细信息
    - 列出破坏性变更（环境变量名称变更）
    - 提供迁移指南
    - _Requirements: 9.5_

- [-] 14. 最终验证和清理
  - [x] 14.1 运行完整测试套件
    - 运行所有单元测试
    - 运行所有属性测试
    - 运行集成测试
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

  - [x] 14.2 性能验证
    - 测试 API 调用延迟
    - 测试多轮对话响应时间
    - 确保性能在可接受范围内

  - [ ] 14.3 代码审查和清理
    - 移除未使用的导入
    - 清理调试代码
    - 确保代码风格一致

- [ ] 15. Final Checkpoint - 完成验证
  - 确保所有测试通过
  - 确保文档完整更新
  - 向用户确认迁移完成并可以使用

## Notes

- 任务标记 `*` 的为可选测试任务，可以根据时间和优先级决定是否实施
- 每个任务都引用了具体的需求编号以确保可追溯性
- Checkpoint 任务用于在关键节点验证系统状态
- 属性测试使用 hypothesis 库，每个测试运行最少 100 次迭代
- 工具功能必须保持向后兼容，确保现有用户不受影响
- 环境变量名称变更是破坏性变更，需要在文档中明确说明
