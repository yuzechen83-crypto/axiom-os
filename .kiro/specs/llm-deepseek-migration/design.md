# Design Document: LLM DeepSeek Migration

## Overview

本设计文档描述了将 Axiom-OS 系统从 Gemini API 迁移到 DeepSeek API 的技术方案。迁移的核心目标是：

1. 使用 DeepSeek 的 OpenAI 兼容接口替换 Google Generative AI SDK
2. 保持现有工具调用（function calling）功能完整性
3. 最小化对现有代码结构的影响
4. 提供清晰的配置和错误处理

DeepSeek API 使用 OpenAI 兼容的接口格式，这意味着我们可以使用 OpenAI Python SDK 进行调用。系统中已有的 `llm_brain.py` 模块已经使用了 OpenAI 客户端，可以作为实现参考。

## Architecture

### 当前架构

```
axiom_os/agent/
├── gemini_agent.py      # 使用 google-generativeai SDK
├── chat_ui.py           # Streamlit UI，调用 gemini_agent
└── tools.py             # 工具定义和执行逻辑

axiom_os/orchestrator/
└── llm_brain.py         # 已使用 OpenAI SDK（参考实现）
```

### 目标架构

```
axiom_os/agent/
├── deepseek_agent.py    # 使用 openai SDK + DeepSeek 端点
├── chat_ui.py           # 更新为调用 deepseek_agent
└── tools.py             # 保持不变
```

### 关键变更

1. **API 客户端**: `google.generativeai` → `openai.OpenAI`
2. **端点配置**: 默认端点 → `https://api.deepseek.com`
3. **认证方式**: `GEMINI_API_KEY` → `DEEPSEEK_API_KEY`
4. **消息格式**: Gemini 格式 → OpenAI Chat Completions 格式
5. **工具调用**: 自定义解析 → OpenAI 原生 function calling

## Components and Interfaces

### 1. DeepSeek Agent Module

**文件**: `axiom_os/agent/deepseek_agent.py`

**核心函数**:

```python
def run_agent_loop(
    user_message: str,
    api_key: str,
    model: str = "deepseek-chat",
    max_tool_rounds: int = 5,
) -> str:
    """
    与 DeepSeek 多轮对话：用户输入 -> LLM 可能返回工具调用 -> 执行工具 -> 结果回传 -> 直至最终回复。
    
    Args:
        user_message: 用户输入的消息
        api_key: DeepSeek API 密钥
        model: 模型名称，默认 "deepseek-chat"
        max_tool_rounds: 最大工具调用轮数
        
    Returns:
        最终的自然语言回复
    """
```

```python
def run_agent_single(
    user_message: str,
    api_key: str,
    model: str = "deepseek-chat",
) -> str:
    """
    单轮对话：仅让 DeepSeek 根据当前上下文回复（不执行工具）。
    
    Args:
        user_message: 用户输入的消息
        api_key: DeepSeek API 密钥
        model: 模型名称
        
    Returns:
        LLM 生成的回复
    """
```

**内部辅助函数**:

```python
def _invoke_deepseek(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
) -> Any:
    """
    调用 DeepSeek API，返回 ChatCompletion 对象。
    
    Args:
        client: OpenAI 客户端实例
        model: 模型名称
        messages: 消息历史
        tools: 工具定义（OpenAI 格式）
        temperature: 生成温度
        max_tokens: 最大 token 数
        
    Returns:
        ChatCompletion 响应对象
    """
```

```python
def _convert_tools_to_openai_format(tool_defs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将 Axiom 工具定义转换为 OpenAI function calling 格式。
    
    输入格式（TOOL_DEFS）:
    {
        "name": "run_rar",
        "description": "运行 RAR 星系旋转曲线发现",
        "params": {"n_galaxies": "int, 星系数", "epochs": "int, 训练轮数"}
    }
    
    输出格式（OpenAI tools）:
    {
        "type": "function",
        "function": {
            "name": "run_rar",
            "description": "运行 RAR 星系旋转曲线发现",
            "parameters": {
                "type": "object",
                "properties": {
                    "n_galaxies": {"type": "integer", "description": "星系数"},
                    "epochs": {"type": "integer", "description": "训练轮数"}
                },
                "required": []
            }
        }
    }
    """
```

### 2. Chat UI Updates

**文件**: `axiom_os/agent/chat_ui.py`

**变更点**:

1. 导入语句更新:
   ```python
   # 旧: from axiom_os.agent.gemini_agent import run_agent_loop
   # 新: from axiom_os.agent.deepseek_agent import run_agent_loop
   ```

2. UI 文本更新:
   ```python
   # 旧: use_gemini = st.sidebar.checkbox("使用 Gemini 扩展", ...)
   # 新: use_deepseek = st.sidebar.checkbox("使用 DeepSeek 扩展", ...)
   ```

3. 环境变量检查:
   ```python
   # 旧: deepseek_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
   # 新: deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
   ```

4. 错误提示更新:
   ```python
   # 旧: "❌ 请设置环境变量 GEMINI_API_KEY 或 GOOGLE_API_KEY"
   # 新: "❌ 请设置环境变量 DEEPSEEK_API_KEY"
   ```

### 3. Tool Definitions

**文件**: `axiom_os/agent/tools.py`

**保持不变**: 工具定义和执行逻辑无需修改，因为工具调用的实现逻辑在 agent 层处理。

## Data Models

### Message Format

**OpenAI Chat Completions 消息格式**:

```python
{
    "role": "system" | "user" | "assistant" | "tool",
    "content": str,  # 消息内容
    "name": str,     # 可选，工具名称（role=tool 时）
    "tool_calls": [  # 可选，LLM 请求的工具调用（role=assistant 时）
        {
            "id": str,
            "type": "function",
            "function": {
                "name": str,
                "arguments": str  # JSON 字符串
            }
        }
    ],
    "tool_call_id": str  # 可选，工具调用 ID（role=tool 时）
}
```

### Tool Definition Format

**Axiom 内部格式** (TOOL_DEFS):

```python
{
    "name": str,
    "description": str,
    "params": Dict[str, str]  # 参数名 -> 描述字符串
}
```

**OpenAI Function Calling 格式**:

```python
{
    "type": "function",
    "function": {
        "name": str,
        "description": str,
        "parameters": {
            "type": "object",
            "properties": {
                param_name: {
                    "type": "string" | "integer" | "number" | "boolean",
                    "description": str
                }
            },
            "required": List[str]  # 必需参数列表
        }
    }
}
```

### API Configuration

```python
@dataclass
class DeepSeekConfig:
    """DeepSeek API 配置"""
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 60  # 秒
```

## 
Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

在开始编写 Correctness Properties 之前，我需要分析需求文档中的验收标准，确定哪些可以转化为可测试的属性。


### Property 1: API 客户端配置正确性

*For any* DeepSeek 客户端初始化，客户端的 base_url 应该指向 https://api.deepseek.com，并且 API 密钥应该从环境变量 DEEPSEEK_API_KEY 中读取。

**Validates: Requirements 1.2, 1.3, 4.1**

### Property 2: OpenAI 消息格式一致性

*For any* 用户输入消息和系统提示，构造的请求消息数组中的每个消息对象都应该包含有效的 role 字段（system、user 或 assistant）和 content 字段，并且整体结构符合 OpenAI Chat Completions API 规范。

**Validates: Requirements 2.1, 2.2, 2.4**

### Property 3: API 响应解析正确性

*For any* 有效的 OpenAI ChatCompletion 响应对象，系统应该能够从 response.choices[0].message.content 正确提取生成的文本内容。

**Validates: Requirements 2.3**

### Property 4: 工具定义格式转换正确性

*For any* Axiom TOOL_DEFS 格式的工具定义，转换后的 OpenAI tools 格式应该包含正确的 type="function" 字段、function.name、function.description 和 function.parameters 结构，并且参数类型应该正确映射（int → integer, str → string）。

**Validates: Requirements 3.1, 3.2**

### Property 5: 工具调用解析和执行正确性

*For any* LLM 返回的包含 tool_calls 字段的响应，系统应该能够正确解析工具名称和参数，执行对应的工具函数，并将结果以 tool 角色消息格式返回给 LLM。

**Validates: Requirements 3.3, 3.4**

### Property 6: 多轮工具调用对话状态一致性

*For any* 多轮工具调用序列（用户消息 → 工具调用 → 工具结果 → 最终回复），系统应该保持正确的消息历史顺序，每个工具调用都应该有对应的工具结果消息，并且 tool_call_id 应该正确匹配。

**Validates: Requirements 3.5, 10.3**

### Property 7: 工具功能回归一致性

*For any* 现有工具（run_benchmark_quick, get_benchmark_report, run_rar, run_discovery_demo, list_domains, apply_domain_extension），使用 DeepSeek 代理调用该工具的结果应该与使用原 Gemini 代理调用的结果在功能上等价（相同的输入产生相同的输出结构和数据）。

**Validates: Requirements 3.6, 10.2**

### Property 8: 异常处理健壮性

*For any* API 调用过程中抛出的异常（包括网络错误、超时、认证失败等），系统应该捕获该异常并返回包含错误类型和上下文信息的错误消息字符串，而不是让异常传播导致程序崩溃。

**Validates: Requirements 2.5, 7.5, 10.4**

### Property 9: 生成参数配置灵活性

*For any* 有效的生成参数值（temperature, max_tokens, model），当调用者提供这些参数时，系统应该使用提供的值；当调用者不提供时，系统应该使用文档化的默认值（temperature=0.7, max_tokens=2000, model="deepseek-chat"）。

**Validates: Requirements 6.2, 6.3, 6.4**

## Error Handling

### 错误类型和处理策略

1. **认证错误** (401 Unauthorized)
   - 检测: API 返回 401 状态码或 AuthenticationError
   - 处理: 返回错误消息 "认证失败，请检查 DEEPSEEK_API_KEY 环境变量是否正确设置"
   - 用户操作: 验证 API 密钥是否有效

2. **速率限制错误** (429 Too Many Requests)
   - 检测: API 返回 429 状态码或 RateLimitError
   - 处理: 返回错误消息 "API 速率限制，请稍后重试"
   - 用户操作: 等待后重试或升级 API 配额

3. **超时错误** (Timeout)
   - 检测: 请求超过配置的超时时间（默认 60 秒）
   - 处理: 返回错误消息 "API 调用超时，请检查网络连接或稍后重试"
   - 用户操作: 检查网络连接，考虑增加超时时间

4. **网络连接错误** (ConnectionError)
   - 检测: 无法连接到 API 端点
   - 处理: 返回错误消息 "无法连接到 DeepSeek API，请检查网络连接"
   - 用户操作: 检查网络连接和防火墙设置

5. **API 密钥缺失**
   - 检测: DEEPSEEK_API_KEY 环境变量未设置或为空
   - 处理: 在初始化时立即返回错误消息 "请设置环境变量 DEEPSEEK_API_KEY"
   - 用户操作: 设置环境变量

6. **工具调用解析错误**
   - 检测: LLM 返回的 tool_calls 格式不正确或工具名称不存在
   - 处理: 返回错误消息给 LLM，说明工具调用格式错误或工具不存在
   - 系统操作: 继续对话循环，让 LLM 有机会纠正

7. **工具执行错误**
   - 检测: 工具函数执行时抛出异常
   - 处理: 捕获异常，将错误信息作为工具结果返回给 LLM
   - 系统操作: 继续对话循环，让 LLM 根据错误信息调整策略

### 错误消息格式

所有错误消息应该遵循统一格式：

```python
{
    "error_type": str,  # 错误类型标识
    "message": str,     # 用户友好的错误描述
    "details": str,     # 技术细节（可选）
    "suggestion": str   # 建议的解决方案（可选）
}
```

对于返回给用户的字符串格式错误消息：

```
❌ [错误类型]: 错误描述
建议: 解决方案
```

### 日志记录

所有错误应该记录到日志系统，包含：
- 时间戳
- 错误类型
- 完整的异常堆栈（如果有）
- 请求上下文（用户消息、工具调用等）

## Testing Strategy

### 测试方法论

本项目采用双重测试策略：

1. **单元测试**: 验证特定示例、边缘情况和错误条件
2. **属性测试**: 验证跨所有输入的通用属性

两种测试方法是互补的：单元测试捕获具体的 bug，属性测试验证通用的正确性。

### 单元测试范围

单元测试应该专注于：

1. **具体示例**:
   - 测试特定的用户输入和预期输出
   - 测试特定的工具调用序列
   - 测试特定的错误场景

2. **边缘情况**:
   - 空消息处理
   - 超长消息处理
   - 特殊字符处理
   - 环境变量缺失
   - API 各种错误响应

3. **集成点**:
   - Chat UI 与 DeepSeek Agent 的集成
   - DeepSeek Agent 与工具系统的集成
   - API 客户端初始化和配置

### 属性测试配置

**测试库**: 使用 `hypothesis` (Python 的属性测试库)

**配置要求**:
- 每个属性测试最少运行 100 次迭代
- 使用 `@given` 装饰器定义输入生成策略
- 每个测试必须引用对应的设计文档属性

**标签格式**:
```python
# Feature: llm-deepseek-migration, Property 1: API 客户端配置正确性
@given(api_key=st.text(min_size=1))
def test_client_configuration(api_key):
    ...
```

### 属性测试实现指南

#### Property 1: API 客户端配置正确性

```python
from hypothesis import given
import hypothesis.strategies as st

# Feature: llm-deepseek-migration, Property 1: API 客户端配置正确性
@given(api_key=st.text(min_size=1, max_size=100))
def test_client_configuration_property(api_key):
    """验证客户端配置的正确性"""
    # 设置环境变量
    os.environ["DEEPSEEK_API_KEY"] = api_key
    
    # 初始化客户端
    client = create_deepseek_client()
    
    # 验证 base_url
    assert client.base_url == "https://api.deepseek.com"
    
    # 验证 API 密钥
    assert client.api_key == api_key
```

#### Property 2: OpenAI 消息格式一致性

```python
# Feature: llm-deepseek-migration, Property 2: OpenAI 消息格式一致性
@given(
    user_message=st.text(min_size=1, max_size=1000),
    system_prompt=st.text(min_size=1, max_size=500)
)
def test_message_format_consistency(user_message, system_prompt):
    """验证消息格式符合 OpenAI 规范"""
    messages = construct_messages(system_prompt, user_message)
    
    for msg in messages:
        # 每个消息必须有 role 和 content
        assert "role" in msg
        assert "content" in msg
        
        # role 必须是有效值
        assert msg["role"] in ["system", "user", "assistant", "tool"]
        
        # content 必须是字符串
        assert isinstance(msg["content"], str)
```

#### Property 4: 工具定义格式转换正确性

```python
# Feature: llm-deepseek-migration, Property 4: 工具定义格式转换正确性
def test_tool_conversion_property():
    """验证工具定义转换的正确性"""
    from axiom_os.agent.tools import TOOL_DEFS
    
    openai_tools = convert_tools_to_openai_format(TOOL_DEFS)
    
    for tool in openai_tools:
        # 验证顶层结构
        assert tool["type"] == "function"
        assert "function" in tool
        
        func = tool["function"]
        # 验证函数定义
        assert "name" in func
        assert "description" in func
        assert "parameters" in func
        
        params = func["parameters"]
        # 验证参数结构
        assert params["type"] == "object"
        assert "properties" in params
```

#### Property 7: 工具功能回归一致性

```python
# Feature: llm-deepseek-migration, Property 7: 工具功能回归一致性
@given(
    tool_name=st.sampled_from([
        "run_benchmark_quick",
        "get_benchmark_report",
        "list_domains"
    ])
)
def test_tool_functionality_regression(tool_name):
    """验证工具功能在迁移后保持一致"""
    # 直接调用工具（不通过 LLM）
    result = run_tool(tool_name)
    
    # 验证结果结构
    assert isinstance(result, dict)
    assert "ok" in result
    
    # 如果成功，应该有预期的字段
    if result["ok"]:
        if tool_name == "list_domains":
            assert "domains" in result
        elif tool_name == "get_benchmark_report":
            assert "content" in result or "error" in result
```

#### Property 8: 异常处理健壮性

```python
# Feature: llm-deepseek-migration, Property 8: 异常处理健壮性
@given(
    error_type=st.sampled_from([
        "timeout",
        "connection_error",
        "auth_error",
        "rate_limit"
    ])
)
def test_exception_handling_robustness(error_type):
    """验证异常处理的健壮性"""
    # Mock API 调用抛出不同类型的异常
    with mock_api_error(error_type):
        result = invoke_deepseek_safe(client, model, messages)
        
        # 应该返回错误消息而不是抛出异常
        assert isinstance(result, str)
        assert "❌" in result or "error" in result.lower()
        
        # 错误消息应该包含有用信息
        assert len(result) > 10
```

### 测试覆盖率目标

- 核心函数代码覆盖率: ≥ 90%
- 错误处理路径覆盖率: ≥ 80%
- 工具调用流程覆盖率: 100%

### 集成测试

除了单元测试和属性测试，还需要进行端到端集成测试：

1. **完整对话流程测试**:
   - 用户输入 → LLM 响应
   - 用户输入 → 工具调用 → 工具结果 → LLM 最终响应
   - 多轮对话保持上下文

2. **Chat UI 集成测试**:
   - UI 启动和配置
   - 用户交互和响应显示
   - 错误消息显示

3. **性能测试**:
   - API 调用延迟
   - 多轮对话响应时间
   - 并发请求处理

### 测试数据

使用以下测试数据源：

1. **真实用户查询**: 从现有 Gemini 集成收集的实际用户输入
2. **工具调用场景**: 覆盖所有 6 个工具的典型使用场景
3. **错误场景**: 各种 API 错误和异常情况
4. **边缘情况**: 空输入、超长输入、特殊字符等

### 回归测试

在迁移完成后，运行完整的回归测试套件：

1. 所有现有的 Axiom-OS 测试应该继续通过
2. 所有工具功能应该保持不变
3. Chat UI 的所有功能应该正常工作
4. 性能指标应该在可接受范围内（响应时间、成功率等）
