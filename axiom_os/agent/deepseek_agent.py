"""
Axiom DeepSeek Agent - DeepSeek LLM 集成模块

本模块提供 Axiom-OS 与 DeepSeek API 的集成，使用 OpenAI 兼容接口实现 LLM 功能。
DeepSeek Agent 可以通过工具调用（function calling）与 Axiom 系统交互，执行以下操作：

功能特性：
- 收集与评估：运行基准测试、读取报告、执行 RAR/公式发现、查看当前领域
- 扩展 Axiom：根据需求生成新领域或新协议的 Python 代码并提交保存
- 优化建议：根据基准报告与指标给出超参数建议、数据增强或协议顺序建议

技术实现：
- API 端点：https://api.deepseek.com
- 认证方式：通过环境变量 DEEPSEEK_API_KEY 提供 API 密钥
- 接口格式：OpenAI Chat Completions API 兼容格式
- 默认模型：deepseek-chat
- 工具调用：支持 OpenAI function calling 格式的工具定义和执行

主要函数：
- run_agent_loop: 多轮对话，支持工具调用和结果回传
- run_agent_single: 单轮对话，仅生成回复不执行工具

错误处理：
- 认证错误（401）：提示检查 API 密钥
- 速率限制（429）：提示稍后重试
- 超时错误：提示检查网络连接
- 连接错误：提示检查网络和防火墙设置

使用示例：
    import os
    from axiom_os.agent.deepseek_agent import run_agent_loop
    
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    response = run_agent_loop(
        user_message="运行一次快速基准测试",
        api_key=api_key,
        model="deepseek-chat",
        max_tool_rounds=5
    )
    print(response)

依赖要求：
- openai>=1.0.0 (OpenAI Python SDK)
- axiom_os.agent.tools (工具定义和执行逻辑)

注意事项：
- 确保设置环境变量 DEEPSEEK_API_KEY
- 工具调用最多执行 max_tool_rounds 轮（默认 5 轮）
- 所有 API 错误都会被捕获并返回用户友好的错误消息
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI, AuthenticationError, RateLimitError, APITimeoutError, APIConnectionError
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    # 定义占位符异常类，避免代码中引用时出错
    class AuthenticationError(Exception):
        pass
    class RateLimitError(Exception):
        pass
    class APITimeoutError(Exception):
        pass
    class APIConnectionError(Exception):
        pass

from .tools import TOOL_DEFS, run_tool


def _format_error_message(
    error: Exception,
    error_type: Optional[str] = None,
    context: str = "",
) -> str:
    """
    格式化错误消息，生成用户友好的错误提示。
    
    Args:
        error: 异常对象
        error_type: 错误类型（可选，如果不提供则从异常类型推断）
        context: 错误上下文信息（可选）
        
    Returns:
        格式化的错误消息字符串，包含错误类型、描述和解决建议
    """
    if error_type is None:
        error_type = type(error).__name__
    
    error_msg = str(error)
    
    # 根据错误类型生成友好的错误消息和建议
    if isinstance(error, AuthenticationError) or "401" in error_msg or "Unauthorized" in error_msg:
        return (
            "❌ 认证失败: 请检查 DEEPSEEK_API_KEY 环境变量是否正确设置\n"
            f"建议: 验证 API 密钥是否有效，可以在 DeepSeek 控制台重新生成密钥\n"
            f"详细信息: {error_msg}"
        )
    
    elif isinstance(error, RateLimitError) or "429" in error_msg or "rate" in error_msg.lower():
        return (
            "❌ API 速率限制: 请求过于频繁，请稍后重试\n"
            f"建议: 等待几分钟后重试，或考虑升级 API 配额\n"
            f"详细信息: {error_msg}"
        )
    
    elif isinstance(error, APITimeoutError) or "timeout" in error_msg.lower():
        return (
            "❌ API 调用超时: 请求超过了最大等待时间\n"
            f"建议: 检查网络连接是否稳定，或稍后重试\n"
            f"详细信息: {error_msg}"
        )
    
    elif isinstance(error, APIConnectionError) or "connection" in error_msg.lower():
        return (
            "❌ 无法连接到 DeepSeek API: 网络连接失败\n"
            f"建议: 检查网络连接和防火墙设置，确保可以访问 https://api.deepseek.com\n"
            f"详细信息: {error_msg}"
        )
    
    else:
        # 通用错误消息
        context_info = f" ({context})" if context else ""
        return (
            f"❌ API 调用错误{context_info}: {error_type}\n"
            f"建议: 请检查错误详情，如果问题持续请联系技术支持\n"
            f"详细信息: {error_msg}"
        )


def _check_api_key(api_key: Optional[str]) -> tuple[bool, str]:
    """
    检查 API 密钥是否有效。
    
    Args:
        api_key: API 密钥字符串
        
    Returns:
        (is_valid, error_message) 元组
        - is_valid: 密钥是否有效（非空）
        - error_message: 如果无效，返回错误消息；否则为空字符串
    """
    if not api_key or not api_key.strip():
        return False, (
            "❌ API 密钥缺失: 请设置环境变量 DEEPSEEK_API_KEY\n"
            "建议: 在终端中运行 'export DEEPSEEK_API_KEY=your_api_key' (Linux/Mac) "
            "或 'set DEEPSEEK_API_KEY=your_api_key' (Windows)\n"
            "获取密钥: 访问 https://platform.deepseek.com 注册并获取 API 密钥"
        )
    return True, ""


SYSTEM_PROMPT_BASE = """你是 Axiom-OS 的扩展与优化助手。你可以通过工具完成以下能力：

1. **收集与评估**：运行基准 (run_benchmark_quick)、读取报告 (get_benchmark_report)、运行 RAR/公式发现 (run_rar, run_discovery_demo)、查看当前领域 (list_domains)。
2. **CAD 建模**：run_cad_model 构建 3D 形状（box/cylinder/sphere/l_bracket/simple_gear）并导出 STL；list_cad_shapes 查看支持的形状；支持参数 width/height/depth/radius。
3. **知识学习**：read_workspace_doc 读取 workspace 文档；retrieve_hippocampus 检索已结晶的物理定律；web_search 搜索关键词发现资源；fetch_url 抓取网页正文并可选保存到 workspace。
4. **扩展 Axiom**：根据需求生成新领域或新协议的 Python 代码，通过 apply_domain_extension 提交保存。
5. **优化建议**：根据基准报告与 R² 等指标，给出超参数建议（如 rar_epochs、n_galaxies）、数据增强或协议顺序建议。

规则：
- 若用户要求「跑基准」「看报告」「分析一下」：先调用相应工具，再根据结果用自然语言总结或建议。
- 若用户要求「3D 建模」「CAD」「建个 XXX 模型」：用 run_cad_model(shape="...") 或先 list_cad_shapes 查看支持的形状。
- 若用户要求「学习」「读文档」「发动机」「芯片」：可先用 web_search 搜索，再用 fetch_url 抓取并 save_to_workspace 保存；或 read_workspace_doc 读取已有文档。
- 若用户要求「扩展」「新领域」「加一个 XXX 协议」：可先 list_domains 了解现有领域，再生成符合 Axiom 接口的代码，用 apply_domain_extension 提交。
- 使用工具时，系统会自动调用并返回结果。
- 收到工具返回后，用中文总结结果并给出下一步建议或直接回答用户。
"""


def _build_system_prompt_with_workspace() -> str:
    """注入 workspace 文档到 system prompt，实现「先学习再使用」"""
    try:
        from axiom_os.config.loader import get_workspace_path
        ws = get_workspace_path()
        if not ws.exists():
            return SYSTEM_PROMPT_BASE
        # 加载 IDENTITY + PROJECTS 作为上下文（控制长度）
        parts = []
        for name in ["IDENTITY.md", "PROJECTS.md"]:
            p = ws / name
            if p.exists():
                text = p.read_text(encoding="utf-8")[:1500]
                parts.append(f"## {name}\n{text}")
        if not parts:
            return SYSTEM_PROMPT_BASE
        context = "\n\n".join(parts)
        return f"""以下为 Axiom workspace 知识库（供参考）：

{context}

---

{SYSTEM_PROMPT_BASE}"""
    except Exception:
        return SYSTEM_PROMPT_BASE


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
    openai_tools = []
    
    for tool in tool_defs:
        properties = {}
        params = tool.get("params", {})
        
        for param_name, param_desc in params.items():
            # 解析参数描述，提取类型和说明
            # 格式: "int, 星系数" 或 "str, Python 代码"
            param_type = "string"  # 默认类型
            description = param_desc
            
            if isinstance(param_desc, str) and "," in param_desc:
                type_part, desc_part = param_desc.split(",", 1)
                type_part = type_part.strip().lower()
                description = desc_part.strip()
                
                # 类型映射
                if type_part == "int":
                    param_type = "integer"
                elif type_part == "str":
                    param_type = "string"
                elif type_part == "float":
                    param_type = "number"
                elif type_part == "bool":
                    param_type = "boolean"
            
            properties[param_name] = {
                "type": param_type,
                "description": description
            }
        
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": []  # Axiom 工具参数都是可选的，有默认值
                }
            }
        }
        
        openai_tools.append(openai_tool)
    
    return openai_tools


def _construct_messages(
    system_prompt: str,
    user_message: str,
    message_history: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    构造 OpenAI 格式的消息数组。
    
    Args:
        system_prompt: 系统提示词
        user_message: 用户消息
        message_history: 可选的消息历史（用于多轮对话）
        
    Returns:
        符合 OpenAI Chat Completions API 规范的消息数组
        每个消息包含 role 和 content 字段
        支持 system、user、assistant、tool 四种角色
    """
    messages = []
    
    # 如果有消息历史，使用历史记录
    if message_history:
        messages = message_history.copy()
    else:
        # 否则创建新的消息数组
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
    
    return messages


def _parse_api_response(response: Any) -> tuple[str, Optional[List[Any]]]:
    """
    解析 DeepSeek API 响应，提取文本内容和工具调用。
    
    Args:
        response: OpenAI ChatCompletion 响应对象
        
    Returns:
        (content, tool_calls) 元组
        - content: 生成的文本内容（可能为空字符串）
        - tool_calls: 工具调用列表（如果没有则为 None）
        
    Raises:
        ValueError: 响应格式不正确或为空
    """
    # 检查响应是否有效
    if not response or not hasattr(response, "choices") or not response.choices:
        raise ValueError("API 响应为空或格式不正确")
    
    # 提取第一个选择的消息
    message = response.choices[0].message
    
    # 提取文本内容
    content = message.content or ""
    
    # 提取工具调用（如果有）
    tool_calls = None
    if hasattr(message, "tool_calls") and message.tool_calls:
        tool_calls = message.tool_calls
    
    return content, tool_calls


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
        
    Raises:
        AuthenticationError: 认证失败（401）
        RateLimitError: 速率限制（429）
        APITimeoutError: 请求超时
        APIConnectionError: 网络连接错误
        Exception: 其他 API 错误
    """
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    if tools:
        kwargs["tools"] = tools
    
    # 调用 API，让异常向上传播以便统一处理
    response = client.chat.completions.create(**kwargs)
    return response


def run_agent_loop(
    user_message: str,
    api_key: str,
    model: str = "deepseek-chat",
    max_tool_rounds: int = 5,
    temperature: float = 0.7,
    max_tokens: int = 2000,
) -> str:
    """
    与 DeepSeek 多轮对话：用户输入 -> LLM 可能返回工具调用 -> 执行工具 -> 结果回传 -> 直至最终回复。
    
    Args:
        user_message: 用户输入的消息
        api_key: DeepSeek API 密钥
        model: 模型名称，默认 "deepseek-chat"
        max_tool_rounds: 最大工具调用轮数
        temperature: 生成温度，默认 0.7
        max_tokens: 最大 token 数，默认 2000
        
    Returns:
        最终的自然语言回复
    """
    # 检查 OpenAI 库
    if not HAS_OPENAI:
        return "❌ 错误: 请安装 OpenAI 库: pip install openai"
    
    # 检查 API 密钥
    is_valid, error_msg = _check_api_key(api_key)
    if not is_valid:
        return error_msg
    
    # 初始化客户端
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
    except Exception as e:
        return _format_error_message(e, context="客户端初始化")
    
    # 转换工具定义
    openai_tools = _convert_tools_to_openai_format(TOOL_DEFS)
    
    # 初始化消息历史（注入 workspace 上下文）
    system_prompt = _build_system_prompt_with_workspace()
    messages = _construct_messages(system_prompt, user_message)
    
    # 多轮对话循环
    for round_num in range(max_tool_rounds):
        try:
            # 调用 API
            response = _invoke_deepseek(
                client=client,
                model=model,
                messages=messages,
                tools=openai_tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # 解析响应
            try:
                content, tool_calls = _parse_api_response(response)
            except ValueError as e:
                return f"❌ 响应解析错误: {e}"
            
            # 构造 assistant 消息
            assistant_message = {"role": "assistant", "content": content}
            
            # 检查是否有工具调用
            if tool_calls:
                # 添加工具调用信息到消息
                assistant_message["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in tool_calls
                ]
                messages.append(assistant_message)
                
                # 执行所有工具调用
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    tool_args_str = tool_call.function.arguments
                    
                    # 解析参数
                    try:
                        tool_args = json.loads(tool_args_str) if tool_args_str else {}
                    except json.JSONDecodeError:
                        tool_args = {}
                    
                    # 执行工具
                    tool_result = run_tool(tool_name, **tool_args)
                    
                    # 将工具结果添加到消息历史
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result, ensure_ascii=False)
                    })
            else:
                # 没有工具调用，返回最终回复
                messages.append(assistant_message)
                return content or "未得到有效回复"
        
        except (AuthenticationError, RateLimitError, APITimeoutError, APIConnectionError) as e:
            # 捕获特定的 OpenAI 异常
            return _format_error_message(e, context=f"第 {round_num + 1} 轮对话")
        
        except Exception as e:
            # 捕获其他异常
            return _format_error_message(e, context=f"第 {round_num + 1} 轮对话")
    
    # 达到最大轮数限制
    return "已达到最大工具调用轮数限制，对话结束。"


def run_agent_single(
    user_message: str,
    api_key: str,
    model: str = "deepseek-chat",
    temperature: float = 0.7,
    max_tokens: int = 2000,
) -> str:
    """
    单轮对话：仅让 DeepSeek 根据当前上下文回复（不执行工具）。
    
    Args:
        user_message: 用户输入的消息
        api_key: DeepSeek API 密钥
        model: 模型名称
        temperature: 生成温度，默认 0.7
        max_tokens: 最大 token 数，默认 2000
        
    Returns:
        LLM 生成的回复
    """
    # 检查 OpenAI 库
    if not HAS_OPENAI:
        return "❌ 错误: 请安装 OpenAI 库: pip install openai"
    
    # 检查 API 密钥
    is_valid, error_msg = _check_api_key(api_key)
    if not is_valid:
        return error_msg
    
    # 初始化客户端
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
    except Exception as e:
        return _format_error_message(e, context="客户端初始化")
    
    # 构造消息（注入 workspace 上下文）
    system_prompt = _build_system_prompt_with_workspace()
    messages = _construct_messages(system_prompt, user_message)
    
    # 调用 API（不传递工具）
    try:
        response = _invoke_deepseek(
            client=client,
            model=model,
            messages=messages,
            tools=None,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # 解析响应
        try:
            content, _ = _parse_api_response(response)
            return content or "未得到有效回复"
        except ValueError as e:
            return f"❌ 响应解析错误: {e}"
    
    except (AuthenticationError, RateLimitError, APITimeoutError, APIConnectionError) as e:
        # 捕获特定的 OpenAI 异常
        return _format_error_message(e, context="API 调用")
    
    except Exception as e:
        # 捕获其他异常
        return _format_error_message(e, context="API 调用")
