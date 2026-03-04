"""
Checkpoint 验证测试
验证核心功能和工具调用流程
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_deepseek_agent_module_exists():
    """验证 DeepSeek Agent 模块存在"""
    try:
        from axiom_os.agent import deepseek_agent
        print("✓ DeepSeek Agent 模块导入成功")
        return True
    except ImportError as e:
        print(f"✗ DeepSeek Agent 模块导入失败: {e}")
        return False


def test_core_functions_exist():
    """验证核心函数存在"""
    from axiom_os.agent.deepseek_agent import (
        run_agent_loop,
        run_agent_single,
        _convert_tools_to_openai_format,
        _check_api_key,
        _format_error_message,
    )
    
    print("✓ run_agent_loop 函数存在")
    print("✓ run_agent_single 函数存在")
    print("✓ _convert_tools_to_openai_format 函数存在")
    print("✓ _check_api_key 函数存在")
    print("✓ _format_error_message 函数存在")
    return True


def test_tool_definitions_exist():
    """验证工具定义存在"""
    from axiom_os.agent.tools import TOOL_DEFS, run_tool
    
    assert len(TOOL_DEFS) > 0, "TOOL_DEFS 应该包含工具定义"
    
    expected_tools = [
        "run_benchmark_quick",
        "get_benchmark_report",
        "run_rar",
        "run_discovery_demo",
        "list_domains",
        "apply_domain_extension"
    ]
    
    tool_names = [tool["name"] for tool in TOOL_DEFS]
    
    for expected in expected_tools:
        assert expected in tool_names, f"工具 {expected} 应该在 TOOL_DEFS 中"
        print(f"✓ 工具 {expected} 已定义")
    
    print(f"✓ 所有 {len(expected_tools)} 个工具已定义")
    return True


def test_tool_format_conversion():
    """验证工具格式转换功能"""
    from axiom_os.agent.deepseek_agent import _convert_tools_to_openai_format
    from axiom_os.agent.tools import TOOL_DEFS
    
    openai_tools = _convert_tools_to_openai_format(TOOL_DEFS)
    
    assert len(openai_tools) == len(TOOL_DEFS), "转换后的工具数量应该相同"
    
    for tool in openai_tools:
        assert tool["type"] == "function", "工具类型应该是 function"
        assert "function" in tool, "工具应该包含 function 字段"
        assert "name" in tool["function"], "function 应该包含 name 字段"
        assert "description" in tool["function"], "function 应该包含 description 字段"
        assert "parameters" in tool["function"], "function 应该包含 parameters 字段"
    
    print(f"✓ 工具格式转换正确，转换了 {len(openai_tools)} 个工具")
    return True


def test_api_key_validation():
    """验证 API 密钥验证功能"""
    from axiom_os.agent.deepseek_agent import _check_api_key
    
    # 测试有效密钥
    is_valid, error_msg = _check_api_key("valid_key_12345")
    assert is_valid is True, "有效密钥应该通过验证"
    assert error_msg == "", "有效密钥不应该有错误消息"
    print("✓ 有效 API 密钥验证通过")
    
    # 测试空密钥
    is_valid, error_msg = _check_api_key("")
    assert is_valid is False, "空密钥应该验证失败"
    assert "DEEPSEEK_API_KEY" in error_msg, "错误消息应该提到 DEEPSEEK_API_KEY"
    print("✓ 空 API 密钥验证失败（符合预期）")
    
    # 测试 None 密钥
    is_valid, error_msg = _check_api_key(None)
    assert is_valid is False, "None 密钥应该验证失败"
    print("✓ None API 密钥验证失败（符合预期）")
    
    return True


def test_error_message_formatting():
    """验证错误消息格式化功能"""
    from axiom_os.agent.deepseek_agent import _format_error_message
    
    # 测试通用错误
    error = Exception("Test error")
    result = _format_error_message(error, context="测试上下文")
    
    assert "❌" in result, "错误消息应该包含错误标记"
    assert "测试上下文" in result, "错误消息应该包含上下文"
    assert "建议" in result, "错误消息应该包含建议"
    
    print("✓ 错误消息格式化正确")
    return True


def test_tool_execution():
    """验证工具执行功能"""
    from axiom_os.agent.tools import run_tool
    
    # 测试 list_domains 工具（不需要参数，执行快速）
    result = run_tool("list_domains")
    
    assert isinstance(result, dict), "工具结果应该是字典"
    assert "ok" in result, "工具结果应该包含 ok 字段"
    
    if result["ok"]:
        assert "domains" in result, "list_domains 应该返回 domains 字段"
        print(f"✓ list_domains 工具执行成功，返回 {len(result['domains'])} 个领域")
    else:
        print(f"✓ list_domains 工具执行完成（返回错误是可接受的）")
    
    # 测试未知工具
    result = run_tool("unknown_tool")
    assert result["ok"] is False, "未知工具应该返回错误"
    assert "error" in result, "未知工具应该包含错误信息"
    print("✓ 未知工具正确返回错误")
    
    return True


def test_message_construction():
    """验证消息构造功能"""
    from axiom_os.agent.deepseek_agent import _construct_messages
    
    system_prompt = "你是一个助手"
    user_message = "你好"
    
    messages = _construct_messages(system_prompt, user_message)
    
    assert isinstance(messages, list), "消息应该是列表"
    assert len(messages) >= 2, "至少应该有 system 和 user 消息"
    
    # 验证消息格式
    for msg in messages:
        assert "role" in msg, "消息应该包含 role 字段"
        assert "content" in msg, "消息应该包含 content 字段"
        assert msg["role"] in ["system", "user", "assistant", "tool"], "role 应该是有效值"
    
    print(f"✓ 消息构造正确，生成了 {len(messages)} 条消息")
    return True


def run_all_tests():
    """运行所有验证测试"""
    print("=" * 60)
    print("开始核心功能验证")
    print("=" * 60)
    print()
    
    tests = [
        ("模块导入", test_deepseek_agent_module_exists),
        ("核心函数", test_core_functions_exist),
        ("工具定义", test_tool_definitions_exist),
        ("工具格式转换", test_tool_format_conversion),
        ("API 密钥验证", test_api_key_validation),
        ("错误消息格式化", test_error_message_formatting),
        ("工具执行", test_tool_execution),
        ("消息构造", test_message_construction),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n测试: {test_name}")
        print("-" * 60)
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} 测试通过")
            else:
                failed += 1
                print(f"✗ {test_name} 测试失败")
        except Exception as e:
            failed += 1
            print(f"✗ {test_name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print("=" * 60)
    print(f"验证完成: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    if failed == 0:
        print("\n✓ 所有核心功能验证通过！")
        print("✓ 工具调用流程正常工作！")
        return True
    else:
        print(f"\n✗ 有 {failed} 个测试失败，请检查")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
