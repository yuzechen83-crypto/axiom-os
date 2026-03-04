"""
Property-based tests for DeepSeek Agent
使用 hypothesis 进行属性测试，验证 DeepSeek Agent 的正确性属性
"""

import os
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st

# 检查是否安装了 hypothesis
try:
    import hypothesis
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False
    pytest.skip("hypothesis not installed", allow_module_level=True)

# 检查是否安装了 openai
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    pytest.skip("openai not installed", allow_module_level=True)


# Feature: llm-deepseek-migration, Property 1: API 客户端配置正确性
# Validates: Requirements 1.2, 1.3, 4.1
@given(api_key=st.text(min_size=1, max_size=100))
@settings(max_examples=100, deadline=None)
def test_client_configuration_property(api_key):
    """
    Property 1: API 客户端配置正确性
    
    For any DeepSeek 客户端初始化，客户端的 base_url 应该指向 https://api.deepseek.com，
    并且 API 密钥应该从提供的参数中正确设置。
    
    Validates: Requirements 1.2, 1.3, 4.1
    """
    # 初始化客户端
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )
    
    # 验证 base_url 配置正确
    assert str(client.base_url) == "https://api.deepseek.com/" or str(client.base_url) == "https://api.deepseek.com", \
        f"Expected base_url to be 'https://api.deepseek.com', got '{client.base_url}'"
    
    # 验证 API 密钥正确设置
    assert client.api_key == api_key, \
        f"Expected api_key to be '{api_key}', got '{client.api_key}'"


# Feature: llm-deepseek-migration, Property 1: API 客户端配置正确性
# Validates: Requirements 1.2, 1.3, 4.1
def test_client_configuration_from_env():
    """
    Property 1: API 客户端配置正确性（环境变量场景）
    
    验证从环境变量 DEEPSEEK_API_KEY 读取 API 密钥的场景。
    
    Validates: Requirements 1.2, 1.3, 4.1
    """
    # 保存原始环境变量
    original_key = os.environ.get("DEEPSEEK_API_KEY")
    
    try:
        # 设置测试环境变量
        test_api_key = "test_key_12345"
        os.environ["DEEPSEEK_API_KEY"] = test_api_key
        
        # 从环境变量读取
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        
        # 初始化客户端
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        
        # 验证 base_url 配置正确
        assert str(client.base_url) == "https://api.deepseek.com/" or str(client.base_url) == "https://api.deepseek.com", \
            f"Expected base_url to be 'https://api.deepseek.com', got '{client.base_url}'"
        
        # 验证 API 密钥正确设置
        assert client.api_key == test_api_key, \
            f"Expected api_key to be '{test_api_key}', got '{client.api_key}'"
    
    finally:
        # 恢复原始环境变量
        if original_key is not None:
            os.environ["DEEPSEEK_API_KEY"] = original_key
        elif "DEEPSEEK_API_KEY" in os.environ:
            del os.environ["DEEPSEEK_API_KEY"]


# Feature: llm-deepseek-migration, Property 1: API 客户端配置正确性
# Validates: Requirements 4.2
def test_missing_api_key_error():
    """
    Property 1: API 客户端配置正确性（缺失密钥场景）
    
    验证当 DEEPSEEK_API_KEY 未设置时，系统返回明确的错误提示。
    
    Validates: Requirements 4.2
    """
    from axiom_os.agent.deepseek_agent import run_agent_loop, run_agent_single
    
    # 测试 run_agent_loop
    result = run_agent_loop(
        user_message="Hello",
        api_key="",  # 空密钥
        model="deepseek-chat"
    )
    
    # 验证返回错误消息
    assert "❌" in result, "Expected error message with ❌"
    assert "DEEPSEEK_API_KEY" in result, "Expected error message to mention DEEPSEEK_API_KEY"
    
    # 测试 run_agent_single
    result = run_agent_single(
        user_message="Hello",
        api_key="",  # 空密钥
        model="deepseek-chat"
    )
    
    # 验证返回错误消息
    assert "❌" in result, "Expected error message with ❌"
    assert "DEEPSEEK_API_KEY" in result, "Expected error message to mention DEEPSEEK_API_KEY"


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
