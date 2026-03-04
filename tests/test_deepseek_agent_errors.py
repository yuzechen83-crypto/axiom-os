"""
Unit tests for DeepSeek Agent error handling
测试错误处理机制的正确性
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock

# 检查是否安装了 openai
try:
    from openai import OpenAI, AuthenticationError, RateLimitError, APITimeoutError, APIConnectionError
    from openai._models import FinalRequestOptions
    from httpx import Response, Request
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    pytest.skip("openai not installed", allow_module_level=True)

from axiom_os.agent.deepseek_agent import (
    _format_error_message,
    _check_api_key,
    run_agent_loop,
    run_agent_single
)


def create_mock_response(status_code=401):
    """创建 mock HTTP 响应对象"""
    mock_request = Mock(spec=Request)
    mock_request.url = "https://api.deepseek.com/v1/chat/completions"
    mock_request.method = "POST"
    
    mock_response = Mock(spec=Response)
    mock_response.status_code = status_code
    mock_response.request = mock_request
    mock_response.headers = {}
    return mock_response


class TestErrorMessageFormatting:
    """测试错误消息格式化功能"""
    
    def test_authentication_error_formatting(self):
        """测试认证错误的格式化"""
        mock_response = create_mock_response(401)
        error = AuthenticationError(
            "Invalid API key",
            response=mock_response,
            body={"error": {"message": "Invalid API key"}}
        )
        result = _format_error_message(error)
        
        assert "❌ 认证失败" in result
        assert "DEEPSEEK_API_KEY" in result
        assert "建议" in result
    
    def test_rate_limit_error_formatting(self):
        """测试速率限制错误的格式化"""
        mock_response = create_mock_response(429)
        error = RateLimitError(
            "Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded"}}
        )
        result = _format_error_message(error)
        
        assert "❌ API 速率限制" in result
        assert "建议" in result
    
    def test_timeout_error_formatting(self):
        """测试超时错误的格式化"""
        error = APITimeoutError(request=Mock())
        result = _format_error_message(error)
        
        assert "❌ API 调用超时" in result
        assert "建议" in result
    
    def test_connection_error_formatting(self):
        """测试连接错误的格式化"""
        error = APIConnectionError(message="Connection failed", request=Mock())
        result = _format_error_message(error)
        
        assert "❌ 无法连接到 DeepSeek API" in result
        assert "建议" in result
    
    def test_generic_error_formatting(self):
        """测试通用错误的格式化"""
        error = Exception("Unknown error")
        result = _format_error_message(error, context="测试上下文")
        
        assert "❌ API 调用错误" in result
        assert "测试上下文" in result
        assert "建议" in result
        assert "Unknown error" in result


class TestAPIKeyValidation:
    """测试 API 密钥验证功能"""
    
    def test_valid_api_key(self):
        """测试有效的 API 密钥"""
        is_valid, error_msg = _check_api_key("valid_key_12345")
        
        assert is_valid is True
        assert error_msg == ""
    
    def test_empty_api_key(self):
        """测试空 API 密钥"""
        is_valid, error_msg = _check_api_key("")
        
        assert is_valid is False
        assert "❌ API 密钥缺失" in error_msg
        assert "DEEPSEEK_API_KEY" in error_msg
        assert "建议" in error_msg
    
    def test_whitespace_api_key(self):
        """测试仅包含空格的 API 密钥"""
        is_valid, error_msg = _check_api_key("   ")
        
        assert is_valid is False
        assert "❌ API 密钥缺失" in error_msg
    
    def test_none_api_key(self):
        """测试 None API 密钥"""
        is_valid, error_msg = _check_api_key(None)
        
        assert is_valid is False
        assert "❌ API 密钥缺失" in error_msg


class TestAgentErrorHandling:
    """测试 Agent 函数的错误处理"""
    
    @patch('axiom_os.agent.deepseek_agent.OpenAI')
    def test_run_agent_loop_authentication_error(self, mock_openai_class):
        """测试 run_agent_loop 处理认证错误"""
        # Mock 客户端抛出认证错误
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = create_mock_response(401)
        error = AuthenticationError(
            "Invalid API key",
            response=mock_response,
            body={"error": {"message": "Invalid API key"}}
        )
        mock_client.chat.completions.create.side_effect = error
        
        result = run_agent_loop(
            user_message="Hello",
            api_key="invalid_key",
            model="deepseek-chat"
        )
        
        assert "❌ 认证失败" in result
        assert "DEEPSEEK_API_KEY" in result
    
    @patch('axiom_os.agent.deepseek_agent.OpenAI')
    def test_run_agent_loop_rate_limit_error(self, mock_openai_class):
        """测试 run_agent_loop 处理速率限制错误"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = create_mock_response(429)
        error = RateLimitError(
            "Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded"}}
        )
        mock_client.chat.completions.create.side_effect = error
        
        result = run_agent_loop(
            user_message="Hello",
            api_key="valid_key",
            model="deepseek-chat"
        )
        
        assert "❌ API 速率限制" in result
    
    @patch('axiom_os.agent.deepseek_agent.OpenAI')
    def test_run_agent_loop_timeout_error(self, mock_openai_class):
        """测试 run_agent_loop 处理超时错误"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        error = APITimeoutError(request=Mock())
        mock_client.chat.completions.create.side_effect = error
        
        result = run_agent_loop(
            user_message="Hello",
            api_key="valid_key",
            model="deepseek-chat"
        )
        
        assert "❌ API 调用超时" in result
    
    @patch('axiom_os.agent.deepseek_agent.OpenAI')
    def test_run_agent_loop_connection_error(self, mock_openai_class):
        """测试 run_agent_loop 处理连接错误"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        error = APIConnectionError(message="Connection failed", request=Mock())
        mock_client.chat.completions.create.side_effect = error
        
        result = run_agent_loop(
            user_message="Hello",
            api_key="valid_key",
            model="deepseek-chat"
        )
        
        assert "❌ 无法连接到 DeepSeek API" in result
    
    @patch('axiom_os.agent.deepseek_agent.OpenAI')
    def test_run_agent_single_authentication_error(self, mock_openai_class):
        """测试 run_agent_single 处理认证错误"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = create_mock_response(401)
        error = AuthenticationError(
            "Invalid API key",
            response=mock_response,
            body={"error": {"message": "Invalid API key"}}
        )
        mock_client.chat.completions.create.side_effect = error
        
        result = run_agent_single(
            user_message="Hello",
            api_key="invalid_key",
            model="deepseek-chat"
        )
        
        assert "❌ 认证失败" in result
        assert "DEEPSEEK_API_KEY" in result
    
    def test_run_agent_loop_missing_api_key(self):
        """测试 run_agent_loop 处理缺失的 API 密钥"""
        result = run_agent_loop(
            user_message="Hello",
            api_key="",
            model="deepseek-chat"
        )
        
        assert "❌ API 密钥缺失" in result
        assert "DEEPSEEK_API_KEY" in result
    
    def test_run_agent_single_missing_api_key(self):
        """测试 run_agent_single 处理缺失的 API 密钥"""
        result = run_agent_single(
            user_message="Hello",
            api_key="",
            model="deepseek-chat"
        )
        
        assert "❌ API 密钥缺失" in result
        assert "DEEPSEEK_API_KEY" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
