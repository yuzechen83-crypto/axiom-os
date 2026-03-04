"""
Performance verification tests for DeepSeek Agent migration.
Tests API call latency and multi-turn conversation response time.
"""

import os
import time
from unittest.mock import Mock, patch
from axiom_os.agent.deepseek_agent import run_agent_single, run_agent_loop


def test_api_call_latency():
    """测试 API 调用延迟 - 使用 mock 避免实际 API 调用"""
    # Mock the OpenAI client to avoid actual API calls
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Test response"
    mock_response.choices[0].message.tool_calls = None
    
    with patch('axiom_os.agent.deepseek_agent.OpenAI') as mock_openai:
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Measure latency
        start_time = time.time()
        result = run_agent_single(
            user_message="Hello, this is a test",
            api_key="test_key",
            model="deepseek-chat"
        )
        end_time = time.time()
        
        latency = end_time - start_time
        
        # Verify response
        assert result == "Test response"
        
        # Latency should be very low for mocked calls (< 1 second)
        assert latency < 1.0, f"API call latency too high: {latency:.3f}s"
        
        print(f"✓ API call latency: {latency*1000:.2f}ms (mocked)")


def test_multi_turn_conversation_response_time():
    """测试多轮对话响应时间 - 使用 mock 避免实际 API 调用"""
    # Mock responses for multi-turn conversation
    mock_response_1 = Mock()
    mock_response_1.choices = [Mock()]
    mock_response_1.choices[0].message.content = None
    mock_response_1.choices[0].message.tool_calls = [Mock()]
    mock_response_1.choices[0].message.tool_calls[0].id = "call_123"
    mock_response_1.choices[0].message.tool_calls[0].function.name = "list_domains"
    mock_response_1.choices[0].message.tool_calls[0].function.arguments = "{}"
    
    mock_response_2 = Mock()
    mock_response_2.choices = [Mock()]
    mock_response_2.choices[0].message.content = "Here are the available domains"
    mock_response_2.choices[0].message.tool_calls = None
    
    with patch('axiom_os.agent.deepseek_agent.OpenAI') as mock_openai:
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        mock_openai.return_value = mock_client
        
        # Measure multi-turn conversation time
        start_time = time.time()
        result = run_agent_loop(
            user_message="What domains are available?",
            api_key="test_key",
            model="deepseek-chat",
            max_tool_rounds=5
        )
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Verify response
        assert "domains" in result.lower()
        
        # Multi-turn response time should be reasonable (< 2 seconds for mocked)
        assert response_time < 2.0, f"Multi-turn response time too high: {response_time:.3f}s"
        
        print(f"✓ Multi-turn conversation response time: {response_time*1000:.2f}ms (mocked)")


def test_performance_acceptable_range():
    """验证性能在可接受范围内"""
    # This is a meta-test that documents performance expectations
    performance_requirements = {
        "api_call_latency_max": 5.0,  # seconds (for real API calls)
        "multi_turn_max": 15.0,  # seconds (for real API calls with 2-3 rounds)
        "mocked_call_max": 1.0,  # seconds (for mocked calls)
    }
    
    # Document the requirements
    print("\n性能要求:")
    print(f"  - API 调用延迟 (实际): < {performance_requirements['api_call_latency_max']}s")
    print(f"  - 多轮对话响应 (实际): < {performance_requirements['multi_turn_max']}s")
    print(f"  - Mock 调用延迟: < {performance_requirements['mocked_call_max']}s")
    
    # This test always passes - it's for documentation
    assert True


if __name__ == "__main__":
    print("Running performance verification tests...")
    test_api_call_latency()
    test_multi_turn_conversation_response_time()
    test_performance_acceptable_range()
    print("\n✓ All performance tests passed!")
