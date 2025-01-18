import json
import time
import pytest
from unittest.mock import Mock, patch
import requests
from neurotrace.ollama_ai_adapter import OllamaAIAdapter, OllamaConfig

@pytest.fixture
def mock_session():
    """Create a mock session with configurable response."""
    with patch('requests.Session') as mock:
        session = Mock()
        mock.return_value = session
        yield session

@pytest.fixture
def adapter(mock_session):
    """Create an adapter instance with a mocked session."""
    return OllamaAIAdapter()

@pytest.fixture
def mock_session():
    """Create a mock session with configurable response."""
    with patch('requests.Session') as mock:
        session = Mock()
        mock.return_value = session
        # Configure default streaming response
        response = Mock()
        response.iter_lines.return_value = [
            b'{"response": "Analysis of log data"}'
        ]
        response.raise_for_status.return_value = None
        session.post.return_value = response
        yield session

def test_successful_analysis(adapter, mock_session):
    """Test successful AI analysis request."""
    mock_response = Mock()
    mock_response.iter_lines.return_value = [
        b'{"response": "Analysis of log data"}'
    ]
    mock_response.raise_for_status.return_value = None
    mock_session.post.return_value = mock_response

    result = adapter.get_ai_analysis("Test log data")
    
    assert result["success"] is True
    assert result["error"] is None
    # Response should be colored
    assert "\033[32m" in result["analysis"]["response"]
    assert "Analysis of log data" in result["analysis"]["response"]
    mock_session.post.assert_called_once()

def test_empty_log_data(adapter):
    """Test handling of empty log data."""
    result = adapter.get_ai_analysis("")
    assert result["success"] is False
    assert "Empty log data" in result["error"]
    assert result["analysis"] is None

def test_server_error(adapter, mock_session):
    """Test handling of server errors."""
    mock_session.post.side_effect = requests.exceptions.RequestException("Server error")
    
    result = adapter.get_ai_analysis("Test log data")
    
    assert result["success"] is False
    assert "Failed to get AI analysis" in result["error"]
    assert result["analysis"] is None

def test_timeout_error(adapter, mock_session):
    """Test handling of timeout errors."""
    mock_session.post.side_effect = requests.exceptions.Timeout("Request timed out")
    
    result = adapter.get_ai_analysis("Test log data")
    
    assert result["success"] is False
    assert "Failed to get AI analysis" in result["error"]
    assert result["analysis"] is None

def test_chunking_large_data(adapter, mock_session):
    """Test that large log data is properly chunked."""
    def create_streaming_response(chunk_num):
        mock_response = Mock()
        mock_response.iter_lines.return_value = [
            f'{{"response": "Chunk {chunk_num} analysis"}}'.encode('utf-8')
        ]
        mock_response.raise_for_status.return_value = None
        return mock_response

    mock_session.post.side_effect = [
        create_streaming_response(i) for i in range(3)
    ]

    # Create log data larger than max_chunk_size
    large_log_data = "x" * (adapter.config.max_chunk_size * 2 + 100)
    
    result = adapter.get_ai_analysis(large_log_data)
    
    assert result["success"] is True
    assert result["error"] is None
    assert result["analysis"]["total_chunks"] == 3
    assert len(result["analysis"]["chunks"]) == 3
    assert mock_session.post.call_count == 3
    
    # Verify each chunk was processed
    for i, chunk in enumerate(result["analysis"]["chunks"]):
        assert f"Chunk {i} analysis" in chunk["response"]

def test_streaming_response(adapter, mock_session):
    """Test handling of streaming responses."""
    mock_response = Mock()
    mock_response.iter_lines.return_value = [
        b'{"response": "Part 1"}',
        b'{"response": " of"}',
        b'{"response": " analysis"}'
    ]
    mock_response.raise_for_status.return_value = None
    mock_session.post.return_value = mock_response

    result = adapter.get_ai_analysis("Test data")
    
    assert result["success"] is True
    # Response should be colored and contain all parts
    colored_response = result["analysis"]["response"]
    assert "\033[32m" in colored_response
    assert "Part 1 of analysis" in colored_response.replace("\033[32m", "").replace("\033[0m", "")

def test_retry_mechanism_with_backoff(adapter, mock_session):
    """Test retry mechanism with exponential backoff."""
    success_response = Mock()
    success_response.iter_lines.return_value = [b'{"response": "Success"}']
    success_response.raise_for_status.return_value = None
    
    # First two calls fail, third succeeds
    mock_session.post.side_effect = [
        requests.exceptions.ConnectionError("Network error"),
        requests.exceptions.ConnectionError("Network error"),
        success_response
    ]
    
    start_time = time.time()
    result = adapter.get_ai_analysis("Test data")
    elapsed_time = time.time() - start_time
    
    assert result["success"] is True
    assert mock_session.post.call_count == 3
    # Should have waited at least 1.5 seconds (0.5 + 1.0 for two retries)
    assert elapsed_time >= 1.5

def test_error_response_handling(adapter, mock_session):
    """Test handling of different error responses."""
    # Test malformed JSON response
    mock_response = Mock()
    mock_response.iter_lines.return_value = [b'invalid json']
    mock_response.raise_for_status.return_value = None
    mock_session.post.return_value = mock_response
    
    result = adapter.get_ai_analysis("Test data")
    assert result["success"] is False
    assert "Unexpected error during analysis" in result["error"]
    
    # Test empty response
    mock_response.iter_lines.return_value = []
    result = adapter.get_ai_analysis("Test data")
    assert result["success"] is False
    assert "No response received" in result["error"]
    
    # Test response without required fields
    mock_response.iter_lines.return_value = [b'{"other": "field"}']
    result = adapter.get_ai_analysis("Test data")
    assert result["success"] is False
    assert "Invalid response format" in result["error"]

def test_custom_model_and_system_prompt(adapter, mock_session):
    """Test using custom model and system prompt."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "model": "codellama",
        "response": "Analysis",
        "context": []
    }
    mock_response.raise_for_status.return_value = None
    mock_session.post.return_value = mock_response

    result = adapter.get_ai_analysis(
        "Test log data",
        model="codellama",
        system_prompt="Analyze debug logs"
    )
    
    assert result["success"] is True
    # Verify the correct payload was sent
    call_args = mock_session.post.call_args[1]
    assert call_args["json"]["model"] == "codellama"
    assert call_args["json"]["system"] == "Analyze debug logs"

def test_retry_mechanism(adapter, mock_session):
    """Test that retry mechanism works for temporary failures."""
    # First two calls fail, third succeeds
    mock_response_success = Mock()
    mock_response_success.json.return_value = {
        "model": "phi4",
        "response": "Analysis after retry",
        "context": []
    }
    mock_response_success.raise_for_status.return_value = None
    
    mock_session.post.side_effect = [
        requests.exceptions.RequestException("Temporary error"),
        requests.exceptions.RequestException("Temporary error"),
        mock_response_success
    ]
    
    result = adapter.get_ai_analysis("Test log data")
    
    assert result["success"] is True
    assert result["analysis"]["response"] == "Analysis after retry"
    assert mock_session.post.call_count >= 3

def test_context_manager(mock_session):
    """Test that context manager properly closes session."""
    mock_response = Mock()
    mock_response.json.return_value = {"response": "test"}
    mock_response.raise_for_status.return_value = None
    mock_session.post.return_value = mock_response
    
    with OllamaAIAdapter() as adapter:
        adapter.get_ai_analysis("Test log data")
    
    mock_session.close.assert_called_once()
