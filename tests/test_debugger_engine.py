import logging
import sys
import threading
import pytest
from unittest.mock import Mock, patch, call
from neurotrace.debugger_engine import DebuggerEngine, DebuggerConfig
from neurotrace.runtime_visualizer import TraceEvent
from neurotrace.ollama_ai_adapter import OllamaConfig

@pytest.fixture
def mock_log_interceptor():
    """Mock LogInterceptor."""
    with patch('neurotrace.debugger_engine.LogInterceptor') as mock:
        instance = mock.return_value
        instance.get_logs.return_value = ["Test log 1", "Test log 2"]
        yield instance

@pytest.fixture
def mock_ai_adapter():
    """Mock OllamaAIAdapter."""
    with patch('neurotrace.debugger_engine.OllamaAIAdapter') as mock:
        instance = mock.return_value
        instance.get_ai_analysis.return_value = {
            "success": True,
            "error": None,
            "analysis": {"response": "\033[32mTest analysis\033[0m"}
        }
        yield instance

@pytest.fixture
def mock_visualizer():
    """Mock RuntimeVisualizer."""
    with patch('neurotrace.debugger_engine.RuntimeVisualizer') as mock:
        instance = mock.return_value
        instance.generate_diagram.return_value = "test_diagram.png"
        yield instance

@pytest.fixture
def debugger(mock_log_interceptor, mock_ai_adapter, mock_visualizer):
    """Create a DebuggerEngine instance with mocked dependencies."""
    return DebuggerEngine()

def test_initialization():
    """Test debugger initialization with different configurations."""
    # Default initialization
    debugger = DebuggerEngine()
    assert debugger.config.ai_model == "phi4"
    assert debugger.config.enable_visualizer is True
    assert debugger.config.live_tracing is False
    assert debugger.config.line_level_tracing is False
    assert debugger.config.verbose_mode is False
    
    # Custom configuration with all features enabled
    config = DebuggerConfig(
        enable_visualizer=True,
        ai_model="codellama",
        log_level="DEBUG",
        live_tracing=True,
        line_level_tracing=True,
        verbose_mode=True,
        max_log_size=5000,
        ai_timeout=30,
        diagram_format="svg"
    )
    debugger = DebuggerEngine(config=config)
    assert debugger.config.ai_model == "codellama"
    assert debugger.config.live_tracing is True
    assert debugger.config.line_level_tracing is True
    assert debugger.config.verbose_mode is True
    assert debugger.config.max_log_size == 5000
    assert debugger.visualizer is not None

def test_start_stop_debugging(debugger, mock_log_interceptor):
    """Test starting and stopping debugging sessions."""
    original_tracer = sys.gettrace()
    
    # Start debugging with live tracing
    debugger.config.live_tracing = True
    debugger.start_debugging()
    mock_log_interceptor.start.assert_called_once()
    assert debugger._is_debugging is True
    assert sys.gettrace() == debugger._trace_callback
    
    # Try starting again (should warn but not error)
    debugger.start_debugging()
    assert mock_log_interceptor.start.call_count == 1
    
    # Stop debugging
    debugger.stop_debugging()
    mock_log_interceptor.stop.assert_called_once()
    assert debugger._is_debugging is False
    assert sys.gettrace() == original_tracer
    
    # Try stopping again (should warn but not error)
    debugger.stop_debugging()
    assert mock_log_interceptor.stop.call_count == 1

def test_analyze_logs_success(debugger, mock_log_interceptor, mock_ai_adapter):
    """Test successful log analysis."""
    debugger.start_debugging()
    
    # Test with verbose mode
    debugger.config.verbose_mode = True
    result = debugger.analyze_logs(system_prompt="Analyze these logs")
    assert result["success"] is True
    mock_log_interceptor.get_logs.assert_called_once()
    mock_ai_adapter.get_ai_analysis.assert_called_once()
    
    # Test with custom model
    debugger.config.ai_model = "codellama"
    result = debugger.analyze_logs()
    assert result["success"] is True
    assert mock_ai_adapter.get_ai_analysis.call_args[1]["model"] == "codellama"

def test_analyze_logs_no_session(debugger):
    """Test analyzing logs without an active session."""
    result = debugger.analyze_logs()
    assert result["success"] is False
    assert "No active debugging session" in result["error"]

def test_analyze_logs_no_data(debugger, mock_log_interceptor):
    """Test analyzing logs when no logs are captured."""
    mock_log_interceptor.get_logs.return_value = []
    debugger.start_debugging()
    result = debugger.analyze_logs()
    
    assert result["success"] is False
    assert "No logs captured" in result["error"]

def test_analyze_logs_ai_failure(debugger, mock_ai_adapter):
    """Test handling AI analysis failure."""
    mock_ai_adapter.get_ai_analysis.side_effect = Exception("AI server error")
    debugger.start_debugging()
    result = debugger.analyze_logs()
    
    assert result["success"] is False
    assert "Analysis failed" in result["error"]
    assert "AI server error" in result["error"]

def test_visualization_disabled():
    """Test behavior when visualization is disabled."""
    config = DebuggerConfig(enable_visualizer=False)
    debugger = DebuggerEngine(config=config)
    
    debugger.start_debugging()
    debugger.add_trace_event(TraceEvent("call", "test", "module", 1))
    result = debugger.generate_visual()
    
    assert result is None

def test_visualization_success(debugger, mock_visualizer):
    """Test successful visualization generation."""
    debugger.start_debugging()
    
    # Test with line-level tracing
    debugger.config.line_level_tracing = True
    events = [
        TraceEvent("call", "main", "test_module", 1),
        TraceEvent("line", "main", "test_module", 2),
        TraceEvent("call", "helper", "test_module", 5, "test_module.main"),
        TraceEvent("line", "helper", "test_module", 6),
        TraceEvent("return", "helper", "test_module", 7),
        TraceEvent("return", "main", "test_module", 3)
    ]
    for event in events:
        debugger.add_trace_event(event)
    
    result = debugger.generate_visual("test_output")
    assert result == "test_diagram.png"
    mock_visualizer.generate_diagram.assert_called_once()
    
    # Verify all events were captured
    assert len(debugger._trace_data) == 6

def test_context_manager(debugger, mock_log_interceptor):
    """Test using the debugger as a context manager."""
    with debugger:
        assert debugger._is_debugging is True
        mock_log_interceptor.start.assert_called_once()
    
    assert debugger._is_debugging is False
    mock_log_interceptor.stop.assert_called_once()

def test_clear_logs(debugger, mock_log_interceptor):
    """Test clearing logs and trace data."""
    debugger.start_debugging()
    debugger.add_trace_event(TraceEvent("call", "test", "module", 1))
    
    debugger.clear_logs()
    mock_log_interceptor.clear.assert_called_once()
    assert len(debugger._trace_data) == 0

def test_error_handling(mock_log_interceptor):
    """Test error handling during start/stop."""
    mock_log_interceptor.start.side_effect = Exception("Start error")
    mock_log_interceptor.stop.side_effect = Exception("Stop error")
    
    config = DebuggerConfig(
        enable_visualizer=False,  # Disable visualizer to simplify test
        live_tracing=True
    )
    debugger = DebuggerEngine(config=config)
    original_tracer = sys.gettrace()
    
    # Test start error
    with pytest.raises(Exception, match="Start error"):
        debugger.start_debugging()
    assert debugger._is_debugging is False
    assert sys.gettrace() == original_tracer
    
    # Force debugging state to test stop error
    debugger._is_debugging = True
    sys.settrace(debugger._trace_callback)
    with pytest.raises(Exception, match="Stop error"):
        debugger.stop_debugging()
    # Should restore original tracer even after error
    assert sys.gettrace() == original_tracer

def test_line_level_tracing(debugger):
    """Test line-level tracing functionality."""
    debugger.config.live_tracing = True
    debugger.config.line_level_tracing = True
    
    def test_function():
        x = 1  # Line event
        y = 2  # Line event
        return x + y  # Line event
    
    with debugger:
        test_function()
    
    # Should capture call, multiple line events, and return
    events = debugger._trace_data
    assert len(events) > 3  # At least call + 3 lines + return
    
    # Verify event types
    assert events[0].event_type == "call"
    assert any(e.event_type == "line" for e in events)
    assert events[-1].event_type == "return"

def test_user_code_filtering(debugger):
    """Test that only user code is traced."""
    debugger.config.live_tracing = True
    debugger._script_path = "/test/script.py"
    
    # Should be ignored (stdlib)
    assert not debugger._is_user_code(Mock(
        f_code=Mock(co_filename="/usr/lib/python3.8/json/__init__.py")
    ))
    
    # Should be ignored (site-packages)
    assert not debugger._is_user_code(Mock(
        f_code=Mock(co_filename="/usr/local/lib/python3.8/site-packages/requests/api.py")
    ))
    
    # Should be ignored (neurotrace)
    assert not debugger._is_user_code(Mock(
        f_code=Mock(co_filename="/path/to/neurotrace/debugger_engine.py")
    ))
    
    # Should be traced (user script directory)
    assert debugger._is_user_code(Mock(
        f_code=Mock(co_filename="/test/utils.py")
    ))

def test_trace_callback_error_handling(debugger):
    """Test error handling in trace callback."""
    debugger.config.live_tracing = True
    debugger.config.verbose_mode = True
    
    def problematic_function():
        raise ValueError("Test error")
    
    with debugger:
        try:
            problematic_function()
        except ValueError:
            pass
    
    # Should have captured the error event
    error_events = [e for e in debugger._trace_data if e.event_type == "error"]
    assert len(error_events) == 1
    assert error_events[0].function_name == "problematic_function"

def test_trace_callback_threading(debugger):
    """Test trace callback with multiple threads."""
    debugger.config.live_tracing = True
    
    def worker_function():
        return "worker result"
    
    with debugger:
        threads = [
            threading.Thread(target=worker_function)
            for _ in range(3)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    
    # Should have captured events from all threads
    call_events = [e for e in debugger._trace_data if e.event_type == "call"]
    assert len(call_events) >= 3  # At least one call per thread
