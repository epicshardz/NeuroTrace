"""Tests for live tracing functionality."""
import sys
from unittest.mock import Mock, patch
import pytest
from neurotrace.debugger_engine import DebuggerEngine, DebuggerConfig

def test_function():
    """Test function for tracing."""
    return "test"

def recursive_function(n):
    """Recursive test function."""
    if n <= 0:
        return
    recursive_function(n - 1)

@pytest.fixture
def debugger():
    """Create a DebuggerEngine instance with live tracing enabled."""
    config = DebuggerConfig(
        live_tracing=True,
        line_level_tracing=False
    )
    return DebuggerEngine(config=config)

def test_live_tracing_setup(debugger):
    """Test that live tracing is properly set up."""
    original_tracer = sys.gettrace()
    
    debugger.start_debugging()
    assert sys.gettrace() == debugger._trace_callback
    
    debugger.stop_debugging()
    assert sys.gettrace() == original_tracer

def test_function_call_tracing(debugger):
    """Test that function calls are traced."""
    with debugger:
        test_function()
    
    # Should have at least two events: call and return
    assert len(debugger._trace_data) >= 2
    
    # Verify call event
    call_event = next(e for e in debugger._trace_data if e.event_type == "call")
    assert call_event.function_name == "test_function"
    assert call_event.module_name == "__main__"
    
    # Verify return event
    return_event = next(e for e in debugger._trace_data if e.event_type == "return")
    assert return_event.function_name == "test_function"
    assert return_event.module_name == "__main__"

def test_recursive_function_tracing(debugger):
    """Test tracing of recursive function calls."""
    with debugger:
        recursive_function(2)
    
    # Should have events for each recursive call
    call_events = [e for e in debugger._trace_data if e.event_type == "call"]
    assert len(call_events) == 3  # Initial call + 2 recursive calls
    
    # Verify call chain
    for event in call_events[1:]:  # Skip first call
        assert event.caller and "recursive_function" in event.caller

def test_line_level_tracing():
    """Test line-level tracing when enabled."""
    config = DebuggerConfig(
        live_tracing=True,
        line_level_tracing=True
    )
    debugger = DebuggerEngine(config=config)
    
    with debugger:
        test_function()
    
    # Should have line events in addition to call/return
    line_events = [e for e in debugger._trace_data if e.event_type == "line"]
    assert len(line_events) > 0

def test_tracing_disabled():
    """Test that no tracing occurs when live_tracing is disabled."""
    config = DebuggerConfig(live_tracing=False)
    debugger = DebuggerEngine(config=config)
    
    original_tracer = sys.gettrace()
    
    with debugger:
        test_function()
    
    assert len(debugger._trace_data) == 0
    assert sys.gettrace() == original_tracer

def test_error_handling(debugger):
    """Test error handling in trace callback."""
    def problematic_function():
        raise ValueError("Test error")
    
    with debugger:
        try:
            problematic_function()
        except ValueError:
            pass
    
    # Should have captured the call despite the error
    assert any(
        e.event_type == "call" and e.function_name == "problematic_function"
        for e in debugger._trace_data
    )

def test_restore_tracer_after_error(debugger):
    """Test that the original tracer is restored even after errors."""
    original_tracer = sys.gettrace()
    
    try:
        with debugger:
            raise ValueError("Test error")
    except ValueError:
        pass
    
    assert sys.gettrace() == original_tracer

def test_concurrent_tracing():
    """Test that tracing works with concurrent operations."""
    import threading
    import time
    
    config = DebuggerConfig(live_tracing=True)
    debugger = DebuggerEngine(config=config)
    
    def worker():
        time.sleep(0.1)
        test_function()
    
    with debugger:
        threads = [
            threading.Thread(target=worker)
            for _ in range(3)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    
    # Should have captured events from all threads
    call_events = [e for e in debugger._trace_data if e.event_type == "call"]
    assert len(call_events) >= 3  # At least one call per thread
