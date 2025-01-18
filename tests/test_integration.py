"""Integration tests for NeuroTrace."""
import os
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
from typer.testing import CliRunner

from neurotrace.cli.cli import app
from neurotrace.debugger_engine import DebuggerEngine
from neurotrace.ollama_ai_adapter import OllamaAIAdapter

@pytest.fixture
def mock_ollama():
    """Mock Ollama AI server responses."""
    with patch('neurotrace.ollama_ai_adapter.OllamaAIAdapter') as mock:
        instance = mock.return_value
        instance.get_ai_analysis.return_value = {
            "success": True,
            "error": None,
            "analysis": {"response": "Test analysis of the error"}
        }
        yield instance

@pytest.fixture
def test_script():
    """Create a temporary test script."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('''
def recursive_function(n):
    if n <= 0:
        return
    print(f"Level {n}")
    recursive_function(n - 1)
    if n == 2:
        raise ValueError("Test error at level 2")

if __name__ == "__main__":
    recursive_function(3)
''')
        return Path(f.name)

@pytest.fixture
def runner():
    """Create a CLI runner."""
    return CliRunner()

def test_ollama_real_integration():
    """Test real Ollama integration without mocks."""
    print("\nStarting real Ollama integration test...")
    start_time = time.time()
    
    adapter = OllamaAIAdapter()
    test_log = "Test log message for analysis"
    
    print("Sending request to Ollama...")
    result = adapter.get_ai_analysis(test_log, model="phi4")
    end_time = time.time()
    
    print(f"Ollama request completed in {end_time - start_time:.2f} seconds")
    print(f"Success: {result['success']}")
    if not result['success']:
        print(f"Error: {result['error']}")
    else:
        print(f"Response length: {len(str(result['analysis']))}")
    
    assert result["success"] is True
    assert result["error"] is None
    assert isinstance(result["analysis"], dict)
    assert "response" in result["analysis"]

def test_end_to_end_flow(runner, test_script, mock_ollama, tmp_path):
    """Test complete workflow: init -> run -> diagram."""
    # 1. Initialize configuration
    config_path = tmp_path / ".neurotrace/config.json"
    result = runner.invoke(app, ["init", "--config-path", str(config_path)])
    assert result.exit_code == 0
    assert config_path.exists()

    # 2. Run script with error (should trigger diagram generation)
    output_diagram = tmp_path / "error_diagram.png"
    result = runner.invoke(app, [
        "run",
        str(test_script),
        "--output", str(output_diagram)
    ])
    
    # Should capture the error but not fail the command
    assert result.exit_code == 0
    assert "Script execution failed" in result.stdout
    assert "Test error at level 2" in result.stdout
    assert mock_ollama.get_ai_analysis.called
    
    # 3. Generate diagram explicitly
    result = runner.invoke(app, [
        "diagram",
        "--output", str(tmp_path / "call_diagram.png")
    ])
    assert result.exit_code == 0
    assert "Diagram generated" in result.stdout

def test_debugger_engine_integration(mock_ollama):
    """Test DebuggerEngine integration with its components."""
    engine = DebuggerEngine(
        enable_visualizer=True,
        ai_adapter_config={"model": "phi4"},
        log_interceptor_config={"level": "DEBUG"}
    )
    
    with engine:
        # Simulate some logging activity
        print("Test log message")
        engine.add_trace_event({
            "event_type": "call",
            "function_name": "test_function",
            "module_name": "test_module",
            "line_number": 1
        })
        
        # Test log analysis
        analysis = engine.analyze_logs()
        assert analysis["success"] is True
        assert "Test analysis" in str(analysis["analysis"])
        
        # Test visualization
        diagram_path = engine.generate_visual([], "test_diagram.png")
        assert diagram_path is not None

def test_error_handling_integration(runner, mock_ollama, tmp_path):
    """Test error handling across components."""
    # 1. Test with non-existent script
    result = runner.invoke(app, ["run", "nonexistent.py"])
    assert result.exit_code == 1
    assert "Script not found" in result.stdout
    
    # 2. Test with AI server failure
    mock_ollama.get_ai_analysis.return_value = {
        "success": False,
        "error": "AI server error",
        "analysis": None
    }
    
    script_content = "print('Test'); raise ValueError('Test error')"
    script_path = tmp_path / "error_script.py"
    script_path.write_text(script_content)
    
    result = runner.invoke(app, ["run", str(script_path)])
    assert result.exit_code == 0  # Command succeeds but notes the error
    assert "Test error" in result.stdout

def test_visualization_integration(runner, test_script, tmp_path):
    """Test visualization with different output formats."""
    # Test PNG output
    png_output = tmp_path / "diagram.png"
    result = runner.invoke(app, [
        "run",
        str(test_script),
        "--output", str(png_output)
    ])
    assert result.exit_code == 0
    
    # Ensure the diagram command can process the generated trace
    result = runner.invoke(app, [
        "diagram",
        "--output", str(tmp_path / "new_diagram.png")
    ])
    assert result.exit_code == 0
    assert "Diagram generated" in result.stdout
