import json
import os
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, ANY
import pytest
from typer.testing import CliRunner

from neurotrace.cli.cli import app
from neurotrace.debugger_engine import DebuggerConfig
from neurotrace.ollama_ai_adapter import OllamaConfig

runner = CliRunner()

@pytest.fixture
def mock_engine():
    """Mock DebuggerEngine for testing."""
    with patch('neurotrace.cli.cli.DebuggerEngine') as mock:
        engine_instance = Mock()
        engine_instance.analyze_logs.return_value = {
            "success": True,
            "analysis": {"response": "\033[32mTest analysis\033[0m"}
        }
        mock.return_value = engine_instance
        yield engine_instance

@pytest.fixture
def mock_subprocess():
    """Mock subprocess for testing script execution."""
    with patch('subprocess.run') as mock:
        mock.return_value = Mock(
            returncode=0,
            stdout="Test output",
            stderr=""
        )
        yield mock

def test_run_script_not_found():
    """Test run command with non-existent script."""
    result = runner.invoke(app, ["run", "nonexistent.py"])
    assert result.exit_code == 1
    assert "Script not found" in result.stdout

def test_run_script_success(mock_engine, mock_subprocess, tmp_path):
    """Test successful script execution with all options."""
    script_path = tmp_path / "test.py"
    script_path.write_text('print("Hello, NeuroTrace!")')
    
    result = runner.invoke(app, [
        "run",
        str(script_path),
        "--enable-visualizer",
        "--live-tracing",
        "--line-level-tracing",
        "--log-level", "DEBUG",
        "--output", "error.png",
        "--ollama-model", "codellama",
        "--ollama-url", "http://localhost:11434",
        "--ollama-timeout", "15",
        "--ollama-retries", "5",
        "--ollama-chunk-size", "8192"
    ])
    
    assert result.exit_code == 0
    assert mock_engine.start_debugging.called
    assert mock_engine.stop_debugging.called
    
    # Verify subprocess configuration
    mock_subprocess.assert_called_once_with(
        [ANY, "-u", str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=ANY,
        timeout=30
    )
    
    # Verify engine configuration
    engine_call = mock_engine.call_args[1]
    config = engine_call["config"]
    assert isinstance(config, DebuggerConfig)
    assert config.enable_visualizer is True
    assert config.live_tracing is True
    assert config.line_level_tracing is True
    assert config.log_level == "DEBUG"
    assert config.ai_model == "codellama"
    
    # Verify Ollama configuration
    ai_config = engine_call["ai_adapter_config"]
    assert isinstance(ai_config["config"], OllamaConfig)
    assert ai_config["config"].base_url == "http://localhost:11434"
    assert ai_config["config"].timeout == 15
    assert ai_config["config"].max_retries == 5
    assert ai_config["config"].max_chunk_size == 8192

def test_run_script_with_error(mock_engine, mock_subprocess, tmp_path):
    """Test script execution with error and diagram generation."""
    script_path = tmp_path / "error.py"
    script_path.write_text('raise ValueError("Test error")')
    
    # Configure subprocess to simulate error
    mock_subprocess.return_value = Mock(
        returncode=1,
        stdout="",
        stderr="ValueError: Test error"
    )
    
    result = runner.invoke(app, [
        "run",
        str(script_path),
        "--output", "error_diagram.png"
    ])
    
    assert result.exit_code == 0  # Command succeeds but notes the error
    assert mock_engine.start_debugging.called
    assert mock_engine.stop_debugging.called
    assert mock_engine.generate_visual.called
    assert "Script execution failed" in result.stdout
    assert "Test error" in result.stdout

def test_run_script_with_ai_analysis(mock_engine, mock_subprocess, tmp_path):
    """Test script execution with AI analysis."""
    script_path = tmp_path / "test.py"
    script_path.write_text('print("Test")')
    
    result = runner.invoke(app, ["run", str(script_path)])
    
    assert result.exit_code == 0
    assert mock_engine.analyze_logs.called
    assert "Test analysis" in str(result.stdout)

def test_run_script_with_timeout(mock_engine, mock_subprocess, tmp_path):
    """Test script execution with timeout."""
    script_path = tmp_path / "test.py"
    script_path.write_text('import time; time.sleep(31)')
    
    mock_subprocess.side_effect = subprocess.TimeoutExpired(["python"], 30)
    
    result = runner.invoke(app, ["run", str(script_path)])
    assert result.exit_code == 1
    assert "Script execution timed out" in result.stdout

def test_diagram_generation(mock_engine, tmp_path):
    """Test diagram generation with different formats."""
    # Test PNG output
    result = runner.invoke(app, [
        "diagram",
        "--output", str(tmp_path / "test.png")
    ])
    assert result.exit_code == 0
    assert mock_engine.generate_visual.called
    
    # Test with trace file
    trace_path = tmp_path / "trace.json"
    trace_path.write_text('{"trace": "data"}')
    result = runner.invoke(app, [
        "diagram",
        str(trace_path),
        "--output", str(tmp_path / "test.png")
    ])
    assert result.exit_code == 0
    assert "Diagram generated" in result.stdout

def test_diagram_errors():
    """Test diagram generation error cases."""
    # Test with non-existent trace file
    result = runner.invoke(app, ["diagram", "nonexistent.json"])
    assert result.exit_code == 1
    assert "Trace file not found" in result.stdout
    
    # Test with invalid output path
    result = runner.invoke(app, [
        "diagram",
        "--output", "/invalid/path/test.png"
    ])
    assert result.exit_code == 1
    assert "Failed to generate diagram" in result.stdout
