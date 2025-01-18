"""NeuroTrace CLI - AI-driven debugging companion."""
import os
import shutil
import subprocess
import sys
import typer
import logging
from pathlib import Path
from typing import Optional

# Import DebuggerEngine and configs
from neurotrace.debugger_engine import DebuggerEngine, DebuggerConfig
from neurotrace.ollama_ai_adapter import OllamaConfig

app = typer.Typer(help="NeuroTrace CLI - AI-driven debugging companion")

@app.command()
def run(
    script: str = typer.Argument(..., help="Python script to run with NeuroTrace debugging"),
    enable_visualizer: bool = typer.Option(True, help="Enable runtime visualization"),
    live_tracing: bool = typer.Option(False, help="Enable live tracing of function calls"),
    line_level_tracing: bool = typer.Option(False, help="Enable line-by-line tracing (requires live_tracing)"),
    log_level: str = typer.Option("INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)"),
    output: Optional[str] = typer.Option(None, help="Output file for diagram if error occurs"),
    ollama_model: str = typer.Option("phi4", help="Ollama model to use for analysis"),
    ollama_url: str = typer.Option("http://localhost:11434", help="Ollama server URL"),
    ollama_timeout: int = typer.Option(10, help="Ollama request timeout in seconds"),
    ollama_retries: int = typer.Option(3, help="Maximum Ollama retry attempts"),
    ollama_chunk_size: int = typer.Option(4096, help="Maximum Ollama chunk size")
):
    """
    Run a Python script with NeuroTrace debugging enabled.
    By default, after the script finishes, NeuroTrace will automatically
    run AI analysis on captured logs.
    """
    if not os.path.exists(script):
        typer.echo(f"Error: Script not found: {script}", err=True)
        raise typer.Exit(code=1)

    # Build configs
    debugger_config = DebuggerConfig(
        enable_visualizer=enable_visualizer,
        ai_model=ollama_model,
        log_level=log_level,
        live_tracing=live_tracing,
        line_level_tracing=line_level_tracing
    )

    ollama_config = OllamaConfig(
        base_url=ollama_url,
        max_retries=ollama_retries,
        timeout=ollama_timeout,
        max_chunk_size=ollama_chunk_size,
        model=ollama_model
    )

    try:
        # Initialize DebuggerEngine with the configs
        engine = DebuggerEngine(
            config=debugger_config,
            ai_adapter_config={"config": ollama_config},
            log_interceptor_config={"max_logs": 2000},
            script_path=os.path.abspath(script)
        )

        engine.start_debugging()

        # Run the user script in a subprocess so logs are captured properly
        result = subprocess.run(
            [sys.executable, "-u", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1"
            },
            timeout=30  # Adjust as needed
        )
        
        # Print script output
        if result.stdout:
            typer.echo(result.stdout, nl=False)
        if result.stderr:
            typer.echo(result.stderr, err=True, nl=False)
        
        if result.returncode != 0:
            typer.echo("Script execution failed:", err=True)
            if result.stderr:
                typer.echo(result.stderr, err=True)

            # If there's an output file, try to generate a diagram
            if output:
                path = engine.generate_visual(output_path=output)
                if path:
                    typer.echo(f"Error diagram generated at: {path}")
        else:
            # Script succeeded
            if result.stdout:
                typer.echo(result.stdout)
        
        # Automatically run AI analysis on the captured logs
        analysis = engine.analyze_logs()
        if analysis["success"]:
            typer.echo(str(analysis["analysis"]))
        else:
            typer.echo(f"AI Analysis not available: {analysis['error']}")

    except Exception as e:
        raise typer.Exit(code=1)
    finally:
        try:
            engine.stop_debugging()
        except Exception as e:
            # Log but don't display cleanup errors to avoid confusion
            logging.error(f"Error during cleanup: {e}")

@app.command()
def diagram(
    trace_file: Optional[Path] = typer.Argument(
        None,
        help="Trace file to generate diagram from (uses latest if not specified)"
    ),
    output: str = typer.Option(
        "diagram.png",
        help="Output file path for the generated diagram"
    )
):
    """Generate a call stack diagram from trace data."""
    try:
        config = DebuggerConfig(enable_visualizer=True)
        engine = DebuggerEngine(config=config)

        if trace_file is None:
            typer.echo("No trace file specified; using empty trace data.")
            engine._trace_data = []
        else:
            if not trace_file.exists():
                typer.echo(f"Error: Trace file not found: {trace_file}", err=True)
                raise typer.Exit(code=1)
            # In a real scenario, parse actual trace data from file
            engine._trace_data = []

        path = engine.generate_visual(output_path=output)
        if path:
            typer.echo(f"Diagram generated at: {path}")
        else:
            typer.echo("No diagram generated (no trace data or visualizer disabled).")

    except Exception as e:
        typer.echo(f"Error: Failed to generate diagram: {e}", err=True)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
