# NeuroTrace 🐍
AI-driven debugging companion for Python

NeuroTrace is an experimental debugging tool that combines traditional Python debugging capabilities with AI-powered analysis. It helps developers understand and fix issues by providing intelligent insights into runtime behavior, call stacks, and error patterns.

Key features:
- Real-time code execution monitoring
- AI-powered error analysis using local LLMs via Ollama
- Visual call stack and execution path diagrams
- Log interception and analysis
- Live tracing capabilities
- Intelligent error context gathering

## Installation

Install the package using pip:
```bash
pip install neurotrace
```

## Configuration

NeuroTrace can be configured through command line arguments or programmatically.

### Command Line Configuration

```bash
neurotrace run script.py \
  --ollama-model phi4 \
  --ollama-url http://localhost:11434 \
  --ollama-timeout 10 \
  --ollama-retries 3 \
  --ollama-chunk-size 4096
```

### Programmatic Configuration

```python
from neurotrace import DebuggerEngine, DebuggerConfig
from neurotrace.ollama_ai_adapter import OllamaConfig

# Configure Ollama settings
ollama_config = OllamaConfig(
    base_url="http://localhost:11434",
    max_retries=3,
    timeout=10,
    max_chunk_size=4096,
    model="phi4"
)

# Initialize debugger with configuration
engine = DebuggerEngine(
    config=DebuggerConfig(enable_visualizer=True),
    ai_adapter_config={"config": ollama_config}
)
```

## Usage

NeuroTrace provides a command-line interface with two main commands:

1. Run a Python script with NeuroTrace debugging:
```bash
neurotrace run your_script.py
```
Options:
- `--enable-visualizer`: Enable/disable runtime visualization (default: enabled)
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `--output`: Specify output file for error diagrams
- `--live-tracing` (Experimental): Enable live tracing of function calls
- `--line-level-tracing` (Experimental): Enable line-by-line tracing (requires live_tracing)

2. Generate call stack diagrams:
```bash
neurotrace diagram --output diagram.png
```