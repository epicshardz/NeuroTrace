# NeuroTrace 🐍
AI-driven debugging companion for Python - Your terminal's intelligent error analyst

NeuroTrace is a terminal-based debugging tool that automatically analyzes Python errors and execution patterns in real-time. It combines traditional debugging capabilities with AI-powered analysis, providing immediate insights right in your terminal when errors occur. No need to manually inspect logs or set breakpoints - NeuroTrace actively monitors your code execution and gives you intelligent feedback the moment something goes wrong.

Key features:
- Automatic error detection and AI-powered analysis in your terminal
- Real-time code execution monitoring with instant feedback
- Intelligent insights provided directly in the console when errors occur
- Visual call stack and execution path diagrams generated on error
- Advanced log interception and analysis
- Local AI processing using Ollama - no data leaves your machine

## Video Demo of NeuroTraceAI in action
Watch NeuroTrace in action:

[![Demo](https://img.youtube.com/vi/OwtPNvdxMyY/0.jpg)](https://www.youtube.com/watch?v=OwtPNvdxMyY)

## Installation
Ollama is the expert running the llm behind the scenes so the project assumes you have followed the steps provided by the Ollama team here:
https://github.com/ollama/ollama
Default model is phi4, but you can changes as needed.

## Installation
First download the repo to "your directory"
Then run the following command in your project
Install the package using pip:
```bash
pip install <your neurotrace directory>/neurotrace
```

## Configuration

NeuroTrace can be configured through command line arguments or programmatically.

## Usage

NeuroTrace runs directly in your terminal and automatically activates when errors occur. Here are the main commands:

Run a Python script with automatic error analysis:
```bash
neurotrace run your_script.py
```

When an error occurs, NeuroTrace will automatically:
- Capture the full error context and stack trace
- Generate a visual diagram of the error path
- Provide AI-powered analysis and insights directly in your terminal
- Suggest potential fixes and improvements

Command Options (Experimental):
- `--enable-visualizer`: Enable/disable runtime visualization (default: enabled)
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `--output`: Specify output file for error diagrams
- `--live-tracing` (Experimental): Enable live tracing of function calls
- `--line-level-tracing` (Experimental): Enable line-by-line tracing (requires live_tracing)

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

### What is Neuro Trace AI Coin (NTAI)?
Neuro Trace AI is associated with the NTAI coin, which is designed to support funding for experimental research and development of advanced features within the AI platform. While not offering financial incentives or returns, the coin reflects the commitment to driving innovation and growth in the AI ecosystem.

For more details, visit the whitepaper: [Neuro Trace AI Whitepaper](https://docs.google.com/document/d/1D2uIFPc0a2PgDG3uRbGnbcaSO0QcRwmygBrQWdDbjTc/edit?usp=sharing)