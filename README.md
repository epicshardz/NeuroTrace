# NeuroTrace 🐍
AI-driven debugging companion

## Installation

Install the package using pip:
```bash
pip install neurotrace
```

## Configuration

NeuroTrace can be configured through command line arguments or programmatically:

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

2. Generate call stack diagrams:
```bash
neurotrace diagram --output diagram.png
```

## GitHub Setup

To enable the CI pipeline:

1. Create a GitHub repository
2. Push this code to your repository:
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin your-repo-url
git push -u origin main
```

3. The GitHub Actions workflow will automatically run on push/PR

## Testing

### Running Tests Locally

1. Install test dependencies:
```bash
pip install -e ".[test]"
```

2. Run tests with coverage:
```bash
coverage run -m pytest
coverage report -m
```

3. Run specific test files:
```bash
pytest tests/test_integration.py  # Run integration tests
pytest tests/test_cli.py         # Run CLI tests
```

4. Run linting checks:
```bash
flake8 .                # Check code style
black . --check        # Check formatting
black .               # Apply formatting
```

### Continuous Integration

The project uses GitHub Actions for continuous integration, running on every push and pull request:

1. **Test Matrix**: Tests run on Python 3.8, 3.9, and 3.10
2. **Coverage Requirements**: Minimum 80% code coverage
3. **Automated Checks**:
   - Unit tests
   - Integration tests
   - Code coverage reporting
   - Flake8 linting
   - Black formatting (optional)

### Test Structure

1. **Unit Tests**: Individual module tests in `tests/`
   - `test_log_interceptor.py`
   - `test_ollama_ai_adapter.py`
   - `test_runtime_visualizer.py`
   - `test_debugger_engine.py`
   - `test_cli.py`

2. **Integration Tests**: End-to-end scenarios in `tests/test_integration.py`
   - Complete workflow testing
   - Component interaction verification
   - Error handling scenarios

### Coverage Reports

Coverage reports exclude:
- Test files
- `__init__.py` files
- Specific lines marked with `# pragma: no cover`
- Standard boilerplate (e.g., `if __name__ == "__main__":`)

View detailed coverage:
```bash
coverage html  # Creates htmlcov/index.html
