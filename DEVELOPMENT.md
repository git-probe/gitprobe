# GitProbe Development Guide

## Quick Setup

### 1. Install in Development Mode
```bash
# Clone the repository
git clone https://github.com/yourusername/gitprobe.git
cd gitprobe

# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### 2. Run GitProbe

Once installed, you can use GitProbe from anywhere:

```bash
# Analyze a repository
gitprobe analyze microsoft/vscode

# Start the server
gitprobe server

# Start server with custom settings
gitprobe server --port 8080 --reload
```

## Alternative Development Setup

If you prefer not to install the package:

```bash
# Set Python path and run directly
PYTHONPATH=src python -m gitprobe.cli analyze user/repo
PYTHONPATH=src python -m gitprobe.web.server
```

## Project Structure

```
gitprobe/
├── src/gitprobe/           # Main package
│   ├── analyzers/          # Language-specific analyzers
│   ├── analysis/           # Business logic & orchestration
│   ├── core/              # Shared utilities
│   ├── models/            # Data models
│   ├── utils/             # Helper functions
│   ├── web/               # FastAPI server
│   └── cli.py             # Command-line interface
├── pyproject.toml          # Package configuration
├── requirements.txt        # Dependencies
└── README.md              # User documentation
```

## Development Commands

```bash
# Run tests
pytest

# Format code
black src/
isort src/

# Type checking
mypy src/

# Install pre-commit hooks
pre-commit install
```

## Adding New Languages

1. Create analyzer in `src/gitprobe/analyzers/`
2. Add language limits in `src/gitprobe/core/analysis_limits.py`
3. Update `src/gitprobe/analysis/call_graph_analyzer.py`
4. Add tests and documentation 