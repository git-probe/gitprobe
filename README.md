# GitProbe 🔍

**Interactive GitHub Repository Analysis Tool** - The backend engine powering [gitprobe.com](https://gitprobe.com)

Transform any GitHub repository into interactive visualizations and LLM-optimized structured data. Perfect for understanding codebases through file trees, function call graphs, and AI-ready exports.

## ✨ Features

### 📁 **File Tree Analysis**
- **Smart filtering** with include/exclude patterns
- **File statistics** (size, tokens, complexity)
- **Advanced filters** by file size and type
- **Hierarchical structure** with metadata

### 🕸️ **Call Graph Analysis**
- **Interactive visualizations** with Cytoscape.js
- **Function relationships** mapping
- **Code snippet extraction** for each function
- **Multiple export formats** (HTML, SVG, JSON)

### 🤖 **LLM-Optimized Exports**
- **Structured JSON** perfect for AI analysis
- **Architectural insights** (entry points, utility functions)
- **Code complexity metrics**
- **Refactoring suggestions**

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/gitprobe.git
cd gitprobe
pip install -r requirements.txt
```

### Basic Usage
```bash
# File tree analysis
python main.py tree https://github.com/user/repo

# Interactive call graph visualization  
python main.py callgraph https://github.com/user/repo --viz callgraph.html

# Export LLM-optimized data
python main.py callgraph https://github.com/user/repo --llm-json analysis.json
```

## 📖 Documentation

### File Tree Analysis

Analyze repository structure with advanced filtering:

```bash
# Basic analysis
python main.py tree https://github.com/fastapi/fastapi

# Filter by file patterns
python main.py tree https://github.com/fastapi/fastapi \
  --include "*.py" "docs/" \
  --exclude "*test*" "*__pycache__*"

# Filter by file size (KB)
python main.py tree https://github.com/fastapi/fastapi \
  --min-size 1 --max-size 100

# Export to JSON
python main.py tree https://github.com/fastapi/fastapi \
  --output fastapi-structure.json
```

**Output includes:**
- Hierarchical file tree structure
- File metadata (size, extension, estimated tokens)
- Directory statistics
- Total repository metrics

### Call Graph Analysis

Analyze function relationships and generate interactive visualizations:

```bash
# Interactive HTML visualization
python main.py callgraph https://github.com/miguelgrinberg/microblog \
  --viz callgraph.html

# Export all formats
python main.py callgraph https://github.com/miguelgrinberg/microblog \
  --viz interactive.html \
  --svg static.svg \
  --llm-json llm-analysis.json

# Python files only
python main.py callgraph https://github.com/miguelgrinberg/microblog \
  --include "*.py" \
  --exclude "*test*" "*migration*"
```

**Interactive Features:**
- **Click nodes** → View function details + code
- **Multiple layouts** (circular, force-directed, hierarchical)
- **Download buttons** for SVG and LLM JSON
- **Zoom, pan, focus** on specific functions

### LLM-Optimized JSON Structure

Perfect for AI code analysis:

```json
{
  "repository_info": {
    "name": "owner/repo",
    "total_functions": 109,
    "total_files": 32,
    "languages": ["python"]
  },
  "architecture_summary": {
    "entry_points": [...],      // Functions not called by others
    "utility_functions": [...], // Frequently reused functions
    "complex_functions": [...], // Large functions (>20 lines)
    "isolated_functions": [...] // Unused/dead code candidates
  },
  "functions": {
    "file.py:function_name": {
      "name": "function_name",
      "file": "file.py", 
      "parameters": [...],
      "complexity": 15,
      "times_called": 3,
      "code_snippet": "def function_name():\n    ...",
      "docstring": "Function description"
    }
  },
  "call_relationships": [
    {
      "from": "caller.py:function_a",
      "to": "callee.py:function_b", 
      "line": 42
    }
  ],
  "insights": {
    "most_called_functions": [...],
    "largest_functions": [...],
    "potential_refactoring_candidates": [...]
  }
}
```

## 🏗️ Project Structure

```
gitprobe/
├── main.py                 # Main CLI entry point
├── repo_analyzer.py        # File tree analysis
├── call_graph_analyzer.py  # Call graph analysis  
├── simple_analyzer.py      # Interactive CLI version
├── requirements.txt        # Dependencies
├── tests/                  # Test files
│   ├── test_analyzer.py
│   ├── test_advanced_analyzer.py
│   └── test_call_graph.py
└── examples/               # Example outputs
    └── tiangolo-fastapi_structure.txt
```

## 🧪 Testing

```bash
# Run file tree tests
python tests/test_analyzer.py

# Run call graph tests  
python tests/test_call_graph.py

# Interactive testing
python tests/test_advanced_analyzer.py
```

## 🎯 Use Cases

### For Developers
- **Code exploration** of unfamiliar repositories
- **Architecture understanding** through visual call graphs
- **Refactoring planning** with complexity insights

### For AI/LLMs
- **Structured codebase understanding** vs. raw file dumps
- **Architectural pattern recognition**
- **Code quality analysis** and suggestions

### For gitprobe.com
- **Backend API** for repository analysis
- **Interactive frontend** data source
- **Export functionality** for users

## 🔧 API Integration

The analyzers can be used programmatically:

```python
from repo_analyzer import RepoAnalyzer
from call_graph_analyzer import CallGraphAnalyzer

# File tree analysis
repo_analyzer = RepoAnalyzer(
    include_patterns=['*.py'],
    exclude_patterns=['*test*']
)
result = repo_analyzer.analyze_repository('https://github.com/user/repo')

# Call graph analysis
cg_analyzer = CallGraphAnalyzer(repo_analyzer)
call_graph = cg_analyzer.analyze_repository('https://github.com/user/repo')
```

## 🚧 Roadmap

- [ ] **Multi-language support** (JavaScript, TypeScript, Java, Go)
- [ ] **Web API** for gitprobe.com integration
- [ ] **Real-time analysis** for large repositories
- [ ] **Advanced metrics** (cyclomatic complexity, maintainability index)
- [ ] **Diff analysis** for pull requests

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

---

**Built for [gitprobe.com](https://gitprobe.com)** - The future of interactive code exploration 🚀 