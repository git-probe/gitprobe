# GitProbe - Repository Analysis & Interactive Call Graph Visualization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Pydantic](https://img.shields.io/badge/Pydantic-2.0+-red.svg)](https://docs.pydantic.dev/)

GitProbe is a powerful repository analysis tool that generates interactive call graphs and provides deep insights into code architecture. Transform any GitHub repository URL into comprehensive visualizations and AI-optimized analysis data.

## 🌟 Features

### 🔍 **Interactive Analysis**
- **Real-time Progress Tracking**: Monitor analysis progress with live updates
- **Advanced Filtering**: Include/exclude patterns, file types, size limits
- **Session Management**: Persistent analysis sessions with state management
- **Filter Preview**: See what files will be analyzed before running

### 🎯 **Interactive Node Management**
- **Node Renaming**: Custom display names for functions
- **Node Selection**: Multi-select nodes for focused analysis
- **Tagging System**: Organize functions with custom tags
- **Notes & Annotations**: Add contextual notes to functions
- **Detailed Inspection**: Deep-dive into function metrics and relationships

### 📊 **Rich Visualizations**
- **Cytoscape.js Integration**: Interactive, zoomable call graphs
- **D3.js Support**: Force-directed layouts and custom visualizations
- **Multiple Layout Options**: Circle, force, hierarchical layouts
- **SVG Export**: Vector graphics for presentations
- **Real-time Updates**: Live graph manipulation and filtering

### 🤖 **AI-Optimized Exports**
- **LLM-Ready JSON**: Structured data optimized for AI analysis
- **Architectural Insights**: Auto-generated code quality assessments
- **Partial Exports**: Export selected subsets of the graph
- **Code Snippets**: Include full function source code
- **Relationship Mapping**: Detailed call relationship analysis

### 🔧 **Developer-Friendly API**
- **RESTful API**: Clean, documented endpoints
- **Pydantic Models**: Type-safe data validation
- **FastAPI Integration**: Auto-generated docs and validation
- **CORS Support**: Ready for frontend integration
- **Background Processing**: Non-blocking analysis

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/gitprobe.git
cd gitprobe

# Install dependencies
pip install -r requirements.txt
```

### Running the API Server

```bash
# Start the development server
python server.py

# Or with uvicorn directly
uvicorn api.app:create_app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000/api
- **Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### Quick Analysis

```bash
# Using the CLI (legacy)
python main.py callgraph https://github.com/user/repo

# Using the API
curl -X POST "http://localhost:8000/api/analyze-repo" \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/user/repo"}'
```

## 📚 API Usage

### 1. Start Analysis

```python
import requests

# Start repository analysis
response = requests.post("http://localhost:8000/api/analyze-repo", json={
    "repo_url": "https://github.com/miguelgrinberg/microblog",
    "filter_options": {
        "include_patterns": ["*.py"],
        "exclude_patterns": ["*test*", "migrations/"]
    }
})

session = response.json()
session_id = session["session_id"]
```

### 2. Monitor Progress

```python
# Check analysis progress
progress = requests.get(f"http://localhost:8000/api/session/{session_id}")
print(f"Progress: {progress.json()['progress']*100:.1f}%")
```

### 3. Interactive Node Management

```python
# Rename a function
requests.put(f"http://localhost:8000/api/session/{session_id}/rename-node", 
             params={"node_id": "app.py:create_app", "new_name": "Application Factory"})

# Select multiple nodes
requests.post(f"http://localhost:8000/api/session/{session_id}/select-nodes", 
              json=["app.py:create_app", "app.py:register_blueprints"])

# Get detailed node information
details = requests.get(f"http://localhost:8000/api/session/{session_id}/node/app.py:create_app")
print(details.json()["function_info"]["code_snippet"])
```

### 4. Export Analysis

```python
# Export LLM-optimized JSON
export_request = {
    "export_format": "json",
    "optimize_for_llm": True,
    "include_code_snippets": True,
    "include_relationships": True
}

export_result = requests.post(
    f"http://localhost:8000/api/session/{session_id}/export",
    json=export_request
)

llm_data = export_result.json()["data"]
```

## 🏗️ Architecture

### Clean Architecture Design

```
gitprobe/
├── 📁 api/                  # FastAPI application & routes
│   ├── app.py              # Application factory
│   └── routes.py           # API endpoints
├── 📁 models/              # Pydantic data models
│   ├── function_models.py  # Function & relationship models
│   ├── session_models.py   # Session & user interaction models
│   ├── analysis_models.py  # Analysis result models
│   └── export_models.py    # Export & node detail models
├── 📁 services/            # Business logic layer
│   ├── analysis_service.py # Call graph analysis engine
│   ├── node_service.py     # Interactive node management
│   ├── session_service.py  # Session state management
│   └── export_service.py   # Data export functionality
├── 📄 server.py            # Development server
├── 📄 main.py              # CLI interface
└── 📄 repo_analyzer.py     # Repository file analysis
```

### Key Design Principles

- **🔷 Separation of Concerns**: Clear boundaries between API, business logic, and data models
- **🔷 Type Safety**: Full Pydantic integration for runtime validation
- **🔷 Dependency Injection**: Clean service composition and testing
- **🔷 Open Source Ready**: Core analysis engine can be used independently
- **🔷 API-First**: Designed for frontend integration (React, Vue, etc.)

## 🎨 Frontend Integration

### React Example

```javascript
// Start analysis
const startAnalysis = async (repoUrl) => {
  const response = await fetch('/api/analyze-repo', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ repo_url: repoUrl })
  });
  return response.json();
};

// Real-time progress updates
const pollProgress = async (sessionId) => {
  const response = await fetch(`/api/session/${sessionId}`);
  const session = await response.json();
  return session.progress;
};

// Interactive node selection
const selectNodes = async (sessionId, nodeIds) => {
  await fetch(`/api/session/${sessionId}/select-nodes`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(nodeIds)
  });
};
```

### Cytoscape.js Integration

```javascript
// Load call graph visualization
const loadCallGraph = async (sessionId) => {
  const response = await fetch(`/api/session/${sessionId}`);
  const session = await response.json();
  const elements = session.analysis_results.visualization.cytoscape.elements;
  
  const cy = cytoscape({
    container: document.getElementById('cy'),
    elements: elements,
    style: session.analysis_results.visualization.cytoscape.style,
    layout: { name: 'cose' }
  });
  
  // Interactive node clicking
  cy.on('tap', 'node', async (evt) => {
    const nodeId = evt.target.id();
    const details = await fetch(`/api/session/${sessionId}/node/${nodeId}`);
    showNodeDetails(await details.json());
  });
};
```

## 📊 Data Models

### Function Information
```python
class FunctionInfo(BaseModel):
    name: str                           # Function name
    file_path: str                      # Source file path
    function_id: str                    # Unique identifier
    line_start: int                     # Starting line number
    line_end: int                       # Ending line number
    parameters: List[str]               # Parameter names
    docstring: Optional[str]            # Function documentation
    code_snippet: str                   # Full source code
    function_type: FunctionType         # function, method, async_function, async_method
    scope: FunctionScope                # module or class
    class_name: Optional[str]           # Parent class if method
    complexity_score: int               # Lines of code complexity
    
    # User modifications
    custom_name: Optional[str]          # User-defined display name
    tags: List[str]                     # User tags
    notes: Optional[str]                # User notes
```

### Call Relationships
```python
class CallRelationship(BaseModel):
    caller_id: str                      # Function making the call
    callee_id: str                      # Function being called
    call_line: int                      # Line number of call
    is_resolved: bool                   # Whether callee was found
    call_type: str                      # Type of call (direct, indirect)
    context: Optional[str]              # Code context
```

## 🔧 Configuration

### Filter Options
```python
class FilterOptions(BaseModel):
    include_patterns: List[str] = []     # Files to include (*.py, src/)
    exclude_patterns: List[str] = []     # Files to exclude (*test*, docs/)
    file_extensions: List[str] = []      # Specific extensions
    min_file_size_kb: Optional[float]    # Minimum file size
    max_file_size_kb: Optional[float]    # Maximum file size
    regex_mode: bool = False             # Enable regex patterns
```

### Export Configuration
```python
class ExportRequest(BaseModel):
    export_format: str                   # json, svg, cytoscape
    export_scope: str = "full"           # full, selection, filtered
    optimize_for_llm: bool = False       # LLM-optimized output
    include_code_snippets: bool = True   # Include source code
    include_relationships: bool = True    # Include call relationships
    max_nodes: Optional[int] = None      # Limit number of nodes
```

## 🧪 Testing

```bash
# Run the test suite (when available)
python -m pytest tests/

# Test the API endpoints
python -m pytest tests/test_api.py

# Test the analysis engine
python -m pytest tests/test_analysis.py
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install

# Start development server with auto-reload
python server.py
```

### Code Style

- **Type Hints**: All public functions must have type hints
- **Pydantic Models**: Use Pydantic for all data structures
- **Docstrings**: Google-style docstrings for all public methods
- **Testing**: Write tests for new functionality

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FastAPI**: For the excellent web framework
- **Pydantic**: For powerful data validation
- **Cytoscape.js**: For interactive graph visualization
- **AST**: Python's Abstract Syntax Tree for code analysis

## 🚀 Roadmap

- [ ] **Multi-language Support**: JavaScript, TypeScript, Java support
- [ ] **Advanced Graph Analytics**: Strongly connected components, cycle detection
- [ ] **AI Integration**: GPT-4 powered code analysis and suggestions
- [ ] **Real-time Collaboration**: Multi-user analysis sessions
- [ ] **Performance Optimization**: Caching and incremental analysis
- [ ] **Plugin System**: Extensible analysis pipeline

---

**GitProbe** - Transform your codebase understanding with interactive visualizations and AI-powered insights.

🔗 **Links**: [Documentation](https://docs.gitprobe.com) | [API Reference](https://api.gitprobe.com) | [Examples](https://examples.gitprobe.com) 