#!/usr/bin/env python3
"""
GitProbe Call Graph Analyzer
Extracts function definitions and call relationships from code files.
"""

import ast
import json
import os
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, asdict
import tempfile
import shutil

from repo_analyzer import RepoAnalyzer


@dataclass
class FunctionInfo:
    """Information about a function."""

    name: str
    file_path: str
    line_start: int
    line_end: int
    parameters: List[str]
    docstring: Optional[str]
    scope: str  # 'module' or 'class:ClassName'
    calls: List[str]  # Functions this function calls
    is_method: bool
    class_name: Optional[str] = None
    code_snippet: Optional[str] = None  # Actual function code
    complexity_score: Optional[int] = None  # Lines of code as complexity metric


@dataclass
class CallRelationship:
    """Represents a function call relationship."""

    caller: str  # "file_path:function_name"
    callee: str  # "file_path:function_name" or just "function_name" for unresolved
    call_line: int
    is_resolved: bool  # Whether we found the callee definition


class PythonASTAnalyzer(ast.NodeVisitor):
    """AST visitor to extract function information from Python code."""

    def __init__(self, file_path: str, content: str):
        self.file_path = file_path
        self.content = content
        self.lines = content.split("\n")
        self.functions: List[FunctionInfo] = []
        self.current_class = None
        self.call_relationships: List[CallRelationship] = []

    def visit_ClassDef(self, node):
        """Visit class definition."""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node):
        """Visit function definition."""
        # Extract function information
        func_info = self._extract_function_info(node)
        self.functions.append(func_info)

        # Find function calls within this function
        self._extract_function_calls(node, func_info.name)

        # Don't visit nested functions for now (keep it simple)
        # self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        """Visit async function definition."""
        self.visit_FunctionDef(node)  # Treat same as regular function

    def _extract_function_info(self, node) -> FunctionInfo:
        """Extract information from a function definition node."""
        # Get parameters
        params = []
        for arg in node.args.args:
            params.append(arg.arg)

        # Get docstring
        docstring = None
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            docstring = node.body[0].value.value

        # Extract code snippet
        start_line = node.lineno - 1  # Convert to 0-based
        end_line = (node.end_lineno or node.lineno) - 1
        code_snippet = "\n".join(self.lines[start_line : end_line + 1])

        # Calculate complexity (simple metric: lines of code)
        complexity_score = end_line - start_line + 1

        # Determine scope
        scope = "module"
        class_name = None
        is_method = False

        if self.current_class:
            scope = f"class:{self.current_class}"
            class_name = self.current_class
            is_method = True

        return FunctionInfo(
            name=node.name,
            file_path=self.file_path,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            parameters=params,
            docstring=docstring,
            scope=scope,
            calls=[],  # Will be populated by _extract_function_calls
            is_method=is_method,
            class_name=class_name,
            code_snippet=code_snippet,
            complexity_score=complexity_score,
        )

    def _extract_function_calls(self, func_node, func_name: str):
        """Extract function calls from within a function."""

        class CallVisitor(ast.NodeVisitor):
            def __init__(self, analyzer):
                self.analyzer = analyzer
                self.caller_name = func_name

            def visit_Call(self, node):
                call_name = self._get_call_name(node.func)
                if call_name:
                    # Create call relationship
                    relationship = CallRelationship(
                        caller=f"{self.analyzer.file_path}:{self.caller_name}",
                        callee=call_name,
                        call_line=node.lineno,
                        is_resolved=False,  # Will resolve later
                    )
                    self.analyzer.call_relationships.append(relationship)

                self.generic_visit(node)

            def _get_call_name(self, node) -> Optional[str]:
                """Extract function name from call node."""
                if isinstance(node, ast.Name):
                    return node.id
                elif isinstance(node, ast.Attribute):
                    # Handle method calls like obj.method()
                    if isinstance(node.value, ast.Name):
                        return f"{node.value.id}.{node.attr}"
                    else:
                        return node.attr
                return None

        call_visitor = CallVisitor(self)
        call_visitor.visit(func_node)


class CallGraphAnalyzer:
    """Main class for analyzing function call graphs."""

    # File extensions for different languages
    CODE_EXTENSIONS = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".rs": "rust",
        ".go": "go",
    }

    def __init__(self, repo_analyzer: RepoAnalyzer):
        """Initialize with a configured RepoAnalyzer."""
        self.repo_analyzer = repo_analyzer
        self.functions: Dict[str, FunctionInfo] = {}  # "file:function" -> FunctionInfo
        self.call_relationships: List[CallRelationship] = []

    def analyze_repository(self, github_url: str) -> Dict:
        """
        Analyze repository and extract call graph.

        Args:
            github_url: GitHub repository URL

        Returns:
            Dictionary with call graph data and visualization formats
        """
        # First get the file tree
        print("🔍 Analyzing file structure...")
        repo_result = self.repo_analyzer.analyze_repository(github_url)

        # Filter to code files only
        code_files = self._extract_code_files(repo_result["file_tree"])
        print(f"📁 Found {len(code_files)} code files")

        # Clone repo again for code analysis (since RepoAnalyzer cleans up)
        temp_dir = tempfile.mkdtemp(prefix="gitprobe_callgraph_")

        try:
            # Clone repository
            repo_info = self.repo_analyzer._parse_github_url(github_url)
            self.repo_analyzer._clone_repository(github_url, temp_dir)

            # Analyze each code file
            for file_info in code_files:
                self._analyze_code_file(temp_dir, file_info)

            # Resolve call relationships
            self._resolve_call_relationships()

            # Generate visualization data
            viz_data = self._generate_visualization_data()

            result = {
                "repository": repo_result["repository"],
                "call_graph": {
                    "total_functions": len(self.functions),
                    "total_calls": len(self.call_relationships),
                    "languages_found": list(set(f.get("language") for f in code_files)),
                    "files_analyzed": len(code_files),
                },
                "functions": [asdict(func) for func in self.functions.values()],
                "relationships": [asdict(rel) for rel in self.call_relationships],
                "visualization": viz_data,
            }

            return result

        finally:
            # Clean up
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def _extract_code_files(self, file_tree: Dict) -> List[Dict]:
        """Extract code files from file tree."""
        code_files = []

        def traverse(tree, current_path=""):
            if tree["type"] == "file":
                ext = tree.get("extension", "").lower()
                if ext in self.CODE_EXTENSIONS:
                    # Skip common non-code files even if they have code extensions
                    name = tree["name"].lower()
                    if not any(
                        skip in name for skip in ["test", "spec", "config", "setup"]
                    ):
                        code_files.append(
                            {
                                "path": tree["path"],
                                "name": tree["name"],
                                "extension": ext,
                                "language": self.CODE_EXTENSIONS[ext],
                                "size_kb": tree.get("size_kb", 0),
                                "estimated_tokens": tree.get("estimated_tokens", 0),
                            }
                        )
            elif tree["type"] == "directory" and tree.get("children"):
                for child in tree["children"]:
                    traverse(child)

        traverse(file_tree)
        return code_files

    def _analyze_code_file(self, repo_dir: str, file_info: Dict):
        """Analyze a single code file."""
        file_path = Path(repo_dir) / file_info["path"]

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Currently only support Python
            if file_info["language"] == "python":
                self._analyze_python_file(file_info["path"], content)

        except Exception as e:
            print(f"⚠️ Error analyzing {file_info['path']}: {str(e)}")

    def _analyze_python_file(self, file_path: str, content: str):
        """Analyze Python file using AST."""
        try:
            tree = ast.parse(content)
            analyzer = PythonASTAnalyzer(file_path, content)
            analyzer.visit(tree)

            # Store functions
            for func in analyzer.functions:
                func_id = f"{file_path}:{func.name}"
                self.functions[func_id] = func

            # Store call relationships
            self.call_relationships.extend(analyzer.call_relationships)

        except SyntaxError as e:
            print(f"⚠️ Syntax error in {file_path}: {str(e)}")
        except Exception as e:
            print(f"⚠️ Error parsing {file_path}: {str(e)}")

    def _resolve_call_relationships(self):
        """Resolve function call relationships."""
        # Create a lookup of function names to their full identifiers
        func_lookup = {}
        for func_id, func_info in self.functions.items():
            func_lookup[func_info.name] = func_id

        # Resolve relationships
        for relationship in self.call_relationships:
            callee_name = relationship.callee

            # Try exact match first
            if callee_name in func_lookup:
                relationship.callee = func_lookup[callee_name]
                relationship.is_resolved = True
            # Try without module prefix for method calls
            elif "." in callee_name:
                method_name = callee_name.split(".")[-1]
                if method_name in func_lookup:
                    relationship.callee = func_lookup[method_name]
                    relationship.is_resolved = True

    def _generate_visualization_data(self) -> Dict:
        """Generate data for different visualization libraries."""

        # Cytoscape.js format
        cytoscape_elements = []

        # Add function nodes
        for func_id, func_info in self.functions.items():
            node_data = {
                "id": func_id,
                "label": func_info.name,
                "file": func_info.file_path,
                "type": "method" if func_info.is_method else "function",
                "class": func_info.class_name,
                "params": len(func_info.parameters),
                "lines": func_info.line_end - func_info.line_start + 1,
            }

            cytoscape_elements.append(
                {"data": node_data, "classes": f"node-{node_data['type']}"}
            )

        # Add call edges
        for relationship in self.call_relationships:
            if relationship.is_resolved:
                edge_data = {
                    "id": f"{relationship.caller}->{relationship.callee}",
                    "source": relationship.caller,
                    "target": relationship.callee,
                    "line": relationship.call_line,
                }

                cytoscape_elements.append({"data": edge_data, "classes": "edge-call"})

        # D3.js format
        d3_nodes = []
        d3_links = []

        node_ids = list(self.functions.keys())
        for i, (func_id, func_info) in enumerate(self.functions.items()):
            d3_nodes.append(
                {
                    "id": func_id,
                    "name": func_info.name,
                    "file": func_info.file_path,
                    "group": func_info.class_name or "module",
                    "type": "method" if func_info.is_method else "function",
                }
            )

        for relationship in self.call_relationships:
            if relationship.is_resolved:
                d3_links.append(
                    {
                        "source": relationship.caller,
                        "target": relationship.callee,
                        "value": 1,
                    }
                )

        return {
            "cytoscape": {
                "elements": cytoscape_elements,
                "style": self._get_cytoscape_style(),
            },
            "d3": {"nodes": d3_nodes, "links": d3_links},
            "summary": {
                "total_nodes": len(self.functions),
                "total_edges": len(
                    [r for r in self.call_relationships if r.is_resolved]
                ),
                "unresolved_calls": len(
                    [r for r in self.call_relationships if not r.is_resolved]
                ),
            },
        }

    def _get_cytoscape_style(self) -> List[Dict]:
        """Get Cytoscape.js styling."""
        return [
            {
                "selector": "node",
                "style": {
                    "background-color": "#3498db",
                    "label": "data(label)",
                    "width": "60px",
                    "height": "60px",
                    "text-valign": "center",
                    "text-halign": "center",
                    "font-size": "12px",
                },
            },
            {
                "selector": ".node-method",
                "style": {"background-color": "#e74c3c", "shape": "ellipse"},
            },
            {
                "selector": ".node-function",
                "style": {"background-color": "#2ecc71", "shape": "rectangle"},
            },
            {
                "selector": "edge",
                "style": {
                    "width": 2,
                    "line-color": "#34495e",
                    "target-arrow-color": "#34495e",
                    "target-arrow-shape": "triangle",
                    "curve-style": "bezier",
                },
            },
        ]


def generate_llm_optimized_json(call_graph_data: Dict) -> Dict:
    """Generate LLM-optimized JSON for code analysis."""

    # Create simplified structure for LLM consumption
    llm_data = {
        "repository_info": {
            "name": f"{call_graph_data['repository']['owner']}/{call_graph_data['repository']['name']}",
            "total_functions": call_graph_data["call_graph"]["total_functions"],
            "total_files": call_graph_data["call_graph"]["files_analyzed"],
            "languages": call_graph_data["call_graph"]["languages_found"],
        },
        "architecture_summary": {
            "entry_points": [],
            "utility_functions": [],
            "complex_functions": [],
            "isolated_functions": [],
        },
        "functions": {},
        "call_relationships": [],
        "insights": {
            "most_called_functions": [],
            "largest_functions": [],
            "potential_refactoring_candidates": [],
        },
    }

    # Process functions
    call_counts = {}
    for rel in call_graph_data["relationships"]:
        if rel["is_resolved"]:
            callee = rel["callee"]
            call_counts[callee] = call_counts.get(callee, 0) + 1

    called_functions = set(call_counts.keys())

    for func in call_graph_data["functions"]:
        func_id = f"{func['file_path']}:{func['name']}"

        # Simplified function info for LLM
        func_info = {
            "name": func["name"],
            "file": func["file_path"],
            "type": "method" if func["is_method"] else "function",
            "parameters": func["parameters"],
            "line_range": [func["line_start"], func["line_end"]],
            "complexity": func.get("complexity_score", 0),
            "calls_made": len(
                [
                    r
                    for r in call_graph_data["relationships"]
                    if r["caller"] == func_id and r["is_resolved"]
                ]
            ),
            "times_called": call_counts.get(func_id, 0),
            "docstring": func.get("docstring"),
            "code_snippet": func.get("code_snippet", ""),
            "class_context": func.get("class_name"),
        }

        llm_data["functions"][func_id] = func_info

        # Categorize functions
        if func_id not in called_functions:
            llm_data["architecture_summary"]["entry_points"].append(func_id)

        if call_counts.get(func_id, 0) > 3:  # Called by multiple functions
            llm_data["architecture_summary"]["utility_functions"].append(func_id)

        if func.get("complexity_score", 0) > 20:  # Large functions
            llm_data["architecture_summary"]["complex_functions"].append(func_id)

        if func_info["calls_made"] == 0 and func_info["times_called"] == 0:
            llm_data["architecture_summary"]["isolated_functions"].append(func_id)

    # Add call relationships in simplified format
    for rel in call_graph_data["relationships"]:
        if rel["is_resolved"]:
            llm_data["call_relationships"].append(
                {"from": rel["caller"], "to": rel["callee"], "line": rel["call_line"]}
            )

    # Generate insights
    sorted_by_calls = sorted(call_counts.items(), key=lambda x: x[1], reverse=True)
    llm_data["insights"]["most_called_functions"] = [
        {"function": func_id, "call_count": count}
        for func_id, count in sorted_by_calls[:10]
    ]

    sorted_by_complexity = sorted(
        [(fid, f) for fid, f in llm_data["functions"].items()],
        key=lambda x: x[1]["complexity"],
        reverse=True,
    )
    llm_data["insights"]["largest_functions"] = [
        {"function": func_id, "lines": func_info["complexity"]}
        for func_id, func_info in sorted_by_complexity[:10]
    ]

    # Potential refactoring candidates (large + highly called)
    refactoring_candidates = []
    for func_id, func_info in llm_data["functions"].items():
        score = func_info["complexity"] * func_info["times_called"]
        if score > 50:  # Arbitrary threshold
            refactoring_candidates.append(
                {
                    "function": func_id,
                    "complexity": func_info["complexity"],
                    "call_count": func_info["times_called"],
                    "refactor_score": score,
                }
            )

    llm_data["insights"]["potential_refactoring_candidates"] = sorted(
        refactoring_candidates, key=lambda x: x["refactor_score"], reverse=True
    )[:10]

    return llm_data


def generate_svg_export(call_graph_data: Dict) -> str:
    """Generate SVG representation of the call graph."""

    functions = call_graph_data["functions"]
    relationships = [r for r in call_graph_data["relationships"] if r["is_resolved"]]

    # Simple layout algorithm - circular layout
    import math

    num_functions = len(functions)
    if num_functions == 0:
        return "<svg></svg>"

    # SVG dimensions
    width, height = 800, 600
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 3

    svg_elements = []
    svg_elements.append(
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
    )

    # Add styles
    svg_elements.append(
        """
    <style>
        .function-node { fill: #2ecc71; stroke: #27ae60; stroke-width: 2; }
        .method-node { fill: #e74c3c; stroke: #c0392b; stroke-width: 2; }
        .function-text { font-family: Arial; font-size: 10px; text-anchor: middle; }
        .call-edge { stroke: #34495e; stroke-width: 1; fill: none; marker-end: url(#arrowhead); }
    </style>
    """
    )

    # Add arrow marker
    svg_elements.append(
        """
    <defs>
        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="#34495e" />
        </marker>
    </defs>
    """
    )

    # Position nodes in circle
    node_positions = {}
    for i, func in enumerate(functions):
        angle = 2 * math.pi * i / num_functions
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        node_positions[f"{func['file_path']}:{func['name']}"] = (x, y)

        # Draw node
        node_class = "method-node" if func["is_method"] else "function-node"
        svg_elements.append(f'<circle cx="{x}" cy="{y}" r="20" class="{node_class}" />')

        # Add text label
        func_name = (
            func["name"][:10] + "..." if len(func["name"]) > 10 else func["name"]
        )
        svg_elements.append(
            f'<text x="{x}" y="{y + 4}" class="function-text">{func_name}</text>'
        )

    # Draw edges
    for rel in relationships:
        caller_pos = node_positions.get(rel["caller"])
        callee_pos = node_positions.get(rel["callee"])

        if caller_pos and callee_pos:
            x1, y1 = caller_pos
            x2, y2 = callee_pos
            svg_elements.append(
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="call-edge" />'
            )

    svg_elements.append("</svg>")

    return "\n".join(svg_elements)


def generate_html_visualization(call_graph_data: Dict, output_file: str):
    """Generate HTML file with Cytoscape.js visualization."""

    cytoscape_data = call_graph_data["visualization"]["cytoscape"]
    repo_name = f"{call_graph_data['repository']['owner']}/{call_graph_data['repository']['name']}"

    # Generate LLM-optimized data for download
    llm_data = generate_llm_optimized_json(call_graph_data)
    svg_data = generate_svg_export(call_graph_data)

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Call Graph: {repo_name}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.21.0/cytoscape.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
        #cy {{ width: 100%; height: 600px; border: 1px solid #ddd; }}
        .info {{ margin-bottom: 20px; padding: 15px; background: #f5f5f5; border-radius: 5px; }}
        .controls {{ margin: 20px 0; }}
        button {{ margin: 5px; padding: 8px 15px; background: #3498db; color: white; border: none; border-radius: 3px; cursor: pointer; }}
        button:hover {{ background: #2980b9; }}
        .download-btn {{ background: #27ae60; }}
        .download-btn:hover {{ background: #229954; }}
        #code-panel {{ 
            margin-top: 20px; 
            padding: 15px; 
            background: #2c3e50; 
            color: #ecf0f1; 
            border-radius: 5px; 
            font-family: 'Courier New', monospace; 
            white-space: pre-wrap; 
            max-height: 300px; 
            overflow-y: auto;
            display: none;
        }}
        .panel-container {{ display: flex; gap: 20px; margin-top: 20px; }}
        #info-panel {{ 
            flex: 1; 
            padding: 15px; 
            background: #f9f9f9; 
            border-radius: 5px; 
        }}
    </style>
</head>
<body>
    <h1>🔍 Function Call Graph: {repo_name}</h1>
    
    <div class="info">
        <strong>Functions:</strong> {call_graph_data['call_graph']['total_functions']} | 
        <strong>Calls:</strong> {call_graph_data['call_graph']['total_calls']} |
        <strong>Files:</strong> {call_graph_data['call_graph']['files_analyzed']}
    </div>
    
    <div class="controls">
        <button onclick="cy.layout({{name: 'circle'}}).run()">Circle Layout</button>
        <button onclick="cy.layout({{name: 'cose'}}).run()">Force Layout</button>
        <button onclick="cy.layout({{name: 'breadthfirst'}}).run()">Hierarchical</button>
        <button onclick="cy.fit()">Fit to Screen</button>
        <button onclick="cy.center()">Center</button>
        <button class="download-btn" onclick="downloadSVG()">📥 Download SVG</button>
        <button class="download-btn" onclick="downloadLLMJSON()">🤖 Download LLM JSON</button>
    </div>
    
    <div id="cy"></div>
    
    <div class="panel-container">
        <div id="info-panel">
            Click on a node to see details and code
        </div>
    </div>
    
    <div id="code-panel"></div>

    <script>
        // Store function data for code lookup
        const functionData = {json.dumps({f['name']: f for f in call_graph_data['functions']})};
        const llmData = {json.dumps(llm_data)};
        const svgData = `{svg_data}`;
        
        const cy = cytoscape({{
            container: document.getElementById('cy'),
            elements: {json.dumps(cytoscape_data['elements'])},
            style: {json.dumps(cytoscape_data['style'])},
            layout: {{
                name: 'cose',
                animate: true,
                randomize: false,
                componentSpacing: 100,
                nodeOverlap: 20,
                idealEdgeLength: 100,
                edgeElasticity: 100,
                nestingFactor: 5,
                gravity: 80,
                numIter: 1000,
                initialTemp: 200,
                coolingFactor: 0.95,
                minTemp: 1.0
            }}
        }});
        
        // Add click handler for nodes
        cy.on('tap', 'node', function(evt) {{
            const node = evt.target;
            const data = node.data();
            
            // Find function info
            const funcInfo = Object.values(functionData).find(f => 
                f.name === data.label && f.file_path === data.file
            );
            
            const info = `
                <h3>${{data.label}}()</h3>
                <strong>File:</strong> ${{data.file}}<br>
                <strong>Type:</strong> ${{data.type}}<br>
                <strong>Parameters:</strong> ${{data.params}}<br>
                <strong>Lines:</strong> ${{data.lines}}<br>
                ${{funcInfo && funcInfo.docstring ? `<strong>Doc:</strong> ${{funcInfo.docstring.substring(0, 100)}}...` : ''}}
            `;
            
            document.getElementById('info-panel').innerHTML = info;
            
            // Show code snippet if available
            if (funcInfo && funcInfo.code_snippet) {{
                const codePanel = document.getElementById('code-panel');
                codePanel.innerHTML = `<h4>📝 Code:</h4>${{funcInfo.code_snippet}}`;
                codePanel.style.display = 'block';
            }} else {{
                document.getElementById('code-panel').style.display = 'none';
            }}
        }});
        
        // Add double-click to focus
        cy.on('dblclick', 'node', function(evt) {{
            const node = evt.target;
            cy.animate({{
                center: {{ eles: node }},
                zoom: 2
            }}, {{
                duration: 500
            }});
        }});
        
        // Download functions
        function downloadSVG() {{
            const blob = new Blob([svgData], {{ type: 'image/svg+xml' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = '{repo_name.replace("/", "-")}-callgraph.svg';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }}
        
        function downloadLLMJSON() {{
            const blob = new Blob([JSON.stringify(llmData, null, 2)], {{ type: 'application/json' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = '{repo_name.replace("/", "-")}-llm-optimized.json';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }}
        
        console.log('Call graph loaded with', cy.nodes().length, 'nodes and', cy.edges().length, 'edges');
        console.log('Click nodes to see code snippets! Download options available.');
    </script>
</body>
</html>
    """

    with open(output_file, "w") as f:
        f.write(html_content)


def main():
    """Command-line interface for call graph analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze function call graphs in repositories"
    )
    parser.add_argument("url", help="GitHub repository URL")
    parser.add_argument(
        "--include", nargs="*", help='Include patterns (e.g., "*.py" "src/")'
    )
    parser.add_argument(
        "--exclude", nargs="*", help='Exclude patterns (e.g., "*test*" "docs/")'
    )
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--viz", help="Generate HTML visualization file")
    parser.add_argument("--svg", help="Export SVG file")
    parser.add_argument("--llm-json", help="Export LLM-optimized JSON file")
    parser.add_argument("--summary", action="store_true", help="Show summary only")

    args = parser.parse_args()

    try:
        # Create repo analyzer with filters
        repo_analyzer = RepoAnalyzer(
            include_patterns=args.include,
            exclude_patterns=args.exclude or ["*test*", "*spec*", "docs/", "*.md"],
        )

        # Create call graph analyzer
        cg_analyzer = CallGraphAnalyzer(repo_analyzer)

        print(f"🚀 Analyzing call graph for: {args.url}")
        result = cg_analyzer.analyze_repository(args.url)

        # Show summary
        cg = result["call_graph"]
        repo = result["repository"]

        print(f"\n📊 CALL GRAPH ANALYSIS")
        print("=" * 50)
        print(f"Repository: {repo['owner']}/{repo['name']}")
        print(f"Functions found: {cg['total_functions']}")
        print(f"Function calls: {cg['total_calls']}")
        print(f"Files analyzed: {cg['files_analyzed']}")
        print(f"Languages: {', '.join(cg['languages_found'])}")

        viz = result["visualization"]["summary"]
        print(f"Resolved calls: {viz['total_edges']}")
        print(f"Unresolved calls: {viz['unresolved_calls']}")

        if not args.summary:
            # Show top functions by call count
            call_counts = {}
            for rel in result["relationships"]:
                if rel["is_resolved"]:
                    caller = rel["caller"]
                    call_counts[caller] = call_counts.get(caller, 0) + 1

            if call_counts:
                print(f"\nTop functions by outgoing calls:")
                sorted_funcs = sorted(
                    call_counts.items(), key=lambda x: x[1], reverse=True
                )
                for func_id, count in sorted_funcs[:5]:
                    func_name = func_id.split(":")[-1]
                    print(f"  {func_name}: {count} calls")

        # Save output
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\n💾 Analysis saved to: {args.output}")

        # Generate visualization
        if args.viz:
            generate_html_visualization(result, args.viz)
            print(f"🎨 Visualization saved to: {args.viz}")

        # Export SVG
        if args.svg:
            svg_data = generate_svg_export(result)
            with open(args.svg, "w") as f:
                f.write(svg_data)
            print(f"📐 SVG export saved to: {args.svg}")

        # Export LLM-optimized JSON
        if args.llm_json:
            llm_data = generate_llm_optimized_json(result)
            with open(args.llm_json, "w") as f:
                json.dump(llm_data, f, indent=2, default=str)
            print(f"🤖 LLM-optimized JSON saved to: {args.llm_json}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
