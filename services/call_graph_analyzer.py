"""
Call Graph Analyzer

Central orchestrator for multi-language call graph analysis.
Coordinates language-specific analyzers to build comprehensive call graphs
across different programming languages in a repository.
"""

from pathlib import Path
from typing import Dict, List
from models.core import Function, CallRelationship
from utils.patterns import CODE_EXTENSIONS


class CallGraphAnalyzer:
    """
    Multi-language call graph analyzer.
    
    This analyzer orchestrates language-specific AST analyzers to build
    comprehensive call graphs across different programming languages.
    
    Supported languages:
    - Python (fully supported)
    - JavaScript (planned)
    - TypeScript (planned)
    """

    def __init__(self):
        """Initialize the call graph analyzer."""
        self.functions: Dict[str, Function] = {}
        self.call_relationships: List[CallRelationship] = []

    def analyze_code_files(self, code_files: List[Dict], base_dir: str) -> Dict:
        """
        Analyze a list of code files from multiple languages.

        Args:
            code_files: List of file info dicts with path, language, etc.
            base_dir: Base directory path where files are located

        Returns:
            Dict with functions, relationships, and visualization data
        """
        # Reset state for new analysis
        self.functions = {}
        self.call_relationships = []

        # Analyze each code file based on its language
        for file_info in code_files:
            self._analyze_code_file(base_dir, file_info)

        # Resolve cross-language relationships
        self._resolve_call_relationships()

        # Generate visualization data
        viz_data = self._generate_visualization_data()

        return {
            "call_graph": {
                "total_functions": len(self.functions),
                "total_calls": len(self.call_relationships),
                "languages_found": list(set(f.get("language") for f in code_files)),
                "files_analyzed": len(code_files),
            },
            "functions": [func.model_dump() for func in self.functions.values()],
            "relationships": [rel.model_dump() for rel in self.call_relationships],
            "visualization": viz_data,
        }

    def extract_code_files(self, file_tree: Dict) -> List[Dict]:
        """
        Extract code files from file tree structure.
        
        Filters files based on supported extensions and excludes test/config files.
        
        Args:
            file_tree: Nested dictionary representing file structure
            
        Returns:
            List of code file information dictionaries
        """
        code_files = []

        def traverse(tree):
            if tree["type"] == "file":
                ext = tree.get("extension", "").lower()
                if ext in CODE_EXTENSIONS:
                    name = tree["name"].lower()
                    # Skip test, spec, config, and setup files
                    if not any(
                        skip in name for skip in ["test", "spec", "config", "setup"]
                    ):
                        code_files.append(
                            {
                                "path": tree["path"],
                                "name": tree["name"],
                                "extension": ext,
                                "language": CODE_EXTENSIONS[ext],
                            }
                        )
            elif tree["type"] == "directory" and tree.get("children"):
                for child in tree["children"]:
                    traverse(child)

        traverse(file_tree)
        return code_files

    def _analyze_code_file(self, repo_dir: str, file_info: Dict):
        """
        Analyze a single code file based on its language.
        
        Routes to appropriate language-specific analyzer.
        
        Args:
            repo_dir: Repository directory path
            file_info: File information dictionary
        """
        file_path = Path(repo_dir) / file_info["path"]

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Route to appropriate language analyzer
            if file_info["language"] == "python":
                self._analyze_python_file(file_info["path"], content)
            elif file_info["language"] == "javascript":
                self._analyze_javascript_file(file_info["path"], content)
            elif file_info["language"] == "typescript":
                self._analyze_typescript_file(file_info["path"], content)
            else:
                print(f"⚠️ Unsupported language: {file_info['language']} for {file_info['path']}")
                
        except Exception as e:
            print(f"⚠️ Error analyzing {file_info['path']}: {str(e)}")

    def _analyze_python_file(self, file_path: str, content: str):
        """
        Analyze Python file using Python AST analyzer.
        
        Args:
            file_path: Relative path to the Python file
            content: File content string
        """
        from .python_analyzer import analyze_python_file
        
        functions, relationships = analyze_python_file(file_path, content)
        
        # Store functions with unique identifiers
        for func in functions:
            func_id = f"{file_path}:{func.name}"
            self.functions[func_id] = func

        # Store call relationships
        self.call_relationships.extend(relationships)

    def _analyze_javascript_file(self, file_path: str, content: str):
        """
        Analyze JavaScript file using JavaScript AST analyzer.
        
        Args:
            file_path: Relative path to the JavaScript file
            content: File content string
        """
        from .js_analyzer import analyze_javascript_file
        
        functions, relationships = analyze_javascript_file(file_path, content)
        
        # Store functions with unique identifiers
        for func in functions:
            func_id = f"{file_path}:{func.name}"
            self.functions[func_id] = func

        # Store call relationships
        self.call_relationships.extend(relationships)

    def _analyze_typescript_file(self, file_path: str, content: str):
        """
        Analyze TypeScript file using TypeScript AST analyzer.
        
        Args:
            file_path: Relative path to the TypeScript file
            content: File content string
        """
        from .js_analyzer import analyze_typescript_file
        
        functions, relationships = analyze_typescript_file(file_path, content)
        
        # Store functions with unique identifiers
        for func in functions:
            func_id = f"{file_path}:{func.name}"
            self.functions[func_id] = func

        # Store call relationships
        self.call_relationships.extend(relationships)

    def _resolve_call_relationships(self):
        """
        Resolve function call relationships across all languages.
        
        Attempts to match function calls to actual function definitions,
        handling cross-language calls where possible.
        """
        # Build lookup table of all functions
        func_lookup = {}
        for func_id, func_info in self.functions.items():
            func_lookup[func_info.name] = func_id

        # Resolve relationships
        for relationship in self.call_relationships:
            callee_name = relationship.callee

            # Direct name match
            if callee_name in func_lookup:
                relationship.callee = func_lookup[callee_name]
                relationship.is_resolved = True
            # Method call resolution (obj.method -> method)
            elif "." in callee_name:
                method_name = callee_name.split(".")[-1]
                if method_name in func_lookup:
                    relationship.callee = func_lookup[method_name]
                    relationship.is_resolved = True

    def _generate_visualization_data(self) -> Dict:
        """
        Generate visualization data for graph rendering.
        
        Creates Cytoscape.js compatible graph data with nodes and edges.
        
        Returns:
            Dict: Visualization data with cytoscape elements and summary
        """
        cytoscape_elements = []

        # Add function nodes
        for func_id, func_info in self.functions.items():
            # Determine node styling based on function type and language
            node_classes = []
            if func_info.is_method:
                node_classes.append("node-method")
            else:
                node_classes.append("node-function")
            
            # Add language-specific styling
            file_ext = Path(func_info.file_path).suffix.lower()
            if file_ext == ".py":
                node_classes.append("lang-python")
            elif file_ext == ".js":
                node_classes.append("lang-javascript")
            elif file_ext == ".ts":
                node_classes.append("lang-typescript")

            cytoscape_elements.append(
                {
                    "data": {
                        "id": func_id,
                        "label": func_info.name,
                        "file": func_info.file_path,
                        "type": "method" if func_info.is_method else "function",
                        "language": CODE_EXTENSIONS.get(file_ext, "unknown")
                    },
                    "classes": " ".join(node_classes),
                }
            )

        # Add call relationship edges
        for rel in self.call_relationships:
            if rel.is_resolved:
                cytoscape_elements.append(
                    {
                        "data": {
                            "id": f"{rel.caller}->{rel.callee}",
                            "source": rel.caller,
                            "target": rel.callee,
                            "line": rel.call_line,
                        },
                        "classes": "edge-call",
                    }
                )

        return {
            "cytoscape": {"elements": cytoscape_elements},
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