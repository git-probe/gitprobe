"""
GitProbe Core Analyzer
"""

import ast
from pathlib import Path
from typing import Dict, List, Set, Optional

from models.core import Function, CallRelationship


class PythonASTAnalyzer(ast.NodeVisitor):
    """AST visitor to extract function information from Python code."""

    def __init__(self, file_path: str, content: str):
        self.file_path = file_path
        self.content = content
        self.lines = content.split("\n")
        self.functions: List[Function] = []
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
        func_info = self._extract_function_info(node)
        self.functions.append(func_info)
        self._extract_function_calls(node, func_info.name)

    def visit_AsyncFunctionDef(self, node):
        """Visit async function definition."""
        self.visit_FunctionDef(node)

    def _extract_function_info(self, node) -> Function:
        """Extract information from a function definition node."""
        # Get parameters
        params = [arg.arg for arg in node.args.args]

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
        start_line = node.lineno - 1
        end_line = (node.end_lineno or node.lineno) - 1
        code_snippet = "\n".join(self.lines[start_line : end_line + 1])

        # Calculate complexity
        complexity_score = end_line - start_line + 1

        # Determine scope
        scope = "module"
        class_name = None
        is_method = False

        if self.current_class:
            scope = f"class:{self.current_class}"
            class_name = self.current_class
            is_method = True

        return Function(
            name=node.name,
            file_path=self.file_path,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            parameters=params,
            docstring=docstring,
            is_method=is_method,
            class_name=class_name,
            code_snippet=code_snippet,
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
                    relationship = CallRelationship(
                        caller=f"{self.analyzer.file_path}:{self.caller_name}",
                        callee=call_name,
                        call_line=node.lineno,
                        is_resolved=False,
                    )
                    self.analyzer.call_relationships.append(relationship)
                self.generic_visit(node)

            def _get_call_name(self, node) -> Optional[str]:
                """Extract function name from call node."""
                if isinstance(node, ast.Name):
                    return node.id
                elif isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name):
                        return f"{node.value.id}.{node.attr}"
                    else:
                        return node.attr
                return None

        call_visitor = CallVisitor(self)
        call_visitor.visit(func_node)


class CallGraphAnalyzer:
    """Analyzer for extracting function call graphs from code files."""

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

    def __init__(self):
        """Initialize the call graph analyzer."""
        self.functions: Dict[str, Function] = {}
        self.call_relationships: List[CallRelationship] = []

    def analyze_code_files(self, code_files: List[Dict], base_dir: str) -> Dict:
        """
        Analyze a list of code files from a local directory.

        Args:
            code_files: List of file info dicts with path, language, etc.
            base_dir: Base directory path where files are located

        Returns:
            Dict with functions, relationships, and visualization data
        """
        # Reset state for new analysis
        self.functions = {}
        self.call_relationships = []

        # Analyze each code file
        for file_info in code_files:
            self._analyze_code_file(base_dir, file_info)

        # Resolve relationships
        self._resolve_call_relationships()

        # Generate visualization
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
        """Extract code files from file tree structure."""
        code_files = []

        def traverse(tree):
            if tree["type"] == "file":
                ext = tree.get("extension", "").lower()
                if ext in self.CODE_EXTENSIONS:
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
        func_lookup = {}
        for func_id, func_info in self.functions.items():
            func_lookup[func_info.name] = func_id

        for relationship in self.call_relationships:
            callee_name = relationship.callee

            if callee_name in func_lookup:
                relationship.callee = func_lookup[callee_name]
                relationship.is_resolved = True
            elif "." in callee_name:
                method_name = callee_name.split(".")[-1]
                if method_name in func_lookup:
                    relationship.callee = func_lookup[method_name]
                    relationship.is_resolved = True

    def _generate_visualization_data(self) -> Dict:
        """Generate visualization data."""
        # Simple visualization data
        cytoscape_elements = []

        # Add nodes
        for func_id, func_info in self.functions.items():
            cytoscape_elements.append(
                {
                    "data": {
                        "id": func_id,
                        "label": func_info.name,
                        "file": func_info.file_path,
                        "type": "method" if func_info.is_method else "function",
                    },
                    "classes": f"node-{'method' if func_info.is_method else 'function'}",
                }
            )

        # Add edges
        for rel in self.call_relationships:
            if rel.is_resolved:
                cytoscape_elements.append(
                    {
                        "data": {
                            "id": f"{rel.caller}->{rel.callee}",
                            "source": rel.caller,
                            "target": rel.callee,
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
            },
        }
