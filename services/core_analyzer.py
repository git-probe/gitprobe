"""
GitProbe Core Analyzer
Simplified, consolidated analyzer classes moved from root files.
"""

import ast
import json
import os
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, asdict


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
    code_snippet: Optional[str] = None
    complexity_score: Optional[int] = None


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
        func_info = self._extract_function_info(node)
        self.functions.append(func_info)
        self._extract_function_calls(node, func_info.name)

    def visit_AsyncFunctionDef(self, node):
        """Visit async function definition."""
        self.visit_FunctionDef(node)

    def _extract_function_info(self, node) -> FunctionInfo:
        """Extract information from a function definition node."""
        # Get parameters
        params = [arg.arg for arg in node.args.args]

        # Get docstring
        docstring = None
        if (node.body and isinstance(node.body[0], ast.Expr) 
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)):
            docstring = node.body[0].value.value

        # Extract code snippet
        start_line = node.lineno - 1
        end_line = (node.end_lineno or node.lineno) - 1
        code_snippet = "\n".join(self.lines[start_line:end_line + 1])

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

        return FunctionInfo(
            name=node.name,
            file_path=self.file_path,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            parameters=params,
            docstring=docstring,
            scope=scope,
            calls=[],
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


class RepoAnalyzer:
    """Simple repository analyzer."""
    
    def __init__(self, include_patterns=None, exclude_patterns=None):
        self.include_patterns = include_patterns or ["*.py"]
        self.exclude_patterns = exclude_patterns or ["*test*", "*spec*", "docs/", "*.md"]

    def analyze_repository(self, github_url: str) -> Dict:
        """Analyze repository and return file structure."""
        temp_dir = tempfile.mkdtemp(prefix="gitprobe_repo_")
        
        try:
            self._clone_repository(github_url, temp_dir)
            repo_info = self._parse_github_url(github_url)
            file_tree = self._build_file_tree(temp_dir)
            
            return {
                "repository": repo_info,
                "file_tree": file_tree,
                "summary": {
                    "total_files": self._count_files(file_tree),
                    "total_size_kb": self._calculate_size(file_tree)
                }
            }
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def _clone_repository(self, github_url: str, target_dir: str):
        """Clone repository to target directory."""
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", github_url, target_dir],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to clone repository: {e.stderr}")

    def _parse_github_url(self, github_url: str) -> Dict:
        """Parse GitHub URL to extract owner and repo name."""
        parts = github_url.rstrip('/').split('/')
        if len(parts) >= 2:
            owner = parts[-2]
            name = parts[-1].replace('.git', '')
            return {
                "owner": owner,
                "name": name,
                "full_name": f"{owner}/{name}",
                "url": github_url
            }
        return {"owner": "unknown", "name": "unknown", "full_name": "unknown", "url": github_url}

    def _build_file_tree(self, repo_dir: str) -> Dict:
        """Build file tree structure."""
        def build_tree(path: Path, base_path: Path) -> Dict:
            relative_path = path.relative_to(base_path)
            
            if path.is_file():
                size = path.stat().st_size
                return {
                    "type": "file",
                    "name": path.name,
                    "path": str(relative_path),
                    "extension": path.suffix,
                    "size_kb": round(size / 1024, 2),
                    "estimated_tokens": size // 4  # Rough estimate
                }
            else:
                children = []
                try:
                    for child in sorted(path.iterdir()):
                        if not child.name.startswith('.'):
                            children.append(build_tree(child, base_path))
                except PermissionError:
                    pass
                
                return {
                    "type": "directory",
                    "name": path.name,
                    "path": str(relative_path),
                    "children": children
                }

        return build_tree(Path(repo_dir), Path(repo_dir))

    def _count_files(self, tree: Dict) -> int:
        """Count total files in tree."""
        if tree["type"] == "file":
            return 1
        return sum(self._count_files(child) for child in tree.get("children", []))

    def _calculate_size(self, tree: Dict) -> float:
        """Calculate total size in KB."""
        if tree["type"] == "file":
            return tree.get("size_kb", 0)
        return sum(self._calculate_size(child) for child in tree.get("children", []))


class CallGraphAnalyzer:
    """Main class for analyzing function call graphs."""

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
        self.functions: Dict[str, FunctionInfo] = {}
        self.call_relationships: List[CallRelationship] = []

    def analyze_repository(self, github_url: str) -> Dict:
        """Analyze repository and extract call graph."""
        # Get file structure
        repo_result = self.repo_analyzer.analyze_repository(github_url)
        
        # Extract code files
        code_files = self._extract_code_files(repo_result["file_tree"])
        
        # Clone and analyze
        temp_dir = tempfile.mkdtemp(prefix="gitprobe_callgraph_")
        try:
            self.repo_analyzer._clone_repository(github_url, temp_dir)
            
            # Analyze each code file
            for file_info in code_files:
                self._analyze_code_file(temp_dir, file_info)
            
            # Resolve relationships
            self._resolve_call_relationships()
            
            # Generate visualization
            viz_data = self._generate_visualization_data()
            
            return {
                "repository": repo_result["repository"],
                "file_tree": repo_result["file_tree"],  # Include file tree!
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
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def _extract_code_files(self, file_tree: Dict) -> List[Dict]:
        """Extract code files from file tree."""
        code_files = []
        
        def traverse(tree):
            if tree["type"] == "file":
                ext = tree.get("extension", "").lower()
                if ext in self.CODE_EXTENSIONS:
                    name = tree["name"].lower()
                    if not any(skip in name for skip in ["test", "spec", "config", "setup"]):
                        code_files.append({
                            "path": tree["path"],
                            "name": tree["name"],
                            "extension": ext,
                            "language": self.CODE_EXTENSIONS[ext],
                            "size_kb": tree.get("size_kb", 0),
                            "estimated_tokens": tree.get("estimated_tokens", 0),
                        })
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
            cytoscape_elements.append({
                "data": {
                    "id": func_id,
                    "label": func_info.name,
                    "file": func_info.file_path,
                    "type": "method" if func_info.is_method else "function",
                },
                "classes": f"node-{'method' if func_info.is_method else 'function'}"
            })
        
        # Add edges
        for rel in self.call_relationships:
            if rel.is_resolved:
                cytoscape_elements.append({
                    "data": {
                        "id": f"{rel.caller}->{rel.callee}",
                        "source": rel.caller,
                        "target": rel.callee,
                    },
                    "classes": "edge-call"
                })
        
        return {
            "cytoscape": {"elements": cytoscape_elements},
            "summary": {
                "total_nodes": len(self.functions),
                "total_edges": len([r for r in self.call_relationships if r.is_resolved]),
            }
        } 