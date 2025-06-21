"""
Python AST Analyzer

Analyzes Python source code using the Abstract Syntax Tree (AST) to extract
function definitions, method information, and function call relationships.
"""

import ast
import logging
from typing import List, Tuple, Optional
from pathlib import Path
from models.core import Function, CallRelationship

logger = logging.getLogger(__name__)


class GlobalNodeCounter:
    """Shared counter for tracking nodes across all files of the same language."""
    
    def __init__(self, max_nodes: int = 800):
        self.max_nodes = max_nodes
        self.nodes_processed = 0
        self.limit_reached = False
    
    def increment(self) -> bool:
        """Increment counter and return True if limit reached."""
        if self.limit_reached:
            return True
        
        self.nodes_processed += 1
        if self.nodes_processed >= self.max_nodes:
            self.limit_reached = True
            logger.warning(f"Global Python node limit of {self.max_nodes} reached. Stopping all Python analysis.")
            return True
        return False
    
    def should_stop(self) -> bool:
        """Check if analysis should stop."""
        return self.limit_reached


class PythonASTAnalyzer(ast.NodeVisitor):
    """
    AST visitor to extract function information from Python code.

    This analyzer traverses Python AST nodes to identify:
    - Function and method definitions
    - Function parameters and docstrings
    - Function call relationships
    - Class context for methods
    - Code snippets and line numbers
    """

    def __init__(self, file_path: str, content: str, global_counter: Optional[GlobalNodeCounter] = None):
        """
        Initialize the Python AST analyzer.

        Args:
            file_path: Path to the Python file being analyzed
            content: Raw content of the Python file
            global_counter: Shared counter for tracking nodes across all Python files
        """
        self.file_path = file_path
        self.content = content
        self.lines = content.splitlines()
        self.functions: List[Function] = []
        self.call_relationships: List[CallRelationship] = []
        self.current_class_name: str | None = None
        self.current_function_name: str | None = None
        self.global_counter = global_counter or GlobalNodeCounter()

    def generic_visit(self, node):
        """Override generic_visit to continue AST traversal without counting every node."""
        # Don't count every node - that's too aggressive
        # Only count specific meaningful nodes in visit methods
        if self.global_counter.should_stop():
            return
        super().generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        """
        Visit class definition and track current class context.

        Args:
            node: AST ClassDef node
        """
        if self.global_counter.should_stop():
            return
            
        # Count class definitions as meaningful nodes
        if self.global_counter.increment():
            return
            
        self.current_class_name = node.name
        self.generic_visit(node)
        self.current_class_name = None

    def _process_function_node(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        """Helper to process both sync and async function definitions."""
        if self.global_counter.should_stop():
            return
        
        # Count function definitions as meaningful nodes
        if self.global_counter.increment():
            return
            
        self.current_function_name = node.name
        
        file_path_str = str(self.file_path)

        function_obj = Function(
            name=node.name,
            file_path=file_path_str,
            line_start=node.lineno,
            line_end=node.end_lineno,
            parameters=[arg.arg for arg in node.args.args],
            docstring=ast.get_docstring(node),
            is_method=self.current_class_name is not None,
            class_name=self.current_class_name,
            code_snippet="\n".join(self.lines[node.lineno - 1:node.end_lineno or node.lineno])
        )
        self.functions.append(function_obj)
        self.generic_visit(node)
        self.current_function_name = None

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definition and extract function information."""
        if self.global_counter.should_stop():
            return
        self._process_function_node(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Visit async function definition and extract function information."""
        if self.global_counter.should_stop():
            return
        self._process_function_node(node)

    def visit_Call(self, node: ast.Call):
        """Visit function call nodes and record relationships."""
        if self.global_counter.should_stop():
            return
        
        # Count function calls as meaningful nodes
        if self.global_counter.increment():
            return
            
        if self.current_function_name:
            call_name = self._get_call_name(node.func)
            if call_name:
                relationship = CallRelationship(
                    caller=f"{self.file_path}:{self.current_function_name}",
                    callee=call_name,
                    call_line=node.lineno,
                    is_resolved=False,
                )
                self.call_relationships.append(relationship)
        self.generic_visit(node)

    def _get_call_name(self, node) -> str | None:
        """
        Extract function name from a call node.
        Handles simple names, attributes (obj.method), and filters built-ins.
        """
        PYTHON_BUILTINS = {
            "print", "len", "str", "int", "float", "bool", "list", "dict",
            "set", "tuple", "range", "enumerate", "zip", "map", "filter",
            "sorted", "reversed", "sum", "min", "max", "abs", "round",
            "isinstance", "hasattr", "getattr", "setattr", "open"
        }

        if isinstance(node, ast.Name):
            if node.id in PYTHON_BUILTINS:
                return None
            return node.id
        elif isinstance(node, ast.Attribute):
            # This can be complex, e.g., self.foo.bar(). We want 'bar' or 'foo.bar'
            # This simplified version will get the rightmost part.
            if isinstance(node.value, ast.Name):
                return f"{node.value.id}.{node.attr}"
            return node.attr # Fallback for nested attributes
        return None

    def analyze(self):
        if self.global_counter.should_stop():
            logger.info(f"Skipping {self.file_path} - global Python node limit already reached")
            return
            
        try:
            tree = ast.parse(self.content)
            self.visit(tree)
            
            logger.info(
                f"Python analysis complete for {self.file_path}: {len(self.functions)} functions, "
                f"{len(self.call_relationships)} relationships, "
                f"global_nodes_processed={self.global_counter.nodes_processed}"
            )
        except SyntaxError as e:
            logger.warning(f"⚠️ Could not parse {self.file_path}: {e}")
        except Exception as e:
            logger.error(f"⚠️ Error analyzing {self.file_path}: {e}", exc_info=True)


def analyze_python_file(file_path: str, content: str, global_counter: Optional[GlobalNodeCounter] = None) -> Tuple[List[Function], List[CallRelationship]]:
    """
    Analyze a Python file and return functions and relationships.

    This is the main entry point for Python file analysis that can be
    called by the CallGraphAnalyzer.

    Args:
        file_path: Path to the Python file
        content: Content of the Python file
        global_counter: Shared counter for tracking nodes across all Python files

    Returns:
        tuple: (functions, call_relationships)

    Raises:
        SyntaxError: If the Python file has syntax errors
        Exception: If parsing fails for other reasons
    """
    analyzer = PythonASTAnalyzer(file_path, content, global_counter)
    analyzer.analyze()
    return analyzer.functions, analyzer.call_relationships
