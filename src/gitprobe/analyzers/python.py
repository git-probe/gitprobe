"""
Python AST Analyzer

Analyzes Python source code using the Abstract Syntax Tree (AST) to extract
function definitions, method information, and function call relationships.
"""

import ast
import logging
from typing import List, Tuple, Optional
from pathlib import Path

from gitprobe.models.core import Function, CallRelationship
from gitprobe.core.analysis_limits import AnalysisLimits, create_python_limits

logger = logging.getLogger(__name__)


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

    def __init__(self, file_path: str, content: str, limits: Optional[AnalysisLimits] = None):
        """
        Initialize the Python AST analyzer.

        Args:
            file_path: Path to the Python file being analyzed
            content: Raw content of the Python file
            limits: Analysis limits configuration
        """
        self.file_path = file_path
        self.content = content
        self.lines = content.splitlines()
        self.functions: List[Function] = []
        self.call_relationships: List[CallRelationship] = []
        self.current_class_name: str | None = None
        self.current_function_name: str | None = None
        self.limits = limits or create_python_limits()

    def generic_visit(self, node):
        """Override generic_visit to continue AST traversal with limit checks."""
        if self.limits.should_stop():
            return
        super().generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definition and track current class context."""
        if self.limits.should_stop():
            return

        # Count class definitions as meaningful nodes
        if self.limits.increment():
            return

        self.current_class_name = node.name
        self.generic_visit(node)
        self.current_class_name = None

    def _process_function_node(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        """Helper to process both sync and async function definitions."""
        if self.limits.should_stop():
            return

        # Count function definitions as meaningful nodes
        if self.limits.increment():
            return

        self.current_function_name = node.name

        # Extract function information
        function_obj = Function(
            name=node.name,
            file_path=str(self.file_path),
            line_start=node.lineno,
            line_end=node.end_lineno,
            parameters=[arg.arg for arg in node.args.args],
            docstring=ast.get_docstring(node),
            is_method=self.current_class_name is not None,
            class_name=self.current_class_name,
            code_snippet="\n".join(self.lines[node.lineno - 1 : node.end_lineno or node.lineno]),
        )

        # Apply minimal filtering - only exclude obvious noise
        if self._should_include_function(function_obj):
            # Check global limits before adding function
            if self.limits.can_add_function():
                self.functions.append(function_obj)
                # Track in global counter
                if self.limits.add_function():
                    return  # Global limit reached, stop analysis
            else:
                return  # Can't add more functions, stop analysis

        self.generic_visit(node)
        self.current_function_name = None

    def _should_include_function(self, func: Function) -> bool:
        """Determine if a function should be included in analysis."""
        # Very minimal filtering - keep almost everything

        # Skip functions that are clearly test fixtures or internal
        if func.name.startswith("_test_") or func.name in ["setUp", "tearDown"]:
            return False

        # Keep everything else - let the graph cleanup handle disconnected functions
        return True

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definition and extract function information."""
        if self.limits.should_stop():
            return
        self._process_function_node(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Visit async function definition and extract function information."""
        if self.limits.should_stop():
            return
        self._process_function_node(node)

    def visit_Call(self, node: ast.Call):
        """Visit function call nodes and record relationships."""
        if self.limits.should_stop():
            return

        # Count function calls as meaningful nodes
        if self.limits.increment():
            return

        if self.current_function_name:
            call_name = self._get_call_name(node.func)
            if call_name:
                # Check global limits before adding relationship
                if self.limits.can_add_relationship():
                    relationship = CallRelationship(
                        caller=f"{self.file_path}:{self.current_function_name}",
                        callee=call_name,
                        call_line=node.lineno,
                        is_resolved=False,
                    )
                    self.call_relationships.append(relationship)
                    # Track in global counter
                    if self.limits.add_relationship():
                        return  # Global limit reached, stop analysis
                else:
                    return  # Can't add more relationships, stop analysis
        self.generic_visit(node)

    def _get_call_name(self, node) -> str | None:
        """
        Extract function name from a call node.
        Handles simple names, attributes (obj.method), and filters built-ins.
        """
        # Reduced builtin list - only exclude obvious globals
        PYTHON_BUILTINS = {
            "print",
            "len",
            "str",
            "int",
            "float",
            "bool",
            "list",
            "dict",
            "range",
            "enumerate",
            "zip",
            "isinstance",
            "hasattr",
            "open",
            "super",
            "__import__",
        }

        if isinstance(node, ast.Name):
            if node.id in PYTHON_BUILTINS:
                return None
            return node.id
        elif isinstance(node, ast.Attribute):
            # For obj.method(), try to preserve context
            if isinstance(node.value, ast.Name):
                return f"{node.value.id}.{node.attr}"
            return node.attr  # Fallback for nested attributes
        return None

    def analyze(self):
        """Analyze the Python file and extract functions and relationships."""
        if not self.limits.start_new_file():
            logger.info(f"Skipping {self.file_path} - global limits reached")
            return

        try:
            tree = ast.parse(self.content)
            self.visit(tree)

            logger.info(
                f"Python analysis complete for {self.file_path}: {len(self.functions)} functions, "
                f"{len(self.call_relationships)} relationships, "
                f"nodes_processed={self.limits.nodes_processed}"
            )
        except SyntaxError as e:
            logger.warning(f"⚠️ Could not parse {self.file_path}: {e}")
        except Exception as e:
            logger.error(f"⚠️ Error analyzing {self.file_path}: {e}", exc_info=True)


def analyze_python_file(
    file_path: str, content: str, limits: Optional[AnalysisLimits] = None
) -> Tuple[List[Function], List[CallRelationship]]:
    """
    Analyze a Python file and return functions and relationships.

    Args:
        file_path: Path to the Python file
        content: Content of the Python file
        limits: Analysis limits configuration

    Returns:
        tuple: (functions, call_relationships)
    """
    if limits is None:
        limits = create_python_limits()

    analyzer = PythonASTAnalyzer(file_path, content, limits)
    analyzer.analyze()
    return analyzer.functions, analyzer.call_relationships
