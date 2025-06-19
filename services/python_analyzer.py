"""
Python AST Analyzer

Analyzes Python source code using the Abstract Syntax Tree (AST) to extract
function definitions, method information, and function call relationships.
"""

import ast
from typing import List, Optional
from models.core import Function, CallRelationship


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

    def __init__(self, file_path: str, content: str):
        """
        Initialize the Python AST analyzer.

        Args:
            file_path: Path to the Python file being analyzed
            content: Raw content of the Python file
        """
        self.file_path = file_path
        self.content = content
        self.lines = content.split("\n")
        self.functions: List[Function] = []
        self.current_class = None
        self.call_relationships: List[CallRelationship] = []

    def visit_ClassDef(self, node):
        """
        Visit class definition and track current class context.

        Args:
            node: AST ClassDef node
        """
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node):
        """
        Visit function definition and extract function information.

        Args:
            node: AST FunctionDef node
        """
        func_info = self._extract_function_info(node)
        self.functions.append(func_info)
        self._extract_function_calls(node, func_info.name)

    def visit_AsyncFunctionDef(self, node):
        """
        Visit async function definition.

        Args:
            node: AST AsyncFunctionDef node
        """
        func_info = self._extract_function_info(node)
        self.functions.append(func_info)
        self._extract_function_calls(node, func_info.name)

    def _extract_function_info(self, node) -> Function:
        """
        Extract comprehensive information from a function definition node.

        Args:
            node: AST FunctionDef or AsyncFunctionDef node

        Returns:
            Function: Complete function metadata
        """
        # Extract parameters
        params = [arg.arg for arg in node.args.args]

        # Extract docstring
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

        # Determine context and scope
        class_name = None
        is_method = False

        if self.current_class:
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
        """
        Extract function calls from within a function's body.

        Args:
            func_node: AST function node to analyze
            func_name: Name of the containing function
        """

        class CallVisitor(ast.NodeVisitor):
            """Inner visitor to find function calls within a function."""

            def __init__(self, analyzer):
                self.analyzer = analyzer
                self.caller_name = func_name

            def visit_Call(self, node):
                """Visit function call nodes and record relationships."""
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
                """
                Extract function name from call node.

                Args:
                    node: AST node representing the function being called

                Returns:
                    str: Function name or None if not extractable
                """
                # Add this set at the top of _get_call_name method:
                PYTHON_BUILTINS = {
                    "print",
                    "len",
                    "str",
                    "int",
                    "float",
                    "bool",
                    "list",
                    "dict",
                    "set",
                    "tuple",
                    "range",
                    "enumerate",
                    "zip",
                    "map",
                    "filter",
                    "sorted",
                    "reversed",
                    "sum",
                    "min",
                    "max",
                    "abs",
                    "round",
                    "isinstance",
                    "hasattr",
                    "getattr",
                    "setattr",
                }

                # Then modify the return logic:
                if isinstance(node, ast.Name):
                    name = node.id
                    if name in PYTHON_BUILTINS:
                        return None  # Skip built-ins
                    return name
                elif isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name):
                        return f"{node.value.id}.{node.attr}"
                    else:
                        return node.attr
                return None

        call_visitor = CallVisitor(self)
        call_visitor.visit(func_node)


def analyze_python_file(
    file_path: str, content: str
) -> tuple[List[Function], List[CallRelationship]]:
    """
    Analyze a Python file and return functions and relationships.

    This is the main entry point for Python file analysis that can be
    called by the CallGraphAnalyzer.

    Args:
        file_path: Path to the Python file
        content: Content of the Python file

    Returns:
        tuple: (functions, call_relationships)

    Raises:
        SyntaxError: If the Python file has syntax errors
        Exception: If parsing fails for other reasons
    """
    try:
        tree = ast.parse(content)
        analyzer = PythonASTAnalyzer(file_path, content)
        analyzer.visit(tree)
        return analyzer.functions, analyzer.call_relationships
    except SyntaxError as e:
        print(f"⚠️ Syntax error in {file_path}: {str(e)}")
        return [], []
    except Exception as e:
        print(f"⚠️ Error parsing {file_path}: {str(e)}")
        return [], []
