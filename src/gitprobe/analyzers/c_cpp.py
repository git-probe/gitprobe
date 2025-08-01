"""
Advanced C/C++ analyzer using Tree-sitter for accurate AST parsing.

This module provides C and C++ source code analysis using tree-sitter,
which is faster and more reliable than libclang or pycparser for basic
function and call relationship extraction.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional, Set
from pathlib import Path

from tree_sitter import Parser, Language
import tree_sitter_c
import tree_sitter_cpp

from gitprobe.models.core import Function, CallRelationship
from gitprobe.core.analysis_limits import AnalysisLimits, create_c_cpp_limits

logger = logging.getLogger(__name__)


class TreeSitterCAnalyzer:
    """C/C++ analyzer using tree-sitter for proper AST parsing."""

    def __init__(
        self,
        file_path: str,
        content: str,
        language: str = "c",
        limits: Optional[AnalysisLimits] = None,
    ):
        self.file_path = str(file_path)
        self.content = content
        self.language = language.lower()
        self.lines = content.splitlines()
        self.functions: List[Function] = []
        self.call_relationships: List[CallRelationship] = []
        self.limits = limits or create_c_cpp_limits()

        is_cpp = (
            self.language == "cpp"
            or self.language == "c++"
            or Path(file_path).suffix.lower()
            in [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hxx", ".h++"]
        )

        try:
            if is_cpp:
                language_capsule = tree_sitter_cpp.language()
                self.language_obj = Language(language_capsule)
                self.parser = Parser(self.language_obj)
                logger.debug(
                    f"C++ parser initialized with language object: {type(self.language_obj)}"
                )
            else:
                language_capsule = tree_sitter_c.language()
                self.language_obj = Language(language_capsule)
                self.parser = Parser(self.language_obj)
                logger.debug(
                    f"C parser initialized with language object: {type(self.language_obj)}"
                )

            test_code = "int main() { return 0; }" if not is_cpp else "int main() { return 0; }"
            test_tree = self.parser.parse(bytes(test_code, "utf8"))
            if test_tree is None or test_tree.root_node is None:
                raise RuntimeError(f"Parser setup test failed for {self.language.upper()}")
            logger.debug(f"Parser test successful - root node type: {test_tree.root_node.type}")

        except Exception as e:
            logger.error(f"Failed to initialize {self.language.upper()} parser: {e}")
            self.parser = None
            self.language_obj = None

        logger.info(
            f"TreeSitterCAnalyzer initialized for {file_path} ({self.language.upper()}) with limits: {self.limits}"
        )

    def analyze(self) -> None:
        """Analyze C/C++ code using tree-sitter."""
        if not self.limits.start_new_file():
            logger.info(f"Skipping {self.file_path} - global limits reached")
            return

        if self.parser is None:
            logger.warning(f"Skipping {self.file_path} - parser initialization failed")
            return

        try:
            logger.debug(
                f"Attempting to parse {len(self.content)} bytes of {self.language.upper()} code"
            )
            logger.debug(f"Parser language object: {self.language_obj}")

            tree = self.parser.parse(bytes(self.content, "utf8"))

            if tree is None:
                raise ValueError("Parser returned None tree")

            root_node = tree.root_node
            if root_node is None:
                raise ValueError("Tree has no root node")

            logger.info(f"Parsed AST with root node type: {root_node.type}")

            if root_node.has_error:
                logger.warning(f"Parse tree contains errors for {self.file_path}")

            self._extract_functions(root_node)

            if not self.limits.should_stop():
                self._extract_calls(root_node)

            logger.info(
                f"Tree-sitter {self.language.upper()} analysis complete: "
                f"{len(self.functions)} functions, {len(self.call_relationships)} calls, "
                f"{self.limits.nodes_processed} nodes processed"
            )

        except Exception as e:
            logger.error(
                f"Tree-sitter {self.language.upper()} analysis failed for {self.file_path}: {e}",
                exc_info=True,
            )

    def _extract_functions(self, node):
        """Extract function definitions from the AST."""
        if self.limits.should_stop():
            return

        if node.type == "function_definition":
            func = self._create_function_from_node(node)
            if func:
                if self.limits.can_add_function():
                    self.functions.append(func)
                    if self.limits.add_function():
                        return
                else:
                    return
        elif node.type == "function_declarator":
            func = self._create_function_from_declarator(node)
            if func:
                if self.limits.can_add_function():
                    self.functions.append(func)
                    if self.limits.add_function():
                        return
                else:
                    return
        elif self.language in ["cpp", "c++"] and node.type in [
            "method_definition",
            "constructor_definition",
            "destructor_definition",
        ]:
            func = self._create_method_from_node(node)
            if func:
                if self.limits.can_add_function():
                    self.functions.append(func)
                    if self.limits.add_function():
                        return
                else:
                    return

        for child in node.children:
            self._extract_functions(child)
            if self.limits.should_stop():
                break

    def _create_function_from_node(self, node) -> Optional[Function]:
        """Create a Function object from a function_definition node."""
        try:
            declarator = self._find_child_by_type(node, "function_declarator")
            if not declarator:
                return None

            identifier = self._find_child_by_type(declarator, "identifier")
            if not identifier:
                return None

            func_name = self._get_node_text(identifier)

            line_start = node.start_point[0] + 1
            line_end = node.end_point[0] + 1

            params = self._extract_parameters(declarator)

            code_snippet = self._get_node_text(node)

            is_method = self._is_method(node)
            class_name = self._get_class_name(node) if is_method else None

            return Function(
                name=func_name,
                file_path=self.file_path,
                line_start=line_start,
                line_end=line_end,
                parameters=params,
                code_snippet=code_snippet,
                is_method=is_method,
                class_name=class_name,
                docstring=None,
            )

        except Exception as e:
            logger.warning(f"Failed to create function from node: {e}")
            return None

    def _create_function_from_declarator(self, node) -> Optional[Function]:
        """Create a Function object from a function_declarator node (for declarations)."""
        try:
            identifier = self._find_child_by_type(node, "identifier")
            if not identifier:
                return None

            func_name = self._get_node_text(identifier)

            line_start = node.start_point[0] + 1
            line_end = node.end_point[0] + 1

            params = self._extract_parameters(node)

            code_snippet = (
                self._get_node_text(node.parent) if node.parent else self._get_node_text(node)
            )

            return Function(
                name=func_name,
                file_path=self.file_path,
                line_start=line_start,
                line_end=line_end,
                parameters=params,
                code_snippet=code_snippet,
                is_method=False,
                class_name=None,
                docstring=None,
            )

        except Exception as e:
            logger.warning(f"Failed to create function from declarator: {e}")
            return None

    def _create_method_from_node(self, node) -> Optional[Function]:
        """Create a Function object from a method_definition node."""
        try:
            declarator = self._find_child_by_type(node, "function_declarator")
            if not declarator:
                return None

            identifier = self._find_child_by_type(declarator, "identifier")
            if not identifier:
                if node.type == "destructor_definition":
                    destructor_name = self._find_child_by_type(node, "destructor_name")
                    if destructor_name:
                        identifier = self._find_child_by_type(destructor_name, "identifier")

                if not identifier:
                    return None

            func_name = self._get_node_text(identifier)

            line_start = node.start_point[0] + 1
            line_end = node.end_point[0] + 1

            params = self._extract_parameters(declarator)

            code_snippet = self._get_node_text(node)

            class_name = self._get_class_name(node)

            return Function(
                name=func_name,
                file_path=self.file_path,
                line_start=line_start,
                line_end=line_end,
                parameters=params,
                code_snippet=code_snippet,
                is_method=True,
                class_name=class_name,
                docstring=None,
            )

        except Exception as e:
            logger.warning(f"Failed to create method from node: {e}")
            return None

    def _extract_parameters(self, declarator_node) -> List[str]:
        """Extract parameter names from function declarator."""
        params = []

        param_list = self._find_child_by_type(declarator_node, "parameter_list")
        if param_list:
            for child in param_list.children:
                if child.type == "parameter_declaration":
                    param_name = self._extract_parameter_name(child)
                    if param_name:
                        params.append(param_name)

        return params

    def _extract_parameter_name(self, param_node) -> Optional[str]:
        """Extract parameter name from parameter_declaration node."""
        for child in param_node.children:
            if child.type == "identifier":
                return self._get_node_text(child)
            elif child.type in ["pointer_declarator", "array_declarator"]:
                identifier = self._find_child_by_type(child, "identifier")
                if identifier:
                    return self._get_node_text(identifier)
        return None

    def _extract_calls(self, node):
        """Extract function calls from the AST."""
        if self.limits.should_stop():
            return

        if node.type == "call_expression":
            self._process_call_expression(node)
            if self.limits.increment():
                return

        for child in node.children:
            self._extract_calls(child)
            if self.limits.should_stop():
                break

    def _process_call_expression(self, node):
        """Process a call_expression node to extract call relationships."""
        try:
            function_node = node.children[0] if node.children else None
            if not function_node:
                return

            callee_name = None

            if function_node.type == "identifier":
                callee_name = self._get_node_text(function_node)
            elif function_node.type == "field_expression":
                field = self._find_child_by_type(function_node, "field_identifier")
                if field:
                    callee_name = self._get_node_text(field)
            elif function_node.type == "scoped_identifier":
                identifier = self._find_child_by_type(function_node, "identifier")
                if identifier:
                    callee_name = self._get_node_text(identifier)

            if callee_name and not self._is_builtin_function(callee_name):
                containing_func = self._find_containing_function(node.start_point[0] + 1)
                if containing_func and containing_func.name != callee_name:
                    call_line = node.start_point[0] + 1

                    relationship = CallRelationship(
                        caller=f"{self.file_path}:{containing_func.name}",
                        callee=callee_name,
                        call_line=call_line,
                        is_resolved=False,
                    )
                    if self.limits.can_add_relationship():
                        self.call_relationships.append(relationship)
                        self.limits.add_relationship()

        except Exception as e:
            logger.warning(f"Failed to process call expression: {e}")

    def _find_containing_function(self, line_number: int) -> Optional[Function]:
        """Find the function that contains the given line number."""
        for func in self.functions:
            if func.line_start is not None and func.line_end is not None:
                if func.line_start <= line_number <= func.line_end:
                    return func
        return None

    def _is_method(self, node) -> bool:
        """Check if the function is a method (inside a class/struct)."""
        parent = node.parent
        while parent:
            if parent.type in ["class_specifier", "struct_specifier"]:
                return True
            parent = parent.parent
        return False

    def _get_class_name(self, node) -> Optional[str]:
        """Get the class name containing this method."""
        parent = node.parent
        while parent:
            if parent.type in ["class_specifier", "struct_specifier"]:
                for child in parent.children:
                    if child.type == "type_identifier":
                        return self._get_node_text(child)
            parent = parent.parent
        return None

    def _is_builtin_function(self, name: str) -> bool:
        """Check if function name is a C/C++ built-in."""
        builtins = {
            "printf",
            "scanf",
            "malloc",
            "free",
            "calloc",
            "realloc",
            "strlen",
            "strcpy",
            "strcmp",
            "strcat",
            "memcpy",
            "memset",
            "exit",
            "abort",
            "assert",
            "sizeof",
        }
        return name in builtins

    def _find_child_by_type(self, node, target_type: str):
        """Find the first child node of the specified type."""
        for child in node.children:
            if child.type == target_type:
                return child
        return None

    def _get_node_text(self, node) -> str:
        """Get the text content of a node."""
        return self.content[node.start_byte : node.end_byte]


def analyze_c_file_treesitter(
    file_path: str, content: str, limits: Optional[AnalysisLimits] = None
) -> Tuple[List[Function], List[CallRelationship]]:
    """
    Analyze a C file using Tree-sitter.

    Args:
        file_path: Path to the C file
        content: Content of the C file
        limits: Analysis limits

    Returns:
        Tuple of (functions, call_relationships)
    """
    try:
        logger.info(f"Tree-sitter C analysis for {file_path}")
        if limits is None:
            limits = create_c_cpp_limits()
        analyzer = TreeSitterCAnalyzer(file_path, content, language="c", limits=limits)
        analyzer.analyze()
        logger.info(
            f"Found {len(analyzer.functions)} functions, {len(analyzer.call_relationships)} calls, {analyzer.limits.nodes_processed} nodes processed"
        )
        return analyzer.functions, analyzer.call_relationships
    except Exception as e:
        logger.error(f"Error in tree-sitter C analysis for {file_path}: {e}", exc_info=True)
        return [], []


def analyze_cpp_file_treesitter(
    file_path: str, content: str, limits: Optional[AnalysisLimits] = None
) -> Tuple[List[Function], List[CallRelationship]]:
    """
    Analyze a C++ file using Tree-sitter.

    Args:
        file_path: Path to the C++ file
        content: Content of the C++ file
        limits: Analysis limits

    Returns:
        Tuple of (functions, call_relationships)
    """
    try:
        logger.info(f"Tree-sitter C++ analysis for {file_path}")
        if limits is None:
            limits = create_c_cpp_limits()
        analyzer = TreeSitterCAnalyzer(file_path, content, language="cpp", limits=limits)
        analyzer.analyze()
        logger.info(
            f"Found {len(analyzer.functions)} functions, {len(analyzer.call_relationships)} calls, {analyzer.limits.nodes_processed} nodes processed"
        )
        return analyzer.functions, analyzer.call_relationships
    except Exception as e:
        logger.error(f"Error in tree-sitter C++ analysis for {file_path}: {e}", exc_info=True)
        return [], []


def analyze_c_file(
    file_path: str, content: str, limits: Optional[AnalysisLimits] = None
) -> Tuple[List[Function], List[CallRelationship]]:
    """Main entry point for C file analysis."""
    return analyze_c_file_treesitter(file_path, content, limits)


def analyze_cpp_file(
    file_path: str, content: str, limits: Optional[AnalysisLimits] = None
) -> Tuple[List[Function], List[CallRelationship]]:
    """Main entry point for C++ file analysis."""
    return analyze_cpp_file_treesitter(file_path, content, limits)
