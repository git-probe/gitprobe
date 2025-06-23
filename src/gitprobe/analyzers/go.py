"""
Go analyzer using tree-sitter for accurate AST parsing and function extraction.
"""

import logging
from typing import List, Set, Optional
from pathlib import Path

from tree_sitter import Parser, Language
import tree_sitter_go

from gitprobe.models.core import Function, CallRelationship
from gitprobe.core.analysis_limits import AnalysisLimits, create_go_limits

logger = logging.getLogger(__name__)


class TreeSitterGoAnalyzer:
    """Go analyzer using tree-sitter for proper AST parsing."""

    def __init__(self, file_path: str, content: str, limits: Optional[AnalysisLimits] = None):
        self.file_path = Path(file_path)
        self.content = content
        self.functions: List[Function] = []
        self.call_relationships: List[CallRelationship] = []
        self.limits = limits or create_go_limits()

        try:
            language_capsule = tree_sitter_go.language()
            self.go_language = Language(language_capsule)
            self.parser = Parser(self.go_language)
            logger.debug(f"Go parser initialized with language object: {type(self.go_language)}")

            test_code = 'package main\nfunc main() { println("test") }'
            test_tree = self.parser.parse(bytes(test_code, "utf8"))
            if test_tree is None or test_tree.root_node is None:
                raise RuntimeError("Parser setup test failed for Go")
            logger.debug(f"Go parser test successful - root node type: {test_tree.root_node.type}")

        except Exception as e:
            logger.error(f"Failed to initialize Go parser: {e}")
            self.parser = None
            self.go_language = None

        logger.info(f"TreeSitterGoAnalyzer initialized for {file_path} with limits: {self.limits}")

    def analyze(self) -> None:
        """Analyze the Go content and extract functions and call relationships."""
        if not self.limits.start_new_file():
            logger.info(f"Skipping {self.file_path} - global limits reached")
            return

        if self.parser is None:
            logger.warning(f"Skipping {self.file_path} - parser initialization failed")
            return

        try:
            tree = self.parser.parse(bytes(self.content, "utf8"))
            root_node = tree.root_node

            logger.info(f"Parsed AST with root node type: {root_node.type}")

            self._extract_functions(root_node)

            if not self.limits.should_stop():
                self._extract_call_relationships(root_node)

            logger.info(
                f"Analysis complete: {len(self.functions)} functions, {len(self.call_relationships)} relationships, {self.limits.nodes_processed} nodes processed"
            )

        except Exception as e:
            logger.error(f"Error analyzing Go file {self.file_path}: {e}", exc_info=True)

    def _extract_functions(self, node) -> None:
        """Extract all function definitions from the AST."""
        self._traverse_for_functions(node)
        self.functions.sort(key=lambda f: f.line_start)

    def _traverse_for_functions(self, node) -> None:
        """Recursively traverse AST nodes to find functions."""
        if self.limits.should_stop():
            return

        if node.type == "function_declaration":
            func = self._extract_function_declaration(node)
            if func and self._should_include_function(func):
                if self.limits.can_add_function():
                    self.functions.append(func)
                    if self.limits.add_function():
                        return
                else:
                    return

        elif node.type == "method_declaration":
            func = self._extract_method_declaration(node)
            if func and self._should_include_function(func):
                if self.limits.can_add_function():
                    self.functions.append(func)
                    if self.limits.add_function():
                        return
                else:
                    return

        elif node.type == "func_literal":
            func = self._extract_func_literal(node)
            if func and self._should_include_function(func):
                if self.limits.can_add_function():
                    self.functions.append(func)
                    if self.limits.add_function():
                        return
                else:
                    return

        for child in node.children:
            self._traverse_for_functions(child)
            if self.limits.should_stop():
                break

    def _extract_function_declaration(self, node) -> Optional[Function]:
        """Extract regular function declaration: func name() {}"""
        try:
            name_node = self._find_child_by_type(node, "identifier")
            if not name_node:
                return None

            func_name = self._get_node_text(name_node)
            line_start = node.start_point[0] + 1
            line_end = node.end_point[0] + 1
            parameters = self._extract_parameters(node)
            code_snippet = self._get_node_text(node)

            return Function(
                name=func_name,
                file_path=str(self.file_path),
                line_start=line_start,
                line_end=line_end,
                parameters=parameters,
                docstring=self._extract_docstring(node),
                is_method=False,
                class_name=None,
                code_snippet=code_snippet,
            )
        except Exception as e:
            logger.warning(f"Error extracting function declaration: {e}")
            return None

    def _extract_method_declaration(self, node) -> Optional[Function]:
        """Extract method declaration: func (receiver) methodName() {}"""
        try:
            name_node = self._find_child_by_type(node, "identifier")
            if not name_node:
                return None

            func_name = self._get_node_text(name_node)
            line_start = node.start_point[0] + 1
            line_end = node.end_point[0] + 1
            parameters = self._extract_parameters(node)
            code_snippet = self._get_node_text(node)
            receiver_type = self._extract_receiver_type(node)

            return Function(
                name=func_name,
                file_path=str(self.file_path),
                line_start=line_start,
                line_end=line_end,
                parameters=parameters,
                docstring=self._extract_docstring(node),
                is_method=True,
                class_name=receiver_type,
                code_snippet=code_snippet,
            )
        except Exception as e:
            logger.warning(f"Error extracting method declaration: {e}")
            return None

    def _extract_func_literal(self, node) -> Optional[Function]:
        """Extract anonymous function/closure: func() {}"""
        try:
            line_start = node.start_point[0] + 1
            line_end = node.end_point[0] + 1
            parameters = self._extract_parameters(node)
            code_snippet = self._get_node_text(node)

            func_name = f"anonymous_func_line_{line_start}"

            return Function(
                name=func_name,
                file_path=str(self.file_path),
                line_start=line_start,
                line_end=line_end,
                parameters=parameters,
                docstring=None,
                is_method=False,
                class_name=None,
                code_snippet=code_snippet,
            )
        except Exception as e:
            logger.warning(f"Error extracting func literal: {e}")
            return None

    def _should_include_function(self, func: Function) -> bool:
        """Determine if a function should be included in the analysis."""
        excluded_names = {
            "init",
            "main",
        }

        if func.name.lower() in excluded_names:
            logger.debug(f"Skipping excluded function: {func.name}")
            return False

        if func.line_end - func.line_start < 2:
            logger.debug(f"Skipping short function: {func.name}")
            return False

        if func.name.startswith("anonymous_func") and func.line_end - func.line_start < 3:
            logger.debug(f"Skipping simple anonymous function: {func.name}")
            return False

        return True

    def _extract_parameters(self, node) -> List[str]:
        """Extract parameter names from a function node."""
        parameters = []
        params_node = self._find_child_by_type(node, "parameter_list")
        if params_node:
            for child in params_node.children:
                if child.type == "parameter_declaration":
                    param_name = self._find_child_by_type(child, "identifier")
                    if param_name:
                        parameters.append(self._get_node_text(param_name))
                elif child.type == "variadic_parameter_declaration":
                    param_name = self._find_child_by_type(child, "identifier")
                    if param_name:
                        parameters.append(f"...{self._get_node_text(param_name)}")
        return parameters

    def _extract_receiver_type(self, node) -> Optional[str]:
        """Extract receiver type from method declaration."""
        receiver_node = self._find_child_by_type(node, "parameter_list")
        if receiver_node and receiver_node.children:
            first_param = receiver_node.children[0] if receiver_node.children else None
            if first_param and first_param.type == "parameter_declaration":
                type_nodes = [
                    child
                    for child in first_param.children
                    if child.type in ["type_identifier", "pointer_type"]
                ]
                if type_nodes:
                    return self._get_node_text(type_nodes[0])
        return None

    def _extract_docstring(self, node) -> Optional[str]:
        """Extract Go doc comment from function."""
        if node.prev_sibling and node.prev_sibling.type == "comment":
            comment_text = self._get_node_text(node.prev_sibling)
            lines = comment_text.split("\n")
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line.startswith("//"):
                    cleaned_lines.append(line[2:].strip())
                elif line.startswith("/*") and line.endswith("*/"):
                    cleaned_lines.append(line[2:-2].strip())
            return "\n".join(cleaned_lines) if cleaned_lines else None
        return None

    def _extract_call_relationships(self, node) -> None:
        """Extract function call relationships from the AST."""
        func_ranges = {}
        for func in self.functions:
            for line in range(func.line_start, func.line_end + 1):
                func_ranges[line] = func

        self._traverse_for_calls(node, func_ranges)

    def _traverse_for_calls(self, node, func_ranges: dict) -> None:
        """Recursively find function calls."""
        if self.limits.should_stop():
            return

        if node.type == "call_expression":
            call_info = self._extract_call_from_node(node, func_ranges)
            if call_info:
                if self.limits.can_add_relationship():
                    self.call_relationships.append(call_info)
                    if self.limits.add_relationship():
                        return
                else:
                    return

        for child in node.children:
            self._traverse_for_calls(child, func_ranges)
            if self.limits.should_stop():
                break

    def _extract_call_from_node(self, node, func_ranges: dict) -> Optional[CallRelationship]:
        """Extract call relationship from a call_expression node."""
        try:
            call_line = node.start_point[0] + 1
            caller_func = func_ranges.get(call_line)
            if not caller_func:
                return None

            callee_name = self._extract_callee_name(node)
            if not callee_name or self._is_builtin_function(callee_name):
                return None

            caller_id = f"{self.file_path}:{caller_func.name}"
            return CallRelationship(
                caller=caller_id,
                callee=callee_name,
                call_line=call_line,
                is_resolved=False,
            )
        except Exception as e:
            logger.warning(f"Error extracting call relationship: {e}")
            return None

    def _extract_callee_name(self, call_node) -> Optional[str]:
        """Extract the name of the called function."""
        if call_node.children:
            callee_node = call_node.children[0]

            if callee_node.type == "identifier":
                return self._get_node_text(callee_node)
            elif callee_node.type == "selector_expression":
                field_node = self._find_child_by_type(callee_node, "field_identifier")
                if field_node:
                    return self._get_node_text(field_node)
            elif callee_node.type == "qualified_type":
                name_node = self._find_child_by_type(callee_node, "type_identifier")
                if name_node:
                    return self._get_node_text(name_node)
        return None

    def _is_builtin_function(self, name: str) -> bool:
        """Check if function name is a Go built-in."""
        builtins = {
            "append",
            "cap",
            "close",
            "complex",
            "copy",
            "delete",
            "imag",
            "len",
            "make",
            "new",
            "panic",
            "print",
            "println",
            "real",
            "recover",
            "fmt",
            "log",
            "os",
            "io",
            "strings",
            "strconv",
            "time",
            "context",
            "errors",
            "sync",
            "http",
            "json",
            "encoding",
            "reflect",
            "sort",
            "math",
            "rand",
            "crypto",
            "hash",
            "net",
            "url",
            "path",
            "filepath",
            "buffer",
            "bytes",
            "regexp",
            "template",
            "html",
            "xml",
            "sql",
            "runtime",
            "unsafe",
            "atomic",
            "testing",
            "flag",
            "tar",
            "zip",
            "gzip",
            "base64",
            "hex",
            "pprof",
            "debug",
            "trace",
            "plugin",
        }
        return name in builtins

    # Helper methods
    def _find_child_by_type(self, node, node_type: str):
        """Find first child node of specified type."""
        for child in node.children:
            if child.type == node_type:
                return child
        return None

    def _find_children_by_type(self, node, node_type: str):
        """Find all child nodes of specified type."""
        return [child for child in node.children if child.type == node_type]

    def _get_node_text(self, node) -> str:
        """Get the text content of a node."""
        start_byte = node.start_byte
        end_byte = node.end_byte
        return self.content.encode("utf8")[start_byte:end_byte].decode("utf8")


# Integration functions
def analyze_go_file_treesitter(
    file_path: str, content: str, limits: Optional[AnalysisLimits] = None
) -> tuple[List[Function], List[CallRelationship]]:
    """Analyze a Go file using tree-sitter."""
    try:
        logger.info(f"Tree-sitter Go analysis for {file_path}")
        analyzer = TreeSitterGoAnalyzer(file_path, content, limits)
        analyzer.analyze()
        logger.info(
            f"Found {len(analyzer.functions)} functions, {len(analyzer.call_relationships)} calls, {analyzer.limits.nodes_processed} nodes processed"
        )
        return analyzer.functions, analyzer.call_relationships
    except Exception as e:
        logger.error(f"Error in tree-sitter Go analysis for {file_path}: {e}", exc_info=True)
        return [], []
