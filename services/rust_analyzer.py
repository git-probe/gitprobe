"""
Advanced Rust analyzer using Tree-sitter for accurate AST parsing.

This module provides proper AST-based analysis for Rust files,
using tree-sitter for accurate function and call relationship extraction.
"""

import logging
from typing import List, Set, Optional
from pathlib import Path

import tree_sitter
import tree_sitter_rust

from models.core import Function, CallRelationship

logger = logging.getLogger(__name__)


class TreeSitterRustAnalyzer:
    """Rust analyzer using tree-sitter for proper AST parsing."""

    def __init__(self, file_path: str, content: str):
        self.file_path = Path(file_path)
        self.content = content
        self.functions: List[Function] = []
        self.call_relationships: List[CallRelationship] = []

        # Initialize tree-sitter
        self.rust_language = tree_sitter.Language(tree_sitter_rust.language())
        self.parser = tree_sitter.Parser(self.rust_language)

        logger.info(f"TreeSitterRustAnalyzer initialized for {file_path}")

    def analyze(self) -> None:
        """Analyze the Rust content and extract functions and call relationships."""
        try:
            # Parse the content into an AST
            tree = self.parser.parse(bytes(self.content, "utf8"))
            root_node = tree.root_node

            logger.info(f"Parsed AST with root node type: {root_node.type}")

            # Extract functions
            self._extract_functions(root_node)

            # Extract call relationships
            self._extract_call_relationships(root_node)

            logger.info(
                f"Analysis complete: {len(self.functions)} functions, {len(self.call_relationships)} relationships"
            )

        except Exception as e:
            logger.error(
                f"Error analyzing Rust file {self.file_path}: {e}", exc_info=True
            )

    def _extract_functions(self, node) -> None:
        """Extract all function definitions from the AST."""
        self._traverse_for_functions(node)
        self.functions.sort(key=lambda f: f.line_start)

    def _traverse_for_functions(self, node) -> None:
        """Recursively traverse AST nodes to find functions."""

        # Handle different function types
        if node.type == "function_item":
            func = self._extract_function_item(node)
            if func and self._should_include_function(func):
                self.functions.append(func)

        elif node.type == "impl_item":
            # Extract methods from impl blocks
            impl_type = self._extract_impl_type(node)
            for child in node.children:
                if child.type == "declaration_list":
                    for method_child in child.children:
                        if method_child.type == "function_item":
                            func = self._extract_method_from_impl(method_child, node)
                            if func and self._should_include_function(func):
                                self.functions.append(func)

        elif node.type == "closure_expression":
            func = self._extract_closure(node)
            if func and self._should_include_function(func):
                self.functions.append(func)

        # Recursively process all child nodes
        for child in node.children:
            self._traverse_for_functions(child)

    def _extract_function_item(self, node) -> Optional[Function]:
        """Extract regular function: fn name() {}"""
        try:
            # Find function name
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
            logger.warning(f"Error extracting function item: {e}")
            return None

    def _extract_method_from_impl(self, node, impl_node) -> Optional[Function]:
        """Extract method from impl block: impl Struct { fn method() {} }"""
        try:
            # Find method name
            name_node = self._find_child_by_type(node, "identifier")
            if not name_node:
                return None

            func_name = self._get_node_text(name_node)
            line_start = node.start_point[0] + 1
            line_end = node.end_point[0] + 1
            parameters = self._extract_parameters(node)
            code_snippet = self._get_node_text(node)
            impl_type = self._extract_impl_type(impl_node)

            return Function(
                name=func_name,
                file_path=str(self.file_path),
                line_start=line_start,
                line_end=line_end,
                parameters=parameters,
                docstring=self._extract_docstring(node),
                is_method=True,
                class_name=impl_type,
                code_snippet=code_snippet,
            )
        except Exception as e:
            logger.warning(f"Error extracting method from impl: {e}")
            return None

    def _extract_closure(self, node) -> Optional[Function]:
        """Extract closure/lambda: |x| x + 1"""
        try:
            line_start = node.start_point[0] + 1
            line_end = node.end_point[0] + 1
            parameters = self._extract_closure_parameters(node)
            code_snippet = self._get_node_text(node)

            # Generate a name for the closure
            func_name = f"closure_line_{line_start}"

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
            logger.warning(f"Error extracting closure: {e}")
            return None

    def _should_include_function(self, func: Function) -> bool:
        """Determine if a function should be included in the analysis."""
        # Filter out certain functions (but be more selective than before)
        excluded_names = {
            # "main",  # Don't exclude main for testing
            # "new",   # Don't exclude new - it's important in Rust
        }

        if func.name.lower() in excluded_names:
            logger.debug(f"Skipping excluded function: {func.name}")
            return False

        # Skip very short functions (likely trivial getters/setters)
        # But be less aggressive - only skip 1-line functions
        if func.line_end and func.line_start and func.line_end - func.line_start < 1:
            logger.debug(f"Skipping short function: {func.name}")
            return False

        return True

    def _extract_parameters(self, node) -> List[str]:
        """Extract parameter names from a function node."""
        parameters = []
        params_node = self._find_child_by_type(node, "parameters")
        if params_node:
            for child in params_node.children:
                if child.type == "parameter":
                    # Find identifier in parameter
                    param_name = self._find_child_by_type(child, "identifier")
                    if param_name:
                        parameters.append(self._get_node_text(param_name))
                elif child.type == "self_parameter":
                    # Handle self parameter
                    parameters.append("self")
        return parameters

    def _extract_closure_parameters(self, node) -> List[str]:
        """Extract parameter names from a closure node."""
        parameters = []
        # Look for closure parameters pattern |param1, param2|
        for child in node.children:
            if child.type == "closure_parameters":
                for param_child in child.children:
                    if param_child.type == "identifier":
                        parameters.append(self._get_node_text(param_child))
        return parameters

    def _extract_impl_type(self, impl_node) -> Optional[str]:
        """Extract the type name from an impl block."""
        for child in impl_node.children:
            if child.type == "type_identifier":
                return self._get_node_text(child)
            elif child.type == "generic_type":
                # Handle generic types like Vec<T>
                type_id = self._find_child_by_type(child, "type_identifier")
                if type_id:
                    return self._get_node_text(type_id)
        return None

    def _extract_docstring(self, node) -> Optional[str]:
        """Extract Rust doc comment from function."""
        # Rust doc comments are /// or //! and appear before the function
        if node.prev_sibling and node.prev_sibling.type == "line_comment":
            comment_text = self._get_node_text(node.prev_sibling)
            # Clean up the comment
            lines = comment_text.split("\n")
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line.startswith("///"):
                    cleaned_lines.append(line[3:].strip())
                elif line.startswith("//!"):
                    cleaned_lines.append(line[3:].strip())
                elif line.startswith("//"):
                    cleaned_lines.append(line[2:].strip())
            return "\n".join(cleaned_lines) if cleaned_lines else None
        return None

    def _extract_call_relationships(self, node) -> None:
        """Extract function call relationships from the AST."""
        # Build function range map
        func_ranges = {}
        for func in self.functions:
            for line in range(func.line_start, func.line_end + 1):
                func_ranges[line] = func

        self._traverse_for_calls(node, func_ranges)

    def _traverse_for_calls(self, node, func_ranges: dict) -> None:
        """Recursively find function calls."""
        if node.type == "call_expression":
            call_info = self._extract_call_from_node(node, func_ranges)
            if call_info:
                self.call_relationships.append(call_info)

        for child in node.children:
            self._traverse_for_calls(child, func_ranges)

    def _extract_call_from_node(
        self, node, func_ranges: dict
    ) -> Optional[CallRelationship]:
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
            elif callee_node.type == "field_expression":
                # For method calls like obj.method(), extract just 'method'
                field_node = self._find_child_by_type(callee_node, "field_identifier")
                if field_node:
                    return self._get_node_text(field_node)
            elif callee_node.type == "scoped_identifier":
                # For module::function() calls
                name_node = self._find_child_by_type(callee_node, "identifier")
                if name_node:
                    return self._get_node_text(name_node)
        return None

    def _is_builtin_function(self, name: str) -> bool:
        """Check if function name is a Rust built-in."""
        builtins = {
            "println",
            "print",
            "panic",
            "assert",
            "assert_eq",
            "assert_ne",
            "vec",
            "format",
            "write",
            "writeln",
            "dbg",
            "todo",
            "unimplemented",
            "unreachable",
            "compile_error",
            "include",
            "include_str",
            "include_bytes",
            "concat",
            "stringify",
            "env",
            "option_env",
            "cfg",
            "line",
            "column",
            "file",
            "module_path",
            "std",
            "core",
            "alloc",
            "collections",
            "iter",
            "slice",
            "str",
            "string",
            "vec",
            "hash",
            "sync",
            "thread",
            "fs",
            "io",
            "net",
            "path",
            "process",
            "time",
            "mem",
            "ptr",
            "fmt",
            "ops",
            "cmp",
            "clone",
            "copy",
            "drop",
            "default",
            "debug",
            "display",
            "from",
            "into",
            "try_from",
            "try_into",
            "as_ref",
            "as_mut",
            "deref",
            "deref_mut",
            "index",
            "range",
            "option",
            "result",
            "ok",
            "err",
            "some",
            "none",
            "unwrap",
            "expect",
            "map",
            "and_then",
            "or_else",
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
def analyze_rust_file_treesitter(
    file_path: str, content: str
) -> tuple[List[Function], List[CallRelationship]]:
    """Analyze a Rust file using tree-sitter."""
    try:
        logger.info(f"Tree-sitter Rust analysis for {file_path}")
        analyzer = TreeSitterRustAnalyzer(file_path, content)
        analyzer.analyze()
        logger.info(
            f"Found {len(analyzer.functions)} functions, {len(analyzer.call_relationships)} calls"
        )
        return analyzer.functions, analyzer.call_relationships
    except Exception as e:
        logger.error(
            f"Error in tree-sitter Rust analysis for {file_path}: {e}", exc_info=True
        )
        return [], []
