"""
Rust analyzer using tree-sitter for accurate AST parsing and function extraction.
"""

import logging
from typing import List, Set, Optional
from pathlib import Path

from tree_sitter import Parser, Language
import tree_sitter_rust

from gitprobe.models.core import Function, CallRelationship
from gitprobe.core.analysis_limits import AnalysisLimits, create_rust_limits

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
            logger.warning(
                f"Global Rust node limit of {self.max_nodes} reached. Stopping all Rust analysis."
            )
            return True
        return False

    def should_stop(self) -> bool:
        """Check if analysis should stop."""
        return self.limit_reached


class TreeSitterRustAnalyzer:
    """Rust analyzer using tree-sitter for proper AST parsing."""

    def __init__(self, file_path: str, content: str, limits: Optional[AnalysisLimits] = None):
        self.file_path = Path(file_path)
        self.content = content
        self.functions: List[Function] = []
        self.call_relationships: List[CallRelationship] = []
        self.limits = limits or create_rust_limits()

        # Initialize tree-sitter
        try:
            language_capsule = tree_sitter_rust.language()
            self.rust_language = Language(language_capsule)
            self.parser = Parser(self.rust_language)
            logger.debug(f"Rust parser initialized with language object: {type(self.rust_language)}")
            
            # Test parse with simple code to verify setup
            test_code = "fn main() { println!(\"test\"); }"
            test_tree = self.parser.parse(bytes(test_code, "utf8"))
            if test_tree is None or test_tree.root_node is None:
                raise RuntimeError("Parser setup test failed for Rust")
            logger.debug(f"Rust parser test successful - root node type: {test_tree.root_node.type}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Rust parser: {e}")
            # Fallback - create a dummy parser that will skip analysis
            self.parser = None
            self.rust_language = None

        logger.info(
            f"TreeSitterRustAnalyzer initialized for {file_path} with limits: {self.limits}"
        )

    def analyze(self) -> None:
        """Analyze the Rust content and extract functions and call relationships."""
        if not self.limits.start_new_file():
            logger.info(f"Skipping {self.file_path} - global limits reached")
            return
            
        if self.parser is None:
            logger.warning(f"Skipping {self.file_path} - parser initialization failed")
            return

        try:
            # Parse the content into an AST
            tree = self.parser.parse(bytes(self.content, "utf8"))
            root_node = tree.root_node

            logger.info(f"Parsed AST with root node type: {root_node.type}")

            # Extract functions
            self._extract_functions(root_node)

            # Extract call relationships (only if we haven't hit limits)
            if not self.limits.should_stop():
                self._extract_call_relationships(root_node)

            logger.info(
                f"Analysis complete: {len(self.functions)} functions, {len(self.call_relationships)} relationships, {self.limits.nodes_processed} nodes processed"
            )

        except Exception as e:
            logger.error(f"Error analyzing Rust file {self.file_path}: {e}", exc_info=True)

    def _check_limits(self) -> bool:
        """Check if we've hit analysis limits."""
        return not self.limits.should_stop()

    def _extract_functions(self, node) -> None:
        """Extract all function definitions from the AST."""
        self._traverse_for_functions(node, depth=0)

    def _traverse_for_functions(self, node, depth: int = 0) -> None:
        """Recursively traverse AST nodes to find functions."""
        if self.limits.should_stop():
            return

        # Handle different function types
        if node.type == "function_item":
            func = self._extract_function_item(node)
            if func and self._should_include_function(func):
                # Check global limits before adding function
                if self.limits.can_add_function():
                    self.functions.append(func)
                    # Track in global counter
                    if self.limits.add_function():
                        return  # Global limit reached, stop analysis
                else:
                    return  # Can't add more functions, stop analysis

        elif node.type == "impl_item":
            # Extract methods from impl blocks
            for child in node.children:
                if child.type == "declaration_list":
                    for method_child in child.children:
                        if method_child.type == "function_item":
                            func = self._extract_method_from_impl(method_child, node)
                            if func and self._should_include_function(func):
                                # Check global limits before adding function
                                if self.limits.can_add_function():
                                    self.functions.append(func)
                                    # Track in global counter
                                    if self.limits.add_function():
                                        return  # Global limit reached, stop analysis
                                else:
                                    return  # Can't add more functions, stop analysis

        elif node.type == "closure_expression" and depth < 10:  # Limit closure depth
            func = self._extract_closure(node)
            if func and self._should_include_function(func):
                # Check global limits before adding function
                if self.limits.can_add_function():
                    self.functions.append(func)
                    # Track in global counter
                    if self.limits.add_function():
                        return  # Global limit reached, stop analysis
                else:
                    return  # Can't add more functions, stop analysis

        # Recursively process child nodes
        for child in node.children:
            self._traverse_for_functions(child, depth + 1)
            if self.limits.should_stop():
                break

    def _is_likely_public(self, func: Function) -> bool:
        """Determine if a function is likely public (part of the API)."""
        # Check if function name suggests it's public
        public_indicators = ["pub ", "main", "new", "from_", "into_", "as_"]

        # Check the code snippet for pub keyword
        if func.code_snippet and "pub " in func.code_snippet:
            return True

        # Main functions are always important
        if func.name == "main":
            return True

        # Constructor-like functions
        if func.name in ["new", "create", "build", "from", "into"]:
            return True

        # Test functions are less important unless we're analyzing tests
        if func.name.startswith("test_") or "test" in func.name.lower():
            return False

        return False

    def _is_trivial_function(self, func: Function) -> bool:
        """Check if function is trivial (getter, setter, simple wrapper)."""
        if not func.code_snippet:
            return False

        lines = func.code_snippet.strip().split("\n")

        # Single line functions might be trivial
        if len(lines) <= 2:
            # But keep important ones
            if func.name in ["main", "new"] or self._is_likely_public(func):
                return False
            return True

        # Simple getter patterns
        getter_patterns = ["self.", "return ", "&self", "-> &"]
        if any(pattern in func.code_snippet for pattern in getter_patterns):
            if len(lines) <= 3:
                return True

        return False

    def _apply_function_sampling(self) -> None:
        """Apply intelligent sampling to reduce function count."""
        original_count = len(self.functions)
        target_count = int(self.limits.max_functions * self.limits.sample_ratio)

        if original_count <= target_count:
            return

        # Separate functions by priority
        high_priority = []
        medium_priority = []
        low_priority = []

        for func in self.functions:
            if self._is_likely_public(func) or func.name == "main":
                high_priority.append(func)
            elif self._is_trivial_function(func):
                low_priority.append(func)
            else:
                medium_priority.append(func)

        # Keep all high priority, sample medium, skip most low priority
        sampled = high_priority.copy()

        # Add medium priority functions
        medium_slots = max(0, target_count - len(high_priority))
        if medium_slots > 0 and medium_priority:
            step = max(1, len(medium_priority) // medium_slots)
            sampled.extend(medium_priority[::step][:medium_slots])

        # Add a few low priority if we have room
        low_slots = max(0, target_count - len(sampled))
        if low_slots > 0 and low_priority:
            step = max(1, len(low_priority) // low_slots)
            sampled.extend(low_priority[::step][:low_slots])

        self.functions = sampled[:target_count]
        logger.info(
            f"Sampled functions: {original_count} -> {len(self.functions)} "
            f"(high: {len(high_priority)}, medium: {len([f for f in sampled if f in medium_priority])}, "
            f"low: {len([f for f in sampled if f in low_priority])})"
        )

    def _is_important_disconnected_function(self, func: Function) -> bool:
        """Check if a disconnected function is still architecturally important."""
        # Always keep entry points
        if func.name == "main":
            return True

        # Keep public APIs
        if self._is_likely_public(func):
            return True

        # Keep constructors and factory functions
        if func.name in ["new", "create", "build", "from", "into", "default"]:
            return True

        # Keep trait implementations (common patterns)
        if func.name in ["fmt", "clone", "debug", "display", "drop"]:
            return True

        # Keep test entry points if analyzing tests
        if func.name.startswith("test_") and len(func.name) > 10:  # Meaningful test names
            return True

        return False

    def _has_ast_context(self, func: Function, root_node) -> bool:
        """Use tree-sitter to check if function has meaningful AST context."""
        func_name = func.name

        # Skip closures - they're always contextual if we found them
        if func_name.startswith("closure_line_"):
            return True

        # Check for various types of references in the AST
        contexts_found = []

        # Look for the function being referenced in different contexts
        self._find_function_references(root_node, func_name, contexts_found)

        # Function has context if it's referenced in meaningful ways
        meaningful_contexts = {
            "attribute",  # #[test], #[derive], etc.
            "use_declaration",  # use statements
            "macro_invocation",  # Used in macros
            "field_expression",  # obj.func references
            "path_expression",  # module::func references
            "type_arguments",  # Generic type usage
            "struct_expression",  # Struct construction
            "enum_variant",  # Enum variant
            "trait_impl",  # Trait implementation
        }

        has_meaningful_context = any(ctx in meaningful_contexts for ctx in contexts_found)

        if has_meaningful_context:
            logger.debug(f"Function {func_name} has AST context: {contexts_found}")
            return True

        # If no meaningful context found, it's truly isolated
        return False

    def _find_function_references(
        self, node, func_name: str, contexts_found: list, depth: int = 0
    ) -> None:
        """Recursively find references to a function in the AST."""
        if depth > 20:  # Prevent deep recursion
            return

        # Check if this node contains our function name
        if hasattr(node, "text") or hasattr(node, "type"):
            node_text = self._get_node_text(node) if node.type == "identifier" else ""

            if node_text == func_name:
                # Found a reference, record the parent context
                parent_type = node.parent.type if node.parent else "root"
                if parent_type not in contexts_found:
                    contexts_found.append(parent_type)

                # Also check grandparent for more context
                if node.parent and node.parent.parent:
                    grandparent_type = node.parent.parent.type
                    if grandparent_type not in contexts_found:
                        contexts_found.append(grandparent_type)

        # Recursively check children (but limit depth)
        for child in node.children:
            self._find_function_references(child, func_name, contexts_found, depth + 1)

    def _keep_largest_connected_components(self) -> None:
        """Keep only functions in the largest connected components of the call graph."""
        if not self.call_relationships:
            return

        # Build adjacency graph
        graph = {}
        for rel in self.call_relationships:
            caller = rel.caller.split(":")[-1] if ":" in rel.caller else rel.caller
            callee = rel.callee.split(":")[-1] if ":" in rel.callee else rel.callee

            if caller not in graph:
                graph[caller] = set()
            if callee not in graph:
                graph[callee] = set()

            graph[caller].add(callee)
            graph[callee].add(caller)  # Treat as undirected for component analysis

        # Find connected components using DFS
        visited = set()
        components = []

        def dfs(node, component):
            visited.add(node)
            component.add(node)
            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, component)

        for node in graph:
            if node not in visited:
                component = set()
                dfs(node, component)
                components.append(component)

        # Sort components by size (largest first)
        components.sort(key=len, reverse=True)

        if not components:
            return

        # Keep functions from the largest 2-3 components
        max_components = min(3, len(components))
        keep_functions = set()

        for i in range(max_components):
            keep_functions.update(components[i])

        # Always keep main if it exists
        for func in self.functions:
            if func.name == "main":
                keep_functions.add(func.name)

        # Filter functions to keep only those in selected components
        original_count = len(self.functions)
        self.functions = [f for f in self.functions if f.name in keep_functions]

        logger.info(
            f"Component analysis: kept {len(self.functions)} functions from {max_components} "
            f"largest components (sizes: {[len(c) for c in components[:max_components]]}), "
            f"filtered {original_count - len(self.functions)} functions"
        )

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

        # Reset call analysis
        self._traverse_for_calls(node, func_ranges, depth=0)

    def _traverse_for_calls(self, node, func_ranges: dict, depth: int = 0) -> None:
        """Recursively find function calls with limits."""
        if self.limits.should_stop():
            return

        # Increment global counter
        if self.limits.increment():
            return

        if node.type == "call_expression":
            call_info = self._extract_call_from_node(node, func_ranges)
            if call_info:
                # Check global limits before adding relationship
                if self.limits.can_add_relationship():
                    self.call_relationships.append(call_info)
                    # Track in global counter
                    if self.limits.add_relationship():
                        return  # Global limit reached, stop analysis
                else:
                    return  # Can't add more relationships, stop analysis

        for child in node.children:
            self._traverse_for_calls(child, func_ranges, depth + 1)
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
    file_path: str, content: str, limits: Optional[AnalysisLimits] = None
) -> tuple[List[Function], List[CallRelationship]]:
    """Analyze a Rust file using tree-sitter with configurable limits."""
    try:
        logger.info(f"Tree-sitter Rust analysis for {file_path}")
        analyzer = TreeSitterRustAnalyzer(file_path, content, limits)
        analyzer.analyze()
        logger.info(
            f"Found {len(analyzer.functions)} functions, {len(analyzer.call_relationships)} calls, {analyzer.limits.nodes_processed} nodes processed"
        )
        return analyzer.functions, analyzer.call_relationships
    except Exception as e:
        logger.error(f"Error in tree-sitter Rust analysis for {file_path}: {e}", exc_info=True)
        return [], []
