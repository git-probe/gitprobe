"""
C/C++ AST Analyzer using Tree-sitter

This module uses the tree-sitter library to parse C and C++ source code
and build an Abstract Syntax Tree (AST). It then traverses the AST
to identify function definitions and call relationships, providing a more
accurate and robust analysis than regex-based approaches.

**A Note on Installation:**
The `tree-sitter` Python bindings for C and C++ require a C++ compiler
to be available on the system during installation to build the language
parsers. If you encounter errors during installation or at runtime, please
ensure you have a working C++ compiler (e.g., MSVC on Windows, GCC on Linux).

On Windows, you may need to install the "Desktop development with C++" workload
from the Visual Studio Installer.

If you continue to face issues, a forced reinstall from source may be necessary:
`pip install tree-sitter-c tree-sitter-cpp --no-binary :all: --no-cache-dir`
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
from models.core import Function, CallRelationship

# Configure logging
logger = logging.getLogger(__name__)

# --- Tree-sitter Language Setup ---
# Attempt to load the C and C++ languages.
# These libraries are expected to be in the environment.
try:
    from tree_sitter import Language, Parser, Node
    from tree_sitter_c import C_LANGUAGE
    from tree_sitter_cpp import CPP_LANGUAGE

    TREE_SITTER_AVAILABLE = True
    logger.info("Successfully loaded tree-sitter C and C++ languages.")
except (ImportError, ModuleNotFoundError):
    logger.warning(
        "Tree-sitter C/C++ languages not found. Fallback unavailable for C/C++. "
        "Please install them and ensure a C++ compiler is available."
    )
    TREE_SITTER_AVAILABLE = False
except Exception as e:
    logger.error(
        f"An unexpected error occurred while loading tree-sitter languages: {e}",
        exc_info=True,
    )
    TREE_SITTER_AVAILABLE = False


if TREE_SITTER_AVAILABLE:

    class TreeSitterCAnalyzer:
        """
        AST analyzer for C and C++ files using tree-sitter.
        """

        def __init__(self, file_path: str, content: str, language: str):
            """
            Initialize the analyzer with the file content and language.
            """
            if language not in ["c", "cpp"]:
                raise ValueError("Language must be 'c' or 'cpp'")

            self.file_path = file_path
            self.content = content
            self.language = language
            self.tree: Optional[Node] = None
            self.functions: List[Function] = []
            self.call_relationships: List[CallRelationship] = []

            # Initialize parser for the specified language
            lang = C_LANGUAGE if language == "c" else CPP_LANGUAGE
            self.parser = Parser()
            self.parser.set_language(lang)

        def analyze(self):
            """
            Parse the source code and extract functions and call relationships.
            """
            logger.info(f"Starting tree-sitter analysis for {self.file_path}")
            self.tree = self.parser.parse(bytes(self.content, "utf8"))
            self._discover_functions()
            self._discover_calls()
            logger.info(
                f"Tree-sitter analysis complete: Found {len(self.functions)} functions and {len(self.call_relationships)} calls."
            )

        def _discover_functions(self):
            """
            Locate function definitions using a tree-sitter query.
            """
            assert (
                self.tree is not None
            ), "Tree must be parsed before discovering functions."

            query_code = """
            (function_definition
              declarator: (function_declarator
                declarator: [
                  (identifier) @function_name
                  (pointer_declarator declarator: (identifier) @function_name)
                  (reference_declarator declarator: (identifier) @function_name)
                ]
                parameters: (parameter_list) @params
              )
              body: (compound_statement) @body
            )
            """

            if self.language == "cpp":
                query_code += """
                (declaration
                  (function_definition
                    declarator: (function_declarator
                      declarator: [
                        (field_identifier) @function_name
                        (qualified_identifier name: (identifier) @function_name)
                      ]
                      parameters: (parameter_list) @params
                    )
                  )
                )
                (constructor_or_destructor_definition
                    declarator: [
                        (qualified_identifier) @function_name
                        (destructor_name) @function_name
                    ]
                    parameters: (parameter_list) @params
                    body: (compound_statement) @body
                )
                """

            lang = C_LANGUAGE if self.language == "c" else CPP_LANGUAGE
            query = lang.query(query_code)
            captures = query.captures(self.tree.root_node)

            func_nodes: Dict[str, Dict[str, Any]] = {}
            for node, capture_name in captures:
                if capture_name == "function_name":
                    func_name = node.text.decode("utf8")
                    unique_key = f"{func_name}_{node.start_point[0]}"
                    if unique_key not in func_nodes:
                        func_nodes[unique_key] = {}
                    func_nodes[unique_key]["node"] = node
                    func_nodes[unique_key]["name"] = func_name
                elif capture_name == "params":
                    if func_nodes:
                        last_key = list(func_nodes.keys())[-1]
                        func_nodes[last_key]["params_node"] = node
                elif capture_name == "body":
                    if func_nodes:
                        last_key = list(func_nodes.keys())[-1]
                        func_nodes[last_key]["body_node"] = node

            for key, data in func_nodes.items():
                if "node" not in data or "body_node" not in data:
                    continue

                func_node = data["node"]
                body_node = data["body_node"]
                params_node = data.get("params_node")
                definition_node = func_node
                while (
                    definition_node.parent
                    and definition_node.type != "function_definition"
                ):
                    definition_node = definition_node.parent

                params = self._extract_parameters(params_node) if params_node else []
                code_snippet = self._get_node_text(definition_node)

                func = Function(
                    name=data["name"],
                    file_path=self.file_path,
                    line_start=definition_node.start_point[0] + 1,
                    line_end=definition_node.end_point[0] + 1,
                    parameters=params,
                    code_snippet=code_snippet,
                    is_method=self._is_method(func_node),
                )
                self.functions.append(func)

            self.functions.sort(key=lambda f: f.line_start)

        def _is_method(self, node: Node) -> bool:
            if self.language == "cpp":
                current = node.parent
                while current:
                    if current.type in ["class_specifier", "struct_specifier"]:
                        return True
                    current = current.parent
            return False

        def _get_node_text(self, node: Node) -> str:
            return self.content.encode("utf8")[node.start_byte : node.end_byte].decode(
                "utf8"
            )

        def _extract_parameters(self, params_node: Node) -> List[str]:
            params = []
            param_query_code = """
            (parameter_declaration
              declarator: [
                (identifier) @param_name
                (pointer_declarator) @param_name
                (reference_declarator) @param_name
              ]
            )
            """
            lang = C_LANGUAGE if self.language == "c" else CPP_LANGUAGE
            param_query = lang.query(param_query_code)
            for node, capture_name in param_query.captures(params_node):
                param_name_node = node
                if node.type == "pointer_declarator":
                    child = node.child_by_field_name("declarator")
                    if child and child.type == "identifier":
                        param_name_node = child
                elif node.type == "reference_declarator":
                    child = node.child_by_field_name("declarator")
                    if child and child.type == "identifier":
                        param_name_node = child
                params.append(param_name_node.text.decode("utf8"))
            return params

        def _discover_calls(self):
            assert (
                self.tree is not None
            ), "Tree must be parsed before discovering calls."
            query_code = """
            (call_expression
              function: [
                (identifier) @call_name
                (field_expression field: (field_identifier) @call_name)
                (qualified_identifier name: (identifier) @call_name)
              ]
            )
            """
            lang = C_LANGUAGE if self.language == "c" else CPP_LANGUAGE
            query = lang.query(query_code)
            captures = query.captures(self.tree.root_node)

            for node, _ in captures:
                callee_name = node.text.decode("utf8")
                call_line = node.start_point[0] + 1

                caller_func = self._find_containing_function(call_line)
                if caller_func:
                    if callee_name == caller_func.name:
                        continue
                    relationship = CallRelationship(
                        caller=f"{self.file_path}:{caller_func.name}",
                        callee=callee_name,
                        call_line=call_line,
                        is_resolved=False,
                    )
                    self.call_relationships.append(relationship)

        def _find_containing_function(self, line_number: int) -> Optional[Function]:
            for func in self.functions:
                if func.line_start is not None and func.line_end is not None:
                    if func.line_start <= line_number <= func.line_end:
                        return func
            return None


def _analyze_file(
    file_path: str, content: str, language: str
) -> Tuple[List[Function], List[CallRelationship]]:
    """
    Generic analysis function for C or C++ files.
    """
    if not TREE_SITTER_AVAILABLE:
        logger.error(
            f"Tree-sitter is not available. Skipping analysis for {file_path}. See install notes in c_analyzer.py."
        )
        return [], []

    try:
        analyzer = TreeSitterCAnalyzer(file_path, content, language)
        analyzer.analyze()
        return analyzer.functions, analyzer.call_relationships
    except Exception as e:
        logger.error(
            f"Failed to analyze {file_path} with tree-sitter: {e}", exc_info=True
        )
        return [], []


def analyze_c_file(
    file_path: str, content: str
) -> tuple[List[Function], List[CallRelationship]]:
    """
    Analyze a C file and return functions and relationships.
    """
    return _analyze_file(file_path, content, "c")


def analyze_cpp_file(
    file_path: str, content: str
) -> tuple[List[Function], List[CallRelationship]]:
    """
    Analyze a C++ file and return functions and relationships.
    """
    return _analyze_file(file_path, content, "cpp")
