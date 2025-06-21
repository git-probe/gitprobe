"""
C/C++ AST Analyzer with Multiple Parsing Backends

This module provides C and C++ source code analysis using multiple parsing
backends in a fallback hierarchy:

1. pycparser - Pure Python C parser (C99 compliant) - Primary for C files
2. libclang - Clang's AST via Python bindings - Fallback for C/C++
3. Regex-based parser - Last resort when AST parsing fails

The module automatically selects the best available parser for each file.

**Installation Notes:**
- For pycparser: `pip install pycparser`
- For libclang: `pip install clang` (requires libclang shared library)
- Regex fallback always available

**Usage Priority:**
- C files (.c, .h): pycparser → libclang → regex
- C++ files (.cpp, .hpp, .cc, .cxx): libclang → regex
"""

import logging
import re
from typing import List, Tuple, Dict, Any, Optional, Set
from pathlib import Path
from models.core import Function, CallRelationship

# Configure logging
logger = logging.getLogger(__name__)

# --- Backend Availability Checks ---
PYCPARSER_AVAILABLE = False
LIBCLANG_AVAILABLE = False

try:
    import pycparser
    from pycparser import c_ast, parse_file
    from pycparser.c_generator import CGenerator

    PYCPARSER_AVAILABLE = True
    logger.info("pycparser is available for C parsing")
except ImportError:
    logger.warning("pycparser not available. Install with: pip install pycparser")

try:
    import clang.cindex
    from clang.cindex import Index, CursorKind, TypeKind

    LIBCLANG_AVAILABLE = True
    logger.info("libclang is available for C/C++ parsing")
except ImportError:
    logger.warning("libclang not available. Install with: pip install clang")


# --- Pycparser Implementation ---
class PycparserAnalyzer:
    """C analyzer using pycparser for pure C code."""

    def __init__(self, file_path: str, content: str):
        self.file_path = str(file_path)  # Ensure it's always a string
        self.content = content
        self.functions: List[Function] = []
        self.call_relationships: List[CallRelationship] = []

    def analyze(self) -> Tuple[List[Function], List[CallRelationship]]:
        """Analyze C code using pycparser."""
        try:
            # Parse the C code
            ast = pycparser.c_parser.CParser().parse(
                self.content, filename=self.file_path
            )

            # Extract functions and calls
            visitor = FunctionVisitor(self.file_path, self.content)
            visitor.visit(ast)

            self.functions = visitor.functions
            self.call_relationships = visitor.call_relationships

            logger.info(
                f"pycparser analysis complete: {len(self.functions)} functions, {len(self.call_relationships)} calls"
            )
            return self.functions, self.call_relationships

        except Exception as e:
            logger.error(f"pycparser analysis failed for {self.file_path}: {e}")
            return [], []


class FunctionVisitor(pycparser.c_ast.NodeVisitor):
    """AST visitor for extracting functions and calls using pycparser."""

    def __init__(self, file_path: str, content: str):
        self.file_path = file_path
        self.content = content
        self.lines = content.splitlines()
        self.functions: List[Function] = []
        self.call_relationships: List[CallRelationship] = []
        self.current_function: Optional[str] = None
        self.struct_context: Optional[str] = None

    def visit_FuncDef(self, node):
        """Visit function definitions."""
        func_name = node.decl.name

        # Get line numbers
        line_start = getattr(node, "coord", None)
        line_start = line_start.line if line_start else 1

        # Estimate end line by looking for closing brace
        line_end = self._find_function_end(line_start)

        # Extract parameters
        params = []
        if node.decl.type.args:
            for param in node.decl.type.args.params:
                if hasattr(param, "name") and param.name:
                    params.append(param.name)

        # Get code snippet
        code_snippet = "\n".join(self.lines[line_start - 1 : line_end])

        # Check if it's a method (inside struct)
        is_method = self.struct_context is not None

        func = Function(
            name=func_name,
            file_path=self.file_path,
            line_start=line_start,
            line_end=line_end,
            parameters=params,
            code_snippet=code_snippet,
            is_method=is_method,
            class_name=self.struct_context,
        )

        self.functions.append(func)

        # Set current function context for call analysis
        old_function = self.current_function
        self.current_function = func_name

        # Visit function body for calls
        if node.body:
            self.visit(node.body)

        self.current_function = old_function

    def visit_Struct(self, node):
        """Visit struct definitions."""
        if node.name:
            old_struct = self.struct_context
            self.struct_context = node.name
            self.generic_visit(node)
            self.struct_context = old_struct
        else:
            self.generic_visit(node)

    def visit_FuncCall(self, node):
        """Visit function calls."""
        if self.current_function and hasattr(node.name, "name"):
            callee_name = node.name.name
            call_line = getattr(node, "coord", None)
            call_line = call_line.line if call_line else 0

            # Skip self-calls
            if callee_name != self.current_function:
                relationship = CallRelationship(
                    caller=f"{self.file_path}:{self.current_function}",
                    callee=callee_name,
                    call_line=call_line,
                    is_resolved=False,
                )
                self.call_relationships.append(relationship)

        self.generic_visit(node)

    def _find_function_end(self, start_line: int) -> int:
        """Find the end line of a function by looking for balanced braces."""
        if start_line > len(self.lines):
            return start_line

        brace_count = 0
        in_function = False

        for i, line in enumerate(self.lines[start_line - 1 :], start_line):
            for char in line:
                if char == "{":
                    brace_count += 1
                    in_function = True
                elif char == "}":
                    brace_count -= 1
                    if in_function and brace_count == 0:
                        return i

        return min(start_line + 50, len(self.lines))  # Fallback


# --- LibClang Implementation ---
class LibclangAnalyzer:
    """C/C++ analyzer using libclang."""

    def __init__(self, file_path: str, content: str, language: str = "c"):
        self.file_path = str(file_path)  # Ensure it's always a string
        self.content = content
        self.language = language
        self.functions: List[Function] = []
        self.call_relationships: List[CallRelationship] = []

    def analyze(self) -> Tuple[List[Function], List[CallRelationship]]:
        """Analyze C/C++ code using libclang."""
        try:
            # Create index and parse
            index = Index.create()

            # Create a temporary file for parsing
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=f".{self.language}", delete=False
            ) as f:
                f.write(self.content)
                temp_path = f.name

            try:
                # Parse with appropriate flags
                args = ["-std=c99"] if self.language == "c" else ["-std=c++17"]
                tu = index.parse(temp_path, args=args)

                if tu:
                    self._extract_functions(tu.cursor)
                    self._extract_calls(tu.cursor)

                    logger.info(
                        f"libclang analysis complete: {len(self.functions)} functions, {len(self.call_relationships)} calls"
                    )

            finally:
                # Clean up temp file
                import os

                os.unlink(temp_path)

            return self.functions, self.call_relationships

        except Exception as e:
            logger.error(f"libclang analysis failed for {self.file_path}: {e}")
            return [], []

    def _extract_functions(self, cursor):
        """Extract function definitions from libclang cursor."""
        if cursor.kind == CursorKind.FUNCTION_DECL:
            if cursor.is_definition():
                func_name = cursor.spelling
                line_start = cursor.location.line
                line_end = cursor.extent.end.line

                # Extract parameters
                params = []
                for arg in cursor.get_arguments():
                    if arg.spelling:
                        params.append(arg.spelling)

                # Get code snippet
                lines = self.content.splitlines()
                code_snippet = "\n".join(lines[line_start - 1 : line_end])

                # Check if it's a method
                is_method = self._is_method(cursor)
                class_name = self._get_class_name(cursor) if is_method else None

                func = Function(
                    name=func_name,
                    file_path=self.file_path,
                    line_start=line_start,
                    line_end=line_end,
                    parameters=params,
                    code_snippet=code_snippet,
                    is_method=is_method,
                    class_name=class_name,
                )

                self.functions.append(func)

        # Recurse through children
        for child in cursor.get_children():
            self._extract_functions(child)

    def _extract_calls(self, cursor):
        """Extract function calls from libclang cursor."""
        if cursor.kind == CursorKind.CALL_EXPR:
            callee_name = cursor.spelling
            call_line = cursor.location.line

            # Find containing function
            caller_func = self._find_containing_function(call_line)
            if caller_func and callee_name != caller_func.name:
                relationship = CallRelationship(
                    caller=f"{self.file_path}:{caller_func.name}",
                    callee=callee_name,
                    call_line=call_line,
                    is_resolved=False,
                )
                self.call_relationships.append(relationship)

        # Recurse through children
        for child in cursor.get_children():
            self._extract_calls(child)

    def _is_method(self, cursor) -> bool:
        """Check if cursor represents a method."""
        parent = cursor.semantic_parent
        while parent:
            if parent.kind in [CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL]:
                return True
            parent = parent.semantic_parent
        return False

    def _get_class_name(self, cursor) -> Optional[str]:
        """Get the class name for a method."""
        parent = cursor.semantic_parent
        while parent:
            if parent.kind in [CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL]:
                return parent.spelling
            parent = parent.semantic_parent
        return None

    def _find_containing_function(self, line_number: int) -> Optional[Function]:
        """Find the function containing a given line number."""
        for func in self.functions:
            if func.line_start and func.line_end:
                if func.line_start <= line_number <= func.line_end:
                    return func
        return None


# --- Regex Fallback Implementation ---
class RegexAnalyzer:
    """Regex-based C/C++ analyzer as fallback."""

    def __init__(self, file_path: str, content: str, language: str = "c"):
        self.file_path = str(file_path)  # Ensure it's always a string
        self.content = content
        self.language = language
        self.lines = content.splitlines()

    def analyze(self) -> Tuple[List[Function], List[CallRelationship]]:
        """Analyze using regex patterns."""
        try:
            functions = self._extract_functions_regex()
            calls = self._extract_calls_regex(functions)

            logger.info(
                f"Regex analysis complete: {len(functions)} functions, {len(calls)} calls"
            )
            return functions, calls

        except Exception as e:
            logger.error(f"Regex analysis failed for {self.file_path}: {e}")
            return [], []

    def _extract_functions_regex(self) -> List[Function]:
        """Extract functions using regex patterns."""
        functions = []

        # C/C++ function patterns
        if self.language == "cpp":
            patterns = [
                # C++ function - FIXED: Don't require brace on same line
                r"^\s*(?:(?:inline|static|virtual|explicit|friend)\s+)*(?:(?:unsigned|signed|const|volatile)\s+)*(?:\w+(?:::\w+)*(?:\s*[*&]+\s*)?)\s+(\w+)\s*\([^)]*\)\s*(?:const\s*)?(?:override\s*)?(?:final\s*)?(?:noexcept\s*)?$",
                # Constructor/destructor - FIXED: Don't require brace on same line
                r"^\s*(?:(?:inline|static|virtual|explicit)\s+)*(?:~)?(\w+)\s*\([^)]*\)\s*(?::\s*[^{]*)?$",
                # Simple function pattern as fallback
                r"^\s*(?:(?:inline|static|virtual|explicit|friend|extern)\s+)*(\w+)\s+(\w+)\s*\([^)]*\)\s*$",
            ]
        else:
            patterns = [
                # C function pattern - FIXED: Don't require brace on same line
                r"^\s*(?:(?:inline|static|extern)\s+)*(?:(?:unsigned|signed|const|volatile)\s+)*(?:\w+\s*[*]*\s+)+(\w+)\s*\([^)]*\)\s*$",
            ]

        for line_num, line in enumerate(self.lines, 1):
            for pattern in patterns:
                match = re.match(pattern, line)
                if match and not line.strip().endswith(";"):  # Skip declarations
                    # Get function name from the last captured group
                    groups = match.groups()
                    func_name = groups[-1]  # Last group is usually the function name

                    # Skip common false positives
                    if func_name.lower() in [
                        "if",
                        "while",
                        "for",
                        "switch",
                        "sizeof",
                        "return",
                    ]:
                        continue

                    # Check if next line has opening brace (common C++ style)
                    has_brace_next_line = False
                    if line_num < len(self.lines):
                        next_line = self.lines[line_num].strip()
                        if next_line.startswith("{"):
                            has_brace_next_line = True

                    # Skip if this looks like a declaration (no brace following)
                    if not has_brace_next_line and not line.strip().endswith(")"):
                        continue

                    # Find function end
                    end_line = self._find_function_end_regex(line_num)

                    # Extract parameters
                    params = self._extract_params_regex(line)

                    # Get code snippet
                    code_snippet = "\n".join(self.lines[line_num - 1 : end_line])

                    func = Function(
                        name=func_name,
                        file_path=self.file_path,
                        line_start=line_num,
                        line_end=end_line,
                        parameters=params,
                        code_snippet=code_snippet,
                        is_method=False,  # Hard to determine with regex
                        class_name=None,
                    )

                    functions.append(func)
                    break

        return functions

    def _extract_calls_regex(self, functions: List[Function]) -> List[CallRelationship]:
        """Extract function calls using regex."""
        calls = []

        # Build function lookup
        func_lookup = {func.line_start: func for func in functions}

        # Pattern for function calls
        call_pattern = r"(\w+)\s*\("

        for line_num, line in enumerate(self.lines, 1):
            # Find containing function
            containing_func = None
            for func in functions:
                if (
                    func.line_start
                    and func.line_end
                    and func.line_start <= line_num <= func.line_end
                ):
                    containing_func = func
                    break

            if containing_func:
                matches = re.finditer(call_pattern, line)
                for match in matches:
                    callee_name = match.group(1)

                    # Skip obvious non-functions
                    if callee_name.lower() in [
                        "if",
                        "while",
                        "for",
                        "switch",
                        "sizeof",
                        "return",
                    ]:
                        continue

                    # Skip self-calls
                    if callee_name != containing_func.name:
                        relationship = CallRelationship(
                            caller=f"{self.file_path}:{containing_func.name}",
                            callee=callee_name,
                            call_line=line_num,
                            is_resolved=False,
                        )
                        calls.append(relationship)

        return calls

    def _find_function_end_regex(self, start_line: int) -> int:
        """Find function end using brace counting."""
        brace_count = 0
        in_function = False

        for i in range(start_line - 1, len(self.lines)):
            line = self.lines[i]
            for char in line:
                if char == "{":
                    brace_count += 1
                    in_function = True
                elif char == "}":
                    brace_count -= 1
                    if in_function and brace_count == 0:
                        return i + 1

        return min(start_line + 50, len(self.lines))

    def _extract_params_regex(self, line: str) -> List[str]:
        """Extract parameter names from function signature."""
        # Find parameter list
        paren_match = re.search(r"\(([^)]*)\)", line)
        if not paren_match:
            return []

        params_str = paren_match.group(1).strip()
        if not params_str or params_str == "void":
            return []

        # Split by comma and extract parameter names
        params = []
        for param in params_str.split(","):
            param = param.strip()
            # Extract the last word as parameter name (simple heuristic)
            words = param.split()
            if words and not words[-1] in ["*", "&", "const", "volatile"]:
                # Remove pointer/reference indicators
                name = words[-1].lstrip("*&")
                if name and name.isalnum():
                    params.append(name)

        return params


# --- Main Analysis Functions ---
def _analyze_file_with_fallback(
    file_path: str, content: str, language: str
) -> Tuple[List[Function], List[CallRelationship]]:
    """
    Analyze file using fallback hierarchy based on language and availability.

    C files: pycparser → libclang → regex
    C++ files: libclang → regex
    """

    if language == "c":
        # Try pycparser first for C files
        if PYCPARSER_AVAILABLE:
            try:
                analyzer = PycparserAnalyzer(file_path, content)
                functions, calls = analyzer.analyze()
                if functions:  # Success if we found functions
                    logger.info(f"Successfully analyzed {file_path} with pycparser")
                    return functions, calls
            except Exception as e:
                logger.warning(f"pycparser failed for {file_path}: {e}")

    # Try libclang (for both C and C++)
    if LIBCLANG_AVAILABLE:
        try:
            analyzer = LibclangAnalyzer(file_path, content, language)
            functions, calls = analyzer.analyze()
            if functions:  # Success if we found functions
                logger.info(f"Successfully analyzed {file_path} with libclang")
                return functions, calls
        except Exception as e:
            logger.warning(f"libclang failed for {file_path}: {e}")

    # Fallback to regex
    logger.info(f"Using regex fallback for {file_path}")
    analyzer = RegexAnalyzer(file_path, content, language)
    return analyzer.analyze()


def analyze_c_file(
    file_path: str, content: str
) -> Tuple[List[Function], List[CallRelationship]]:
    """
    Analyze a C file and return functions and relationships.
    """
    return _analyze_file_with_fallback(file_path, content, "c")


def analyze_cpp_file(
    file_path: str, content: str
) -> Tuple[List[Function], List[CallRelationship]]:
    """
    Analyze a C++ file and return functions and relationships.
    """
    return _analyze_file_with_fallback(file_path, content, "cpp")
