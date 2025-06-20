"""
C/C++ AST Analyzer

Placeholder for C/C++ AST analysis functionality.
This will be integrated into the AnalysisService to support multi-language analysis.
"""

import re
from typing import List
from models.core import Function, CallRelationship


class CASTAnalyzer:
    """
    AST analyzer for C files.

    NOTE:
    -------
    A production-ready call-graph extractor should ideally leverage a full
    C parser (e.g. pycparser, clang, tree-sitter-c, etc.).
    For the purposes of this project we want **good-enough** heuristics that
    work without any native add-ons or external processes so that the test
    suite can run in any environment – including CI systems that may not have
    a C tool-chain.

    Therefore the implementation below uses **regular-expression based
    heuristics** to:
    1. Locate function definitions (function declarations and definitions).
    2. Create `Function` model instances with reasonable metadata (name, file
       path, start/end lines, etc.).
    3. Perform a second pass to discover potential function calls inside the
       detected functions and create `CallRelationship` instances.

    While not 100% accurate, these heuristics are sufficient for small / well
    structured repositories and satisfy the expectations of the accompanying
    tests until a more sophisticated parser is introduced.
    """

    # C keywords that should NOT be treated as functions
    C_KEYWORDS = {
        "if",
        "else",
        "for",
        "while",
        "do",
        "switch",
        "case",
        "default",
        "break",
        "continue",
        "return",
        "goto",
        "typedef",
        "struct",
        "union",
        "enum",
        "const",
        "volatile",
        "static",
        "extern",
        "auto",
        "register",
        "inline",
        "restrict",
        "sizeof",
        "void",
        "char",
        "short",
        "int",
        "long",
        "float",
        "double",
        "signed",
        "unsigned",
        "bool",
        "_Bool",
        "_Complex",
        "_Imaginary",
    }

    # Function definition pattern: return_type function_name(parameters) {
    # Matches: int main(int argc, char *argv[]) {
    #          static void helper_function(void) {
    #          char* get_string(const char* input) {
    FUNCTION_DEF_RE = re.compile(
        r"^\s*(?:(?:static|extern|inline)\s+)?(?:const\s+)?(?:unsigned\s+)?(?:struct\s+\w+\s*\*?|union\s+\w+\s*\*?|enum\s+\w+\s*\*?|\w+\s*\*?)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)\s*\{",
        re.MULTILINE,
    )

    # Function call pattern: function_name(
    CALL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")

    def __init__(self, file_path: str, content: str):
        self.file_path = file_path
        self.content = content
        self.functions: List[Function] = []
        self.call_relationships: List[CallRelationship] = []

    def analyze(self):
        """Parse the C file and populate ``self.functions`` and
        ``self.call_relationships`` using regex-based heuristics.
        """
        self.functions = []
        self.call_relationships = []

        # First pass – discover function definitions
        self._discover_functions()

        # Second pass – discover call relationships inside the known functions
        self._discover_calls()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_functions(self):
        """Locate function definitions."""
        lines = self.content.split("\n")

        for match in self.FUNCTION_DEF_RE.finditer(self.content):
            name = match.group(1)

            # Skip C keywords
            if name in self.C_KEYWORDS:
                continue

            # Find the line number
            line_start = self.content[: match.start()].count("\n") + 1

            # Extract parameters from the match
            func_line = match.group(0)
            params_match = re.search(r"\(([^)]*)\)", func_line)
            params = []
            if params_match:
                param_str = params_match.group(1).strip()
                if param_str and param_str != "void":
                    # Simple parameter parsing - extract parameter names
                    param_parts = [p.strip() for p in param_str.split(",")]
                    for param in param_parts:
                        # Extract just the parameter name (last word)
                        param_words = param.split()
                        if param_words:
                            # Handle pointer parameters like "char *name" or "int* count"
                            param_name = param_words[-1].lstrip("*")
                            if param_name and not param_name in self.C_KEYWORDS:
                                params.append(param_name)

            # Find the end of the function by matching braces
            start_pos = match.end()
            brace_count = 1
            end_pos = start_pos

            while end_pos < len(self.content) and brace_count > 0:
                if self.content[end_pos] == "{":
                    brace_count += 1
                elif self.content[end_pos] == "}":
                    brace_count -= 1
                end_pos += 1

            line_end = self.content[:end_pos].count("\n") + 1

            # Extract code snippet
            code_snippet = "\n".join(lines[line_start - 1 : line_end])

            func = Function(
                name=name,
                file_path=self.file_path,
                line_start=line_start,
                line_end=line_end,
                parameters=params,
                docstring=None,
                is_method=False,
                class_name=None,
                code_snippet=code_snippet,
            )
            self.functions.append(func)

        # Preserve order for later mapping
        self.functions.sort(key=lambda f: f.line_start)

    def _discover_calls(self):
        """Locate function calls within each discovered function."""
        if not self.functions:
            return

        lines = self.content.split("\n")

        # C standard library functions to skip
        C_BUILTINS = {
            "printf",
            "fprintf",
            "sprintf",
            "snprintf",
            "scanf",
            "fscanf",
            "sscanf",
            "malloc",
            "calloc",
            "realloc",
            "free",
            "alloca",
            "strlen",
            "strcpy",
            "strncpy",
            "strcat",
            "strncat",
            "strcmp",
            "strncmp",
            "strchr",
            "strrchr",
            "strstr",
            "strtok",
            "strtol",
            "strtod",
            "atoi",
            "atof",
            "memcpy",
            "memmove",
            "memset",
            "memcmp",
            "memchr",
            "fopen",
            "fclose",
            "fread",
            "fwrite",
            "fseek",
            "ftell",
            "rewind",
            "fflush",
            "getchar",
            "putchar",
            "gets",
            "puts",
            "fgets",
            "fputs",
            "abs",
            "labs",
            "fabs",
            "ceil",
            "floor",
            "sqrt",
            "pow",
            "sin",
            "cos",
            "tan",
            "exit",
            "abort",
            "atexit",
            "system",
            "getenv",
            "rand",
            "srand",
            "time",
            "sizeof",
            "offsetof",
            "va_start",
            "va_end",
            "va_arg",
        }

        for func in self.functions:
            caller_id = f"{self.file_path}:{func.name}"

            # Look for calls within this function's line range
            end_line = func.line_end or func.line_start
            for line_num in range(func.line_start, end_line + 1):
                if line_num <= len(lines):
                    line = lines[line_num - 1]

                    for call_match in self.CALL_RE.finditer(line):
                        callee = call_match.group(1)

                        # Skip built-ins and keywords
                        if callee in C_BUILTINS or callee in self.C_KEYWORDS:
                            continue

                        # Don't create self-calls
                        if callee == func.name:
                            continue

                        relationship = CallRelationship(
                            caller=caller_id,
                            callee=callee,
                            call_line=line_num,
                            is_resolved=False,
                        )
                        self.call_relationships.append(relationship)


class CppASTAnalyzer(CASTAnalyzer):
    """
    AST analyzer for C++ files.

    TODO: Implement using C++ AST parser like:
    - clang (via subprocess)
    - tree-sitter-cpp
    - or a Python-based C++ parser
    """

    # C++ keywords in addition to C keywords
    CPP_KEYWORDS = {
        # C++ specific keywords
        "class",
        "struct",
        "public",
        "private",
        "protected",
        "virtual",
        "override",
        "namespace",
        "using",
        "template",
        "typename",
        "this",
        "new",
        "delete",
        "try",
        "catch",
        "throw",
        "friend",
        "operator",
        "explicit",
        "mutable",
        "constexpr",
        "decltype",
        "nullptr",
        "auto",
        "final",
        "noexcept",
        "thread_local",
        "alignas",
        "alignof",
        "static_assert",
        "concept",
        "requires",
        # STL and common C++ patterns
        "std",
        "vector",
        "string",
        "map",
        "set",
        "list",
        "deque",
        "stack",
        "queue",
        "pair",
        "tuple",
        "shared_ptr",
        "unique_ptr",
        "weak_ptr",
        "make_shared",
        "make_unique",
    }

    # C++ function definition patterns
    CPP_FUNCTION_DEF_RE = re.compile(
        r"^\s*(?:(?:static|extern|inline|virtual|explicit|constexpr)\s+)*(?:const\s+)?(?:unsigned\s+)?(?:template\s*<[^>]*>\s+)?(?:[\w:]+\s*<?[^>]*>?\s*\*?\s*&?\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)\s*(?:const\s+)?(?:override\s+)?(?:final\s+)?\{",
        re.MULTILINE,
    )

    # C++ method definition pattern (inside class)
    CPP_METHOD_DEF_RE = re.compile(
        r"^\s*(?:(?:static|virtual|explicit|constexpr)\s+)*(?:const\s+)?(?:unsigned\s+)?(?:[\w:]+\s*<?[^>]*>?\s*\*?\s*&?\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)\s*(?:const\s+)?(?:override\s+)?(?:final\s+)?(?:\s*:\s*[^{]*)?(?:\{|;)",
        re.MULTILINE,
    )

    def __init__(self, file_path: str, content: str):
        super().__init__(file_path, content)
        # Combine C and C++ keywords
        self.ALL_KEYWORDS = self.C_KEYWORDS | self.CPP_KEYWORDS

    def _discover_functions(self):
        """Locate C++ function and method definitions."""
        lines = self.content.split("\n")

        # Find regular function definitions
        for match in self.CPP_FUNCTION_DEF_RE.finditer(self.content):
            name = match.group(1)

            # Skip keywords and operators
            if name in self.ALL_KEYWORDS or name.startswith("operator"):
                continue

            self._create_function_from_match(match, name, lines, is_method=False)

        # Find method definitions (simpler pattern)
        for match in self.CPP_METHOD_DEF_RE.finditer(self.content):
            name = match.group(1)

            # Skip keywords, operators, and constructors/destructors
            if (
                name in self.ALL_KEYWORDS
                or name.startswith("operator")
                or name.startswith("~")
            ):
                continue

            self._create_function_from_match(match, name, lines, is_method=True)

        # Remove duplicates and sort
        seen = set()
        unique_functions = []
        for func in self.functions:
            key = (func.name, func.line_start)
            if key not in seen:
                seen.add(key)
                unique_functions.append(func)

        self.functions = sorted(unique_functions, key=lambda f: f.line_start)

    def _create_function_from_match(self, match, name, lines, is_method=False):
        """Helper to create Function object from regex match."""
        line_start = self.content[: match.start()].count("\n") + 1

        # Extract parameters
        func_line = match.group(0)
        params_match = re.search(r"\(([^)]*)\)", func_line)
        params = []
        if params_match:
            param_str = params_match.group(1).strip()
            if param_str and param_str != "void":
                # Simple parameter parsing for C++
                param_parts = [p.strip() for p in param_str.split(",")]
                for param in param_parts:
                    # Extract parameter name (handle C++ references and pointers)
                    param_words = param.split()
                    if param_words:
                        param_name = param_words[-1].lstrip("*&")
                        if param_name and param_name not in self.ALL_KEYWORDS:
                            params.append(param_name)

        # Find function end
        if match.group(0).rstrip().endswith(";"):
            # Declaration only
            line_end = line_start
        else:
            # Definition with body
            start_pos = match.end()
            brace_count = 1
            end_pos = start_pos

            while end_pos < len(self.content) and brace_count > 0:
                if self.content[end_pos] == "{":
                    brace_count += 1
                elif self.content[end_pos] == "}":
                    brace_count -= 1
                end_pos += 1

            line_end = self.content[:end_pos].count("\n") + 1

        # Extract code snippet
        code_snippet = "\n".join(lines[line_start - 1 : line_end])

        func = Function(
            name=name,
            file_path=self.file_path,
            line_start=line_start,
            line_end=line_end,
            parameters=params,
            docstring=None,
            is_method=is_method,
            class_name=None,  # TODO: Extract class context
            code_snippet=code_snippet,
        )
        self.functions.append(func)

    def _discover_calls(self):
        """Locate function calls within each discovered function."""
        if not self.functions:
            return

        lines = self.content.split("\n")

        # C++ standard library functions to skip
        CPP_BUILTINS = {
            # C standard library
            "printf",
            "fprintf",
            "sprintf",
            "snprintf",
            "scanf",
            "fscanf",
            "sscanf",
            "malloc",
            "calloc",
            "realloc",
            "free",
            "strlen",
            "strcpy",
            "strncpy",
            "strcat",
            "strncat",
            "strcmp",
            "strncmp",
            "strchr",
            "strrchr",
            "strstr",
            "strtok",
            "strtol",
            "strtod",
            "atoi",
            "atof",
            "memcpy",
            "memmove",
            "memset",
            "memcmp",
            "memchr",
            "fopen",
            "fclose",
            "fread",
            "fwrite",
            "fseek",
            "ftell",
            "rewind",
            "fflush",
            "getchar",
            "putchar",
            "gets",
            "puts",
            "fgets",
            "fputs",
            "abs",
            "labs",
            "fabs",
            "ceil",
            "floor",
            "sqrt",
            "pow",
            "sin",
            "cos",
            "tan",
            "exit",
            "abort",
            "atexit",
            "system",
            "getenv",
            "rand",
            "srand",
            "time",
            # C++ standard library
            "cout",
            "cin",
            "cerr",
            "endl",
            "flush",
            "setw",
            "setprecision",
            "push_back",
            "pop_back",
            "push_front",
            "pop_front",
            "insert",
            "erase",
            "find",
            "begin",
            "end",
            "size",
            "empty",
            "clear",
            "resize",
            "reserve",
            "make_pair",
            "make_tuple",
            "get",
            "swap",
            "sort",
            "reverse",
            "unique",
            "min",
            "max",
            "accumulate",
            "transform",
            "for_each",
            "count",
            "find_if",
        }

        for func in self.functions:
            caller_id = f"{self.file_path}:{func.name}"

            # Look for calls within this function's line range
            end_line = func.line_end or func.line_start
            for line_num in range(func.line_start, end_line + 1):
                if line_num <= len(lines):
                    line = lines[line_num - 1]

                    for call_match in self.CALL_RE.finditer(line):
                        callee = call_match.group(1)

                        # Skip built-ins and keywords
                        if callee in CPP_BUILTINS or callee in self.ALL_KEYWORDS:
                            continue

                        # Don't create self-calls
                        if callee == func.name:
                            continue

                        relationship = CallRelationship(
                            caller=caller_id,
                            callee=callee,
                            call_line=line_num,
                            is_resolved=False,
                        )
                        self.call_relationships.append(relationship)


# Helper functions to integrate C/C++ analyzers into CallGraphAnalyzer


def analyze_c_file(
    file_path: str, content: str
) -> tuple[List[Function], List[CallRelationship]]:
    """
    Analyze a C file and return functions and relationships.

    This function is called by CallGraphAnalyzer._analyze_c_file()
    """
    analyzer = CASTAnalyzer(file_path, content)
    analyzer.analyze()
    return analyzer.functions, analyzer.call_relationships


def analyze_cpp_file(
    file_path: str, content: str
) -> tuple[List[Function], List[CallRelationship]]:
    """
    Analyze a C++ file and return functions and relationships.

    This function is called by CallGraphAnalyzer._analyze_cpp_file()
    """
    analyzer = CppASTAnalyzer(file_path, content)
    analyzer.analyze()
    return analyzer.functions, analyzer.call_relationships
