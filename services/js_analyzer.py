"""
JavaScript/TypeScript AST Analyzer

Placeholder for upcoming JavaScript and TypeScript AST analysis functionality.
This will be integrated into the AnalysisService to support multi-language analysis.
"""

import re
from typing import List
from models.core import Function, CallRelationship

class JavaScriptASTAnalyzer:
    """
    AST analyzer for JavaScript files.

    NOTE:
    -------
    A production-ready call-graph extractor should ideally leverage a full
    JavaScript parser (e.g. tree-sitter, esprima, acorn, etc.).
    For the purposes of this project we want **good-enough** heuristics that
    work without any native add-ons or external processes so that the test
    suite can run in any environment – including CI systems that may not have
    node or a C tool-chain.

    Therefore the implementation below uses **regular-expression based
    heuristics** to:
    1. Locate function definitions (declarations, arrow functions, simple
       method definitions inside classes / objects).
    2. Create `Function` model instances with reasonable metadata (name, file
       path, start/end lines, etc.).
    3. Perform a second pass to discover potential function calls inside the
       detected functions and create `CallRelationship` instances.

    While not 100 % accurate, these heuristics are sufficient for small / well
    structured repositories and satisfy the expectations of the accompanying
    tests until a more sophisticated parser is introduced.
    """

    # JavaScript keywords that should NOT be treated as functions
    JS_KEYWORDS = {
        'if', 'else', 'for', 'while', 'do', 'switch', 'try', 'catch', 'finally',
        'return', 'break', 'continue', 'throw', 'new', 'delete', 'typeof', 'instanceof',
        'void', 'null', 'undefined', 'true', 'false', 'var', 'let', 'const',
        'function', 'class', 'extends', 'import', 'export', 'default', 'async', 'await'
    }
    
    FUNCTION_DECL_RE = re.compile(r"^\s*function\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\(")
    ARROW_FUNCTION_RE = re.compile(
        r"^\s*(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(?:\([^)]*\)|[A-Za-z_$][A-Za-z0-9_$]*)\s*=>"
    )
    # Method definition – best-effort: `<name>(...) {` that is not a control keyword
    METHOD_DEF_RE = re.compile(r"^\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*\([^)]*\)\s*{", re.ASCII)
    
    CALL_RE = re.compile(r"([A-Za-z_$][A-Za-z0-9_$]*)\s*\(")

    def __init__(self, file_path: str, content: str):
        self.file_path = file_path
        self.content = content
        self.functions: List[Function] = []
        self.call_relationships: List[CallRelationship] = []
    
    def analyze(self):
        """Parse the JavaScript file and populate ``self.functions`` and
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
        """Locate function definitions (declarations, arrows, methods)."""
        lines = self.content.split("\n")
        for idx, line in enumerate(lines, 1):
            line_stripped = line.rstrip()

            match = self.FUNCTION_DECL_RE.match(line_stripped)
            if match:
                name = match.group(1)
            else:
                match = self.ARROW_FUNCTION_RE.match(line_stripped)
                if match:
                    name = match.group(1)
                else:
                    match = self.METHOD_DEF_RE.match(line_stripped)
                    if match:
                        name = match.group(1)
                        # Skip JavaScript keywords
                        if name in self.JS_KEYWORDS:
                            continue
                    else:
                        continue

            # Extract parameters if possible
            params_match = re.search(r"\(([^)]*)\)", line_stripped)
            params = (
                [p.strip() for p in params_match.group(1).split(",") if p.strip()]
                if params_match
                else []
            )

            # -------- Determine end of function for code snippet ----------
            end_line_idx = idx - 1  # zero-based index into lines list
            brace_depth = line_stripped.count("{") - line_stripped.count("}")

            if brace_depth > 0:
                for j in range(end_line_idx + 1, len(lines)):
                    brace_depth += lines[j].count("{") - lines[j].count("}")
                    if brace_depth <= 0:
                        end_line_idx = j
                        break
            # If we never opened a brace (e.g., concise arrow fn) we keep single line

            code_snippet = "\n".join(lines[idx - 1 : end_line_idx + 1])

            func = Function(
                name=name,
                file_path=self.file_path,
                line_start=idx,
                line_end=end_line_idx + 1,
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
        total_lines = len(lines)

        # Build function boundary ranges
        boundaries = []
        for idx, func in enumerate(self.functions):
            start_line = func.line_start
            end_line = (
                self.functions[idx + 1].line_start - 1
                if idx + 1 < len(self.functions)
                else total_lines
            )
            boundaries.append((func, start_line, end_line))

        for func, start, end in boundaries:
            caller_id = f"{self.file_path}:{func.name}"
            for ln in range(start, end + 1):
                for call_match in self.CALL_RE.finditer(lines[ln - 1]):
                    callee = call_match.group(1)
                    relationship = CallRelationship(
                        caller=caller_id,
                        callee=callee,
                        call_line=ln,
                        is_resolved=False,
                    )
                    self.call_relationships.append(relationship)
    
    def _extract_function_declarations(self):
        """Extract function declarations from JavaScript AST."""
        # TODO: Parse function declarations
        pass
    
    def _extract_arrow_functions(self):
        """Extract arrow function expressions."""
        # TODO: Parse arrow functions
        pass
    
    def _extract_method_definitions(self):
        """Extract method definitions from classes and objects."""
        # TODO: Parse method definitions
        pass
    
    def _extract_function_calls(self):
        """Extract function call relationships."""
        # TODO: Parse function calls and build relationships
        pass


class TypeScriptASTAnalyzer(JavaScriptASTAnalyzer):
    """
    AST analyzer for TypeScript files.
    
    TODO: Implement using TypeScript AST parser like:
    - typescript npm package (via subprocess)
    - ts-morph (via subprocess)
    - or a Python-based TypeScript parser
    """
    
    # The TypeScript analyzer reuses the same lightweight heuristics used for
    # JavaScript.  A dedicated TypeScript parser could improve accuracy (for
    # example by recognising generics and interface methods) but would require
    # external dependencies.  For now the parent implementation is sufficient.

    def __init__(self, file_path: str, content: str):
        super().__init__(file_path, content)

    # The ``analyze`` method from the parent class already discovers functions
    # and calls, so no override is required.

    
    def _extract_typed_functions(self):
        """Extract functions with TypeScript type annotations."""
        # TODO: Parse typed function declarations
        pass
    
    def _extract_interfaces_and_types(self):
        """Extract interface and type definitions."""
        # TODO: Parse interfaces and type aliases
        pass
    
    def _extract_generic_functions(self):
        """Extract generic function definitions."""
        # TODO: Parse generic functions
        pass


# Helper functions to integrate JS/TS analyzers into CallGraphAnalyzer

def analyze_javascript_file(file_path: str, content: str) -> tuple[List[Function], List[CallRelationship]]:
    """
    Analyze a JavaScript file and return functions and relationships.
    
    This function is called by CallGraphAnalyzer._analyze_javascript_file()
    """
    analyzer = JavaScriptASTAnalyzer(file_path, content)
    analyzer.analyze()
    return analyzer.functions, analyzer.call_relationships


def analyze_typescript_file(file_path: str, content: str) -> tuple[List[Function], List[CallRelationship]]:
    """
    Analyze a TypeScript file and return functions and relationships.
    
    This function is called by CallGraphAnalyzer._analyze_typescript_file()
    """
    analyzer = TypeScriptASTAnalyzer(file_path, content)
    analyzer.analyze()
    return analyzer.functions, analyzer.call_relationships 