"""
JavaScript/TypeScript AST Analyzer

Placeholder for upcoming JavaScript and TypeScript AST analysis functionality.
This will be integrated into the AnalysisService to support multi-language analysis.
"""

import re
import logging
from typing import List, Tuple, Optional
from models.core import Function, CallRelationship
from tree_sitter import Language, Parser

# TODO: Add proper error handling and logging
logger = logging.getLogger(__name__)

# --- Tree-sitter Language Setup ---
# You must build the tree-sitter languages before running
# See: https://github.com/tree-sitter/py-tree-sitter#installation
try:
    logger.info("Attempting to load tree-sitter JavaScript language")
    # Modern tree-sitter API - build library first, then load languages
    from tree_sitter import Language
    
    # For now, we'll disable tree-sitter parsing and fall back to regex-based analysis
    # This is because tree-sitter requires pre-built language files that aren't included
    logger.warning("Tree-sitter languages not available - using regex-based analysis fallback")
    JS_LANGUAGE = None
    TS_LANGUAGE = None
    
except Exception as e:
    logger.error(f"Error loading tree-sitter languages: {e}", exc_info=True)
    JS_LANGUAGE = None
    TS_LANGUAGE = None


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
            logger.warning(f"Global JavaScript node limit of {self.max_nodes} reached. Stopping all JavaScript analysis.")
            return True
        return False
    
    def should_stop(self) -> bool:
        """Check if analysis should stop."""
        return self.limit_reached


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
        "if",
        "else",
        "for",
        "while",
        "do",
        "switch",
        "try",
        "catch",
        "finally",
        "return",
        "break",
        "continue",
        "throw",
        "new",
        "delete",
        "typeof",
        "instanceof",
        "void",
        "null",
        "undefined",
        "true",
        "false",
        "var",
        "let",
        "const",
        "function",
        "class",
        "extends",
        "import",
        "export",
        "default",
        "async",
        "await",
    }

    FUNCTION_DECL_RE = re.compile(r"^\s*function\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\(")
    EXPORT_FUNCTION_RE = re.compile(r"^\s*export\s+function\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\(")
    EXPORT_DEFAULT_FUNCTION_RE = re.compile(r"^\s*export\s+default\s+function\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\(")
    EXPORT_DEFAULT_ANON_FUNCTION_RE = re.compile(r"^\s*export\s+default\s+function\s*\(")
    ARROW_FUNCTION_RE = re.compile(
        r"^\s*(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(?:\([^)]*\)|[A-Za-z_$][A-Za-z0-9_$]*)\s*=>"
    )
    EXPORT_ARROW_FUNCTION_RE = re.compile(
        r"^\s*export\s+(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(?:\([^)]*\)|[A-Za-z_$][A-Za-z0-9_$]*)\s*=>"
    )
    # Method definition – best-effort: `<name>(...) {` that is not a control keyword
    METHOD_DEF_RE = re.compile(
        r"^\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*\([^)]*\)\s*{", re.ASCII
    )

    CALL_RE = re.compile(r"([A-Za-z_$][A-Za-z0-9_$]*)\s*\(")

    def __init__(self, file_path: str, content: str, global_counter: Optional[GlobalNodeCounter] = None):
        logger.info(f"JavaScriptASTAnalyzer.__init__ called with file_path='{file_path}', content_length={len(content)}")
        
        self.file_path = file_path
        self.content = content
        self.functions: List[Function] = []
        self.call_relationships: List[CallRelationship] = []
        self.global_counter = global_counter or GlobalNodeCounter()
        
        logger.info(f"JavaScriptASTAnalyzer initialized for {file_path} with global limit: {self.global_counter.max_nodes}")

    def analyze(self):
        """Parse the JavaScript file and populate functions and relationships using regex-based heuristics."""
        if self.global_counter.should_stop():
            logger.info(f"Skipping {self.file_path} - global JavaScript node limit already reached")
            return
            
        logger.info("Starting regex-based JavaScript analysis")
        self.functions = []
        self.call_relationships = []

        # First pass – discover function definitions
        self._discover_functions()

        # Second pass – discover call relationships inside the known functions
        if not self.global_counter.should_stop():
            self._discover_calls()
            
        logger.info(
            f"JavaScript analysis complete for {self.file_path}: {len(self.functions)} functions, "
            f"{len(self.call_relationships)} relationships, "
            f"global_nodes_processed={self.global_counter.nodes_processed}"
        )

    def _discover_functions(self):
        """Locate function definitions (declarations, arrows, methods)."""
        lines = self.content.split("\n")
        for idx, line in enumerate(lines, 1):
            if self.global_counter.should_stop():
                break
                
            line_stripped = line.rstrip()

            # Check different function patterns in order of specificity
            name = None
            
            # 1. Export default function with name
            match = self.EXPORT_DEFAULT_FUNCTION_RE.match(line_stripped)
            if match:
                name = match.group(1)
            else:
                # 2. Export default anonymous function
                match = self.EXPORT_DEFAULT_ANON_FUNCTION_RE.match(line_stripped)
                if match:
                    name = "default"  # Use "default" as the function name
                else:
                    # 3. Export function
                    match = self.EXPORT_FUNCTION_RE.match(line_stripped)
                    if match:
                        name = match.group(1)
                    else:
                        # 4. Export arrow function
                        match = self.EXPORT_ARROW_FUNCTION_RE.match(line_stripped)
                        if match:
                            name = match.group(1)
                        else:
                            # 5. Regular function declaration
                            match = self.FUNCTION_DECL_RE.match(line_stripped)
                            if match:
                                name = match.group(1)
                            else:
                                # 6. Regular arrow function
                                match = self.ARROW_FUNCTION_RE.match(line_stripped)
                                if match:
                                    name = match.group(1)
                                else:
                                    # 7. Method definition
                                    match = self.METHOD_DEF_RE.match(line_stripped)
                                    if match:
                                        name = match.group(1)
                                        # Skip JavaScript keywords
                                        if name in self.JS_KEYWORDS:
                                            continue
                                    else:
                                        continue

            if not name:
                continue

            # Count function definitions as meaningful nodes
            if self.global_counter.increment():
                break

            # Extract parameters if possible
            import re
            params_match = re.search(r"\(([^)]*)\)", line_stripped)
            params = (
                [p.strip() for p in params_match.group(1).split(",") if p.strip()]
                if params_match
                else []
            )

            # Determine end of function for code snippet
            end_line_idx = idx - 1  # zero-based index into lines list
            brace_depth = line_stripped.count("{") - line_stripped.count("}")

            if brace_depth > 0:
                # Add safety limit to prevent infinite loops
                max_lines_to_check = min(len(lines) - end_line_idx - 1, 1000)
                for j in range(end_line_idx + 1, end_line_idx + 1 + max_lines_to_check):
                    if j >= len(lines):
                        break
                    brace_depth += lines[j].count("{") - lines[j].count("}")
                    if brace_depth <= 0:
                        end_line_idx = j
                        break

            code_snippet = "\n".join(lines[idx - 1 : end_line_idx + 1])
            
            # Ensure file_path is a string (fix for Pydantic validation)
            file_path_str = str(self.file_path)

            func = Function(
                name=name,
                file_path=file_path_str,
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

        # Add filtering for built-ins:
        JS_BUILTINS = {
            "console",
            "setTimeout",
            "setInterval",
            "parseInt",
            "parseFloat",
            "JSON",
            "Math",
            "Date",
            "Array",
            "Object",
            "String",
            "Number",
        }

        for func, start, end in boundaries:
            if self.global_counter.should_stop():
                break
                
            caller_id = f"{self.file_path}:{func.name}"
            for ln in range(start, end + 1):
                if self.global_counter.should_stop():
                    break
                    
                for call_match in self.CALL_RE.finditer(lines[ln - 1]):
                    # Count function calls as meaningful nodes
                    if self.global_counter.increment():
                        return
                        
                    callee = call_match.group(1)
                    # Skip built-ins when creating relationships:
                    if callee in JS_BUILTINS:
                        continue
                    relationship = CallRelationship(
                        caller=caller_id,
                        callee=callee,
                        call_line=ln,
                        is_resolved=False,
                    )
                    self.call_relationships.append(relationship)


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

    # Override patterns to ensure TypeScript-specific exports are captured
    FUNCTION_DECL_RE = re.compile(r"^\s*function\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\(")
    EXPORT_FUNCTION_RE = re.compile(r"^\s*export\s+function\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\(")
    EXPORT_DEFAULT_FUNCTION_RE = re.compile(r"^\s*export\s+default\s+function\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\(")
    EXPORT_DEFAULT_ANON_FUNCTION_RE = re.compile(r"^\s*export\s+default\s+function\s*\(")
    ARROW_FUNCTION_RE = re.compile(
        r"^\s*(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(?:\([^)]*\)|[A-Za-z_$][A-Za-z0-9_$]*)\s*=>"
    )
    EXPORT_ARROW_FUNCTION_RE = re.compile(
        r"^\s*export\s+(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(?:\([^)]*\)|[A-Za-z_$][A-Za-z0-9_$]*)\s*=>"
    )
    # Method definition – best-effort: `<name>(...) {` that is not a control keyword
    METHOD_DEF_RE = re.compile(
        r"^\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*\([^)]*\)\s*{", re.ASCII
    )

    def __init__(self, file_path: str, content: str, global_counter: Optional[GlobalNodeCounter] = None):
        logger.info(f"TypeScriptASTAnalyzer.__init__ called with file_path='{file_path}', content_length={len(content)}")
        
        # Just call the parent constructor - no tree-sitter needed
        super().__init__(file_path, content, global_counter)
        
        logger.info(f"TypeScriptASTAnalyzer initialized for {file_path} with global limit: {self.global_counter.max_nodes}")

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


def analyze_javascript_file(
    file_path: str, content: str, global_counter: Optional[GlobalNodeCounter] = None
) -> tuple[List[Function], List[CallRelationship]]:
    """
    Analyze a JavaScript file and return functions and relationships.

    This function is called by CallGraphAnalyzer._analyze_javascript_file()
    """
    try:
        logger.info(f"analyze_javascript_file called with file_path='{file_path}', content_length={len(content)}")
        logger.info(f"About to create JavaScriptASTAnalyzer instance")
        analyzer = JavaScriptASTAnalyzer(file_path, content, global_counter)
        logger.info(f"JavaScriptASTAnalyzer created successfully, calling analyze()")
        analyzer.analyze()
        logger.info(f"Analysis complete, returning {len(analyzer.functions)} functions and {len(analyzer.call_relationships)} relationships")
        return analyzer.functions, analyzer.call_relationships
    except Exception as e:
        logger.error(f"Error in analyze_javascript_file for {file_path}: {e}", exc_info=True)
        return [], []


def analyze_typescript_file(
    file_path: str, content: str, global_counter: Optional[GlobalNodeCounter] = None
) -> tuple[List[Function], List[CallRelationship]]:
    """
    Analyze a TypeScript file and return functions and relationships.
    This function is called by CallGraphAnalyzer._analyze_typescript_file()
    """
    try:
        logger.info(f"analyze_typescript_file called with file_path='{file_path}', content_length={len(content)}")
        logger.info(f"About to create TypeScriptASTAnalyzer instance")
        analyzer = TypeScriptASTAnalyzer(file_path, content, global_counter)
        logger.info(f"TypeScriptASTAnalyzer created successfully, calling analyze()")
        analyzer.analyze()
        logger.info(f"Analysis complete, returning {len(analyzer.functions)} functions and {len(analyzer.call_relationships)} relationships")
        return analyzer.functions, analyzer.call_relationships
    except Exception as e:
        logger.error(f"Error in analyze_typescript_file for {file_path}: {e}", exc_info=True)
        return [], []