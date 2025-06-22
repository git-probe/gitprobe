"""
C/C++ AST Analyzer

Analyzes C and C++ source code using AST parsing to extract
function definitions and function call relationships.

Uses pycparser for C files and libclang for C++ files.
"""

import logging
import re
import os
import tempfile
import subprocess
from typing import List, Tuple, Optional, Set
from pathlib import Path
from models.core import Function, CallRelationship

logger = logging.getLogger(__name__)

# --- Backend Availability Checks ---
PYCPARSER_AVAILABLE = False
LIBCLANG_AVAILABLE = False

try:
    import pycparser
    from pycparser import c_ast, parse_file
    PYCPARSER_AVAILABLE = True
    logger.info("pycparser available for C parsing")
except ImportError:
    logger.warning("pycparser not available. Install with: pip install pycparser")

try:
    import clang.cindex
    from clang.cindex import Index, CursorKind, TypeKind, Config
    
    # Try to configure libclang for different platforms
    try:
        # Common libclang paths for different platforms
        possible_paths = [
            # macOS paths
            '/opt/homebrew/lib/libclang.dylib',
            '/usr/local/lib/libclang.dylib',
            '/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/libclang.dylib',
            '/Library/Developer/CommandLineTools/usr/lib/libclang.dylib',
            # Linux paths
            '/usr/lib/libclang.so',
            '/usr/lib/x86_64-linux-gnu/libclang.so',
            '/usr/lib/llvm/libclang.so',
            # Windows paths
            'C:\\Program Files\\LLVM\\bin\\libclang.dll',
            'libclang.dll',
        ]
        
        configured = False
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    Config.set_library_file(path)
                    logger.info(f"Using libclang from: {path}")
                    configured = True
                    break
                except Exception as e:
                    logger.debug(f"Failed to configure libclang path {path}: {e}")
                    continue
        
        if not configured:
            logger.debug("Could not find libclang in standard locations, trying default configuration")
            
    except Exception as e:
        logger.debug(f"Could not configure libclang path: {e}")
    
    LIBCLANG_AVAILABLE = True
    logger.info("libclang available for C/C++ parsing")
except ImportError:
    logger.warning("libclang not available. Install with: pip install clang")


class GlobalNodeCounter:
    """Shared counter for tracking nodes across all files of the same language."""
    
    def __init__(self, max_nodes: int = 1000):
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
            logger.warning(f"Global C/C++ node limit of {self.max_nodes} reached. Stopping analysis.")
            return True
        return False
    
    def should_stop(self) -> bool:
        """Check if analysis should stop."""
        return self.limit_reached


class CAnalyzer:
    """
    C analyzer using pycparser with direct content sanitization.
    
    Simply removes all problematic constructs and parses the clean C code.
    """

    def __init__(self, file_path: str, content: str, global_counter: Optional[GlobalNodeCounter] = None):
        self.file_path = str(file_path)
        self.content = content
        self.lines = content.splitlines()
        self.functions: List[Function] = []
        self.call_relationships: List[CallRelationship] = []
        self.current_function: Optional[str] = None
        self.global_counter = global_counter or GlobalNodeCounter()

    def analyze(self) -> Tuple[List[Function], List[CallRelationship]]:
        """Analyze C code by sanitizing content and parsing directly."""
        if self.global_counter.should_stop():
            logger.info(f"Skipping {self.file_path} - global C/C++ node limit reached")
            return [], []

        try:
            # Sanitize the content by removing all problematic constructs
            sanitized_content = self._sanitize_content(self.content)
            
            if not sanitized_content.strip():
                logger.debug(f"No content left after sanitization for {self.file_path}")
                return [], []
            
            # Additional validation - ensure content doesn't start with problematic characters
            lines = sanitized_content.splitlines()
            if lines and lines[0].strip().startswith('/'):
                logger.debug(f"Skipping {self.file_path} - content still starts with problematic character")
                return [], []
            
            # Try parsing with multiple fallback strategies
            ast = None
            
            # Strategy 1: Direct parsing
            try:
                ast = pycparser.c_parser.CParser().parse(sanitized_content, filename=self.file_path)
                logger.debug(f"Successfully parsed {self.file_path} with direct parsing")
            except Exception as e1:
                logger.debug(f"Direct parsing failed for {self.file_path}: {e1}")
                
                # Strategy 2: Ultra-aggressive sanitization
                try:
                    ultra_sanitized = self._ultra_sanitize_content(sanitized_content)
                    if ultra_sanitized.strip():
                        ast = pycparser.c_parser.CParser().parse(ultra_sanitized, filename=self.file_path)
                        logger.debug(f"Successfully parsed {self.file_path} with ultra-sanitization")
                except Exception as e2:
                    logger.debug(f"Ultra-sanitization parsing failed for {self.file_path}: {e2}")
                    
                    # Strategy 3: Function-only extraction (fallback)
                    try:
                        self._extract_functions_with_regex(self.content)
                        logger.debug(f"Extracted functions using regex fallback for {self.file_path}")
                        return self.functions, self.call_relationships
                    except Exception as e3:
                        logger.debug(f"Regex extraction failed for {self.file_path}: {e3}")
                        return [], []
            
            if ast:
                # Extract functions and calls
                self._visit_ast(ast)
                
                logger.info(
                    f"C analysis complete for {self.file_path}: {len(self.functions)} functions, "
                    f"{len(self.call_relationships)} calls, "
                    f"global_nodes_processed={self.global_counter.nodes_processed}"
                )
            
            return self.functions, self.call_relationships

        except Exception as e:
            logger.warning(f"Could not parse {self.file_path}: {e}")
            return [], []

    def _sanitize_content(self, content: str) -> str:
        """Sanitize C content by removing all problematic constructs."""
        # Remove BOM if present
        if content.startswith('\ufeff'):
            content = content[1:]
        if content.startswith(b'\xef\xbb\xbf'.decode('utf-8', errors='ignore')):
            content = content[3:]
        
        # Handle potential encoding issues - try to clean non-ASCII characters
        try:
            content = content.encode('ascii', errors='ignore').decode('ascii')
        except (UnicodeDecodeError, UnicodeEncodeError):
            # Fallback: replace problematic characters
            content = re.sub(r'[^\x00-\x7F]+', ' ', content)
        
        # First, remove ALL comments completely using regex
        # Remove multi-line comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        # Remove single-line comments  
        content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
        
        lines = []
        
        for line in content.splitlines():
            stripped = line.strip()
            
            # Skip ALL preprocessor directives
            if stripped.startswith('#'):
                continue
            
            # Skip empty lines
            if not stripped:
                continue
            
            # Skip lines that start with problematic characters that cause "before: /" errors
            if stripped.startswith('/') and not (stripped.startswith('/*') or stripped.startswith('//')):
                continue
            
            # Remove all problematic constructs that pycparser can't handle
            line = self._clean_line(line)
            
            # Skip lines that became empty after cleaning
            cleaned_stripped = line.strip()
            if not cleaned_stripped:
                continue
                
            # Skip lines that contain only problematic characters
            if re.match(r'^[/\*\s]*$', cleaned_stripped):
                continue
                
            # Skip lines that would confuse pycparser
            if any(problematic in cleaned_stripped for problematic in ['/*', '*/', '//']):
                continue
            
            lines.append(line)
        
        # Add minimal typedefs for common types that might be missing
        result = self._add_minimal_typedefs() + '\n\n' + '\n'.join(lines)
        return result
    
    def _clean_line(self, line: str) -> str:
        """Clean a single line of problematic constructs."""
        # Remove all __attribute__ variations
        line = re.sub(r'__attribute__\s*\(\([^)]*\)\)', '', line)
        line = re.sub(r'__attribute__\s*\([^)]*\)', '', line)
        
        # Remove other problematic constructs
        line = re.sub(r'__extension__', '', line)
        line = re.sub(r'__asm__\s*\([^)]*\)', '', line)
        line = re.sub(r'__asm\s*\([^)]*\)', '', line)
        line = re.sub(r'_Pragma\s*\([^)]*\)', '', line)
        line = re.sub(r'__pragma\s*\([^)]*\)', '', line)
        
        # Remove inline assembly blocks
        line = re.sub(r'asm\s*\{[^}]*\}', '', line)
        line = re.sub(r'__asm__\s*\{[^}]*\}', '', line)
        
        # Replace compiler-specific keywords with standard equivalents
        line = re.sub(r'\b__inline__\b', 'inline', line)
        line = re.sub(r'\b__restrict__\b', 'restrict', line)
        line = re.sub(r'\b__restrict\b', 'restrict', line)
        line = re.sub(r'\b__const__\b', 'const', line)
        line = re.sub(r'\b__volatile__\b', 'volatile', line)
        line = re.sub(r'\b__signed__\b', 'signed', line)
        line = re.sub(r'\b__unsigned__\b', 'unsigned', line)
        
        # Remove GCC/Clang builtins
        line = re.sub(r'\b__builtin_\w+\b', 'NULL', line)
        
        # Remove register keyword (deprecated in modern C)
        line = re.sub(r'\bregister\s+', '', line)
        
        # Clean up multiple spaces
        line = re.sub(r'\s+', ' ', line)
        
        return line.strip()
    
    def _add_minimal_typedefs(self) -> str:
        """Add minimal typedefs for common types that might be missing."""
        return """
// Minimal typedefs for common types
typedef long size_t;
typedef long ssize_t;
typedef int pid_t;
typedef unsigned int uint32_t;
typedef int int32_t;
typedef unsigned short uint16_t;
typedef short int16_t;
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef unsigned long long uint64_t;
typedef long long int64_t;
typedef void* FILE;
typedef long off_t;
typedef long time_t;
"""

    def _ultra_sanitize_content(self, content: str) -> str:
        """Ultra-aggressive content sanitization for problematic files."""
        lines = []
        in_function = False
        brace_count = 0
        
        for line in content.splitlines():
            stripped = line.strip()
            
            # Skip everything except function definitions and basic statements
            if not stripped or stripped.startswith('#') or stripped.startswith('//'):
                continue
                
            # Skip complex macro definitions, inline assembly, etc.
            if any(skip in stripped for skip in ['__asm', '_Pragma', '__attribute__', 'register', '__builtin']):
                continue
                
            # Very simple function detection
            if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(', stripped):
                in_function = True
                brace_count = 0
                
            # Track braces to know when function ends
            if in_function:
                brace_count += stripped.count('{') - stripped.count('}')
                lines.append(line)
                if brace_count <= 0 and '}' in stripped:
                    in_function = False
            elif '{' in stripped and not in_function:
                # Potential function start
                lines.append(line)
                in_function = True
                brace_count = stripped.count('{') - stripped.count('}')
                
        result = self._add_minimal_typedefs() + '\n\n' + '\n'.join(lines)
        return result
    
    def _extract_functions_with_regex(self, content: str):
        """Fallback: Extract function definitions using regex patterns."""
        # Simple regex to find function definitions
        func_pattern = r'^\s*(?:static\s+)?(?:inline\s+)?(?:extern\s+)?([a-zA-Z_][a-zA-Z0-9_*\s]+)\s+([a-zA-Z_][a-zA-Z0-9_]+)\s*\([^)]*\)\s*{'
        
        lines = content.splitlines()
        for i, line in enumerate(lines):
            match = re.match(func_pattern, line)
            if match:
                func_name = match.group(2)
                line_start = i + 1
                
                # Find function end by counting braces
                brace_count = line.count('{') - line.count('}')
                line_end = line_start
                
                for j in range(i + 1, min(len(lines), i + 100)):  # Limit search
                    brace_count += lines[j].count('{') - lines[j].count('}')
                    if brace_count <= 0:
                        line_end = j + 1
                        break
                
                # Create function object
                code_snippet = '\n'.join(lines[i:line_end])
                func = Function(
                    name=func_name,
                    file_path=self.file_path,
                    line_start=line_start,
                    line_end=line_end,
                    parameters=[],  # Would need more complex parsing
                    code_snippet=code_snippet,
                    is_method=False,
                    class_name=None
                )
                self.functions.append(func)
                
                # Simple call extraction within this function
                self._extract_calls_with_regex(func_name, code_snippet, line_start)
    
    def _extract_calls_with_regex(self, current_func: str, code: str, start_line: int):
        """Extract function calls using regex patterns."""
        # Simple regex to find function calls
        call_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        
        for i, line in enumerate(code.splitlines()):
            for match in re.finditer(call_pattern, line):
                callee_name = match.group(1)
                
                # Skip common keywords and the function itself
                if callee_name in ['if', 'while', 'for', 'switch', 'return', 'sizeof', current_func]:
                    continue
                    
                relationship = CallRelationship(
                    caller=f"{self.file_path}:{current_func}",
                    callee=callee_name,
                    call_line=start_line + i,
                    is_resolved=False
                )
                self.call_relationships.append(relationship)

    def _visit_ast(self, node):
        """Visit AST nodes to extract functions and calls."""
        if self.global_counter.should_stop():
            return
            
        if isinstance(node, c_ast.FuncDef):
            if self.global_counter.increment():
                return
            self._handle_function_def(node)
        elif isinstance(node, c_ast.FuncCall):
            if self.global_counter.increment():
                return
            self._handle_function_call(node)
        
        # Visit children
        for child in node:
            self._visit_ast(child)

    def _handle_function_def(self, node):
        """Handle function definition."""
        func_name = node.decl.name
        line_start = getattr(node, 'coord', None)
        line_start = line_start.line if line_start else 1
        
        # Find function end in original content
        line_end = self._find_function_end_in_original(func_name, line_start)
        
        # Extract parameters
        params = []
        if node.decl.type.args:
            for param in node.decl.type.args.params:
                if hasattr(param, 'name') and param.name:
                    params.append(param.name)
        
        # Get code snippet from original content
        code_snippet = '\n'.join(self.lines[max(0, line_start - 1):min(len(self.lines), line_end)])
        
        func = Function(
            name=func_name,
            file_path=self.file_path,
            line_start=line_start,
            line_end=line_end,
            parameters=params,
            code_snippet=code_snippet,
            is_method=False,
            class_name=None
        )
        
        self.functions.append(func)
        
        # Set context for call analysis
        old_function = self.current_function
        self.current_function = func_name
        
        # Visit function body
        if node.body:
            self._visit_ast(node.body)
        
        self.current_function = old_function

    def _find_function_end_in_original(self, func_name: str, start_line: int) -> int:
        """Find function end in original content by counting braces."""
        if start_line > len(self.lines):
            return start_line
            
        brace_count = 0
        in_function = False
        
        # Look for the function name and opening brace
        for i in range(max(0, start_line - 5), min(len(self.lines), start_line + 5)):
            line = self.lines[i]
            if func_name in line and '{' in line:
                start_line = i + 1
                break
        
        for i, line in enumerate(self.lines[start_line - 1:], start_line):
            for char in line:
                if char == '{':
                    brace_count += 1
                    in_function = True
                elif char == '}':
                    brace_count -= 1
                    if in_function and brace_count == 0:
                        return i + 1
        
        return min(start_line + 50, len(self.lines))

    def _handle_function_call(self, node):
        """Handle function call."""
        if self.current_function and hasattr(node.name, 'name'):
            callee_name = node.name.name
            call_line = getattr(node, 'coord', None)
            call_line = call_line.line if call_line else 0
            
            # Skip self-calls
            if callee_name != self.current_function:
                relationship = CallRelationship(
                    caller=f"{self.file_path}:{self.current_function}",
                    callee=callee_name,
                    call_line=call_line,
                    is_resolved=False
                )
                self.call_relationships.append(relationship)


class CppAnalyzer:
    """
    C++ analyzer using libclang with proper configuration.
    """

    def __init__(self, file_path: str, content: str, global_counter: Optional[GlobalNodeCounter] = None):
        self.file_path = str(file_path)
        self.content = content
        self.lines = content.splitlines()
        self.functions: List[Function] = []
        self.call_relationships: List[CallRelationship] = []
        self.global_counter = global_counter or GlobalNodeCounter()

    def analyze(self) -> Tuple[List[Function], List[CallRelationship]]:
        """Analyze C++ code using libclang with fallback strategies."""
        if self.global_counter.should_stop():
            logger.info(f"Skipping {self.file_path} - global C/C++ node limit reached")
            return [], []

        # First try libclang
        if LIBCLANG_AVAILABLE:
            try:
                # Create index and parse
                index = Index.create()
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                    f.write(self.content)
                    temp_path = f.name
                
                try:
                    # Parse with comprehensive flags for better compatibility
                    args = [
                        '-std=c++17', 
                        '-w',  # Suppress warnings
                        '-D__PYCPARSER__=1',
                        '-nostdinc',  # Don't use standard includes
                        '-fparse-all-comments'
                    ]
                    
                    tu = index.parse(temp_path, args=args, options=Index.PARSE_DETAILED_PROCESSING_RECORD)
                    
                    if tu and not tu.diagnostics:
                        self._extract_from_cursor(tu.cursor)
                        logger.info(
                            f"C++ analysis complete for {self.file_path}: {len(self.functions)} functions, "
                            f"{len(self.call_relationships)} calls, "
                            f"global_nodes_processed={self.global_counter.nodes_processed}"
                        )
                        return self.functions, self.call_relationships
                    elif tu:
                        # Try with fewer restrictions if there were errors
                        logger.debug(f"Parsing with errors, trying simpler approach for {self.file_path}")
                        args_simple = ['-std=c++17', '-w']
                        tu_simple = index.parse(temp_path, args=args_simple)
                        if tu_simple:
                            self._extract_from_cursor(tu_simple.cursor)
                            if self.functions:  # Success
                                logger.info(f"C++ analysis complete (fallback) for {self.file_path}: {len(self.functions)} functions")
                                return self.functions, self.call_relationships
                    
                finally:
                    os.unlink(temp_path)
                
            except Exception as e:
                logger.debug(f"libclang failed for {self.file_path}: {e}")
        
        # Fallback to regex-based extraction
        try:
            logger.debug(f"Using regex fallback for C++ file {self.file_path}")
            self._extract_cpp_functions_with_regex(self.content)
            logger.info(f"C++ regex analysis complete for {self.file_path}: {len(self.functions)} functions")
            return self.functions, self.call_relationships
        except Exception as e:
            logger.warning(f"Could not parse C++ file {self.file_path}: {e}")
            return [], []

    def _extract_from_cursor(self, cursor):
        """Extract functions and calls from libclang cursor."""
        if self.global_counter.should_stop():
            return
            
        if cursor.kind == CursorKind.FUNCTION_DECL and cursor.is_definition():
            if self.global_counter.increment():
                return
            self._handle_function(cursor)
        elif cursor.kind == CursorKind.CALL_EXPR:
            if self.global_counter.increment():
                return
            self._handle_call(cursor)
        
        # Visit children
        for child in cursor.get_children():
            self._extract_from_cursor(child)

    def _handle_function(self, cursor):
        """Handle function definition."""
        func_name = cursor.spelling
        line_start = cursor.location.line
        line_end = cursor.extent.end.line
        
        # Extract parameters
        params = []
        for arg in cursor.get_arguments():
            if arg.spelling:
                params.append(arg.spelling)
        
        # Get code snippet
        code_snippet = '\n'.join(self.lines[line_start - 1:line_end])
        
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
            class_name=class_name
        )
        
        self.functions.append(func)

    def _handle_call(self, cursor):
        """Handle function call."""
        callee_name = cursor.spelling
        call_line = cursor.location.line
        
        # Find containing function
        caller_func = self._find_containing_function(call_line)
        if caller_func and callee_name != caller_func.name:
            relationship = CallRelationship(
                caller=f"{self.file_path}:{caller_func.name}",
                callee=callee_name,
                call_line=call_line,
                is_resolved=False
            )
            self.call_relationships.append(relationship)

    def _is_method(self, cursor) -> bool:
        """Check if cursor is a method."""
        parent = cursor.semantic_parent
        while parent:
            if parent.kind in [CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL]:
                return True
            parent = parent.semantic_parent
        return False

    def _get_class_name(self, cursor) -> Optional[str]:
        """Get class name for a method."""
        parent = cursor.semantic_parent
        while parent:
            if parent.kind in [CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL]:
                return parent.spelling
            parent = parent.semantic_parent
        return None

    def _find_containing_function(self, line_number: int) -> Optional[Function]:
        """Find function containing given line number."""
        for func in self.functions:
            if func.line_start and func.line_end:
                if func.line_start <= line_number <= func.line_end:
                    return func
        return None
    
    def _extract_cpp_functions_with_regex(self, content: str):
        """Fallback: Extract C++ function definitions using regex patterns."""
        # Enhanced regex to handle C++ functions, methods, constructors
        patterns = [
            # Regular functions: return_type function_name(params) {
            r'^\s*(?:static\s+)?(?:inline\s+)?(?:virtual\s+)?([a-zA-Z_][a-zA-Z0-9_:*\s&]+)\s+([a-zA-Z_][a-zA-Z0-9_:]+)\s*\([^)]*\)\s*(?:const\s*)?{',
            # Methods in classes: return_type ClassName::method(params) {
            r'^\s*(?:static\s+)?(?:inline\s+)?([a-zA-Z_][a-zA-Z0-9_:*\s&]+)\s+([a-zA-Z_][a-zA-Z0-9_:]+::[a-zA-Z_][a-zA-Z0-9_]+)\s*\([^)]*\)\s*(?:const\s*)?{',
            # Constructors: ClassName::ClassName(params) {
            r'^\s*([a-zA-Z_][a-zA-Z0-9_:]+)::([a-zA-Z_][a-zA-Z0-9_]+)\s*\([^)]*\)\s*(?::\s*[^{]+)?{',
        ]
        
        lines = content.splitlines()
        for i, line in enumerate(lines):
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    if len(match.groups()) >= 2:
                        func_name = match.group(2)
                        if '::' in func_name:
                            # Extract just the method name for C++ methods
                            func_name = func_name.split('::')[-1]
                    else:
                        func_name = match.group(1)
                        
                    line_start = i + 1
                    
                    # Find function end by counting braces
                    brace_count = line.count('{') - line.count('}')
                    line_end = line_start
                    
                    for j in range(i + 1, min(len(lines), i + 200)):  # Limit search
                        brace_count += lines[j].count('{') - lines[j].count('}')
                        if brace_count <= 0:
                            line_end = j + 1
                            break
                    
                    # Create function object
                    code_snippet = '\n'.join(lines[i:line_end])
                    is_method = '::' in match.group(2) if len(match.groups()) >= 2 and '::' in match.group(2) else False
                    class_name = match.group(2).split('::')[0] if is_method else None
                    
                    func = Function(
                        name=func_name,
                        file_path=self.file_path,
                        line_start=line_start,
                        line_end=line_end,
                        parameters=[],  # Would need more complex parsing
                        code_snippet=code_snippet,
                        is_method=is_method,
                        class_name=class_name
                    )
                    self.functions.append(func)
                    
                    # Simple call extraction within this function
                    self._extract_calls_with_regex(func_name, code_snippet, line_start)
    
    def _extract_calls_with_regex(self, current_func: str, code: str, start_line: int):
        """Extract function calls using regex patterns for C++."""
        # Enhanced regex to find function calls including method calls
        patterns = [
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',  # Regular function calls
            r'([a-zA-Z_][a-zA-Z0-9_]*)->([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',  # Pointer method calls
            r'([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',  # Object method calls
        ]
        
        for i, line in enumerate(code.splitlines()):
            for pattern in patterns:
                for match in re.finditer(pattern, line):
                    if len(match.groups()) >= 2:
                        callee_name = match.group(2)  # Method name
                    else:
                        callee_name = match.group(1)  # Function name
                    
                    # Skip common keywords and the function itself
                    if callee_name in ['if', 'while', 'for', 'switch', 'return', 'sizeof', 'new', 'delete', current_func]:
                        continue
                        
                    relationship = CallRelationship(
                        caller=f"{self.file_path}:{current_func}",
                        callee=callee_name,
                        call_line=start_line + i,
                        is_resolved=False
                    )
                    self.call_relationships.append(relationship)


def analyze_c_file(file_path: str, content: str, global_counter: Optional[GlobalNodeCounter] = None) -> Tuple[List[Function], List[CallRelationship]]:
    """
    Analyze a C file using the best available method.
    
    Tries multiple approaches in order of reliability:
    1. Tree-sitter (most robust)
    2. Clang command-line tools  
    3. Original pycparser with fallbacks
    
    Args:
        file_path: Path to the C file
        content: Content of the C file
        global_counter: Shared counter for tracking nodes across all C/C++ files
    
    Returns:
        tuple: (functions, call_relationships)
    """
    
    # Strategy 1: Try tree-sitter approach (most robust)
    try:
        from .c_analyzer_treesitter import analyze_c_file_treesitter
        functions, relationships = analyze_c_file_treesitter(file_path, content)
        if functions:  # Success
            logger.info(f"Tree-sitter analysis successful for {file_path}: {len(functions)} functions")
            return functions, relationships
    except Exception as e:
        logger.debug(f"Tree-sitter analysis failed for {file_path}: {e}")
    
    # Strategy 2: Try clang-based approach  
    try:
        from .c_analyzer_clang import analyze_c_file_clang
        functions, relationships = analyze_c_file_clang(file_path, content)
        if functions:  # Success
            logger.info(f"Clang analysis successful for {file_path}: {len(functions)} functions")
            return functions, relationships
    except Exception as e:
        logger.debug(f"Clang analysis failed for {file_path}: {e}")
    
    # Strategy 3: Fallback to original pycparser approach
    logger.debug(f"Using pycparser fallback for {file_path}")
    analyzer = CAnalyzer(file_path, content, global_counter)
    return analyzer.analyze()


def analyze_cpp_file(file_path: str, content: str, global_counter: Optional[GlobalNodeCounter] = None) -> Tuple[List[Function], List[CallRelationship]]:
    """
    Analyze a C++ file using the best available method.
    
    Tries multiple approaches in order of reliability:
    1. Tree-sitter (most robust)
    2. Clang command-line tools
    3. Original libclang with fallbacks
    
    Args:
        file_path: Path to the C++ file
        content: Content of the C++ file
        global_counter: Shared counter for tracking nodes across all C/C++ files
    
    Returns:
        tuple: (functions, call_relationships)
    """
    
    # Strategy 1: Try tree-sitter approach (most robust)
    try:
        from .c_analyzer_treesitter import analyze_cpp_file_treesitter
        functions, relationships = analyze_cpp_file_treesitter(file_path, content)
        if functions:  # Success
            logger.info(f"Tree-sitter C++ analysis successful for {file_path}: {len(functions)} functions")
            return functions, relationships
    except Exception as e:
        logger.debug(f"Tree-sitter C++ analysis failed for {file_path}: {e}")
    
    # Strategy 2: Try clang-based approach
    try:
        from .c_analyzer_clang import analyze_cpp_file_clang
        functions, relationships = analyze_cpp_file_clang(file_path, content)
        if functions:  # Success
            logger.info(f"Clang C++ analysis successful for {file_path}: {len(functions)} functions")
            return functions, relationships
    except Exception as e:
        logger.debug(f"Clang C++ analysis failed for {file_path}: {e}")
    
    # Strategy 3: Fallback to original libclang approach
    logger.debug(f"Using libclang fallback for {file_path}")
    analyzer = CppAnalyzer(file_path, content, global_counter)
    return analyzer.analyze() 