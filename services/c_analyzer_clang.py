"""
C/C++ AST Analyzer using Clang Python Bindings

This approach uses the official Clang Python bindings which are more
robust than pycparser and handle real-world C/C++ code better.
"""

import logging
import subprocess
import tempfile
import os
from typing import List, Tuple, Optional
from pathlib import Path
from models.core import Function, CallRelationship

logger = logging.getLogger(__name__)


class ClangBasedAnalyzer:
    """
    C/C++ analyzer using clang command-line tools with Python parsing.
    
    This approach uses clang's AST dump feature which is more reliable
    than trying to parse C/C++ directly in Python.
    """
    
    def __init__(self, file_path: str, content: str):
        self.file_path = str(file_path)
        self.content = content
        self.functions: List[Function] = []
        self.call_relationships: List[CallRelationship] = []
    
    def analyze(self) -> Tuple[List[Function], List[CallRelationship]]:
        """Analyze using clang AST dump."""
        
        try:
            # Try clang AST dump approach
            if self._analyze_with_clang_ast():
                logger.info(f"Clang AST analysis successful for {self.file_path}")
                return self.functions, self.call_relationships
        except Exception as e:
            logger.debug(f"Clang AST analysis failed: {e}")
        
        # Fallback to ctags approach
        try:
            self._analyze_with_ctags()
            logger.info(f"Ctags analysis successful for {self.file_path}")
        except Exception as e:
            logger.debug(f"Ctags analysis failed: {e}")
        
        return self.functions, self.call_relationships
    
    def _analyze_with_clang_ast(self) -> bool:
        """Use clang to dump AST and parse it."""
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
            f.write(self.content)
            temp_path = f.name
        
        try:
            # Run clang to get AST dump
            cmd = [
                'clang', '-Xclang', '-ast-dump', '-fsyntax-only',
                '-fno-color-diagnostics', '-w',  # Suppress warnings
                temp_path
            ]
            
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            
            if result.returncode == 0:
                self._parse_clang_ast_output(result.stdout)
                return True
            else:
                logger.debug(f"Clang failed: {result.stderr}")
                return False
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.debug("Clang not available or timed out")
            return False
        finally:
            os.unlink(temp_path)
    
    def _parse_clang_ast_output(self, ast_output: str):
        """Parse clang AST output to extract functions."""
        
        lines = ast_output.splitlines()
        current_function = None
        
        for line in lines:
            # Function declarations: FunctionDecl
            if 'FunctionDecl' in line and 'used' in line:
                func_match = self._extract_function_from_ast_line(line)
                if func_match:
                    self.functions.append(func_match)
                    current_function = func_match.name
            
            # Function calls: CallExpr
            elif 'CallExpr' in line and current_function:
                call_match = self._extract_call_from_ast_line(line, current_function)
                if call_match:
                    self.call_relationships.append(call_match)
    
    def _extract_function_from_ast_line(self, line: str) -> Optional[Function]:
        """Extract function info from clang AST line."""
        import re
        
        # Parse function declaration line
        # Example: FunctionDecl 0x... <line:5:1, line:8:1> line:5:5 used main 'int (int, char **)'
        match = re.search(r"line:(\d+):\d+.*?(\w+)\s+'([^']+)'", line)
        if match:
            line_num = int(match.group(1))
            func_name = match.group(2)
            func_type = match.group(3)
            
            return Function(
                name=func_name,
                file_path=self.file_path,
                line_start=line_num,
                line_end=line_num + 10,  # Estimate
                parameters=[],  # Would need more parsing
                code_snippet="",  # Would need to extract from original
                is_method=False,
                class_name=None
            )
        return None
    
    def _extract_call_from_ast_line(self, line: str, current_func: str) -> Optional[CallRelationship]:
        """Extract call info from clang AST line."""
        import re
        
        # Parse call expression line
        match = re.search(r"line:(\d+):\d+.*?'(\w+)'", line)
        if match:
            line_num = int(match.group(1))
            callee_name = match.group(2)
            
            if callee_name != current_func:  # Skip self-calls
                return CallRelationship(
                    caller=f"{self.file_path}:{current_func}",
                    callee=callee_name,
                    call_line=line_num,
                    is_resolved=False
                )
        return None
    
    def _analyze_with_ctags(self):
        """Fallback: Use ctags for function extraction."""
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
            f.write(self.content)
            temp_path = f.name
        
        try:
            # Run ctags
            cmd = ['ctags', '-x', '--kinds-c=f', temp_path]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                self._parse_ctags_output(result.stdout)
            else:
                logger.debug(f"Ctags failed: {result.stderr}")
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.debug("Ctags not available")
        finally:
            os.unlink(temp_path)
    
    def _parse_ctags_output(self, ctags_output: str):
        """Parse ctags output to extract functions."""
        
        for line in ctags_output.strip().splitlines():
            parts = line.split()
            if len(parts) >= 3:
                func_name = parts[0]
                line_num = int(parts[2]) if parts[2].isdigit() else 1
                
                func = Function(
                    name=func_name,
                    file_path=self.file_path,
                    line_start=line_num,
                    line_end=line_num + 10,  # Estimate
                    parameters=[],
                    code_snippet="",
                    is_method=False,
                    class_name=None
                )
                self.functions.append(func)
                
                # Extract calls using regex within function
                self._extract_calls_for_function(func_name, line_num)
    
    def _extract_calls_for_function(self, func_name: str, start_line: int):
        """Extract function calls using simple regex."""
        import re
        
        lines = self.content.splitlines()
        
        # Look for function body around the start line
        for i in range(max(0, start_line - 1), min(len(lines), start_line + 50)):
            line = lines[i]
            
            # Find function calls
            for match in re.finditer(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', line):
                callee = match.group(1)
                
                # Skip keywords and self-calls
                if (callee not in ['if', 'while', 'for', 'switch', 'return', 'sizeof'] and
                    callee != func_name):
                    
                    relationship = CallRelationship(
                        caller=f"{self.file_path}:{func_name}",
                        callee=callee,
                        call_line=i + 1,
                        is_resolved=False
                    )
                    self.call_relationships.append(relationship)


def analyze_c_file_clang(file_path: str, content: str) -> Tuple[List[Function], List[CallRelationship]]:
    """Analyze C file using clang-based approach."""
    analyzer = ClangBasedAnalyzer(file_path, content)
    return analyzer.analyze()


def analyze_cpp_file_clang(file_path: str, content: str) -> Tuple[List[Function], List[CallRelationship]]:
    """Analyze C++ file using clang-based approach."""
    analyzer = ClangBasedAnalyzer(file_path, content)
    return analyzer.analyze()