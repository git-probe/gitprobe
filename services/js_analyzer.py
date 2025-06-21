"""
Advanced JavaScript/TypeScript analyzer using Tree-sitter for accurate AST parsing.

This module provides proper AST-based analysis for JavaScript and TypeScript files,
replacing the regex-based approach with a more accurate tree-sitter implementation.
"""

import logging
import time
from typing import List, Set, Optional
from pathlib import Path

import tree_sitter
import tree_sitter_javascript
import tree_sitter_typescript

from models.core import Function, CallRelationship

logger = logging.getLogger(__name__)


class AnalysisLimits:
    """Configurable limits for JavaScript/TypeScript analysis to prevent excessive resource usage."""
    
    def __init__(self, max_nodes: int = 800, max_time: float = 10.0):
        self.max_nodes = max_nodes
        self.max_time = max_time
        self.nodes_processed = 0
        self.start_time = time.time()
        self.limit_reached = False
    
    def increment(self) -> bool:
        """Increment node counter and check limits. Returns True if limits exceeded."""
        if self.limit_reached:
            return True
        
        self.nodes_processed += 1
        elapsed = time.time() - self.start_time
        
        if self.nodes_processed >= self.max_nodes:
            logger.warning(f"JavaScript analysis hit node limit: {self.max_nodes} nodes")
            self.limit_reached = True
            return True
        
        if elapsed >= self.max_time:
            logger.warning(f"JavaScript analysis hit time limit: {self.max_time}s")
            self.limit_reached = True
            return True
        
        return False
    
    def should_stop(self) -> bool:
        """Check if analysis should stop due to limits."""
        return self.limit_reached


class TreeSitterJSAnalyzer:
    """JavaScript analyzer using tree-sitter for proper AST parsing."""
    
    def __init__(self, file_path: str, content: str, limits: Optional[AnalysisLimits] = None):
        self.file_path = Path(file_path)
        self.content = content
        self.functions: List[Function] = []
        self.call_relationships: List[CallRelationship] = []
        self.limits = limits or AnalysisLimits()
        
        # Initialize tree-sitter
        self.js_language = tree_sitter.Language(tree_sitter_javascript.language())
        self.parser = tree_sitter.Parser(self.js_language)
        
        logger.info(f"TreeSitterJSAnalyzer initialized for {file_path} with limits: {self.limits.max_nodes} nodes, {self.limits.max_time}s")
    
    def analyze(self) -> None:
        """Analyze the JavaScript content and extract functions and call relationships."""
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
            
            logger.info(f"Analysis complete: {len(self.functions)} functions, {len(self.call_relationships)} relationships, {self.limits.nodes_processed} nodes processed")
            
        except Exception as e:
            logger.error(f"Error analyzing JavaScript file {self.file_path}: {e}", exc_info=True)
    
    def _extract_functions(self, node) -> None:
        """Extract all function definitions from the AST."""
        self._traverse_for_functions(node)
        self.functions.sort(key=lambda f: f.line_start)
    
    def _traverse_for_functions(self, node) -> None:
        """Recursively traverse AST nodes to find functions."""
        
        # Check limits before processing
        if self.limits.increment():
            return
        
        # Handle different function types
        if node.type == 'function_declaration':
            func = self._extract_function_declaration(node)
            if func and self._should_include_function(func):
                self.functions.append(func)
        
        elif node.type == 'export_statement':
            func = self._extract_exported_function(node)
            if func and self._should_include_function(func):
                self.functions.append(func)
        
        elif node.type == 'lexical_declaration':
            func = self._extract_arrow_function_from_declaration(node)
            if func and self._should_include_function(func):
                self.functions.append(func)
        
        elif node.type == 'method_definition':
            func = self._extract_method_definition(node)
            if func and self._should_include_function(func):
                self.functions.append(func)
        
        # Recursively process all child nodes (with limit checks)
        for child in node.children:
            if self.limits.should_stop():
                break
            self._traverse_for_functions(child)
    
    def _extract_function_declaration(self, node) -> Optional[Function]:
        """Extract regular function declaration: function name() {}"""
        try:
            # Find function name
            name_node = self._find_child_by_type(node, 'identifier')
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
                docstring=None,
                is_method=False,
                class_name=None,
                code_snippet=code_snippet,
            )
        except Exception as e:
            logger.warning(f"Error extracting function declaration: {e}")
            return None
    
    def _extract_exported_function(self, node) -> Optional[Function]:
        """Extract export function or export default function"""
        try:
            # Look for function_declaration within export_statement
            func_decl = self._find_child_by_type(node, 'function_declaration')
            if func_decl:
                func = self._extract_function_declaration(func_decl)
                if func:
                    # Check if it's export default without a name
                    export_text = self._get_node_text(node)
                    if 'export default' in export_text and 'function (' in export_text:
                        func.name = 'default'
                return func
        except Exception as e:
            logger.warning(f"Error extracting exported function: {e}")
        return None
    
    def _extract_arrow_function_from_declaration(self, node) -> Optional[Function]:
        """Extract arrow function from const/let/var declarations"""
        try:
            # Look for variable_declarator containing arrow_function
            for child in node.children:
                if child.type == 'variable_declarator':
                    name_node = self._find_child_by_type(child, 'identifier')
                    arrow_node = self._find_child_by_type(child, 'arrow_function')
                    
                    if name_node and arrow_node:
                        func_name = self._get_node_text(name_node)
                        line_start = arrow_node.start_point[0] + 1
                        line_end = arrow_node.end_point[0] + 1
                        parameters = self._extract_parameters(arrow_node)
                        code_snippet = self._get_node_text(child)  # Get full declaration
                        
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
            logger.warning(f"Error extracting arrow function: {e}")
        return None
    
    def _extract_method_definition(self, node) -> Optional[Function]:
        """Extract class method definition"""
        try:
            # Find method name
            property_name = self._find_child_by_type(node, 'property_identifier')
            if not property_name:
                return None
            
            func_name = self._get_node_text(property_name)
            line_start = node.start_point[0] + 1
            line_end = node.end_point[0] + 1
            parameters = self._extract_parameters(node)
            code_snippet = self._get_node_text(node)
            class_name = self._find_containing_class_name(node)
            
            return Function(
                name=func_name,
                file_path=str(self.file_path),
                line_start=line_start,
                line_end=line_end,
                parameters=parameters,
                docstring=None,
                is_method=True,
                class_name=class_name,
                code_snippet=code_snippet,
            )
        except Exception as e:
            logger.warning(f"Error extracting method definition: {e}")
            return None
    
    def _should_include_function(self, func: Function) -> bool:
        """Determine if a function should be included in the analysis."""
        # Filter out constructors and other non-useful functions
        excluded_names = {
            'constructor', 'destructor', 'render',
            'componentDidMount', 'componentDidUpdate', 'componentWillUnmount',
            'getInitialState', 'getDefaultProps',
        }
        
        if func.name.lower() in excluded_names:
            logger.debug(f"Skipping excluded function: {func.name}")
            return False
        
        # Skip very short functions (likely getters/setters)
        if func.line_end - func.line_start < 2:
            logger.debug(f"Skipping short function: {func.name}")
            return False
        
        return True
    
    def _extract_parameters(self, node) -> List[str]:
        """Extract parameter names from a function node."""
        parameters = []
        params_node = self._find_child_by_type(node, 'formal_parameters')
        if params_node:
            for child in params_node.children:
                if child.type == 'identifier':
                    parameters.append(self._get_node_text(child))
        return parameters
    
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
        
        # Check limits before processing
        if self.limits.increment():
            return
        
        if node.type == 'call_expression':
            call_info = self._extract_call_from_node(node, func_ranges)
            if call_info:
                self.call_relationships.append(call_info)
        
        for child in node.children:
            if self.limits.should_stop():
                break
            self._traverse_for_calls(child, func_ranges)
    
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
            
            if callee_node.type == 'identifier':
                return self._get_node_text(callee_node)
            elif callee_node.type == 'member_expression':
                # For method calls like obj.method(), extract just 'method'
                property_node = self._find_child_by_type(callee_node, 'property_identifier')
                if property_node:
                    return self._get_node_text(property_node)
        return None
    
    def _is_builtin_function(self, name: str) -> bool:
        """Check if function name is a JavaScript built-in."""
        builtins = {
            'console', 'setTimeout', 'setInterval', 'parseInt', 'parseFloat',
            'JSON', 'Math', 'Date', 'Array', 'Object', 'String', 'Number',
            'log', 'error', 'warn', 'push', 'pop', 'slice', 'trim',
        }
        return name in builtins
    
    # Helper methods
    def _find_child_by_type(self, node, node_type: str):
        """Find first child node of specified type."""
        for child in node.children:
            if child.type == node_type:
                return child
        return None
    
    def _get_node_text(self, node) -> str:
        """Get the text content of a node."""
        start_byte = node.start_byte
        end_byte = node.end_byte
        return self.content.encode('utf8')[start_byte:end_byte].decode('utf8')
    
    def _find_containing_class_name(self, method_node) -> Optional[str]:
        """Find the name of the class containing a method."""
        current = method_node.parent
        while current:
            if current.type == 'class_declaration':
                name_node = self._find_child_by_type(current, 'identifier')
                if name_node:
                    return self._get_node_text(name_node)
            current = current.parent
        return None


class TreeSitterTSAnalyzer(TreeSitterJSAnalyzer):
    """TypeScript analyzer using tree-sitter."""
    
    def __init__(self, file_path: str, content: str, limits: Optional[AnalysisLimits] = None):
        self.file_path = Path(file_path)
        self.content = content
        self.functions: List[Function] = []
        self.call_relationships: List[CallRelationship] = []
        self.limits = limits or AnalysisLimits()
        
        # Initialize tree-sitter for TypeScript
        self.ts_language = tree_sitter.Language(tree_sitter_typescript.language_typescript())
        self.parser = tree_sitter.Parser(self.ts_language)
        
        logger.info(f"TreeSitterTSAnalyzer initialized for {file_path} with limits: {self.limits.max_nodes} nodes, {self.limits.max_time}s")


# Integration functions
def analyze_javascript_file_treesitter(
    file_path: str, content: str
) -> tuple[List[Function], List[CallRelationship]]:
    """Analyze a JavaScript file using tree-sitter."""
    try:
        logger.info(f"Tree-sitter JS analysis for {file_path}")
        limits = AnalysisLimits(max_nodes=800, max_time=10.0)
        analyzer = TreeSitterJSAnalyzer(file_path, content, limits)
        analyzer.analyze()
        logger.info(f"Found {len(analyzer.functions)} functions, {len(analyzer.call_relationships)} calls, {limits.nodes_processed} nodes processed")
        return analyzer.functions, analyzer.call_relationships
    except Exception as e:
        logger.error(f"Error in tree-sitter JS analysis for {file_path}: {e}", exc_info=True)
        return [], []


def analyze_typescript_file_treesitter(
    file_path: str, content: str
) -> tuple[List[Function], List[CallRelationship]]:
    """Analyze a TypeScript file using tree-sitter."""
    try:
        logger.info(f"Tree-sitter TS analysis for {file_path}")
        limits = AnalysisLimits(max_nodes=800, max_time=10.0)
        analyzer = TreeSitterTSAnalyzer(file_path, content, limits)
        analyzer.analyze()
        logger.info(f"Found {len(analyzer.functions)} functions, {len(analyzer.call_relationships)} calls, {limits.nodes_processed} nodes processed")
        return analyzer.functions, analyzer.call_relationships
    except Exception as e:
        logger.error(f"Error in tree-sitter TS analysis for {file_path}: {e}", exc_info=True)
        return [], []
