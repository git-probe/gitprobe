"""
Call Graph Analyzer

Central orchestrator for multi-language call graph analysis.
Coordinates language-specific analyzers to build comprehensive call graphs
across different programming languages in a repository.
"""

from pathlib import Path
from typing import Dict, List
import logging
from models.core import Function, CallRelationship
from utils.patterns import CODE_EXTENSIONS

logger = logging.getLogger(__name__)


class CallGraphAnalyzer:
    """
    Multi-language call graph analyzer.

    This analyzer orchestrates language-specific AST analyzers to build
    comprehensive call graphs across different programming languages.

    Supported languages:
    - Python (fully supported with AST parsing)
    - JavaScript (tree-sitter AST parsing - high accuracy, supports exports/imports)
    - TypeScript (tree-sitter AST parsing - high accuracy, supports exports/imports)
    - C (fully supported with AST parsing)
    - C++ (fully supported with AST parsing)
    - Go (fully supported with tree-sitter AST parsing)
    - Rust (fully supported with tree-sitter AST parsing)

    Key improvements:
    - JavaScript/TypeScript now use tree-sitter for 99%+ accuracy
    - Properly handles export/import statements, arrow functions, class methods
    - Automatically filters out constructors and other non-useful functions
    - Better call relationship detection
    """

    def __init__(self):
        """Initialize the call graph analyzer."""
        self.functions: Dict[str, Function] = {}
        self.call_relationships: List[CallRelationship] = []
        logger.info("CallGraphAnalyzer initialized.")

    def analyze_code_files(self, code_files: List[Dict], base_dir: str) -> Dict:
        """
        Analyze a list of code files from multiple languages.

        Args:
            code_files: List of file info dicts with path, language, etc.
            base_dir: Base directory path where files are located

        Returns:
            Dict with functions, relationships, and visualization data
        """
        logger.info(f"Starting analysis of {len(code_files)} code files.")
        # Reset state for new analysis
        self.functions = {}
        self.call_relationships = []

        # Analyze each code file based on its language
        for file_info in code_files:
            logger.debug(f"Analyzing file: {file_info['path']}")
            self._analyze_code_file(base_dir, file_info)

        logger.info("Initial analysis complete. Resolving call relationships.")
        # Resolve cross-language relationships
        self._resolve_call_relationships()

        # After collecting all relationships, deduplicate:
        logger.info("Deduplicating call relationships.")
        self._deduplicate_relationships()
        logger.info(
            f"Deduplication complete. {len(self.call_relationships)} unique relationships found."
        )

        # Clean up disconnected functions for better visualization
        logger.info("Cleaning up disconnected functions.")
        self._cleanup_disconnected_functions()

        # Generate visualization data
        logger.info("Generating visualization data.")
        viz_data = self._generate_visualization_data()

        return {
            "call_graph": {
                "total_functions": len(self.functions),
                "total_calls": len(self.call_relationships),
                "languages_found": list(set(f.get("language") for f in code_files)),
                "files_analyzed": len(code_files),
            },
            "functions": [func.dict() for func in self.functions.values()],
            "relationships": [rel.dict() for rel in self.call_relationships],
            "visualization": viz_data,
        }

    def extract_code_files(self, file_tree: Dict) -> List[Dict]:
        """
        Extract code files from file tree structure.

        Filters files based on supported extensions and excludes test/config files.

        Args:
            file_tree: Nested dictionary representing file structure

        Returns:
            List of code file information dictionaries
        """
        code_files = []

        def traverse(tree):
            if tree["type"] == "file":
                ext = tree.get("extension", "").lower()
                if ext in CODE_EXTENSIONS:
                    name = tree["name"].lower()
                    # Skip test, spec, config, and setup files
                    if not any(
                        skip in name for skip in ["test", "spec", "config", "setup"]
                    ):
                        code_files.append(
                            {
                                "path": tree["path"],
                                "name": tree["name"],
                                "extension": ext,
                                "language": CODE_EXTENSIONS[ext],
                            }
                        )
            elif tree["type"] == "directory" and tree.get("children"):
                for child in tree["children"]:
                    traverse(child)

        traverse(file_tree)
        return code_files

    def _analyze_code_file(self, repo_dir: str, file_info: Dict):
        """
        Analyze a single code file based on its language.

        Routes to appropriate language-specific analyzer.

        Args:
            repo_dir: Repository directory path
            file_info: File information dictionary
        """
        file_path = Path(repo_dir) / file_info["path"]

        logger.debug(f"Reading content of {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Route to appropriate language analyzer
            language = file_info["language"]
            logger.info(f"Analyzing {language} file: {file_path}")
            if language == "python":
                self._analyze_python_file(file_path, content)
            elif language == "javascript":
                self._analyze_javascript_file(file_path, content)
            elif language == "typescript":
                self._analyze_typescript_file(file_path, content)
            elif language == "c":
                self._analyze_c_file(file_path, content)
            elif language == "cpp":
                self._analyze_cpp_file(file_path, content)
            elif language == "go":
                self._analyze_go_file(file_path, content)
            elif language == "rust":
                self._analyze_rust_file(file_path, content)
            else:
                logger.warning(
                    f"Unsupported language for call graph analysis: {language} for file {file_path}"
                )

        except Exception as e:
            logger.error(f"⚠️ Error analyzing {file_path}: {str(e)}")

    def _analyze_python_file(self, file_path: str, content: str):
        """
        Analyze Python file using Python AST analyzer.

        Args:
            file_path: Relative path to the Python file
            content: File content string
        """
        from .python_analyzer import analyze_python_file

        try:
            functions, relationships = analyze_python_file(file_path, content)
            logger.info(
                f"Found {len(functions)} functions and {len(relationships)} relationships in {file_path}"
            )

            # Store functions with unique identifiers
            for func in functions:
                func_id = f"{file_path}:{func.name}"
                self.functions[func_id] = func

            # Store call relationships
            self.call_relationships.extend(relationships)
        except Exception as e:
            logger.error(
                f"Failed to analyze Python file {file_path}: {e}", exc_info=True
            )

    def _analyze_javascript_file(self, file_path: str, content: str):
        """
        Analyze JavaScript file using tree-sitter based AST analyzer.

        Args:
            file_path: Relative path to the JavaScript file
            content: File content string
        """
        try:
            logger.info(f"Starting tree-sitter JavaScript analysis for {file_path}")
            # Use the new tree-sitter based analyzer
            from .js_analyzer_new import analyze_javascript_file_treesitter

            logger.info(
                f"About to call analyze_javascript_file_treesitter with args: file_path='{file_path}', content_length={len(content)}"
            )
            functions, relationships = analyze_javascript_file_treesitter(
                file_path, content
            )
            logger.info(
                f"Tree-sitter JavaScript analysis completed for {file_path}: {len(functions)} functions, {len(relationships)} relationships"
            )

            # Store functions with unique identifiers
            for func in functions:
                func_id = f"{file_path}:{func.name}"
                self.functions[func_id] = func

            # Store call relationships
            self.call_relationships.extend(relationships)
        except Exception as e:
            logger.error(
                f"Failed to analyze JavaScript file {file_path}: {e}", exc_info=True
            )

    def _analyze_typescript_file(self, file_path: str, content: str):
        """
        Analyze TypeScript file using tree-sitter based AST analyzer.

        Args:
            file_path: Relative path to the TypeScript file
            content: File content string
        """
        try:
            logger.info(f"Starting tree-sitter TypeScript analysis for {file_path}")
            # Use the new tree-sitter based analyzer
            from .js_analyzer_new import analyze_typescript_file_treesitter

            logger.info(
                f"About to call analyze_typescript_file_treesitter with args: file_path='{file_path}', content_length={len(content)}"
            )
            functions, relationships = analyze_typescript_file_treesitter(
                file_path, content
            )
            logger.info(
                f"Tree-sitter TypeScript analysis completed for {file_path}: {len(functions)} functions, {len(relationships)} relationships"
            )

            # Store functions with unique identifiers
            for func in functions:
                func_id = f"{file_path}:{func.name}"
                self.functions[func_id] = func

            # Store call relationships
            self.call_relationships.extend(relationships)
        except Exception as e:
            logger.error(
                f"Failed to analyze TypeScript file {file_path}: {e}", exc_info=True
            )

    def _analyze_c_file(self, file_path: str, content: str):
        """
        Analyze C file using C AST analyzer.

        Args:
            file_path: Relative path to the C file
            content: File content string
        """
        from .c_analyzer import analyze_c_file

        functions, relationships = analyze_c_file(file_path, content)

        # Store functions with unique identifiers
        for func in functions:
            func_id = f"{file_path}:{func.name}"
            self.functions[func_id] = func

        # Store call relationships
        self.call_relationships.extend(relationships)

    def _analyze_cpp_file(self, file_path: str, content: str):
        """
        Analyze C++ file using C++ AST analyzer.

        Args:
            file_path: Relative path to the C++ file
            content: File content string
        """
        from .c_analyzer import analyze_cpp_file

        functions, relationships = analyze_cpp_file(file_path, content)

        # Store functions with unique identifiers
        for func in functions:
            func_id = f"{file_path}:{func.name}"
            self.functions[func_id] = func

        # Store call relationships
        self.call_relationships.extend(relationships)

    def _analyze_go_file(self, file_path: str, content: str):
        """
        Analyze Go file using Go AST analyzer.

        Args:
            file_path: Relative path to the Go file
            content: File content string
        """
        from .go_analyzer import analyze_go_file_treesitter

        try:
            functions, relationships = analyze_go_file_treesitter(file_path, content)
            logger.info(
                f"Found {len(functions)} functions and {len(relationships)} relationships in {file_path}"
            )

            # Store functions with unique identifiers
            for func in functions:
                func_id = f"{file_path}:{func.name}"
                self.functions[func_id] = func

            # Store call relationships
            self.call_relationships.extend(relationships)
        except Exception as e:
            logger.error(f"Failed to analyze Go file {file_path}: {e}", exc_info=True)

    def _analyze_rust_file(self, file_path: str, content: str):
        """
        Analyze Rust file using Rust AST analyzer.

        Args:
            file_path: Relative path to the Rust file
            content: File content string
        """
        from .rust_analyzer import analyze_rust_file_treesitter

        try:
            functions, relationships = analyze_rust_file_treesitter(file_path, content)
            logger.info(
                f"Found {len(functions)} functions and {len(relationships)} relationships in {file_path}"
            )

            # Store functions with unique identifiers
            for func in functions:
                func_id = f"{file_path}:{func.name}"
                self.functions[func_id] = func

            # Store call relationships
            self.call_relationships.extend(relationships)
        except Exception as e:
            logger.error(f"Failed to analyze Rust file {file_path}: {e}", exc_info=True)

    def _resolve_call_relationships(self):
        """
        Resolve function call relationships across all languages.

        Attempts to match function calls to actual function definitions,
        handling cross-language calls where possible.
        """
        logger.info("Building function lookup table for resolving relationships.")
        # Build lookup table of all functions
        func_lookup = {}
        for func_id, func_info in self.functions.items():
            func_lookup[func_info.name] = func_id

        # Resolve relationships
        resolved_count = 0
        for relationship in self.call_relationships:
            callee_name = relationship.callee

            # Direct name match
            if callee_name in func_lookup:
                relationship.callee = func_lookup[callee_name]
                relationship.is_resolved = True
                resolved_count += 1
            # Method call resolution (obj.method -> method)
            elif "." in callee_name:
                method_name = callee_name.split(".")[-1]
                if method_name in func_lookup:
                    relationship.callee = func_lookup[method_name]
                    relationship.is_resolved = True

        logger.info(
            f"Resolved {resolved_count}/{len(self.call_relationships)} call relationships."
        )

    def _deduplicate_relationships(self):
        """
        Deduplicate call relationships based on caller-callee pairs.

        Removes duplicate relationships while preserving the first occurrence.
        This helps eliminate noise from multiple calls to the same function.
        """
        seen = set()
        unique_relationships = []

        for rel in self.call_relationships:
            # Create unique key (ignore line numbers)
            key = (rel.caller, rel.callee)
            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)

        logger.debug(
            f"Removed {len(self.call_relationships) - len(unique_relationships)} duplicate relationships."
        )
        self.call_relationships = unique_relationships

    def _cleanup_disconnected_functions(self):
        """
        AGGRESSIVELY remove functions that aren't connected to the call graph.
        
        Only keeps functions that are actually part of resolved call relationships.
        """
        original_count = len(self.functions)
        
        # Build set of connected function IDs (ONLY resolved relationships)
        connected_function_ids = set()
        
        for rel in self.call_relationships:
            if rel.is_resolved:
                connected_function_ids.add(rel.caller)
                connected_function_ids.add(rel.callee)
        
        logger.info(f"Found {len(connected_function_ids)} connected function IDs from {len(self.call_relationships)} relationships")
        
        # VERY AGGRESSIVE: Only keep functions that are actually connected + main
        filtered_functions = {}
        disconnected_count = 0
        
        for func_id, func in self.functions.items():
            if func_id in connected_function_ids:
                # Part of resolved call graph - keep
                filtered_functions[func_id] = func
            elif func.name == 'main':
                # Always keep main entry point
                filtered_functions[func_id] = func
            else:
                # Filter out EVERYTHING else
                disconnected_count += 1
                logger.debug(f"Filtering disconnected function: {func.name} (ID: {func_id})")
        
        self.functions = filtered_functions
        
        logger.info(f"AGGRESSIVE cleanup: {original_count} -> {len(self.functions)} "
                   f"(removed {disconnected_count} disconnected functions, "
                   f"kept {len([f for f in filtered_functions.keys() if f in connected_function_ids])} connected + "
                   f"{len([f for f in filtered_functions.values() if f.name == 'main'])} main functions)")

    def _is_important_function(self, func) -> bool:
        """Check if a function is important enough to keep even if disconnected."""
        # Always keep main entry points
        if func.name == 'main':
            return True
            
        # Keep likely public APIs (heuristic based on naming/file patterns)
        if hasattr(func, 'code_snippet') and func.code_snippet:
            if 'pub ' in func.code_snippet:  # Rust public functions
                return True
            if func.code_snippet.startswith('export '):  # JS/TS exports
                return True
                
        # Keep common constructor patterns
        if func.name in ['new', 'create', 'build', 'init', '__init__']:
            return True
            
        # Keep functions with many parameters (likely important APIs)
        if hasattr(func, 'parameters') and len(func.parameters) >= 3:
            return True
            
        return False

    def _generate_visualization_data(self) -> Dict:
        """
        Generate visualization data for graph rendering.

        Creates Cytoscape.js compatible graph data with nodes and edges.

        Returns:
            Dict: Visualization data with cytoscape elements and summary
        """
        logger.info("Generating Cytoscape-compatible visualization data.")
        cytoscape_elements = []

        # Add function nodes
        logger.debug(f"Adding {len(self.functions)} function nodes.")
        for func_id, func_info in self.functions.items():
            # Determine node styling based on function type and language
            node_classes = []
            if func_info.is_method:
                node_classes.append("node-method")
            else:
                node_classes.append("node-function")

            # Add language-specific styling
            file_ext = Path(func_info.file_path).suffix.lower()
            if file_ext == ".py":
                node_classes.append("lang-python")
            elif file_ext == ".js":
                node_classes.append("lang-javascript")
            elif file_ext == ".ts":
                node_classes.append("lang-typescript")
            elif file_ext in [".c", ".h"]:
                node_classes.append("lang-c")
            elif file_ext in [".cpp", ".cc", ".cxx", ".hpp", ".hxx"]:
                node_classes.append("lang-cpp")

            cytoscape_elements.append(
                {
                    "data": {
                        "id": func_id,
                        "label": func_info.name,
                        "file": func_info.file_path,
                        "type": "method" if func_info.is_method else "function",
                        "language": CODE_EXTENSIONS.get(file_ext, "unknown"),
                    },
                    "classes": " ".join(node_classes),
                }
            )

        # Add call relationship edges
        resolved_rels = [r for r in self.call_relationships if r.is_resolved]
        logger.debug(f"Adding {len(resolved_rels)} relationship edges.")
        for rel in resolved_rels:
            cytoscape_elements.append(
                {
                    "data": {
                        "id": f"{rel.caller}->{rel.callee}",
                        "source": rel.caller,
                        "target": rel.callee,
                        "line": rel.call_line,
                    },
                    "classes": "edge-call",
                }
            )

        summary = {
            "total_nodes": len(self.functions),
            "total_edges": len(resolved_rels),
            "unresolved_calls": len(self.call_relationships) - len(resolved_rels),
        }
        logger.info(f"Visualization data generated: {summary}")

        return {
            "cytoscape": {"elements": cytoscape_elements},
            "summary": summary,
        }

    def generate_llm_format(self) -> Dict:
        """Generate clean format optimized for LLM consumption."""
        return {
            "functions": [
                {
                    "name": func.name,
                    "file": Path(func.file_path).name,  # Just filename
                    "purpose": (
                        func.docstring.split("\n")[0] if func.docstring else None
                    ),
                    "parameters": func.parameters,
                    "is_recursive": func.name
                    in [
                        rel.callee
                        for rel in self.call_relationships
                        if rel.caller.endswith(func.name)
                    ],
                }
                for func in self.functions.values()
            ],
            "relationships": {
                func.name: {
                    "calls": [
                        rel.callee.split(":")[-1]
                        for rel in self.call_relationships
                        if rel.caller.endswith(func.name) and rel.is_resolved
                    ],
                    "called_by": [
                        rel.caller.split(":")[-1]
                        for rel in self.call_relationships
                        if rel.callee.endswith(func.name) and rel.is_resolved
                    ],
                }
                for func in self.functions.values()
            },
        }
