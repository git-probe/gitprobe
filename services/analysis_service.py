"""
Analysis Service

Centralized service for repository analysis with support for multiple languages.
Handles the orchestration of repository cloning, structure analysis, and multi-language
AST parsing for call graph generation.
"""

import logging
from typing import Dict, List, Optional, Any

from .repo_analyzer import RepoAnalyzer
from .call_graph_analyzer import CallGraphAnalyzer
from .cloning import clone_repository, cleanup_repository, parse_github_url
from models.analysis import AnalysisResult
from models.core import Repository

logger = logging.getLogger(__name__)


class AnalysisService:
    """
    Centralized analysis service supporting multiple programming languages.

    This service orchestrates the complete analysis workflow:
    1. Repository cloning and validation
    2. File structure analysis with filtering
    3. Multi-language AST parsing and call graph generation
    4. Result consolidation and cleanup

    Supports:
    - Python (fully implemented)
    - JavaScript/TypeScript (fully implemented)
    - C/C++ (fully implemented)
    - Additional languages (extensible)
    """

    def __init__(self):
        """Initialize the analysis service with language-specific analyzers."""
        self.call_graph_analyzer = CallGraphAnalyzer()
        self._temp_directories = []  # Track for cleanup

    def analyze_repository_full(
        self,
        github_url: str,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> AnalysisResult:
        """
        Perform complete repository analysis including call graph generation.

        Args:
            github_url: GitHub repository URL to analyze
            include_patterns: File patterns to include (e.g., ['*.py', '*.js'])
            exclude_patterns: Additional patterns to exclude

        Returns:
            AnalysisResult: Complete analysis with functions, relationships, and visualization

        Raises:
            ValueError: If GitHub URL is invalid
            RuntimeError: If analysis fails
        """
        temp_dir = None
        try:
            logger.info(f"Starting full analysis of {github_url}")

            # Step 1: Clone and validate repository
            temp_dir = self._clone_repository(github_url)
            repo_info = self._parse_repository_info(github_url)

            # Step 2: Analyze file structure
            structure_result = self._analyze_structure(
                temp_dir, include_patterns, exclude_patterns
            )

            # Step 3: Multi-language call graph analysis
            call_graph_result = self._analyze_call_graph(
                structure_result["file_tree"], temp_dir
            )

            # Step 4: Build comprehensive result
            analysis_result = AnalysisResult(
                repository=Repository(
                    url=repo_info["url"],
                    name=repo_info["name"],
                    clone_path=temp_dir,
                    analysis_id=f"{repo_info['owner']}-{repo_info['name']}",
                ),
                functions=call_graph_result["functions"],
                relationships=call_graph_result["relationships"],
                file_tree=structure_result["file_tree"],
                summary={
                    **structure_result["summary"],
                    **call_graph_result["call_graph"],
                    "analysis_type": "full",
                    "languages_analyzed": call_graph_result["call_graph"][
                        "languages_found"
                    ],
                },
                visualization=call_graph_result["visualization"],
            )

            # Step 5: Cleanup
            self._cleanup_repository(temp_dir)

            logger.info(
                f"Analysis completed: {analysis_result.summary['total_functions']} functions found"
            )
            return analysis_result

        except Exception as e:
            if temp_dir:
                self._cleanup_repository(temp_dir)
            logger.error(f"Analysis failed for {github_url}: {str(e)}")
            raise RuntimeError(f"Repository analysis failed: {str(e)}") from e

    def analyze_repository_structure_only(
        self,
        github_url: str,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Perform lightweight structure-only analysis without call graph generation.

        Args:
            github_url: GitHub repository URL to analyze
            include_patterns: File patterns to include
            exclude_patterns: Additional patterns to exclude

        Returns:
            Dict: Repository structure with file tree and summary statistics
        """
        temp_dir = None
        try:
            logger.info(f"Starting structure analysis of {github_url}")

            # Step 1: Clone and validate repository
            temp_dir = self._clone_repository(github_url)
            repo_info = self._parse_repository_info(github_url)

            # Step 2: Analyze file structure only
            structure_result = self._analyze_structure(
                temp_dir, include_patterns, exclude_patterns
            )

            # Step 3: Build lightweight result
            result = {
                "repository": repo_info,
                "file_tree": structure_result["file_tree"],
                "file_summary": {
                    **structure_result["summary"],
                    "analysis_type": "structure_only",
                },
            }

            # Step 4: Cleanup
            self._cleanup_repository(temp_dir)

            logger.info(
                f"Structure analysis completed: {result['file_summary']['total_files']} files found"
            )
            return result

        except Exception as e:
            if temp_dir:
                self._cleanup_repository(temp_dir)
            logger.error(f"Structure analysis failed for {github_url}: {str(e)}")
            raise RuntimeError(f"Structure analysis failed: {str(e)}") from e

    def _clone_repository(self, github_url: str) -> str:
        """Clone repository and track for cleanup."""
        temp_dir = clone_repository(github_url)
        self._temp_directories.append(temp_dir)
        return temp_dir

    def _parse_repository_info(self, github_url: str) -> Dict[str, str]:
        """Parse GitHub URL and extract repository metadata."""
        return parse_github_url(github_url)

    def _analyze_structure(
        self,
        repo_dir: str,
        include_patterns: Optional[List[str]],
        exclude_patterns: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Analyze repository file structure with filtering."""
        repo_analyzer = RepoAnalyzer(include_patterns, exclude_patterns)
        return repo_analyzer.analyze_repository_structure(repo_dir)

    def _analyze_call_graph(
        self, file_tree: Dict[str, Any], repo_dir: str
    ) -> Dict[str, Any]:
        """
        Perform multi-language call graph analysis.

        This method will be expanded to handle:
        - Python AST analysis (current)
        - JavaScript/TypeScript AST analysis (planned)
        - Additional language support (future)
        """
        # Extract code files for all supported languages
        code_files = self.call_graph_analyzer.extract_code_files(file_tree)

        # Filter by supported languages (will expand as we add JS/TS support)
        supported_files = self._filter_supported_languages(code_files)

        # Perform multi-language analysis
        result = self.call_graph_analyzer.analyze_code_files(supported_files, repo_dir)

        # Add language-specific metadata
        result["call_graph"]["supported_languages"] = self._get_supported_languages()
        result["call_graph"]["unsupported_files"] = len(code_files) - len(
            supported_files
        )

        return result

    def _filter_supported_languages(self, code_files: List[Dict]) -> List[Dict]:
        """
        Filter code files to only include supported languages.

        Supports Python, JavaScript, TypeScript, C, and C++.
        """
        supported_languages = {"python", "javascript", "typescript", "c", "cpp"}

        return [
            file_info
            for file_info in code_files
            if file_info.get("language") in supported_languages
        ]

    def _get_supported_languages(self) -> List[str]:
        """Get list of currently supported languages for analysis."""
        return ["python", "javascript", "typescript", "c", "cpp"]

    def _cleanup_repository(self, temp_dir: str):
        """Clean up temporary repository directory."""
        try:
            cleanup_repository(temp_dir)
            if temp_dir in self._temp_directories:
                self._temp_directories.remove(temp_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup {temp_dir}: {str(e)}")

    def cleanup_all(self):
        """Clean up all tracked temporary directories."""
        for temp_dir in self._temp_directories[:]:
            self._cleanup_repository(temp_dir)

    def __del__(self):
        """Ensure cleanup on service destruction."""
        self.cleanup_all()


# Convenience functions for backward compatibility
def analyze_repository(
    github_url: str, include_patterns=None, exclude_patterns=None
) -> tuple[AnalysisResult, None]:
    """
    Backward compatibility function.

    Returns:
        tuple: (AnalysisResult, None) - None instead of temp_dir since cleanup is handled internally
    """
    service = AnalysisService()
    result = service.analyze_repository_full(
        github_url, include_patterns, exclude_patterns
    )
    return result, None


def analyze_repository_structure_only(
    github_url: str, include_patterns=None, exclude_patterns=None
) -> tuple[Dict, None]:
    """
    Backward compatibility function.

    Returns:
        tuple: (structure_result, None) - None instead of temp_dir since cleanup is handled internally
    """
    service = AnalysisService()
    result = service.analyze_repository_structure_only(
        github_url, include_patterns, exclude_patterns
    )
    return result, None
