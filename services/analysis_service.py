"""
Analysis Service

Centralized service for repository analysis with support for multiple languages.
Handles the orchestration of repository cloning, structure analysis, and multi-language
AST parsing for call graph generation.
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

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
    - Go (fully implemented)
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
            logger.info("Analyzing repository file structure...")
            structure_result = self._analyze_structure(
                temp_dir, include_patterns, exclude_patterns
            )
            logger.info(
                f"Found {structure_result['summary']['total_files']} files to analyze."
            )

            # Step 3: Multi-language call graph analysis
            logger.info("Starting call graph analysis...")
            call_graph_result = self._analyze_call_graph(
                structure_result["file_tree"], temp_dir
            )
            logger.info(
                f"Call graph analysis complete. Found {call_graph_result['call_graph']['total_functions']} functions."
            )

            # Step 3.5: Read README file
            readme_content = self._read_readme_file(temp_dir)

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
                readme_content=readme_content,
            )

            # Step 5: Cleanup
            logger.info(f"Cleaning up temporary repository directory: {temp_dir}")
            self._cleanup_repository(temp_dir)

            logger.info(
                f"Analysis completed: {analysis_result.summary['total_functions']} functions found"
            )
            return analysis_result

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            # Ensure cleanup happens even on failure
            if "temp_dir" in locals() and Path(temp_dir).exists():
                self._cleanup_repository(temp_dir)
            raise RuntimeError(f"Repository analysis failed: {str(e)}")

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
        """Clone repository and return temp dir path."""
        logger.info(f"Cloning {github_url}...")
        temp_dir = clone_repository(github_url)
        logger.info(f"Repository cloned to {temp_dir}")
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
        logger.info(
            f"Initializing RepoAnalyzer with include: {include_patterns}, exclude: {exclude_patterns}"
        )
        repo_analyzer = RepoAnalyzer(include_patterns, exclude_patterns)
        return repo_analyzer.analyze_repository_structure(repo_dir)

    def _read_readme_file(self, repo_dir: str) -> Optional[str]:
        """Find and read the README file from the repository root."""
        possible_readme_names = ["README.md", "README", "readme.md", "README.txt"]
        for name in possible_readme_names:
            readme_path = Path(repo_dir) / name
            if readme_path.exists():
                try:
                    logger.info(f"Found README file at {readme_path}")
                    return readme_path.read_text(encoding="utf-8")
                except Exception as e:
                    logger.warning(f"Could not read README file at {readme_path}: {e}")
                    return None
        logger.info("No README file found in repository root.")
        return None

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
        logger.info("Extracting code files from file tree...")
        code_files = self.call_graph_analyzer.extract_code_files(file_tree)

        # Filter by supported languages (will expand as we add JS/TS support)
        logger.info(
            f"Found {len(code_files)} total code files. Filtering for supported languages."
        )
        supported_files = self._filter_supported_languages(code_files)
        logger.info(f"Analyzing {len(supported_files)} supported files.")

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

        Supports Python, JavaScript, TypeScript, C, C++, and Go.
        """
        supported_languages = {"python", "javascript", "typescript", "c", "cpp", "go"}

        return [
            file_info
            for file_info in code_files
            if file_info.get("language") in supported_languages
        ]

    def _get_supported_languages(self) -> List[str]:
        """Get list of currently supported languages for analysis."""
        return ["python", "javascript", "typescript", "c", "cpp", "go"]

    def _cleanup_repository(self, temp_dir: str):
        """Clean up cloned repository."""
        logger.info(f"Attempting to clean up {temp_dir}")
        cleanup_repository(temp_dir)
        if temp_dir in self._temp_directories:
            self._temp_directories.remove(temp_dir)

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
