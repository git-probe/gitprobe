"""
GitProbe Analysis Orchestrator
Coordinates between repository analysis and call graph analysis services.
"""

from typing import Dict

from .repo_analyzer import RepoAnalyzer
from .core_analyzer import CallGraphAnalyzer
from .utils import clone_repository, cleanup_repository, parse_github_url


def analyze_repository(
    github_url: str, include_patterns=None, exclude_patterns=None
) -> tuple[Dict, str]:
    """
    Complete repository analysis workflow.

    Args:
        github_url: GitHub repository URL
        include_patterns: File patterns to include
        exclude_patterns: File patterns to exclude

    Returns:
        tuple: (analysis_results, temp_dir_path) - caller must cleanup temp_dir
    """
    # Step 1: Clone repository using utils
    temp_dir = clone_repository(github_url)

    # Step 2: Get repository info
    repo_info = parse_github_url(github_url)

    # Step 3: Analyze repository structure
    repo_analyzer = RepoAnalyzer(include_patterns, exclude_patterns)
    structure_result = repo_analyzer.analyze_repository_structure(temp_dir)

    # Step 4: Extract code files and analyze call graph
    call_graph_analyzer = CallGraphAnalyzer()
    code_files = call_graph_analyzer.extract_code_files(structure_result["file_tree"])
    call_graph_result = call_graph_analyzer.analyze_code_files(code_files, temp_dir)

    # Step 5: Combine results
    analysis_results = {
        "repository": repo_info,
        "file_tree": structure_result["file_tree"],
        "file_summary": structure_result["summary"],
        "call_graph": call_graph_result["call_graph"],
        "functions": call_graph_result["functions"],
        "relationships": call_graph_result["relationships"],
        "visualization": call_graph_result["visualization"],
    }

    return analysis_results, temp_dir


def analyze_repository_structure_only(
    github_url: str, include_patterns=None, exclude_patterns=None
) -> tuple[Dict, str]:
    """Analyze only repository structure without call graph analysis."""
    temp_dir = clone_repository(github_url)
    repo_info = parse_github_url(github_url)

    repo_analyzer = RepoAnalyzer(include_patterns, exclude_patterns)
    structure_result = repo_analyzer.analyze_repository_structure(temp_dir)

    results = {
        "repository": repo_info,
        "file_tree": structure_result["file_tree"],
        "file_summary": structure_result["summary"],
    }

    return results, temp_dir
