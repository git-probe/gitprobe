#!/usr/bin/env python3
"""
GitProbe Repository Analyzer
Extracts file tree structure from GitHub repositories with customizable ignore patterns.
"""

import os
import re
import shutil
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse

import git
import pathspec
import requests
import functools
import time


class RepoAnalyzer:
    """Analyzes GitHub repositories and extracts file tree structures."""

    # Default ignore patterns (similar to common .gitignore patterns)
    DEFAULT_IGNORE_PATTERNS = [
        # Dependencies
        "node_modules/",
        "venv/",
        "env/",
        ".env/",
        "virtualenv/",
        "__pycache__/",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".Python",
        "pip-log.txt",
        "pip-delete-this-directory.txt",
        # Build directories
        "build/",
        "dist/",
        "target/",
        "out/",
        ".cargo/",
        "cmake-build-*/",
        # IDE and editor files
        ".vscode/",
        ".idea/",
        "*.swp",
        "*.swo",
        "*~",
        ".DS_Store",
        "Thumbs.db",
        # Version control
        ".git/",
        ".svn/",
        ".hg/",
        # Logs and temporary files
        "*.log",
        ".tmp/",
        "tmp/",
        ".cache/",
        # Package manager files
        "bower_components/",
        "jspm_packages/",
        ".npm/",
        ".yarn/",
        "vendor/",
        "Pods/",
        # Compiled binaries
        "*.so",
        "*.dylib",
        "*.dll",
        "*.exe",
        "*.o",
        "*.a",
        # Documentation builds
        "docs/_build/",
        "site/",
        "_site/",
    ]

    def __init__(
        self,
        exclude_patterns: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
        max_file_size_kb: Optional[int] = None,
        min_file_size_kb: Optional[int] = None,
    ):
        """
        Initialize the repository analyzer.

        Args:
            exclude_patterns: Patterns to exclude (added to defaults)
            include_patterns: Patterns to include (if specified, only these are included)
            max_file_size_kb: Maximum file size in KB to include
            min_file_size_kb: Minimum file size in KB to include
        """
        # Set up exclude patterns
        self.exclude_patterns = self.DEFAULT_IGNORE_PATTERNS.copy()
        if exclude_patterns:
            self.exclude_patterns.extend(exclude_patterns)

        # Set up include patterns
        self.include_patterns = include_patterns or []

        # File size filters
        self.max_file_size_kb = max_file_size_kb
        self.min_file_size_kb = min_file_size_kb

        # Create pathspec objects for efficient pattern matching
        self.exclude_pathspec = pathspec.PathSpec.from_lines(
            "gitwildmatch", self.exclude_patterns
        )
        self.include_pathspec = (
            pathspec.PathSpec.from_lines("gitwildmatch", self.include_patterns)
            if self.include_patterns
            else None
        )

    def _parse_github_url(self, url: str) -> Dict[str, str]:
        """
        Validate and normalize a GitHub repository URL.

        Supported URL patterns:
        - https://github.com/<owner>/<repo>[.git]
        - git@github.com:<owner>/<repo>.git
        - Branch/commit selectors via:
          * https://github.com/<owner>/<repo>/tree/<branch>
          * https://github.com/<owner>/<repo>/commit/<sha>
          * https://github.com/<owner>/<repo>?ref=<branch>
          * https://github.com/<owner>/<repo>#<branch>

        Returns:
            {
                "owner": str,
                "repo": str,
                "ref": Optional[str],
                "ref_type": Optional[str],
                "clone_url": str,
                "html_url": str,
            }
        """
        original = url.strip()

        ref: Optional[str] = None
        ref_type: Optional[str] = None

        # Extract branch/commit via query string or fragment
        if "?ref=" in original:
            original, ref = original.split("?ref=", 1)
            ref_type = "branch"
        elif "#" in original:
            original, ref = original.split("#", 1)
            ref_type = "branch"

        # /tree/<branch>
        tree_match = re.search(r"/tree/([^/]+)", original)
        if tree_match:
            ref = tree_match.group(1)
            ref_type = "branch"
            original = original[: tree_match.start()]

        # /commit/<sha>
        commit_match = re.search(r"/commit/([0-9a-fA-F]{5,40})", original)
        if commit_match:
            ref = commit_match.group(1)
            ref_type = "commit"
            original = original[: commit_match.start()]

        patterns = [
            r"^https://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$",
            r"^git@github\.com:([^/]+)/([^/]+?)\.git$",
        ]

        owner = repo = None
        for pattern in patterns:
            m = re.match(pattern, original)
            if m:
                owner, repo = m.groups()
                break

        if not owner or not repo:
            raise ValueError(f"Invalid GitHub URL format: {url}")

        if repo.endswith(".git"):
            repo = repo[:-4]

        clone_url = f"https://github.com/{owner}/{repo}.git"
        html_url = f"https://github.com/{owner}/{repo}"

        return {
            "owner": owner,
            "repo": repo,
            "ref": ref,
            "ref_type": ref_type,
            "clone_url": clone_url,
            "html_url": html_url,
        }

    def _get_oauth_token(self) -> Optional[str]:
        """Return GitHub OAuth token from the environment, if set."""
        return os.getenv("GITHUB_TOKEN")

    @functools.lru_cache(maxsize=256)
    def _repository_exists(self, owner: str, repo: str) -> bool:
        """
        Check if a repository exists (and is accessible with provided credentials).
        This call is cached to avoid hitting GitHub rate limits repeatedly.
        """
        api_url = f"https://api.github.com/repos/{owner}/{repo}"
        headers: Dict[str, str] = {"Accept": "application/vnd.github.v3+json"}
        token = self._get_oauth_token()
        if token:
            headers["Authorization"] = f"token {token}"

        while True:
            resp = requests.get(api_url, headers=headers)
            # Handle basic rate limiting
            if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
                reset = int(resp.headers.get("X-RateLimit-Reset", 0))
                wait_seconds = max(0, reset - int(time.time()) + 1)
                print(f"GitHub rate limit exceeded; sleeping for {wait_seconds}s…")
                time.sleep(wait_seconds)
                continue

            return resp.status_code == 200

    def _clone_repository(self, url: str, target_dir: str, ref: Optional[str] = None) -> None:
        """Clone a GitHub repository, supporting private access and ref checkout."""
        token = self._get_oauth_token()
        auth_url = url
        if token and url.startswith("https://github.com"):
            # Inject token to enable private repo cloning
            auth_url = url.replace("https://", f"https://{token}@")

        try:
            print(f"Cloning repository: {url}" + (f" (ref: {ref})" if ref else ""))
            repo = git.Repo.clone_from(auth_url, target_dir, depth=1)
            if ref:
                repo.git.checkout(ref)
            print(f"Repository cloned to: {target_dir}")
        except git.exc.GitCommandError as e:
            raise RuntimeError(f"Failed to clone repository: {str(e)}")

    def _should_include_file(self, path: str, file_size_bytes: int) -> bool:
        """
        Check if a file should be included based on all filters.

        Args:
            path: File or directory path to check
            file_size_bytes: File size in bytes

        Returns:
            True if file should be included, False otherwise
        """
        # Check exclude patterns first
        if self.exclude_pathspec.match_file(path):
            return False

        # If include patterns are specified, file must match one of them
        if self.include_pathspec and not self.include_pathspec.match_file(path):
            return False

        # Check file size filters
        file_size_kb = file_size_bytes / 1024

        if self.max_file_size_kb and file_size_kb > self.max_file_size_kb:
            return False

        if self.min_file_size_kb and file_size_kb < self.min_file_size_kb:
            return False

        return True

    def _get_file_info(self, file_path: Path) -> Dict:
        """
        Get information about a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with file information
        """
        try:
            stat = file_path.stat()
            is_binary = self._is_binary_file(file_path)
            content_length = 0

            # Estimate token count for text files
            if not is_binary:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        content_length = len(content)
                except:
                    content_length = 0

            return {
                "name": file_path.name,
                "size": stat.st_size,
                "size_kb": round(stat.st_size / 1024, 2),
                "extension": file_path.suffix,
                "is_binary": is_binary,
                "content_length": content_length,
                "estimated_tokens": (
                    self._estimate_tokens(content_length) if not is_binary else 0
                ),
            }
        except (OSError, IOError):
            return {
                "name": file_path.name,
                "size": 0,
                "size_kb": 0,
                "extension": file_path.suffix,
                "is_binary": True,
                "content_length": 0,
                "estimated_tokens": 0,
            }

    def _estimate_tokens(self, content_length: int) -> int:
        """
        Estimate token count from content length.

        Args:
            content_length: Number of characters in content

        Returns:
            Estimated number of tokens
        """
        # Rough estimation: ~4 characters per token for code
        return max(1, content_length // 4)

    def _is_binary_file(self, file_path: Path) -> bool:
        """
        Check if a file is binary.

        Args:
            file_path: Path to the file

        Returns:
            True if file is binary, False otherwise
        """
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(1024)
                return b"\0" in chunk
        except (OSError, IOError):
            return True

    def _extract_file_tree(self, repo_dir: str) -> Dict:
        """
        Extract file tree structure from repository directory.

        Args:
            repo_dir: Path to the cloned repository

        Returns:
            Dictionary representing the file tree structure
        """
        repo_path = Path(repo_dir)

        def build_tree(current_path: Path, relative_path: str = "") -> Dict:
            """Recursively build file tree structure."""
            tree = {
                "name": current_path.name,
                "type": "directory" if current_path.is_dir() else "file",
                "path": relative_path,
                "children": [] if current_path.is_dir() else None,
            }

            if current_path.is_file():
                tree.update(self._get_file_info(current_path))
                return tree

            if current_path.is_dir():
                try:
                    for item in sorted(current_path.iterdir()):
                        item_relative_path = str(item.relative_to(repo_path))

                        # For files, check all filters including size
                        if item.is_file():
                            try:
                                file_size = item.stat().st_size
                                if not self._should_include_file(
                                    item_relative_path, file_size
                                ):
                                    continue
                            except (OSError, IOError):
                                continue
                        else:
                            # For directories, only check exclude patterns
                            if self.exclude_pathspec.match_file(item_relative_path):
                                continue

                        child_tree = build_tree(item, item_relative_path)
                        tree["children"].append(child_tree)

                except (PermissionError, OSError):
                    # Handle permission errors gracefully
                    pass

            return tree

        return build_tree(repo_path)

    def analyze_repository(self, github_url: str) -> Dict:
        """
        Analyze a GitHub repository and return its file tree structure.

        Args:
            github_url: GitHub repository URL

        Returns:
            Dictionary containing repository metadata and file tree
        """
        # Parse, validate, and normalize the URL
        repo_info = self._parse_github_url(github_url)

        # Verify repository existence/access
        if not self._repository_exists(repo_info["owner"], repo_info["repo"]):
            raise ValueError(
                f"Repository {repo_info['owner']}/{repo_info['repo']} does not exist "
                "or you do not have access."
            )

        # Create temporary directory for cloning
        temp_dir = tempfile.mkdtemp(prefix="gitprobe_")

        try:
            # Clone repository (with optional branch/commit)
            self._clone_repository(repo_info["clone_url"], temp_dir, ref=repo_info.get("ref"))

            # Extract file tree
            file_tree = self._extract_file_tree(temp_dir)

            # Prepare result
            result = {
                "repository": {
                    "owner": repo_info["owner"],
                    "name": repo_info["repo"],
                    "url": repo_info["html_url"],
                    "ref": repo_info.get("ref"),
                    "ref_type": repo_info.get("ref_type"),
                },
                "analysis": {
                    "exclude_patterns": self.exclude_patterns,
                    "include_patterns": self.include_patterns,
                    "max_file_size_kb": self.max_file_size_kb,
                    "min_file_size_kb": self.min_file_size_kb,
                    "total_exclude_patterns": len(self.exclude_patterns),
                    "total_include_patterns": len(self.include_patterns),
                },
                "file_tree": file_tree,
            }

            return result

        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def get_file_statistics(self, file_tree: Dict) -> Dict:
        """
        Calculate statistics from the file tree.

        Args:
            file_tree: File tree dictionary

        Returns:
            Dictionary with file statistics
        """
        stats = {
            "total_files": 0,
            "total_directories": 0,
            "total_size_bytes": 0,
            "total_size_kb": 0,
            "estimated_total_tokens": 0,
            "file_extensions": {},
            "binary_files": 0,
            "text_files": 0,
            "largest_files": [],  # Top 10 largest files
            "most_token_files": [],  # Top 10 files with most tokens
        }

        all_files = []

        def count_items(tree: Dict):
            if tree["type"] == "file":
                stats["total_files"] += 1
                file_size = tree.get("size", 0)
                stats["total_size_bytes"] += file_size
                stats["total_size_kb"] += tree.get("size_kb", 0)
                stats["estimated_total_tokens"] += tree.get("estimated_tokens", 0)

                ext = tree.get("extension", "").lower()
                if ext:
                    stats["file_extensions"][ext] = (
                        stats["file_extensions"].get(ext, 0) + 1
                    )

                if tree.get("is_binary", False):
                    stats["binary_files"] += 1
                else:
                    stats["text_files"] += 1

                # Collect file info for rankings
                all_files.append(
                    {
                        "path": tree.get("path", ""),
                        "name": tree.get("name", ""),
                        "size_kb": tree.get("size_kb", 0),
                        "estimated_tokens": tree.get("estimated_tokens", 0),
                        "is_binary": tree.get("is_binary", False),
                    }
                )

            elif tree["type"] == "directory":
                stats["total_directories"] += 1
                if tree["children"]:
                    for child in tree["children"]:
                        count_items(child)

        count_items(file_tree)

        # Calculate top files by size and tokens
        stats["largest_files"] = sorted(
            all_files, key=lambda x: x["size_kb"], reverse=True
        )[:10]
        stats["most_token_files"] = sorted(
            [f for f in all_files if not f["is_binary"]],
            key=lambda x: x["estimated_tokens"],
            reverse=True,
        )[:10]

        # Round total size for readability
        stats["total_size_kb"] = round(stats["total_size_kb"], 2)

        return stats


def main():
    """Command-line interface for the repository analyzer."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze GitHub repository file structure"
    )
    parser.add_argument("url", help="GitHub repository URL")
    parser.add_argument(
        "--exclude",
        nargs="*",
        help='Additional exclude patterns (e.g., "*.md" "docs/")',
    )
    parser.add_argument(
        "--include",
        nargs="*",
        help='Include only files matching these patterns (e.g., "*.py" "src/")',
    )
    parser.add_argument("--max-size", type=int, help="Maximum file size in KB")
    parser.add_argument("--min-size", type=int, help="Minimum file size in KB")
    parser.add_argument("--output", "-o", help="Output file (JSON format)")
    parser.add_argument("--stats", action="store_true", help="Show file statistics")
    parser.add_argument("--summary", action="store_true", help="Show summary only")
    parser.add_argument(
        "--pretty", action="store_true", help="Pretty print JSON output"
    )

    args = parser.parse_args()

    try:
        # Create analyzer with filtering options
        analyzer = RepoAnalyzer(
            exclude_patterns=args.exclude,
            include_patterns=args.include,
            max_file_size_kb=args.max_size,
            min_file_size_kb=args.min_size,
        )

        # Analyze repository
        result = analyzer.analyze_repository(args.url)

        # Add statistics if requested
        if args.stats or args.summary:
            result["statistics"] = analyzer.get_file_statistics(result["file_tree"])

        # Show summary if requested
        if args.summary:
            stats = result["statistics"]
            repo = result["repository"]

            print(f"\n📊 SUMMARY")
            print("=" * 50)
            print(f"Repository: {repo['owner']}/{repo['name']}")
            print(f"Files analyzed: {stats['total_files']}")
            print(f"Total size: {stats['total_size_kb']:.1f} KB")
            print(f"Estimated tokens: {stats['estimated_total_tokens']:,}")
            print(f"Text files: {stats['text_files']}")
            print(f"Binary files: {stats['binary_files']}")

            if stats["file_extensions"]:
                print(f"\nTop file types:")
                sorted_exts = sorted(
                    stats["file_extensions"].items(), key=lambda x: x[1], reverse=True
                )
                for ext, count in sorted_exts[:5]:
                    print(f"  {ext}: {count} files")

            if stats["largest_files"]:
                print(f"\nLargest files:")
                for i, file_info in enumerate(stats["largest_files"][:5], 1):
                    print(f"  {i}. {file_info['name']} ({file_info['size_kb']:.1f} KB)")

            return 0

        # Output result
        json_output = json.dumps(result, indent=2 if args.pretty else None)

        if args.output:
            with open(args.output, "w") as f:
                f.write(json_output)
            print(f"Analysis saved to: {args.output}")
        else:
            print(json_output)

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
