#!/usr/bin/env python3
"""
Simple GitProbe Repository Analyzer
Shows clean directory structure for any GitHub repository.
"""

import os
import shutil
import tempfile
from pathlib import Path

import git
import pathspec


class SimpleRepoAnalyzer:
    """Simple analyzer that shows directory structure."""

    # Common ignore patterns
    IGNORE_PATTERNS = [
        "node_modules/",
        "venv/",
        "env/",
        "__pycache__/",
        ".git/",
        "build/",
        "dist/",
        "target/",
        ".vscode/",
        ".idea/",
        "*.pyc",
        "*.log",
        ".DS_Store",
        "Thumbs.db",
    ]

    def __init__(self):
        self.pathspec = pathspec.PathSpec.from_lines(
            "gitwildmatch", self.IGNORE_PATTERNS
        )

    def get_repo_name(self, url):
        """Extract repo name from GitHub URL."""
        if "github.com/" in url:
            parts = url.rstrip("/").split("/")
            if len(parts) >= 2:
                owner, repo = parts[-2], parts[-1]
                if repo.endswith(".git"):
                    repo = repo[:-4]
                return f"{owner}-{repo}"
        return "repository"

    def should_ignore(self, path):
        """Check if path should be ignored."""
        return self.pathspec.match_file(path)

    def analyze_repo(self, github_url):
        """Analyze repository and return directory structure."""
        repo_name = self.get_repo_name(github_url)
        temp_dir = tempfile.mkdtemp(prefix="gitprobe_")

        # Prepare output content
        output_lines = []

        try:
            print(f"📥 Cloning {repo_name}...")
            git.Repo.clone_from(github_url, temp_dir, depth=1)
            print("✅ Clone complete!")

            print(f"\n📁 Directory structure:")
            print("└── " + repo_name + "/")

            # Add to output
            output_lines.append(f"Directory structure for: {github_url}")
            output_lines.append("└── " + repo_name + "/")

            self._print_tree(Path(temp_dir), temp_dir, "    ", output_lines)

            # Write to file
            filename = f"{repo_name}_structure.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write("\n".join(output_lines))

            print(f"\n💾 Directory structure saved to: {filename}")

        except Exception as e:
            print(f"❌ Error: {str(e)}")
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def _print_tree(self, current_path, repo_root, indent, output_lines=None):
        """Print directory tree structure."""
        try:
            items = sorted(current_path.iterdir())

            for i, item in enumerate(items):
                # Get relative path for ignore checking
                rel_path = str(item.relative_to(Path(repo_root)))

                # Skip ignored files/directories
                if self.should_ignore(rel_path):
                    continue

                # Determine if this is the last item
                is_last = i == len(items) - 1

                # Print the current item
                if item.is_dir():
                    line = f"{indent}├── {item.name}/"
                    print(line)
                    if output_lines is not None:
                        output_lines.append(line)
                    # Recurse into subdirectory
                    self._print_tree(item, repo_root, indent + "│   ", output_lines)
                else:
                    connector = "└──" if is_last else "├──"
                    line = f"{indent}{connector} {item.name}"
                    print(line)
                    if output_lines is not None:
                        output_lines.append(line)

        except (PermissionError, OSError):
            pass


def main():
    """Main function with user input."""
    print("🔍 GitProbe - Simple Repository Analyzer")
    print("=" * 50)

    # Get GitHub URL from user
    while True:
        github_url = input("\n📋 Enter GitHub repository URL: ").strip()

        if not github_url:
            print("❌ Please enter a valid URL")
            continue

        if "github.com" not in github_url:
            print("❌ Please enter a valid GitHub URL")
            continue

        break

    # Analyze the repository
    analyzer = SimpleRepoAnalyzer()
    analyzer.analyze_repo(github_url)

    print("\n✨ Analysis complete!")


if __name__ == "__main__":
    main()
