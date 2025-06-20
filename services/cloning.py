"""
GitProbe Utility Functions
Repository cloning and cleanup utilities.
"""

import os
import shutil
import tempfile
import subprocess
from typing import Optional

GIT_EXECUTABLE_PATH = shutil.which("git")


def clone_repository(github_url: str) -> str:
    """
    Clone a GitHub repository to a temporary directory.

    Args:
        github_url: GitHub repository URL

    Returns:
        str: Path to the cloned repository directory

    Raises:
        RuntimeError: If cloning fails or git executable is not found.
    """
    if not GIT_EXECUTABLE_PATH:
        raise RuntimeError(
            "Git executable not found. Please install Git and ensure it is in the system's PATH."
        )

    temp_dir = tempfile.mkdtemp(prefix="gitprobe_")

    try:
        subprocess.run(
            [GIT_EXECUTABLE_PATH, "clone", "--depth", "1", github_url, temp_dir],
            check=True,
            capture_output=True,
            text=True,
        )
        return temp_dir
    except subprocess.CalledProcessError as e:
        # Clean up on failure
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise RuntimeError(f"Failed to clone repository: {e.stderr}")
    except FileNotFoundError:
        # This is a fallback, but shutil.which should prevent this.
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise RuntimeError(
            f"Git executable not found at '{GIT_EXECUTABLE_PATH}'. "
            "Please ensure Git is installed and the path is correct."
        )


def cleanup_repository(repo_dir: str) -> bool:
    """
    Remove the cloned repository directory.

    Args:
        repo_dir: Path to the repository directory to remove

    Returns:
        bool: True if cleanup successful, False otherwise
    """
    try:
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
            return True
        return False
    except Exception as e:
        print(f"⚠️ Warning: Failed to cleanup {repo_dir}: {str(e)}")
        return False


def parse_github_url(github_url: str) -> dict:
    """
    Parse GitHub URL to extract owner and repository name.

    Args:
        github_url: GitHub repository URL

    Returns:
        dict: Repository information
    """
    parts = github_url.rstrip("/").split("/")
    if len(parts) >= 2:
        owner = parts[-2]
        name = parts[-1].replace(".git", "")
        return {
            "owner": owner,
            "name": name,
            "full_name": f"{owner}/{name}",
            "url": github_url,
        }
    return {
        "owner": "unknown",
        "name": "unknown",
        "full_name": "unknown",
        "url": github_url,
    }
