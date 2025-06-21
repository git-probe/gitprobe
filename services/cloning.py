"""
GitProbe Utility Functions
Repository cloning and cleanup utilities.
"""

import os
import shutil
import tempfile
import subprocess
import stat
import time
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
        # Configure git for Windows long paths if on Windows
        if os.name == "nt":  # Windows
            try:
                subprocess.run(
                    [
                        GIT_EXECUTABLE_PATH,
                        "config",
                        "--global",
                        "core.longpaths",
                        "true",
                    ],
                    capture_output=True,
                    text=True,
                )
            except:
                pass  # Ignore if this fails

        # Clone with timeout and sparse-checkout to avoid problematic paths
        subprocess.run(
            [
                GIT_EXECUTABLE_PATH,
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                github_url,
                temp_dir,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for large repos like Sui
        )

        # Configure sparse-checkout to exclude problematic directories on Windows
        if os.name == "nt":  # Windows
            try:
                # Enable sparse-checkout
                subprocess.run(
                    [
                        GIT_EXECUTABLE_PATH,
                        "-C",
                        temp_dir,
                        "config",
                        "core.sparseCheckout",
                        "true",
                    ],
                    capture_output=True,
                    text=True,
                )

                # Create sparse-checkout file to exclude problematic paths
                sparse_checkout_path = os.path.join(
                    temp_dir, ".git", "info", "sparse-checkout"
                )
                os.makedirs(os.path.dirname(sparse_checkout_path), exist_ok=True)
                with open(sparse_checkout_path, "w") as f:
                    f.write("*\n")
                    f.write(
                        "!**/tests/**/CvnF9nAXfESwhrtdkjGhX2wAkKHzwr8N2rjExPK8eZYS/**\n"
                    )
                    f.write(
                        "!**/0x0000000000000000000000000000000000000000000000000000000000000002/**\n"
                    )

                # Re-checkout with sparse-checkout
                subprocess.run(
                    [
                        GIT_EXECUTABLE_PATH,
                        "-C",
                        temp_dir,
                        "read-tree",
                        "-m",
                        "-u",
                        "HEAD",
                    ],
                    capture_output=True,
                    text=True,
                )
            except:
                pass  # Continue if sparse-checkout fails
        return temp_dir
    except subprocess.TimeoutExpired:
        # Clean up on timeout
        if os.path.exists(temp_dir):
            cleanup_repository_safe(temp_dir)
        raise RuntimeError(
            f"Repository cloning timed out after 5 minutes. The repository may be too large or network is slow."
        )
    except subprocess.CalledProcessError as e:
        # Clean up on failure using Windows-safe cleanup
        if os.path.exists(temp_dir):
            cleanup_repository_safe(temp_dir)
        raise RuntimeError(f"Failed to clone repository: {e.stderr}")
    except FileNotFoundError:
        # This is a fallback, but shutil.which should prevent this.
        if os.path.exists(temp_dir):
            cleanup_repository_safe(temp_dir)
        raise RuntimeError(
            f"Git executable not found at '{GIT_EXECUTABLE_PATH}'. "
            "Please ensure Git is installed and the path is correct."
        )


def cleanup_repository_safe(repo_dir: str) -> bool:
    """
    Windows-safe removal of the cloned repository directory.
    Handles read-only files and permission issues common on Windows.

    Args:
        repo_dir: Path to the repository directory to remove

    Returns:
        bool: True if cleanup successful, False otherwise
    """

    def handle_remove_readonly(func, path, exc):
        """Error handler for Windows read-only files."""
        if os.path.exists(path):
            # Make the file writable and try again
            os.chmod(path, stat.S_IWRITE)
            func(path)

    try:
        if os.path.exists(repo_dir):
            # On Windows, git creates read-only files that need special handling
            if os.name == "nt":  # Windows
                shutil.rmtree(repo_dir, onerror=handle_remove_readonly)
            else:
                shutil.rmtree(repo_dir)
            return True
        return False
    except PermissionError as e:
        # If still having permission issues, try waiting and retrying
        try:
            time.sleep(1)  # Wait 1 second for file handles to close
            if os.path.exists(repo_dir):
                # Force remove all read-only attributes
                for root, dirs, files in os.walk(repo_dir):
                    for dir in dirs:
                        os.chmod(os.path.join(root, dir), stat.S_IWRITE)
                    for file in files:
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            os.chmod(file_path, stat.S_IWRITE)
                shutil.rmtree(repo_dir)
            return True
        except Exception as retry_e:
            print(
                f"⚠️ Warning: Failed to cleanup {repo_dir} after retry: {str(retry_e)}"
            )
            return False
    except Exception as e:
        print(f"⚠️ Warning: Failed to cleanup {repo_dir}: {str(e)}")
        return False


def cleanup_repository(repo_dir: str) -> bool:
    """
    Remove the cloned repository directory (wrapper for backward compatibility).

    Args:
        repo_dir: Path to the repository directory to remove

    Returns:
        bool: True if cleanup successful, False otherwise
    """
    return cleanup_repository_safe(repo_dir)


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
