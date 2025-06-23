"""
GitProbe Package Main Entry Point

Allows running GitProbe as a module: python -m gitprobe
"""

import sys
from pathlib import Path


def main():
    """Main entry point for running GitProbe server."""
    try:
        import uvicorn

        print("🚀 Starting GitProbe Server via package...")
        uvicorn.run("gitprobe.web.server:app", host="0.0.0.0", port=8000, reload=True)
    except ImportError:
        print("❌ uvicorn not installed. Please install with: pip install uvicorn")
        sys.exit(1)


if __name__ == "__main__":
    main()
