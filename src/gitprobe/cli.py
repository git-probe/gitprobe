"""
GitProbe Command Line Interface

Provides command-line access to GitProbe functionality.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional, List

from gitprobe.analysis.analysis_service import AnalysisService


def analyze_repo(
    url: str,
    output: Optional[str] = None,
    format: str = "json",
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    structure_only: bool = False,
) -> None:
    """Analyze a repository and output results."""
    print(f"🔍 Analyzing repository: {url}")

    try:
        service = AnalysisService()

        if structure_only:
            result = service.analyze_repository_structure_only(url)
        else:
            analysis_result = service.analyze_repository_full(url, include, exclude)
            result = analysis_result.model_dump()

        # Output results
        if output:
            output_path = Path(output)
            with open(output_path, "w") as f:
                if format == "json":
                    json.dump(result, f, indent=2)
                else:
                    f.write(str(result))
            print(f"✅ Results saved to: {output_path}")
        else:
            if format == "json":
                print(json.dumps(result, indent=2))
            else:
                print(result)

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GitProbe - Advanced repository analysis with call graph generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  gitprobe analyze https://github.com/user/repo
  gitprobe analyze user/repo --output results.json
  gitprobe analyze https://github.com/user/repo --structure-only
  gitprobe server --port 8080
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a repository")
    analyze_parser.add_argument("url", help="GitHub repository URL or owner/repo")
    analyze_parser.add_argument("--output", "-o", help="Output file path")
    analyze_parser.add_argument(
        "--format", choices=["json", "text"], default="json", help="Output format"
    )
    analyze_parser.add_argument("--include", nargs="*", help="File patterns to include")
    analyze_parser.add_argument("--exclude", nargs="*", help="File patterns to exclude")
    analyze_parser.add_argument(
        "--structure-only", action="store_true", help="Analyze structure only (faster)"
    )

    # Server command
    server_parser = subparsers.add_parser("server", help="Start the GitProbe server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    server_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    server_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "analyze":
        analyze_repo(
            url=args.url,
            output=args.output,
            format=args.format,
            include=args.include,
            exclude=args.exclude,
            structure_only=args.structure_only,
        )
    elif args.command == "server":
        start_server(host=args.host, port=args.port, reload=args.reload)


def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the GitProbe server."""
    try:
        import uvicorn

        print(f"🚀 Starting GitProbe server on {host}:{port}")
        uvicorn.run("gitprobe.web.server:app", host=host, port=port, reload=reload)
    except ImportError:
        print("❌ uvicorn not installed. Please install with: pip install uvicorn")
        sys.exit(1)


if __name__ == "__main__":
    main()
