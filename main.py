#!/usr/bin/env python3
"""
GitProbe - GitHub Repository Analysis Tool
Main entry point for the gitprobe.com backend.
"""

import sys
import argparse
from repo_analyzer import RepoAnalyzer
from call_graph_analyzer import CallGraphAnalyzer


def main():
    """Main CLI interface for GitProbe."""
    parser = argparse.ArgumentParser(
        description="GitProbe - Analyze GitHub repositories with file trees and call graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic file tree analysis
  python main.py tree https://github.com/user/repo
  
  # Call graph analysis with visualization
  python main.py callgraph https://github.com/user/repo --viz output.html
  
  # Export LLM-optimized JSON
  python main.py callgraph https://github.com/user/repo --llm-json analysis.json
  
  # File tree with advanced filtering
  python main.py tree https://github.com/user/repo --include "*.py" "src/" --max-size 100
        """,
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Analysis type")

    # File tree analysis
    tree_parser = subparsers.add_parser("tree", help="Analyze file tree structure")
    tree_parser.add_argument("url", help="GitHub repository URL")
    tree_parser.add_argument(
        "--include", nargs="*", help='Include patterns (e.g., "*.py" "src/")'
    )
    tree_parser.add_argument(
        "--exclude", nargs="*", help='Exclude patterns (e.g., "*test*" "docs/")'
    )
    tree_parser.add_argument("--min-size", type=float, help="Minimum file size in KB")
    tree_parser.add_argument("--max-size", type=float, help="Maximum file size in KB")
    tree_parser.add_argument("--output", "-o", help="Output JSON file")
    tree_parser.add_argument("--summary", action="store_true", help="Show summary only")

    # Call graph analysis
    callgraph_parser = subparsers.add_parser(
        "callgraph", help="Analyze function call graphs"
    )
    callgraph_parser.add_argument("url", help="GitHub repository URL")
    callgraph_parser.add_argument(
        "--include", nargs="*", help='Include patterns (e.g., "*.py" "src/")'
    )
    callgraph_parser.add_argument(
        "--exclude", nargs="*", help='Exclude patterns (e.g., "*test*" "docs/")'
    )
    callgraph_parser.add_argument("--output", "-o", help="Output JSON file")
    callgraph_parser.add_argument("--viz", help="Generate HTML visualization file")
    callgraph_parser.add_argument("--svg", help="Export SVG file")
    callgraph_parser.add_argument("--llm-json", help="Export LLM-optimized JSON file")
    callgraph_parser.add_argument(
        "--summary", action="store_true", help="Show summary only"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "tree":
            return run_tree_analysis(args)
        elif args.command == "callgraph":
            return run_callgraph_analysis(args)
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1


def run_tree_analysis(args):
    """Run file tree analysis."""
    from repo_analyzer import main as repo_main

    # Convert args to format expected by repo_analyzer
    sys.argv = ["repo_analyzer.py", args.url]

    if args.include:
        sys.argv.extend(["--include"] + args.include)
    if args.exclude:
        sys.argv.extend(["--exclude"] + args.exclude)
    if args.min_size:
        sys.argv.extend(["--min-size", str(args.min_size)])
    if args.max_size:
        sys.argv.extend(["--max-size", str(args.max_size)])
    if args.output:
        sys.argv.extend(["--output", args.output])
    if args.summary:
        sys.argv.append("--summary")

    return repo_main()


def run_callgraph_analysis(args):
    """Run call graph analysis."""
    from call_graph_analyzer import main as callgraph_main

    # Convert args to format expected by call_graph_analyzer
    sys.argv = ["call_graph_analyzer.py", args.url]

    if args.include:
        sys.argv.extend(["--include"] + args.include)
    if args.exclude:
        sys.argv.extend(["--exclude"] + args.exclude)
    if args.output:
        sys.argv.extend(["--output", args.output])
    if args.viz:
        sys.argv.extend(["--viz", args.viz])
    if args.svg:
        sys.argv.extend(["--svg", args.svg])
    if args.llm_json:
        sys.argv.extend(["--llm-json", args.llm_json])
    if args.summary:
        sys.argv.append("--summary")

    return callgraph_main()


if __name__ == "__main__":
    exit(main())
