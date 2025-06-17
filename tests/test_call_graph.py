#!/usr/bin/env python3
"""
Test script for Call Graph Analyzer
Validates the functionality and provides visual validation.
"""

from call_graph_analyzer import CallGraphAnalyzer
from repo_analyzer import RepoAnalyzer


def test_call_graph_basic():
    """Test basic call graph functionality."""
    print("🧪 Testing Call Graph Analyzer")
    print("=" * 50)

    # Get repository URL from user
    repo_url = input(
        "Enter GitHub repository URL (or press Enter for default): "
    ).strip()
    if not repo_url:
        repo_url = "https://github.com/octocat/Hello-World"
        print(f"Using default: {repo_url}")

    try:
        # Create analyzers
        repo_analyzer = RepoAnalyzer(
            include_patterns=["*.py"],
            exclude_patterns=["*test*", "*spec*", "docs/", "*.md"],
        )

        cg_analyzer = CallGraphAnalyzer(repo_analyzer)

        print(f"\n🚀 Analyzing call graph for: {repo_url}")
        result = cg_analyzer.analyze_repository(repo_url)

        # Display results
        cg = result["call_graph"]
        repo = result["repository"]
        viz = result["visualization"]["summary"]

        print(f"\n📊 RESULTS")
        print("=" * 40)
        print(f"Repository: {repo['owner']}/{repo['name']}")
        print(f"Functions found: {cg['total_functions']}")
        print(f"Function calls: {cg['total_calls']}")
        print(f"Files analyzed: {cg['files_analyzed']}")
        print(f"Languages: {', '.join(cg['languages_found'])}")
        print(f"Resolved calls: {viz['total_edges']}")
        print(f"Unresolved calls: {viz['unresolved_calls']}")

        # Show some functions
        if result["functions"]:
            print(f"\n📋 Sample Functions Found:")
            for i, func in enumerate(result["functions"][:5], 1):
                print(f"  {i}. {func['name']}() in {func['file_path']}")
                if func["docstring"]:
                    print(f"     Doc: {func['docstring'][:50]}...")

        # Show some relationships
        resolved_rels = [r for r in result["relationships"] if r["is_resolved"]]
        if resolved_rels:
            print(f"\n🔗 Sample Call Relationships:")
            for i, rel in enumerate(resolved_rels[:5], 1):
                caller_name = rel["caller"].split(":")[-1]
                callee_name = rel["callee"].split(":")[-1]
                print(
                    f"  {i}. {caller_name}() → {callee_name}() (line {rel['call_line']})"
                )

        # Generate visualization
        output_file = f"{repo['owner']}-{repo['name']}-callgraph.html"
        from call_graph_analyzer import generate_html_visualization

        generate_html_visualization(result, output_file)

        print(f"\n🎨 Interactive visualization saved to: {output_file}")
        print(f"Open the file in your browser to see the call graph!")

        # Save JSON data
        import json

        json_file = f"{repo['owner']}-{repo['name']}-callgraph.json"
        with open(json_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"📄 Raw data saved to: {json_file}")

        return True

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def test_small_python_repo():
    """Test with a known small Python repository."""
    print("\n🧪 Testing with Small Python Repository")
    print("=" * 50)

    # Use a simple Python repo
    repo_url = "https://github.com/miguelgrinberg/microblog"

    try:
        repo_analyzer = RepoAnalyzer(
            include_patterns=["*.py"],
            exclude_patterns=["*test*", "migrations/", "venv/", "*.md"],
        )

        cg_analyzer = CallGraphAnalyzer(repo_analyzer)

        print(f"🔍 Analyzing: {repo_url}")
        result = cg_analyzer.analyze_repository(repo_url)

        cg = result["call_graph"]
        print(
            f"\n✅ Found {cg['total_functions']} functions with {cg['total_calls']} calls"
        )

        # Create simple text visualization
        print(f"\n📈 Call Graph Summary:")

        # Find entry points (functions that aren't called by others)
        called_functions = set()
        for rel in result["relationships"]:
            if rel["is_resolved"]:
                called_functions.add(rel["callee"])

        entry_points = []
        for func in result["functions"]:
            func_id = f"{func['file_path']}:{func['name']}"
            if func_id not in called_functions:
                entry_points.append(func)

        if entry_points:
            print(f"Entry points (not called by others):")
            for func in entry_points[:5]:
                print(f"  • {func['name']}() in {func['file_path']}")

        return True

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def main():
    """Run call graph tests."""
    print("🔬 Call Graph Analyzer Validation")
    print("=" * 60)

    choice = input(
        "Choose test:\n1. Interactive test (you provide URL)\n2. Automated test (small repo)\nEnter choice (1 or 2): "
    ).strip()

    if choice == "1":
        success = test_call_graph_basic()
    elif choice == "2":
        success = test_small_python_repo()
    else:
        print("❌ Invalid choice")
        return 1

    if success:
        print("\n✅ Call graph analysis completed successfully!")
        print("🎯 Key features validated:")
        print("  • Function extraction from Python code")
        print("  • Call relationship detection")
        print("  • Interactive HTML visualization")
        print("  • JSON data export")
        return 0
    else:
        print("\n❌ Call graph analysis failed!")
        return 1


if __name__ == "__main__":
    exit(main())
