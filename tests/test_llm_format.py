"""
Test script for LLM Format Generation

This script demonstrates how to use the CallGraphAnalyzer to generate
LLM-optimized call graph data for better AI code understanding.
"""

import json
import sys
import os

# Add parent directory to path so we can import from services
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.analysis_service import AnalysisService


def test_llm_format(github_url: str):
    """
    Test the LLM format generation with a GitHub repository.

    Args:
        github_url: GitHub repository URL to analyze
    """
    print(f"🤖 Testing LLM Format Generation")
    print(f"Repository: {github_url}")
    print("=" * 60)

    try:
        # Create analysis service
        analysis_service = AnalysisService()

        # Perform full analysis
        print("📊 Analyzing repository...")
        analysis_result = analysis_service.analyze_repository_full(
            github_url,
            include_patterns=["*.py", "*.js", "*.ts"],
            exclude_patterns=["*test*", "*spec*", "node_modules/", "__pycache__/"],
        )

        # Generate LLM format
        print("🧠 Generating LLM-optimized format...")
        llm_data = analysis_service.call_graph_analyzer.generate_llm_format()

        # Display results
        print(f"\n📋 LLM FORMAT SUMMARY")
        print("=" * 40)
        print(f"Repository: {analysis_result.repository.name}")
        print(f"Functions found: {len(llm_data['functions'])}")
        print(f"Functions with relationships: {len(llm_data['relationships'])}")

        # Show sample functions
        print(f"\n🔍 SAMPLE FUNCTIONS")
        print("-" * 40)
        for i, func in enumerate(llm_data["functions"][:5], 1):
            print(f"{i}. {func['name']}() in {func['file']}")
            if func["purpose"]:
                print(f"   Purpose: {func['purpose']}")
            if func["parameters"]:
                print(f"   Parameters: {', '.join(func['parameters'])}")
            if func["is_recursive"]:
                print(f"   🔄 Recursive function")
            print()

        # Show sample relationships
        print(f"🔗 SAMPLE RELATIONSHIPS")
        print("-" * 40)
        sample_funcs = list(llm_data["relationships"].items())[:3]
        for func_name, relations in sample_funcs:
            print(f"{func_name}:")
            if relations["calls"]:
                print(f"  → Calls: {', '.join(relations['calls'][:3])}")
            if relations["called_by"]:
                print(f"  ← Called by: {', '.join(relations['called_by'][:3])}")
            print()

        # Save to file for LLM testing (in tests directory)
        output_file = os.path.join(
            os.path.dirname(__file__),
            f"llm_format_{analysis_result.repository.name}.json",
        )
        with open(output_file, "w") as f:
            json.dump(llm_data, f, indent=2)

        print(f"💾 LLM format saved to: {output_file}")
        print(f"📏 File size: {len(json.dumps(llm_data, indent=2))} characters")

        return llm_data

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


def main():
    """Main test function with example repositories."""
    print("🧪 LLM Format Generation Tester")
    print("=" * 50)

    # Example repositories for testing
    test_repos = [
        "https://github.com/octocat/Hello-World",
        "https://github.com/microsoft/vscode-python",  # Larger Python project
    ]

    # Get repository from user or use example
    repo_url = input(
        f"Enter GitHub repository URL (or press Enter for default): "
    ).strip()

    if not repo_url:
        repo_url = test_repos[0]
        print(f"Using default: {repo_url}")

    # Test LLM format generation
    llm_data = test_llm_format(repo_url)

    if llm_data:
        print(f"\n✅ LLM format generated successfully!")
        print(f"🎯 This format is optimized for AI/LLM consumption")
        print(f"📤 You can now feed this JSON to language models for code analysis")
    else:
        print(f"\n❌ Failed to generate LLM format")


if __name__ == "__main__":
    main()
