#!/usr/bin/env python3
"""
Test script for Advanced GitProbe Repository Analyzer
Demonstrates the new filtering and analysis features.
"""

from repo_analyzer import RepoAnalyzer


def test_basic_analysis():
    """Test basic analysis with summary."""
    print("🔍 TEST 1: Basic Analysis with Summary")
    print("=" * 60)

    repo_url = input("Enter GitHub repository URL: ").strip()
    if not repo_url:
        repo_url = "https://github.com/octocat/Hello-World"
        print(f"Using default: {repo_url}")

    analyzer = RepoAnalyzer()
    result = analyzer.analyze_repository(repo_url)
    stats = analyzer.get_file_statistics(result["file_tree"])

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
        print(f"\nFile types:")
        for ext, count in sorted(
            stats["file_extensions"].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {ext}: {count} files")


def test_include_patterns():
    """Test include patterns."""
    print("\n🔍 TEST 2: Include Only Python Files")
    print("=" * 60)

    repo_url = "https://github.com/psf/requests"
    print(f"Analyzing: {repo_url}")
    print("Include patterns: ['*.py']")

    analyzer = RepoAnalyzer(include_patterns=["*.py"])
    result = analyzer.analyze_repository(repo_url)
    stats = analyzer.get_file_statistics(result["file_tree"])

    print(f"\nFiles found: {stats['total_files']}")
    print(f"Estimated tokens: {stats['estimated_total_tokens']:,}")

    if stats["most_token_files"]:
        print(f"\nFiles with most tokens:")
        for i, file_info in enumerate(stats["most_token_files"][:3], 1):
            print(
                f"  {i}. {file_info['name']} ({file_info['estimated_tokens']:,} tokens)"
            )


def test_exclude_patterns():
    """Test exclude patterns."""
    print("\n🔍 TEST 3: Exclude Documentation Files")
    print("=" * 60)

    repo_url = "https://github.com/tiangolo/fastapi"
    print(f"Analyzing: {repo_url}")
    print("Exclude patterns: ['*.md', 'docs/', '*.rst']")

    analyzer = RepoAnalyzer(exclude_patterns=["*.md", "docs/", "*.rst"])
    result = analyzer.analyze_repository(repo_url)
    stats = analyzer.get_file_statistics(result["file_tree"])

    print(f"\nFiles found: {stats['total_files']}")
    print(f"Estimated tokens: {stats['estimated_total_tokens']:,}")

    print(f"\nFile types found:")
    for ext, count in sorted(
        stats["file_extensions"].items(), key=lambda x: x[1], reverse=True
    )[:5]:
        print(f"  {ext}: {count} files")


def test_file_size_filter():
    """Test file size filtering."""
    print("\n🔍 TEST 4: Small Files Only (Under 10KB)")
    print("=" * 60)

    repo_url = "https://github.com/octocat/Hello-World"
    print(f"Analyzing: {repo_url}")
    print("Max file size: 10 KB")

    analyzer = RepoAnalyzer(max_file_size_kb=10)
    result = analyzer.analyze_repository(repo_url)
    stats = analyzer.get_file_statistics(result["file_tree"])

    print(f"\nFiles found: {stats['total_files']}")
    print(f"Total size: {stats['total_size_kb']:.1f} KB")

    if stats["largest_files"]:
        print(f"\nFiles found (by size):")
        for i, file_info in enumerate(stats["largest_files"], 1):
            print(f"  {i}. {file_info['name']}")


def test_combined_filters():
    """Test combined filtering."""
    print("\n🔍 TEST 5: Combined Filters")
    print("=" * 60)

    repo_url = "https://github.com/psf/requests"
    print(f"Analyzing: {repo_url}")
    print("Include: ['*.py', '*.txt']")
    print("Exclude: ['*test*', '*example*']")
    print("Max size: 50 KB")

    analyzer = RepoAnalyzer(
        include_patterns=["*.py", "*.txt"],
        exclude_patterns=["*test*", "*example*"],
        max_file_size_kb=50,
    )
    result = analyzer.analyze_repository(repo_url)
    stats = analyzer.get_file_statistics(result["file_tree"])

    print(f"\nFiles found: {stats['total_files']}")
    print(f"Estimated tokens: {stats['estimated_total_tokens']:,}")

    if stats["most_token_files"]:
        print(f"\nTop files by tokens:")
        for i, file_info in enumerate(stats["most_token_files"][:3], 1):
            print(
                f"  {i}. {file_info['name']} ({file_info['estimated_tokens']:,} tokens)"
            )


def interactive_test():
    """Interactive test with user input."""
    print("\n🔍 INTERACTIVE TEST")
    print("=" * 60)

    repo_url = input("Repository URL: ").strip()
    if not repo_url:
        print("❌ Please provide a repository URL")
        return

    print("\nOptional filters (press Enter to skip):")

    include = input("Include patterns (e.g., '*.py *.js'): ").strip()
    include_patterns = include.split() if include else None

    exclude = input("Exclude patterns (e.g., '*test* docs/'): ").strip()
    exclude_patterns = exclude.split() if exclude else None

    max_size = input("Max file size (KB): ").strip()
    max_file_size_kb = int(max_size) if max_size.isdigit() else None

    print(f"\n🔄 Analyzing with filters...")

    analyzer = RepoAnalyzer(
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        max_file_size_kb=max_file_size_kb,
    )

    result = analyzer.analyze_repository(repo_url)
    stats = analyzer.get_file_statistics(result["file_tree"])

    repo = result["repository"]
    print(f"\n📊 RESULTS")
    print("=" * 40)
    print(f"Repository: {repo['owner']}/{repo['name']}")
    print(f"Files analyzed: {stats['total_files']}")
    print(f"Total size: {stats['total_size_kb']:.1f} KB")
    print(f"Estimated tokens: {stats['estimated_total_tokens']:,}")

    print(f"\nFile types:")
    for ext, count in sorted(
        stats["file_extensions"].items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {ext}: {count} files")


def main():
    """Main test runner."""
    print("🚀 Advanced GitProbe Repository Analyzer - Test Suite")
    print("=" * 70)

    tests = [
        ("1", "Basic Analysis", test_basic_analysis),
        ("2", "Include Patterns", test_include_patterns),
        ("3", "Exclude Patterns", test_exclude_patterns),
        ("4", "File Size Filter", test_file_size_filter),
        ("5", "Combined Filters", test_combined_filters),
        ("6", "Interactive Test", interactive_test),
    ]

    print("\nSelect a test to run:")
    for code, name, _ in tests:
        print(f"  {code}. {name}")
    print("  0. Run all tests")

    choice = input("\nEnter choice (0-6): ").strip()

    if choice == "0":
        print("\n🔄 Running all tests...")
        for _, name, test_func in tests:
            try:
                test_func()
            except KeyboardInterrupt:
                print("\n❌ Test interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Test failed: {str(e)}")
                continue
    else:
        test_map = {code: test_func for code, _, test_func in tests}
        if choice in test_map:
            test_map[choice]()
        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    main()
