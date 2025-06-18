#!/usr/bin/env python3
"""
Test script for GitProbe Repository Analyzer
Demonstrates usage and tests various scenarios.
"""

import json
import sys
from repo_analyzer import RepoAnalyzer


def test_basic_functionality():
    """Test basic repository analysis functionality."""
    print("=" * 60)
    print("TEST 1: Basic Repository Analysis")
    print("=" * 60)

    # Test with a simple Python repository
    test_repo = "https://github.com/psf/requests"

    try:
        analyzer = RepoAnalyzer()
        print(f"Analyzing repository: {test_repo}")

        result = analyzer.analyze_repository(test_repo)
        stats = analyzer.get_file_statistics(result["file_tree"])

        print(
            f"\nRepository: {result['repository']['owner']}/{result['repository']['name']}"
        )
        print(f"Total files: {stats['total_files']}")
        print(f"Total directories: {stats['total_directories']}")
        print(f"Total size: {stats['total_size_bytes']:,} bytes")
        print(f"Binary files: {stats['binary_files']}")
        print(f"Text files: {stats['text_files']}")

        print("\nFile extensions found:")
        for ext, count in sorted(stats["file_extensions"].items()):
            print(f"  {ext}: {count} files")

        print("\nIgnore patterns used:")
        for pattern in result["analysis"]["ignore_patterns_used"][:10]:  # Show first 10
            print(f"  {pattern}")
        print(f"  ... and {len(result['analysis']['ignore_patterns_used']) - 10} more")

        print("\n✅ Basic functionality test PASSED")
        return True

    except Exception as e:
        print(f"❌ Basic functionality test FAILED: {str(e)}")
        return False


def test_custom_ignore_patterns():
    """Test custom ignore patterns functionality."""
    print("\n" + "=" * 60)
    print("TEST 2: Custom Ignore Patterns")
    print("=" * 60)

    # Test with custom ignore patterns
    test_repo = "https://github.com/octocat/Hello-World"
    custom_patterns = ["*.md", "docs/", "examples/"]

    try:
        analyzer = RepoAnalyzer(custom_ignore_patterns=custom_patterns)
        print(f"Analyzing repository with custom ignore patterns: {custom_patterns}")

        result = analyzer.analyze_repository(test_repo)
        stats = analyzer.get_file_statistics(result["file_tree"])

        print(
            f"\nRepository: {result['repository']['owner']}/{result['repository']['name']}"
        )
        print(f"Total files: {stats['total_files']}")
        print(f"Custom patterns added: {custom_patterns}")

        # Check if custom patterns are in the ignore list
        all_patterns = result["analysis"]["ignore_patterns_used"]
        custom_found = all([pattern in all_patterns for pattern in custom_patterns])

        if custom_found:
            print("✅ Custom ignore patterns were successfully added")
        else:
            print("❌ Custom ignore patterns were not found")
            return False

        print("\n✅ Custom ignore patterns test PASSED")
        return True

    except Exception as e:
        print(f"❌ Custom ignore patterns test FAILED: {str(e)}")
        return False


def test_file_tree_structure():
    """Test file tree structure and navigation."""
    print("\n" + "=" * 60)
    print("TEST 3: File Tree Structure")
    print("=" * 60)

    test_repo = "https://github.com/octocat/Hello-World"

    try:
        analyzer = RepoAnalyzer()
        result = analyzer.analyze_repository(test_repo)
        file_tree = result["file_tree"]

        print(f"Analyzing file tree structure for: {result['repository']['name']}")

        def print_tree(tree, level=0, max_level=3):
            """Print file tree structure with indentation."""
            if level > max_level:
                return

            indent = "  " * level
            if tree["type"] == "file":
                binary_indicator = " [BINARY]" if tree.get("is_binary", False) else ""
                print(f"{indent}📄 {tree['name']}{binary_indicator}")
            else:
                child_count = len(tree.get("children", []))
                print(f"{indent}📁 {tree['name']}/ ({child_count} items)")

                if tree.get("children") and level < max_level:
                    for child in tree["children"][:5]:  # Limit to first 5 items
                        print_tree(child, level + 1, max_level)
                    if len(tree["children"]) > 5:
                        print(
                            f"{indent}  ... and {len(tree['children']) - 5} more items"
                        )

        print("\nFile tree structure (showing first 3 levels):")
        print_tree(file_tree)

        # Validate tree structure
        if file_tree["type"] == "directory" and "children" in file_tree:
            print("\n✅ File tree structure test PASSED")
            return True
        else:
            print("\n❌ File tree structure test FAILED: Invalid tree structure")
            return False

    except Exception as e:
        print(f"❌ File tree structure test FAILED: {str(e)}")
        return False


def test_url_parsing():
    """Test GitHub URL parsing functionality."""
    print("\n" + "=" * 60)
    print("TEST 4: URL Parsing")
    print("=" * 60)

    test_urls = [
        "https://github.com/octocat/Hello-World",
        "https://github.com/octocat/Hello-World/",
        "https://github.com/octocat/Hello-World.git",
        "git@github.com:octocat/Hello-World.git",
    ]

    analyzer = RepoAnalyzer()

    try:
        for url in test_urls:
            print(f"Testing URL: {url}")
            repo_info = analyzer._parse_github_url(url)
            print(f"  Owner: {repo_info['owner']}")
            print(f"  Repo: {repo_info['repo']}")

            if repo_info["owner"] == "octocat" and repo_info["repo"] == "Hello-World":
                print("  ✅ Parsed correctly")
            else:
                print("  ❌ Parsing failed")
                return False

        # Test invalid URL
        try:
            analyzer._parse_github_url("https://invalid-url.com/repo")
            print("❌ Invalid URL test FAILED: Should have raised ValueError")
            return False
        except ValueError:
            print("✅ Invalid URL correctly rejected")

        print("\n✅ URL parsing test PASSED")
        return True

    except Exception as e:
        print(f"❌ URL parsing test FAILED: {str(e)}")
        return False


def test_programmatic_usage():
    """Test programmatic usage with a simple example."""
    print("\n" + "=" * 60)
    print("TEST 5: Programmatic Usage Example")
    print("=" * 60)

    try:
        # Example: Analyze a repository and find all Python files
        analyzer = RepoAnalyzer()
        result = analyzer.analyze_repository("https://github.com/octocat/Hello-World")

        def find_files_by_extension(tree, extension):
            """Find all files with a specific extension."""
            files = []

            if tree["type"] == "file":
                if tree.get("extension", "").lower() == extension.lower():
                    files.append(tree["path"])
            elif tree["type"] == "directory" and tree.get("children"):
                for child in tree["children"]:
                    files.extend(find_files_by_extension(child, extension))

            return files

        # Find all README files
        readme_files = []

        def find_readme_files(tree):
            if tree["type"] == "file":
                if tree["name"].lower().startswith("readme"):
                    readme_files.append(tree["path"])
            elif tree["type"] == "directory" and tree.get("children"):
                for child in tree["children"]:
                    find_readme_files(child)

        find_readme_files(result["file_tree"])

        print("Example: Finding README files in the repository")
        print(f"Found {len(readme_files)} README files:")
        for readme in readme_files:
            print(f"  {readme}")

        print("\n✅ Programmatic usage test PASSED")
        return True

    except Exception as e:
        print(f"❌ Programmatic usage test FAILED: {str(e)}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("GitProbe Repository Analyzer - Test Suite")
    print("=" * 60)

    tests = [
        test_basic_functionality,
        test_custom_ignore_patterns,
        test_file_tree_structure,
        test_url_parsing,
        test_programmatic_usage,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")

    if passed == total:
        print("🎉 All tests PASSED!")
        return 0
    else:
        print("⚠️  Some tests FAILED!")
        return 1


def demo_usage():
    """Demonstrate different ways to use the analyzer."""
    print("\n" + "=" * 60)
    print("USAGE DEMO")
    print("=" * 60)

    print("1. Basic usage:")
    print("   analyzer = RepoAnalyzer()")
    print("   result = analyzer.analyze_repository('https://github.com/user/repo')")

    print("\n2. With custom ignore patterns:")
    print("   analyzer = RepoAnalyzer(custom_ignore_patterns=['*.tmp', 'cache/'])")
    print("   result = analyzer.analyze_repository('https://github.com/user/repo')")

    print("\n3. With statistics:")
    print("   stats = analyzer.get_file_statistics(result['file_tree'])")

    print("\n4. Command line usage:")
    print("   python repo_analyzer.py https://github.com/user/repo --stats --pretty")
    print(
        "   python repo_analyzer.py https://github.com/user/repo --ignore '*.log' 'temp/' -o output.json"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test GitProbe Repository Analyzer")
    parser.add_argument("--demo", action="store_true", help="Show usage examples only")
    parser.add_argument(
        "--test", help="Run specific test (basic, custom, tree, url, programmatic)"
    )

    args = parser.parse_args()

    if args.demo:
        demo_usage()
        sys.exit(0)

    if args.test:
        test_map = {
            "basic": test_basic_functionality,
            "custom": test_custom_ignore_patterns,
            "tree": test_file_tree_structure,
            "url": test_url_parsing,
            "programmatic": test_programmatic_usage,
        }

        if args.test in test_map:
            result = test_map[args.test]()
            sys.exit(0 if result else 1)
        else:
            print(f"Unknown test: {args.test}")
            print(f"Available tests: {', '.join(test_map.keys())}")
            sys.exit(1)

    # Run all tests by default
    sys.exit(run_all_tests())
