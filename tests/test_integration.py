#!/usr/bin/env python3
"""
GitProbe Integration Test Suite

Comprehensive integration tests for all GitProbe tree-sitter language analyzers.
Tests real-world repositories to ensure all language parsers are working correctly.

Usage:
    python tests/test_integration.py                     # Run all tests
    python tests/test_integration.py --language python   # Test specific language
    python tests/test_integration.py --quick             # Run quick subset
    python tests/test_integration.py --verbose           # Detailed output

Author: GitProbe Team
License: MIT
"""

import argparse
import json
import sys
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

try:
    import requests
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    from rich.text import Text
except ImportError:
    print("❌ Missing dependencies. Install with:")
    print("   pip install requests rich")
    sys.exit(1)


@dataclass
class TestResult:
    """Result of a single repository test."""
    repo_name: str
    language: str
    success: bool
    functions: int
    calls: int
    error: Optional[str] = None
    duration: float = 0.0


class GitProbeIntegrationTests:
    """Main integration test runner for GitProbe analyzers."""
    
    # Curated test repositories by language
    TEST_REPOSITORIES = {
        "Python": [
            ("https://github.com/Textualize/rich", "Modern terminal formatting"),
            ("https://github.com/psf/requests", "HTTP library for humans"), 
            ("https://github.com/pallets/flask", "Lightweight web framework"),
            ("https://github.com/python/cpython", "Python interpreter (large)"),
        ],
        "JavaScript": [
            ("https://github.com/lodash/lodash", "Modern utility library"),
            ("https://github.com/axios/axios", "Promise-based HTTP client"),
            ("https://github.com/expressjs/express", "Fast web framework"),
            ("https://github.com/nodejs/node", "Node.js runtime (large)"),
        ],
        "TypeScript": [
            ("https://github.com/microsoft/vscode", "Code editor (large)"),
            ("https://github.com/microsoft/TypeScript", "TypeScript compiler"),
            ("https://github.com/angular/angular", "Angular framework (large)"),
        ],
        "Rust": [
            ("https://github.com/clap-rs/clap", "Command line parser"),
            ("https://github.com/BurntSushi/ripgrep", "Fast grep alternative"),
            ("https://github.com/rust-lang/rust", "Rust compiler (very large)"),
        ],
        "Go": [
            ("https://github.com/spf13/cobra", "CLI library"),
            ("https://github.com/gohugoio/hugo", "Static site generator"),
            ("https://github.com/kubernetes/kubernetes", "Container orchestration (very large)"),
        ],
        "C": [
            ("https://github.com/DaveGamble/cJSON", "JSON parser in C"),
            ("https://github.com/libuv/libuv", "Cross-platform async I/O"),
            ("https://github.com/curl/curl", "Data transfer library"),
        ],
        "C++": [
            ("https://github.com/fmtlib/fmt", "Modern formatting library"),
            ("https://github.com/catchorg/Catch2", "Modern test framework"),
            ("https://github.com/protocolbuffers/protobuf", "Protocol buffers"),
        ]
    }
    
    # Quick subset for fast testing
    QUICK_TEST_SET = {
        "Python": [("https://github.com/psf/requests", "HTTP library")],
        "JavaScript": [("https://github.com/axios/axios", "HTTP client")],
        "Rust": [("https://github.com/clap-rs/clap", "CLI parser")],
        "Go": [("https://github.com/spf13/cobra", "CLI library")],
        "C": [("https://github.com/DaveGamble/cJSON", "JSON parser")],
        "C++": [("https://github.com/fmtlib/fmt", "Formatting library")],
    }

    def __init__(self, server_url: str = "http://localhost:8000", timeout: int = 120):
        """Initialize test runner."""
        self.server_url = server_url
        self.timeout = timeout
        self.console = Console()
        self.results: List[TestResult] = []

    def check_server_health(self) -> bool:
        """Check if GitProbe server is running and healthy."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def test_repository(self, repo_url: str, language: str, description: str = "") -> TestResult:
        """Test analysis of a single repository."""
        repo_name = repo_url.split('/')[-1]
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.server_url}/analyze",
                json={"github_url": repo_url},
                timeout=self.timeout
            )
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                summary = data.get("data", {}).get("summary", {})
                
                functions = summary.get("total_functions", 0)
                calls = summary.get("total_calls", 0)
                
                # Consider success if we found functions and no errors
                has_errors = "error" in data.get("status", "").lower()
                success = functions > 0 and not has_errors
                
                return TestResult(
                    repo_name=repo_name,
                    language=language,
                    success=success,
                    functions=functions,
                    calls=calls,
                    duration=duration
                )
            else:
                return TestResult(
                    repo_name=repo_name,
                    language=language,
                    success=False,
                    functions=0,
                    calls=0,
                    error=f"HTTP {response.status_code}",
                    duration=duration
                )
                
        except requests.exceptions.Timeout:
            return TestResult(
                repo_name=repo_name,
                language=language,
                success=False,
                functions=0,
                calls=0,
                error="Timeout",
                duration=self.timeout
            )
        except Exception as e:
            return TestResult(
                repo_name=repo_name,
                language=language,
                success=False,
                functions=0,
                calls=0,
                error=str(e),
                duration=time.time() - start_time
            )

    def run_tests(self, languages: Optional[List[str]] = None, quick: bool = False, verbose: bool = False) -> Dict:
        """Run integration tests and return detailed results."""
        
        # Check server health first
        if not self.check_server_health():
            self.console.print("❌ [red]GitProbe server is not running or unhealthy[/red]")
            self.console.print("   Start server with: [cyan]./gitprobe server[/cyan]")
            return {"error": "Server not available"}

        # Select test set
        test_set = self.QUICK_TEST_SET if quick else self.TEST_REPOSITORIES
        
        # Filter by languages if specified
        if languages:
            test_set = {lang: repos for lang, repos in test_set.items() 
                       if lang.lower() in [l.lower() for l in languages]}

        if not test_set:
            self.console.print("❌ [red]No tests to run with current filters[/red]")
            return {"error": "No tests selected"}

        # Display test plan
        total_tests = sum(len(repos) for repos in test_set.values())
        self.console.print(Panel(
            f"🧪 [bold blue]GitProbe Integration Test Suite[/bold blue]\n\n"
            f"Testing {len(test_set)} languages, {total_tests} repositories\n"
            f"Server: {self.server_url}\n"
            f"Timeout: {self.timeout}s per repository",
            title="Test Configuration"
        ))

        # Run tests with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            for language, repos in test_set.items():
                lang_task = progress.add_task(f"Testing {language}...", total=len(repos))
                
                for repo_url, description in repos:
                    repo_name = repo_url.split('/')[-1]
                    progress.update(lang_task, description=f"Testing {language}: {repo_name}")
                    
                    result = self.test_repository(repo_url, language, description)
                    self.results.append(result)
                    
                    if verbose:
                        status = "✅" if result.success else "❌"
                        details = f"({result.functions} functions, {result.calls} calls, {result.duration:.1f}s)"
                        if result.error:
                            details = f"Error: {result.error}"
                        self.console.print(f"  {status} {repo_name}: {details}")
                    
                    progress.advance(lang_task)

        return self._generate_report()

    def _generate_report(self) -> Dict:
        """Generate comprehensive test report."""
        # Calculate statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        
        # Group by language
        by_language = {}
        for result in self.results:
            if result.language not in by_language:
                by_language[result.language] = []
            by_language[result.language].append(result)

        # Create summary table
        table = Table(title="📊 Test Results Summary")
        table.add_column("Language", style="cyan", no_wrap=True)
        table.add_column("Passed", style="green", justify="center")
        table.add_column("Failed", style="red", justify="center")
        table.add_column("Success Rate", justify="center")
        table.add_column("Avg Functions", justify="right")
        table.add_column("Total Duration", justify="right")

        overall_success = True
        total_duration = 0

        for language, results in by_language.items():
            passed = sum(1 for r in results if r.success)
            total = len(results)
            failed = total - passed
            success_rate = (passed / total * 100) if total > 0 else 0
            avg_functions = sum(r.functions for r in results if r.success) / max(passed, 1)
            lang_duration = sum(r.duration for r in results)
            total_duration += lang_duration
            
            if failed > 0:
                overall_success = False
            
            status_style = "green" if failed == 0 else "yellow" if passed > 0 else "red"
            table.add_row(
                f"[{status_style}]{language}[/{status_style}]",
                str(passed),
                str(failed),
                f"{success_rate:.0f}%",
                f"{avg_functions:.0f}" if passed > 0 else "0",
                f"{lang_duration:.1f}s"
            )

        self.console.print("\n")
        self.console.print(table)

        # Overall status
        if overall_success:
            self.console.print("\n🎉 [bold green]All analyzers working perfectly![/bold green]")
        elif passed_tests > 0:
            self.console.print(f"\n⚠️  [yellow]Partial success: {passed_tests}/{total_tests} tests passed[/yellow]")
        else:
            self.console.print(f"\n❌ [red]All tests failed - check GitProbe server[/red]")

        # Show failures if any
        failures = [r for r in self.results if not r.success]
        if failures:
            self.console.print(f"\n[red]Failed Tests ({len(failures)}):[/red]")
            for failure in failures:
                error_msg = failure.error or "No functions detected"
                self.console.print(f"  ❌ {failure.language}/{failure.repo_name}: {error_msg}")

        return {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "overall_success": overall_success,
            "duration": total_duration,
            "by_language": {
                lang: {
                    "passed": sum(1 for r in results if r.success),
                    "total": len(results),
                    "results": [
                        {
                            "repo": r.repo_name,
                            "success": r.success,
                            "functions": r.functions,
                            "calls": r.calls,
                            "error": r.error,
                            "duration": r.duration
                        }
                        for r in results
                    ]
                }
                for lang, results in by_language.items()
            }
        }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GitProbe Integration Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/test_integration.py                     # Run all tests
  python tests/test_integration.py --quick             # Quick test subset
  python tests/test_integration.py --language python   # Test Python only
  python tests/test_integration.py --language python --language rust  # Multiple languages
  python tests/test_integration.py --verbose           # Detailed output
  python tests/test_integration.py --server http://localhost:9000  # Custom server
        """
    )
    
    parser.add_argument(
        "--language", 
        action="append", 
        help="Test specific language(s) only (can be used multiple times)"
    )
    parser.add_argument(
        "--quick", 
        action="store_true", 
        help="Run quick test subset (1 repo per language)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Show detailed test progress"
    )
    parser.add_argument(
        "--server", 
        default="http://localhost:8000",
        help="GitProbe server URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=120,
        help="Request timeout in seconds (default: 120)"
    )
    parser.add_argument(
        "--json", 
        action="store_true", 
        help="Output results as JSON"
    )

    args = parser.parse_args()

    # Run tests
    runner = GitProbeIntegrationTests(server_url=args.server, timeout=args.timeout)
    report = runner.run_tests(
        languages=args.language,
        quick=args.quick,
        verbose=args.verbose
    )

    # Output results
    if args.json:
        print(json.dumps(report, indent=2))
    
    # Exit with error code if tests failed
    if "error" in report:
        sys.exit(1)
    elif not report.get("overall_success", False):
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main() 