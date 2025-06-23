# GitProbe Integration Tests

Comprehensive integration test suite for GitProbe's tree-sitter language analyzers. Tests real-world repositories to ensure all language parsers are working correctly.

## 🚀 Quick Start

```bash
# Install dependencies
pip install requests rich

# Start GitProbe server (in another terminal)
./gitprobe server

# Run all tests
python tests/test_integration.py

# Run quick subset (1 repo per language)
python tests/test_integration.py --quick

# Test specific language
python tests/test_integration.py --language python

# Verbose output with detailed progress
python tests/test_integration.py --verbose
```

## 📋 Test Coverage

The integration tests cover **7 languages** with carefully curated real-world repositories:

### Supported Languages
- **Python** - 4 repositories (rich, requests, flask, cpython)
- **JavaScript** - 4 repositories (lodash, axios, express, node)
- **TypeScript** - 3 repositories (vscode, TypeScript, angular)
- **Rust** - 3 repositories (clap, ripgrep, rust)
- **Go** - 3 repositories (cobra, hugo, kubernetes)
- **C** - 3 repositories (cJSON, libuv, curl)
- **C++** - 3 repositories (fmt, Catch2, protobuf)

### Test Repository Selection Criteria
- **Real-world usage**: Popular, actively maintained projects
- **Diverse complexity**: From small libraries to large frameworks
- **Language features**: Covers different language patterns and idioms
- **Performance testing**: Includes large repositories to test scaling

## 🛠️ Usage Examples

### Basic Testing

```bash
# Test all languages with all repositories (~25 repositories)
python tests/test_integration.py

# Quick test with 1 repository per language (6 repositories)
python tests/test_integration.py --quick
```

### Language-Specific Testing

```bash
# Test only Python repositories
python tests/test_integration.py --language python

# Test multiple specific languages
python tests/test_integration.py --language python --language rust

# Test C/C++ analyzers
python tests/test_integration.py --language c --language c++
```

### Advanced Options

```bash
# Verbose output showing each test result
python tests/test_integration.py --verbose

# Custom server URL
python tests/test_integration.py --server http://localhost:9000

# Longer timeout for large repositories
python tests/test_integration.py --timeout 300

# JSON output for CI/CD integration
python tests/test_integration.py --json > test_results.json
```

## 📊 Output Formats

### Standard Output
Beautiful terminal output with:
- Progress indicators with spinners
- Colored summary table by language
- Success/failure statistics
- Performance metrics (functions found, duration)
- Error details for failed tests

### JSON Output
Structured data perfect for CI/CD integration:
```json
{
  "total_tests": 17,
  "passed": 17,
  "failed": 0,
  "success_rate": 100.0,
  "overall_success": true,
  "duration": 125.3,
  "by_language": {
    "Python": {
      "passed": 3,
      "total": 3,
      "results": [...]
    }
  }
}
```

## 🔧 Configuration

### Environment Requirements
- **GitProbe server**: Must be running on specified URL (default: `http://localhost:8000`)
- **Dependencies**: `requests` and `rich` packages
- **Network access**: Required for cloning public GitHub repositories
- **Disk space**: Temporary clones are created and cleaned up automatically

### Timeout Settings
- **Default**: 120 seconds per repository
- **Large repos**: Consider increasing to 300+ seconds for repositories like kubernetes or rust
- **Quick tests**: Usually complete in 30-60 seconds

### Server Health Check
The test suite automatically:
1. Checks if GitProbe server is running
2. Validates server health endpoint
3. Provides clear error messages if server is unavailable

## 🎯 Test Success Criteria

A repository test is considered **successful** if:
- ✅ HTTP 200 response from GitProbe API
- ✅ At least 1 function detected in the codebase
- ✅ No error status in the response
- ✅ Analysis completes within timeout period

## 🔍 Troubleshooting

### Common Issues

**Server not running:**
```
❌ GitProbe server is not running or unhealthy
   Start server with: ./gitprobe server
```
*Solution*: Start GitProbe server in another terminal

**Timeout errors:**
```
❌ rust/rust: Timeout
```
*Solution*: Increase timeout with `--timeout 300` for large repositories

**No functions detected:**
```
❌ python/someproject: No functions detected
```
*Possible causes*:
- Repository has no supported files
- Tree-sitter parser failed to initialize
- Repository structure not recognized

**Network issues:**
```
❌ python/requests: HTTP 500
```
*Solution*: Check internet connection and GitHub API limits

### Debug Mode

For detailed debugging, combine flags:
```bash
python tests/test_integration.py --verbose --language python --timeout 300
```

## 🚦 CI/CD Integration

Perfect for continuous integration pipelines:

```yaml
# GitHub Actions example
- name: Run GitProbe Integration Tests
  run: |
    ./gitprobe server &
    sleep 10  # Wait for server startup
    python tests/test_integration.py --quick --json > results.json
    
- name: Check Test Results
  run: |
    if jq -e '.overall_success == false' results.json; then
      echo "Tests failed"
      exit 1
    fi
```

## 🏗️ Architecture

### Test Structure
- **TestResult**: Dataclass for individual repository results
- **GitProbeIntegrationTests**: Main test runner class
- **Progress tracking**: Real-time progress with rich library
- **Error handling**: Comprehensive timeout and exception handling

### Repository Management
- Repositories are cloned by GitProbe server
- Temporary directories are automatically cleaned up
- No local storage required for test suite

### Extensibility
- Easy to add new test repositories
- Simple language addition process
- Configurable test sets (quick vs. comprehensive)

## 📈 Performance Benchmarks

Typical execution times on modern hardware:

| Test Set | Repositories | Duration | Use Case |
|----------|-------------|----------|----------|
| Quick | 6 repos | 30-60s | Development, quick validation |
| Full | ~25 repos | 5-15min | CI/CD, comprehensive testing |
| Single Language | 3-4 repos | 1-3min | Language-specific debugging |

## 🤝 Contributing

To add new test repositories:

1. Add to appropriate language section in `TEST_REPOSITORIES`
2. Include description for context
3. Test with `--language <your_language>` first
4. Consider adding to `QUICK_TEST_SET` if it's a good representative

Example:
```python
"Python": [
    ("https://github.com/new/repository", "Description of what it tests"),
    # ... existing repos
]
``` 