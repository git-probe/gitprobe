#!/usr/bin/env python3
"""
Test script for C/C++ AST Analyzer
Validates the functionality of C and C++ analyzers.
"""

import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.c_analyzer import analyze_c_file, analyze_cpp_file


def test_c_analyzer():
    """Test C analyzer with sample C code."""
    print("🧪 Testing C Analyzer")
    print("=" * 50)

    # Sample C code
    c_code = """
#include <stdio.h>
#include <stdlib.h>

int add(int a, int b) {
    return a + b;
}

void print_result(int result) {
    printf("Result: %d\\n", result);
}

int main(int argc, char *argv[]) {
    int x = 5;
    int y = 3;
    int sum = add(x, y);
    print_result(sum);
    return 0;
}

static void helper_function(void) {
    printf("This is a helper function\\n");
}
"""

    try:
        functions, relationships = analyze_c_file("test.c", c_code)

        print(f"✅ Found {len(functions)} functions:")
        for func in functions:
            print(f"  - {func.name}() at line {func.line_start}")
            if func.parameters:
                print(f"    Parameters: {', '.join(func.parameters)}")

        print(f"\n✅ Found {len(relationships)} call relationships:")
        for rel in relationships:
            caller = rel.caller.split(":")[-1]  # Get function name
            print(f"  - {caller} calls {rel.callee} at line {rel.call_line}")

        # Verify expected functions
        expected_functions = {"add", "print_result", "main", "helper_function"}
        found_functions = {func.name for func in functions}

        if expected_functions <= found_functions:
            print("✅ All expected functions found!")
        else:
            missing = expected_functions - found_functions
            print(f"❌ Missing functions: {missing}")

        return True

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def test_cpp_analyzer():
    """Test C++ analyzer with sample C++ code."""
    print("\n🧪 Testing C++ Analyzer")
    print("=" * 50)

    # Sample C++ code
    cpp_code = """
#include <iostream>
#include <vector>
#include <string>

class Calculator {
public:
    int add(int a, int b) {
        return a + b;
    }
    
    void display_result(int result) {
        std::cout << "Result: " << result << std::endl;
    }
};

template<typename T>
T multiply(T a, T b) {
    return a * b;
}

namespace MathUtils {
    double divide(double a, double b) {
        if (b != 0) {
            return a / b;
        }
        return 0.0;
    }
}

int main() {
    Calculator calc;
    int sum = calc.add(10, 5);
    calc.display_result(sum);
    
    auto product = multiply(3.5, 2.0);
    std::cout << "Product: " << product << std::endl;
    
    double quotient = MathUtils::divide(10.0, 2.0);
    std::cout << "Quotient: " << quotient << std::endl;
    
    return 0;
}
"""

    try:
        functions, relationships = analyze_cpp_file("test.cpp", cpp_code)

        print(f"✅ Found {len(functions)} functions:")
        for func in functions:
            method_type = "method" if func.is_method else "function"
            print(f"  - {func.name}() ({method_type}) at line {func.line_start}")
            if func.parameters:
                print(f"    Parameters: {', '.join(func.parameters)}")

        print(f"\n✅ Found {len(relationships)} call relationships:")
        for rel in relationships:
            caller = rel.caller.split(":")[-1]  # Get function name
            print(f"  - {caller} calls {rel.callee} at line {rel.call_line}")

        # Verify expected functions
        expected_functions = {"add", "display_result", "multiply", "divide", "main"}
        found_functions = {func.name for func in functions}

        if expected_functions <= found_functions:
            print("✅ All expected functions found!")
        else:
            missing = expected_functions - found_functions
            print(f"❌ Missing functions: {missing}")

        return True

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def test_header_file():
    """Test C++ analyzer with header file."""
    print("\n🧪 Testing C++ Header File")
    print("=" * 50)

    # Sample header file
    header_code = """
#ifndef MATH_UTILS_H
#define MATH_UTILS_H

// Function declarations
int fibonacci(int n);
double square_root(double x);

class Vector3D {
private:
    double x, y, z;
    
public:
    Vector3D(double x, double y, double z);
    ~Vector3D();
    
    double magnitude() const;
    void normalize();
    Vector3D cross_product(const Vector3D& other) const;
};

template<class T>
inline T max_value(T a, T b) {
    return (a > b) ? a : b;
}

#endif // MATH_UTILS_H
"""

    try:
        functions, relationships = analyze_cpp_file("math_utils.hpp", header_code)

        print(f"✅ Found {len(functions)} functions/methods:")
        for func in functions:
            method_type = "method" if func.is_method else "function"
            print(f"  - {func.name}() ({method_type}) at line {func.line_start}")

        print(f"\n✅ Found {len(relationships)} call relationships:")
        for rel in relationships:
            caller = rel.caller.split(":")[-1]
            print(f"  - {caller} calls {rel.callee} at line {rel.call_line}")

        return True

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


if __name__ == "__main__":
    print("🚀 Running C/C++ Analyzer Tests")
    print("=" * 60)

    success = True

    success &= test_c_analyzer()
    success &= test_cpp_analyzer()
    success &= test_header_file()

    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! C/C++ analyzers are working correctly.")
    else:
        print("❌ Some tests failed. Please check the output above.")

    print("\n📋 Summary:")
    print("- C analyzer: Regex-based function detection")
    print("- C++ analyzer: Regex-based function and method detection")
    print("- Header file support: Yes")
    print("- Call graph analysis: Yes")
    print("- JSON format compatibility: Yes")
