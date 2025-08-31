#!/usr/bin/env python3
"""
Test interface for HW1 Part 1: Linear Algebra from Scratch

Usage:
    python part1_test.py                                    # Run all tests on part1_from_scratch.py
    python part1_test.py --file part1_from_scratch_solution # Test reference implementation 
    python part1_test.py func1 func2                        # Run specific functions
"""

import sys
import os
import argparse

# Add parent directory to path to import tester
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tester.runner import run_tests, run_specific_functions

# Points for each function in Part 1
FUNCTION_POINTS = {
    # Vector Operations
    "vector_add": 2,
    "scalar_multiply": 2,
    "dot_product": 3,
    "vector_magnitude": 2,
    "normalize_vector": 3,
    # Matrix Operations
    "matrix_add": 3,
    "matrix_vector_multiply": 3,
    "matrix_multiply": 4,
    "matrix_transpose": 3,
}

ASSIGNMENT_NAME = "HW1 Part 1: Linear Algebra from Scratch"


def main():
    parser = argparse.ArgumentParser(
        description="Test Linear Algebra Part 1 implementations"
    )
    parser.add_argument(
        "--file",
        "-f",
        default="part1_from_scratch",
        help="Python module to test (without .py extension)",
    )
    parser.add_argument(
        "functions",
        nargs="*",
        help="Specific functions to test (test all if none specified)",
    )

    args = parser.parse_args()

    test_cases_file = "part1_test_cases.json"

    if args.functions:
        # Run specific functions
        run_specific_functions(
            test_cases_file, args.file, args.functions, FUNCTION_POINTS
        )
    else:
        # Run all tests
        run_tests(test_cases_file, args.file, FUNCTION_POINTS, ASSIGNMENT_NAME)


if __name__ == "__main__":
    main()
