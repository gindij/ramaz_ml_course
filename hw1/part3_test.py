#!/usr/bin/env python3
"""
Test interface for HW1 Part 3: Advanced Problems
"""

import sys
import os
import argparse

# Add parent directory to path to import tester
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tester.runner import run_tests, run_specific_functions

# Points for each function in Part 3
FUNCTION_POINTS = {
    "create_permutation_matrix": 3,
    "back_substitution": 4,
    "rotation_matrix_2d": 3,
    "scaling_matrix_2d": 3,
    "translation_matrix_2d": 3,
    "complex_transformation_challenge": 5,
}

ASSIGNMENT_NAME = "HW1 Part 3: Advanced Problems"


def main():
    parser = argparse.ArgumentParser(
        description="Test Advanced Problems implementations"
    )
    parser.add_argument(
        "--file",
        "-f",
        default="part3_advanced_problems",
        help="Python module to test (without .py extension)",
    )
    parser.add_argument(
        "functions",
        nargs="*",
        help="Specific functions to test (test all if none specified)",
    )

    args = parser.parse_args()

    test_cases_file = "part3_test_cases.json"

    if args.functions:
        run_specific_functions(
            test_cases_file, args.file, args.functions, FUNCTION_POINTS
        )
    else:
        run_tests(test_cases_file, args.file, FUNCTION_POINTS, ASSIGNMENT_NAME)


if __name__ == "__main__":
    main()
