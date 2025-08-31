#!/usr/bin/env python3
"""
Test interface for HW1 Part 2: NumPy Essentials
"""

import sys
import os
import argparse

# Add parent directory to path to import tester
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tester.runner import run_tests, run_specific_functions

# Points for each function in Part 2
FUNCTION_POINTS = {
    "create_zeros_array": 2,
    "create_scaled_identity_matrix": 3,
    "create_range_matrix": 2,
    "get_diagonal": 2,
    "get_submatrix": 3,
    "matrix_plus_row_vector": 3,
    "matrix_plus_column_vector": 3,
    "conditional_values": 3,
    "numpy_matrix_multiply": 3,
    "compute_row_means": 2,
    "normalize_features": 4,
    "find_max_indices": 2,
    "stack_arrays": 3,
    "filter_positive_values": 2,
}

ASSIGNMENT_NAME = "HW1 Part 2: NumPy Essentials"


def main():
    parser = argparse.ArgumentParser(description='Test NumPy Essentials implementations')
    parser.add_argument('--file', '-f', default='part2_numpy_essentials',
                       help='Python module to test (without .py extension)')
    parser.add_argument('functions', nargs='*',
                       help='Specific functions to test (test all if none specified)')

    args = parser.parse_args()

    test_cases_file = "part2_test_cases.json"

    if args.functions:
        run_specific_functions(test_cases_file, args.file, args.functions, FUNCTION_POINTS)
    else:
        run_tests(test_cases_file, args.file, FUNCTION_POINTS, ASSIGNMENT_NAME)


if __name__ == "__main__":
    main()