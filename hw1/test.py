#!/usr/bin/env python3
"""
Test interface for HW1: Linear Algebra Assignment

Usage:
    python test.py                                    # Run all tests for all parts
    python test.py --part 1                          # Run tests for Part 1 only
    python test.py --part 2                          # Run tests for Part 2 only  
    python test.py --part 3                          # Run tests for Part 3 only
    python test.py --file part1_from_scratch         # Test specific file for all parts
    python test.py --part 1 --file part1_solution    # Test specific file for Part 1
    python test.py vector_add matrix_multiply        # Run specific functions (auto-detect part)
"""

import sys
import os
import argparse

# Add parent directory to path to import tester
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tester.runner import run_tests, run_specific_functions

# Points for each function by part
PART1_FUNCTION_POINTS = {
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

PART2_FUNCTION_POINTS = {
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

PART3_FUNCTION_POINTS = {
    "create_permutation_matrix": 3,
    "back_substitution": 4,
    "rotation_matrix_2d": 3,
    "scaling_matrix_2d": 3,
    "translation_matrix_2d": 3,
    "complex_transformation_challenge": 5,
}

ALL_FUNCTION_POINTS = {
    **PART1_FUNCTION_POINTS,
    **PART2_FUNCTION_POINTS,
    **PART3_FUNCTION_POINTS,
}

PART_CONFIG = {
    1: {
        "functions": PART1_FUNCTION_POINTS,
        "test_cases": "part1_test_cases.json",
        "default_file": "part1_from_scratch",
        "name": "HW1 Part 1: Linear Algebra from Scratch",
    },
    2: {
        "functions": PART2_FUNCTION_POINTS,
        "test_cases": "part2_test_cases.json",
        "default_file": "part2_numpy_essentials",
        "name": "HW1 Part 2: NumPy Essentials",
    },
    3: {
        "functions": PART3_FUNCTION_POINTS,
        "test_cases": "part3_test_cases.json",
        "default_file": "part3_advanced_problems",
        "name": "HW1 Part 3: Advanced Problems",
    },
}


def detect_part_from_functions(functions):
    """Detect which part(s) contain the given functions."""
    parts = []
    for part_num, config in PART_CONFIG.items():
        if any(func in config["functions"] for func in functions):
            parts.append(part_num)
    return parts


def run_part(part_num, file_name=None, specific_functions=None):
    """Run tests for a specific part."""
    config = PART_CONFIG[part_num]

    if file_name is None:
        file_name = config["default_file"]

    if specific_functions:
        # Filter functions that belong to this part
        part_functions = [f for f in specific_functions if f in config["functions"]]
        if part_functions:
            run_specific_functions(
                config["test_cases"], file_name, part_functions, config["functions"]
            )
    else:
        run_tests(config["test_cases"], file_name, config["functions"], config["name"])


def main():
    parser = argparse.ArgumentParser(
        description="Test HW1 Linear Algebra implementations"
    )
    parser.add_argument(
        "--part",
        "-p",
        type=int,
        choices=[1, 2, 3],
        help="Which part to test (1, 2, or 3). If not specified, test all parts.",
    )
    parser.add_argument(
        "--file",
        "-f",
        help="Python module to test (without .py extension). If not specified, uses default file for each part.",
    )
    parser.add_argument(
        "functions",
        nargs="*",
        help="Specific functions to test. If not specified, test all functions.",
    )

    args = parser.parse_args()

    if args.functions:
        # Specific functions specified
        if args.part:
            # Test specific functions in specific part
            run_part(args.part, args.file, args.functions)
        else:
            # Auto-detect which parts contain these functions
            parts = detect_part_from_functions(args.functions)
            if not parts:
                print(f"Error: No functions found matching: {args.functions}")
                sys.exit(1)

            for part_num in sorted(parts):
                run_part(part_num, args.file, args.functions)
    elif args.part:
        # Test specific part, all functions
        run_part(args.part, args.file)
    else:
        # Test all parts
        for part_num in [1, 2, 3]:
            print()  # Add spacing between parts
            run_part(part_num, args.file)


if __name__ == "__main__":
    main()
