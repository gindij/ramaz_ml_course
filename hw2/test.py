#!/usr/bin/env python3
"""
Test interface for HW2: K-Nearest Neighbors from Scratch

Usage:
    python test_simple.py                         # Run all tests on knn_from_scratch.py
    python test_simple.py --file reference_solution # Test reference implementation
    python test_simple.py func1 func2             # Run specific functions
    python test_simple.py --file reference_solution func1 func2  # Test specific functions in specific file
"""

import sys
import os
import argparse

# Add parent directory to path to import tester
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tester.runner import run_tests, run_specific_functions

# Points for each function
FUNCTION_POINTS = {
    # Distance Metrics
    "euclidean_distance": 2,
    "manhattan_distance": 2,
    "minkowski_distance": 3,
    # Neighbor Finding
    "find_k_nearest_neighbors": 4,
    "find_neighbors_within_radius": 3,
    # Classification
    "majority_vote": 2,
    "weighted_majority_vote": 3,
    "knn_classify": 4,
    # Regression
    "mean_prediction": 2,
    "weighted_mean_prediction": 3,
    "knn_regress": 4,
    # Model Validation
    "train_test_split": 3,
    "calculate_accuracy": 2,
    "mean_squared_error": 2,
    "cross_validate_knn": 5,
}

ASSIGNMENT_NAME = "HW2: K-Nearest Neighbors"


def main():
    parser = argparse.ArgumentParser(description="Test K-NN implementations")
    parser.add_argument(
        "--file",
        "-f",
        default="knn_from_scratch",
        help="Python module to test (without .py extension)",
    )
    parser.add_argument(
        "functions",
        nargs="*",
        help="Specific functions to test (test all if none specified)",
    )

    args = parser.parse_args()

    test_cases_file = "test_cases.json"

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
