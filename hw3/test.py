#!/usr/bin/env python3
"""
Test interface for HW3: Linear and Logistic Regression from Scratch

Usage:
    python test.py                          # Run all tests
    python test.py --part utilities         # Test only utilities.py
    python test.py --part linear           # Test only linear_regression.py  
    python test.py --part logistic         # Test only logistic_regression.py
    python test.py --file utilities_solution  # Test reference implementation
    python test.py standardize_features sigmoid linear_gradient  # Test specific functions
"""

import sys
import os
import argparse

# Add parent directory to path to import tester
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tester.runner import run_tests, run_specific_functions

# Points for each function
FUNCTION_POINTS = {
    # utilities.py (20 points)
    "standardize_features": 3,
    "add_intercept": 2,
    "train_test_split": 3,
    "accuracy_score": 2,
    "mean_squared_error": 2,
    "gradient_descent": 5,  # PROVIDED
    "k_fold_split": 2,
    "compute_classification_metrics": 1,
    # linear_regression.py (22 points)
    "compute_cost": 2,
    "linear_gradient": 4,
    "linear_gradient_descent": 3,
    "predict": 4,
    "polynomial_features": 3,
    "fit_polynomial_regression": 3,
    "r_squared": 2,
    "cross_validate_linear_regression": 1,
    # logistic_regression.py (30 points)
    "sigmoid": 2,
    "sigmoid_derivative": 2,
    "compute_logistic_cost": 3,
    "logistic_cost_from_predictions": 2,
    "logistic_gradient": 5,
    "logistic_gradient_descent": 3,
    "predict_proba": 4,
    "predict_classes": 3,
    "cross_validate_logistic_regression": 6,
}

ASSIGNMENT_NAME = "HW3: Linear and Logistic Regression from Scratch"

# File groupings for --part argument
PARTS = {
    "utilities": [
        "standardize_features",
        "add_intercept",
        "train_test_split",
        "accuracy_score",
        "mean_squared_error",
        "gradient_descent",
        "k_fold_split",
        "compute_classification_metrics",
    ],
    "linear": [
        "compute_cost",
        "linear_gradient",
        "linear_gradient_descent",
        "predict",
        "polynomial_features",
        "fit_polynomial_regression",
        "r_squared",
        "cross_validate_linear_regression",
    ],
    "logistic": [
        "sigmoid",
        "sigmoid_derivative",
        "compute_logistic_cost",
        "logistic_cost_from_predictions",
        "logistic_gradient",
        "logistic_gradient_descent",
        "predict_proba",
        "predict_classes",
        "cross_validate_logistic_regression",
    ],
}


def main():
    parser = argparse.ArgumentParser(
        description="Test Linear and Logistic Regression implementations"
    )
    parser.add_argument(
        "--file",
        "-f",
        help="Python module to test (without .py extension)",
    )
    parser.add_argument(
        "--part",
        "-p",
        choices=["utilities", "linear", "logistic"],
        help="Test only functions from specific part",
    )
    parser.add_argument(
        "functions",
        nargs="*",
        help="Specific functions to test (test all if none specified)",
    )

    args = parser.parse_args()

    test_cases_file = "test_cases.json"

    if args.functions:
        # Run specific functions - determine which file(s) to test
        files_to_test = set()
        for func in args.functions:
            if func in PARTS["utilities"]:
                files_to_test.add("utilities")
            elif func in PARTS["linear"]:
                files_to_test.add("linear_regression")
            elif func in PARTS["logistic"]:
                files_to_test.add("logistic_regression")

        for file in files_to_test:
            relevant_functions = [
                f
                for f in args.functions
                if f
                in PARTS.get(
                    (
                        "utilities"
                        if file == "utilities"
                        else "linear" if file == "linear_regression" else "logistic"
                    ),
                    [],
                )
            ]
            if relevant_functions:
                run_specific_functions(
                    test_cases_file, file, relevant_functions, FUNCTION_POINTS
                )

    elif args.part:
        # Run tests for specific part
        if args.part == "utilities":
            run_tests(test_cases_file, "utilities", FUNCTION_POINTS, ASSIGNMENT_NAME)
        elif args.part == "linear":
            run_tests(
                test_cases_file, "linear_regression", FUNCTION_POINTS, ASSIGNMENT_NAME
            )
        elif args.part == "logistic":
            run_tests(
                test_cases_file, "logistic_regression", FUNCTION_POINTS, ASSIGNMENT_NAME
            )

    elif args.file:
        # Run tests for specific file
        run_tests(test_cases_file, args.file, FUNCTION_POINTS, ASSIGNMENT_NAME)

    else:
        # Run all tests for all three files
        print(f"*** {ASSIGNMENT_NAME.upper()} - TEST RESULTS")
        print("=" * len(ASSIGNMENT_NAME) * 2)
        print()

        total_points = 0
        max_points = 0

        for part, file in [
            ("utilities", "utilities"),
            ("linear", "linear_regression"),
            ("logistic", "logistic_regression"),
        ]:
            print(
                f"{part.upper()}.PY ({sum(FUNCTION_POINTS[f] for f in PARTS[part])} points):"
            )
            points_earned, points_possible = run_tests(
                test_cases_file, file, FUNCTION_POINTS, "", return_scores=True
            )
            total_points += points_earned
            max_points += points_possible
            print()

        percentage = (total_points / max_points) * 100 if max_points > 0 else 0
        print(
            f"OVERALL FINAL SCORE: {total_points}/{max_points} points ({percentage:.1f}%)"
        )
        print("=" * len(ASSIGNMENT_NAME) * 2)


if __name__ == "__main__":
    main()
