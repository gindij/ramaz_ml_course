#!/usr/bin/env python3
"""
Test interface for HW5: Bias-Variance Tradeoff and Ensemble Methods

Usage:
    python test.py                                    # Run all tests
    python test.py --file data_utils                 # Test specific file
    python test.py --file data_utils_solution        # Test solution file
    python test.py load_assignment_dataset           # Test specific function
"""

import sys
import os
import argparse

# Add parent directory to path to import tester
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tester.runner import run_tests, run_specific_functions

# Points for each function
DATA_UTILS_POINTS = {
    "load_assignment_dataset": 5,
    "get_dataset_info": 3,
}

METRICS_UTILS_POINTS = {
    "calculate_classification_metrics": 6,
    "get_roc_data": 5,
    "get_confusion_matrix": 4,
    "compare_model_metrics": 7,
}

TREE_UTILS_POINTS = {
    "train_trees_by_depth": 8,
    "bootstrap_sample_predictions": 8,
    "train_ensemble_models": 6,
    "get_complexity_curve": 8,
    "compare_ensemble_sizes": 6,
}

ALL_FUNCTION_POINTS = {
    **DATA_UTILS_POINTS,
    **METRICS_UTILS_POINTS,
    **TREE_UTILS_POINTS,
}

MODULE_CONFIG = {
    "data_utils": {
        "functions": DATA_UTILS_POINTS,
        "test_cases": "data_utils_test_cases.json",
        "name": "HW5: Data Utilities",
    },
    "metrics_utils": {
        "functions": METRICS_UTILS_POINTS,
        "test_cases": "metrics_utils_test_cases.json",
        "name": "HW5: Metrics Utilities",
    },
    "tree_utils": {
        "functions": TREE_UTILS_POINTS,
        "test_cases": "tree_utils_test_cases.json",
        "name": "HW5: Tree Utilities",
    },
}


def detect_module_from_functions(functions):
    """Detect which modules contain the given functions."""
    modules = []
    for module_name, config in MODULE_CONFIG.items():
        if any(func in config["functions"] for func in functions):
            modules.append(module_name)
    return modules


def run_module(module_name, file_name=None, specific_functions=None):
    """Run tests for a specific module."""
    config = MODULE_CONFIG[module_name]

    if file_name is None:
        file_name = module_name

    if specific_functions:
        # Filter functions that belong to this module
        module_functions = [f for f in specific_functions if f in config["functions"]]
        if module_functions:
            run_specific_functions(
                config["test_cases"], file_name, module_functions, config["functions"]
            )
    else:
        run_tests(config["test_cases"], file_name, config["functions"], config["name"])


def main():
    parser = argparse.ArgumentParser(
        description="Test HW5 Bias-Variance Tradeoff implementations"
    )
    parser.add_argument(
        "--module",
        "-m",
        choices=["data_utils", "metrics_utils", "tree_utils"],
        help="Which module to test. If not specified, test all modules.",
    )
    parser.add_argument(
        "--file",
        "-f",
        help="Python module to test (without .py extension). If not specified, uses module name.",
    )
    parser.add_argument(
        "functions",
        nargs="*",
        help="Specific functions to test. If not specified, test all functions.",
    )

    args = parser.parse_args()

    if args.functions:
        # Specific functions specified
        if args.module:
            # Test specific functions in specific module
            run_module(args.module, args.file, args.functions)
        else:
            # Auto-detect which modules contain these functions
            modules = detect_module_from_functions(args.functions)
            if not modules:
                print(f"Error: No functions found matching: {args.functions}")
                sys.exit(1)

            for module_name in modules:
                run_module(module_name, args.file, args.functions)
    elif args.module:
        # Test specific module, all functions
        run_module(args.module, args.file)
    else:
        # Test all modules
        for module_name in ["data_utils", "metrics_utils", "tree_utils"]:
            print()  # Add spacing between modules
            run_module(module_name, args.file)


if __name__ == "__main__":
    main()
