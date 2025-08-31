"""
Secure test runner for the Linear Algebra Programming Assignment.

This system loads pre-computed test cases and runs them against student implementations,
showing which specific inputs failed for each function.
"""

import json
import numpy as np
import argparse
import sys
import importlib
from typing import Dict, List, Any, Optional


def python_to_numpy(obj):
    """Convert Python lists back to NumPy arrays where appropriate."""
    if isinstance(obj, list):
        return np.array(obj)
    return obj


def slice_from_dict(slice_dict):
    """Convert dictionary back to slice object."""
    return slice(slice_dict['start'], slice_dict['stop'], slice_dict['step'])


def load_test_cases(filename='test_cases.json'):
    """Load test cases from JSON file."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {filename} not found. Run generate_test_cases.py first.")
        sys.exit(1)


def import_student_modules():
    """Import student implementation modules."""
    modules = {}
    try:
        modules['part1'] = importlib.import_module('part1_from_scratch')
    except ImportError:
        print("Warning: Could not import part1_from_scratch.py")
        modules['part1'] = None
    
    try:
        modules['part2'] = importlib.import_module('part2_numpy_essentials')
    except ImportError:
        print("Warning: Could not import part2_numpy_essentials.py")
        modules['part2'] = None
    
    try:
        modules['part3'] = importlib.import_module('part3_advanced_problems')
    except ImportError:
        print("Warning: Could not import part3_advanced_problems.py")
        modules['part3'] = None
    
    return modules


def arrays_equal(actual, expected, tolerance=1e-10):
    """Check if two values are equal, handling numpy arrays with tolerance."""
    if isinstance(actual, np.ndarray) and isinstance(expected, (list, np.ndarray)):
        expected_array = np.array(expected) if isinstance(expected, list) else expected
        return np.allclose(actual, expected_array, atol=tolerance, rtol=tolerance)
    elif isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(actual - expected) < tolerance
    elif isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            return False
        return all(arrays_equal(a, e, tolerance) for a, e in zip(actual, expected))
    else:
        return actual == expected


def run_function_test(func, test_case, func_name):
    """Run a single test case for a function."""
    try:
        inputs = test_case['input']
        
        # Handle special cases for input conversion
        if func_name == 'get_submatrix':
            # Convert slice dictionaries back to slice objects
            if len(inputs) == 3:
                arr, rows_dict, cols_dict = inputs
                inputs = [python_to_numpy(arr), slice_from_dict(rows_dict), slice_from_dict(cols_dict)]
            
        # Convert lists to numpy arrays for numpy functions
        elif 'numpy' in func_name or func_name in [
            'create_zeros_array', 'create_scaled_identity_matrix', 'create_range_matrix',
            'get_diagonal', 'matrix_plus_row_vector', 'matrix_plus_column_vector',
            'conditional_values', 'compute_row_means', 'normalize_features',
            'find_max_indices', 'stack_arrays', 'filter_positive_values',
            'create_permutation_matrix', 'back_substitution', 'rotation_matrix_2d',
            'scaling_matrix_2d', 'translation_matrix_2d', 'complex_transformation_challenge'
        ]:
            inputs = [python_to_numpy(inp) if isinstance(inp, list) else inp for inp in inputs]
        
        # Call the function
        if inputs:
            result = func(*inputs)
        else:
            result = func()
        
        # Check if we expected an error
        if 'error' in test_case:
            return False, f"Expected error but got result: {result}"
        
        # Compare result with expected
        expected = test_case['expected']
        if arrays_equal(result, expected):
            return True, None
        else:
            return False, f"Expected: {expected}, Got: {result}"
    
    except Exception as e:
        if 'error' in test_case:
            # We expected an error
            if test_case['error'] == 'ValueError' and isinstance(e, ValueError):
                return True, None
            elif test_case['error'] in str(e):
                return True, None
            else:
                return False, f"Expected error '{test_case['error']}' but got '{str(e)}'"
        else:
            return False, f"Error: {str(e)}"


def run_function_tests(func_name, func, test_cases):
    """Run all test cases for a single function."""
    if func is None:
        return {
            'total': len(test_cases),
            'passed': 0,
            'failed': len(test_cases),
            'status': 'NOT_IMPLEMENTED',
            'failures': []
        }
    
    results = {
        'total': len(test_cases),
        'passed': 0,
        'failed': 0,
        'failures': []
    }
    
    for i, test_case in enumerate(test_cases):
        success, error_msg = run_function_test(func, test_case, func_name)
        
        if success:
            results['passed'] += 1
        else:
            results['failed'] += 1
            results['failures'].append({
                'test_index': i,
                'input': test_case['input'],
                'error': error_msg
            })
    
    results['status'] = 'PASS' if results['failed'] == 0 else 'FAIL'
    return results


def print_function_result(func_name, results, points):
    """Print the results for a single function."""
    status_icon = "[PASS]" if results['status'] == 'PASS' else "[FAIL]" if results['status'] == 'FAIL' else "[WARN]"
    points_earned = points if results['status'] == 'PASS' else 0
    
    print(f"{status_icon} {func_name:<30} {points_earned:>2} / {points:>2} points")
    
    if results['status'] == 'NOT_IMPLEMENTED':
        print(f"     -> NOT IMPLEMENTED")
    elif results['status'] == 'FAIL':
        print(f"     -> {results['passed']}/{results['total']} test cases passed")
        
        # Show first few failures
        max_failures_to_show = 3
        for failure in results['failures'][:max_failures_to_show]:
            input_str = str(failure['input'])
            if len(input_str) > 50:
                input_str = input_str[:47] + "..."
            print(f"     -> FAILED on input {failure['test_index']}: {input_str}")
            print(f"      {failure['error']}")
        
        if len(results['failures']) > max_failures_to_show:
            remaining = len(results['failures']) - max_failures_to_show
            print(f"     -> ... and {remaining} more failures")


def run_part_tests(part_name, part_module, test_cases, function_points):
    """Run all tests for a part."""
    part_results = {}
    total_points = 0
    earned_points = 0
    
    for func_name, points in function_points:
        if func_name in test_cases:
            func = getattr(part_module, func_name, None) if part_module else None
            results = run_function_tests(func_name, func, test_cases[func_name])
            part_results[func_name] = results
            
            print_function_result(func_name, results, points)
            
            total_points += points
            if results['status'] == 'PASS':
                earned_points += points
        else:
            print(f"[WARN]  {func_name:<30} NO TEST CASES")
    
    return part_results, earned_points, total_points


def main():
    parser = argparse.ArgumentParser(description='Run linear algebra assignment tests')
    parser.add_argument('--part', choices=['1', '2', '3', 'all'], default='all',
                       help='Which part to test (default: all)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed failure information')
    args = parser.parse_args()
    
    # Load test cases
    print("Loading test cases...")
    all_test_cases = load_test_cases()
    
    # Import student modules
    print("Importing student modules...")
    modules = import_student_modules()
    
    # Define function points for each part (Easy=1, Medium=2, Hard=3)
    part1_functions = [
        ("vector_add", 1),          # Easy: element-wise addition
        ("scalar_multiply", 1),     # Easy: multiply each element
        ("dot_product", 2),         # Medium: sum of products
        ("vector_magnitude", 2),    # Medium: sqrt of sum of squares
        ("normalize_vector", 3),    # Hard: magnitude + division
        ("matrix_add", 1),          # Easy: element-wise matrix addition
        ("matrix_vector_multiply", 2), # Medium: matrix-vector multiply
        ("matrix_multiply", 3),     # Hard: nested loops with dot products
        ("matrix_transpose", 2)     # Medium: swap rows and columns
    ]
    
    part2_functions = [
        ("create_zeros_array", 1),         # Easy: single np.zeros call
        ("create_scaled_identity_matrix", 2), # Medium: np.eye + scaling
        ("create_range_matrix", 1),        # Easy: np.arange + reshape
        ("get_diagonal", 1),               # Easy: single np.diag call
        ("get_submatrix", 2),              # Medium: array slicing with slice objects
        ("matrix_plus_row_vector", 2),     # Medium: broadcasting understanding
        ("matrix_plus_column_vector", 2),  # Medium: broadcasting with reshape
        ("conditional_values", 2),         # Medium: np.where function
        ("numpy_matrix_multiply", 1),      # Easy: single @ operator
        ("compute_row_means", 2),          # Medium: np.mean with axis parameter
        ("normalize_features", 3),         # Hard: multiple operations (mean, std, broadcasting)
        ("find_max_indices", 1),           # Easy: single np.argmax call
        ("stack_arrays", 1),               # Easy: single np.vstack call
        ("filter_positive_values", 1)      # Easy: boolean indexing
    ]
    
    part3_functions = [
        ("create_permutation_matrix", 3),     # Hard: understanding permutation concept
        ("back_substitution", 3),             # Hard: algorithm implementation with validation
        ("rotation_matrix_2d", 2),            # Medium: requires trig functions
        ("scaling_matrix_2d", 1),             # Easy: simple diagonal matrix
        ("translation_matrix_2d", 2),         # Medium: 3*3 homogeneous matrix
        ("complex_transformation_challenge", 3) # Hard: multiple transformations in sequence
    ]
    
    print('=' * 60)
    print('*** LINEAR ALGEBRA ASSIGNMENT - TEST RESULTS')
    print('=' * 60)
    
    total_earned = 0
    total_possible = 0
    
    # Run tests based on selected part
    if args.part == 'all' or args.part == '1':
        print('\n*** PART 1: FROM SCRATCH IMPLEMENTATION')
        print('-' * 40)
        part1_results, earned, possible = run_part_tests(
            'part1', modules['part1'], all_test_cases['part1'], part1_functions
        )
        print(f'\nPart 1 Score: {earned}/{possible} points ({earned/possible*100:.1f}%)')
        total_earned += earned
        total_possible += possible
    
    if args.part == 'all' or args.part == '2':
        print('\n*** PART 2: NUMPY ESSENTIALS')
        print('-' * 40)
        part2_results, earned, possible = run_part_tests(
            'part2', modules['part2'], all_test_cases['part2'], part2_functions
        )
        print(f'\nPart 2 Score: {earned}/{possible} points ({earned/possible*100:.1f}%)')
        total_earned += earned
        total_possible += possible
    
    if args.part == 'all' or args.part == '3':
        print('\n*** PART 3: ADVANCED PROBLEMS')
        print('-' * 40)
        part3_results, earned, possible = run_part_tests(
            'part3', modules['part3'], all_test_cases['part3'], part3_functions
        )
        print(f'\nPart 3 Score: {earned}/{possible} points ({earned/possible*100:.1f}%)')
        total_earned += earned
        total_possible += possible
    
    # Final score
    if args.part == 'all':
        print('\n' + '=' * 60)
        percentage = (total_earned / total_possible) * 100
        
        if percentage >= 90:
            grade_emoji = "🏆"
            grade_text = "EXCELLENT"
        elif percentage >= 80:
            grade_emoji = "EXCELLENT"
            grade_text = "GREAT"
        elif percentage >= 70:
            grade_emoji = "GOOD"
            grade_text = "GOOD"
        elif percentage >= 60:
            grade_emoji = "PASSING"
            grade_text = "PASSING"
        else:
            grade_emoji = "KEEP WORKING"
            grade_text = "KEEP WORKING"
        
        print(f'{grade_emoji} FINAL SCORE: {total_earned}/{total_possible} points ({percentage:.1f}%) - {grade_text}')
        print('=' * 60)
        
        if percentage < 100:
            print('\n*** Tips:')
            print('   • Make sure all functions are implemented')
            print('   • Check function names match exactly')
            print('   • Read error messages carefully')
            print('   • Part 3 problems require mathematical thinking!')
        else:
            print('\n🎉 Perfect score! You\'ve mastered linear algebra programming!')


if __name__ == "__main__":
    main()