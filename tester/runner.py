"""
Generic test runner.

Takes JSON test cases and student implementation, runs tests and provides scoring.
"""

import json
import sys
import importlib
import traceback
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


def deserialize_from_json(obj):
    """Convert JSON-serialized objects back to their original types."""
    if isinstance(obj, dict):
        if "__numpy_array__" in obj:
            # Reconstruct numpy array
            return np.array(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])
        elif "__slice__" in obj:
            # Reconstruct slice object
            return slice(obj["start"], obj["stop"], obj["step"])
        elif "__tuple__" in obj:
            # Reconstruct tuple
            return tuple(deserialize_from_json(item) for item in obj["data"])
        else:
            # Regular dict - recursively deserialize values
            return {key: deserialize_from_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [deserialize_from_json(item) for item in obj]
    else:
        return obj


def load_test_cases(test_cases_file: str) -> Dict[str, List[Dict]]:
    """Load test cases from JSON file."""
    try:
        with open(test_cases_file, "r") as f:
            raw_data = json.load(f)
            # Deserialize all test cases
            return deserialize_from_json(raw_data)
    except FileNotFoundError:
        print(f"Error: {test_cases_file} not found.")
        print("Run the test case generator first.")
        sys.exit(1)


def import_student_module(module_name: str):
    """Import student implementation module."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        print(f"Error: Could not import {module_name}.py")
        print("Make sure the file exists and has no syntax errors.")
        sys.exit(1)


def safe_call_function(
    func: Any, inputs: List[Any], student_module=None
) -> Dict[str, Any]:
    """Safely call a function and return result or error."""
    try:
        if func is None:
            return {"status": "NOT_IMPLEMENTED", "error": "Function not found"}

        # Handle string function names that need to be converted to actual functions
        processed_inputs = []
        for inp in inputs:
            if isinstance(inp, str) and student_module and hasattr(student_module, inp):
                processed_inputs.append(getattr(student_module, inp))
            else:
                processed_inputs.append(inp)

        result = func(*processed_inputs)
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        return {"status": "ERROR", "error": str(e), "traceback": traceback.format_exc()}


def compare_results(expected: Any, actual: Any, tolerance: float = 1e-6) -> bool:
    """Compare expected and actual results with tolerance for floating point."""
    # Handle NumPy arrays
    if isinstance(expected, np.ndarray) or isinstance(actual, np.ndarray):
        try:
            # Convert both to numpy arrays if they aren't already
            expected_arr = np.asarray(expected)
            actual_arr = np.asarray(actual)

            # Check shapes match
            if expected_arr.shape != actual_arr.shape:
                return False

            # Handle NaN values specially
            if np.any(np.isnan(expected_arr)) or np.any(np.isnan(actual_arr)):
                # Both should have NaN in same positions
                expected_nan_mask = np.isnan(expected_arr)
                actual_nan_mask = np.isnan(actual_arr)
                if not np.array_equal(expected_nan_mask, actual_nan_mask):
                    return False
                # Compare non-NaN values
                non_nan_mask = ~expected_nan_mask
                if np.any(non_nan_mask):
                    return np.allclose(
                        expected_arr[non_nan_mask],
                        actual_arr[non_nan_mask],
                        rtol=tolerance,
                        atol=tolerance,
                    )
                return True
            else:
                # Regular comparison with tolerance
                return np.allclose(
                    expected_arr, actual_arr, rtol=tolerance, atol=tolerance
                )
        except (ValueError, TypeError):
            # Fall back to regular comparison if numpy comparison fails
            pass

    if isinstance(expected, float) and isinstance(actual, (int, float)):
        return abs(expected - actual) < tolerance
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return False
        return all(compare_results(e, a, tolerance) for e, a in zip(expected, actual))
    elif isinstance(expected, tuple) and isinstance(actual, tuple):
        if len(expected) != len(actual):
            return False
        return all(compare_results(e, a, tolerance) for e, a in zip(expected, actual))
    # Handle tuple/list equivalence
    elif isinstance(expected, tuple) and isinstance(actual, list):
        return compare_results(expected, tuple(actual), tolerance)
    elif isinstance(expected, list) and isinstance(actual, tuple):
        return compare_results(tuple(expected), actual, tolerance)
    else:
        try:
            return expected == actual
        except ValueError:
            # This can happen with numpy arrays, fall back to element-wise comparison
            try:
                return np.array_equal(expected, actual)
            except:
                return False


def run_function_tests(
    func_name: str, func: Any, test_cases: List[Dict], student_module=None
) -> Dict[str, Any]:
    """Run all test cases for a single function."""
    if not test_cases:
        return {"status": "NO_TESTS", "passed": 0, "total": 0, "failures": []}

    passed = 0
    failures = []

    for i, test_case in enumerate(test_cases):
        inputs = test_case["input"]

        # Check if this test expects an error
        if "expected_error" in test_case:
            result = safe_call_function(func, inputs, student_module)
            if result["status"] == "ERROR":
                passed += 1
            else:
                failures.append(
                    {
                        "test_index": i,
                        "input": inputs,
                        "expected": f"Error: {test_case['expected_error']}",
                        "actual": result.get("result", "No error raised"),
                    }
                )
        else:
            expected_output = test_case["expected_output"]
            result = safe_call_function(func, inputs, student_module)

            if result["status"] == "SUCCESS":
                if compare_results(expected_output, result["result"]):
                    passed += 1
                else:
                    failures.append(
                        {
                            "test_index": i,
                            "input": inputs,
                            "expected": expected_output,
                            "actual": result["result"],
                        }
                    )
            else:
                failures.append(
                    {
                        "test_index": i,
                        "input": inputs,
                        "expected": expected_output,
                        "actual": f"Error: {result.get('error', 'Unknown error')}",
                    }
                )

    status = "PASS" if passed == len(test_cases) else "FAIL"
    if func is None:
        status = "NOT_IMPLEMENTED"

    return {
        "status": status,
        "passed": passed,
        "total": len(test_cases),
        "failures": failures,
    }


def format_input_for_display(inputs: List[Any], max_length: int = 50) -> str:
    """Format function inputs for display, truncating if too long."""
    input_str = str(inputs)
    if len(input_str) > max_length:
        input_str = input_str[:max_length] + "..."
    return input_str


def print_function_results(func_name: str, results: Dict[str, Any], points: int):
    """Print results for a single function."""
    status_icon = (
        "[PASS]"
        if results["status"] == "PASS"
        else "[FAIL]" if results["status"] == "FAIL" else "[WARN]"
    )

    earned_points = points if results["status"] == "PASS" else 0
    print(f"{status_icon} {func_name:<30} {earned_points:2d} / {points:2d} points")

    if results["status"] == "NOT_IMPLEMENTED":
        print("     -> NOT IMPLEMENTED")
    elif results["total"] > 0:
        print(f"     -> {results['passed']}/{results['total']} test cases passed")

        # Show failures (limit to first 3 for readability)
        failures_to_show = results["failures"][:3]
        for failure in failures_to_show:
            input_str = format_input_for_display(failure["input"])
            print(f"     -> FAILED on input {failure['test_index']}: {input_str}")
            print(f"      Expected: {failure['expected']}")
            print(f"      Got: {failure['actual']}")

        if len(results["failures"]) > 3:
            remaining = len(results["failures"]) - 3
            print(f"     -> ... and {remaining} more failures")
    else:
        print("     -> NO TEST CASES")


def get_grade_info(percentage: float) -> Tuple[str, str]:
    """Get grade description based on percentage."""
    if percentage >= 95:
        return "EXCELLENT", "EXCELLENT"
    elif percentage >= 85:
        return "GOOD", "GOOD"
    elif percentage >= 70:
        return "PASSING", "PASSING"
    else:
        return "KEEP WORKING", "KEEP WORKING"


def print_tips(percentage: float, earned_points: int):
    """Print helpful tips based on performance."""
    if percentage < 100:
        print()
        print("*** Tips:")
        print("   • Make sure all functions are implemented")
        print("   • Check function names match exactly")
        print("   • Read error messages carefully")
        print("   • Test with edge cases (empty lists, invalid inputs)")

    if earned_points == 0:
        print("   • Start with the easier functions first")
        print("   • Build up to the more complex functions")
        print("   • Make sure you understand the concepts")


def run_tests(
    test_cases_file: str,
    student_module_name: str,
    function_points: Dict[str, int],
    assignment_name: str,
) -> Tuple[int, int]:
    """
    Run all tests and display results.

    Args:
        test_cases_file: Path to test_cases.json
        student_module_name: Name of student module to test
        function_points: Dict mapping function_name -> points
        assignment_name: Display name for assignment

    Returns:
        Tuple of (earned_points, total_points)
    """
    print("Loading test cases...")
    test_cases = load_test_cases(test_cases_file)

    print(f"Importing module: {student_module_name}...")
    student_module = import_student_module(student_module_name)

    print("=" * 60)
    print(f"*** {assignment_name.upper()} - TEST RESULTS")
    print("=" * 60)
    print()

    total_points = 0
    earned_points = 0

    # Test individual functions
    for func_name, points in function_points.items():
        func = getattr(student_module, func_name, None)
        func_test_cases = test_cases.get(func_name, [])

        results = run_function_tests(func_name, func, func_test_cases, student_module)
        print_function_results(func_name, results, points)

        total_points += points
        if results["status"] == "PASS":
            earned_points += points

    # Calculate final score and grade
    percentage = (earned_points / total_points) * 100 if total_points > 0 else 0

    print()
    print("=" * 60)

    grade_text, _ = get_grade_info(percentage)
    print(
        f"{grade_text} FINAL SCORE: {earned_points}/{total_points} points ({percentage:.1f}%) - {grade_text}"
    )
    print("=" * 60)

    print_tips(percentage, earned_points)

    return earned_points, total_points


def run_specific_functions(
    test_cases_file: str,
    student_module_name: str,
    function_names: List[str],
    function_points: Dict[str, int],
) -> Tuple[int, int]:
    """
    Run tests for specific functions only.

    Returns:
        Tuple of (earned_points, total_points)
    """
    print("Loading test cases...")
    test_cases = load_test_cases(test_cases_file)

    print(f"Importing module: {student_module_name}...")
    student_module = import_student_module(student_module_name)

    print("=" * 60)
    print(f'*** TESTING FUNCTIONS: {", ".join(function_names)}')
    print("=" * 60)
    print()

    total_points = 0
    earned_points = 0

    for func_name in function_names:
        if func_name not in function_points:
            print(f"Unknown function: {func_name}")
            continue

        func = getattr(student_module, func_name, None)
        func_test_cases = test_cases.get(func_name, [])
        points = function_points[func_name]

        results = run_function_tests(func_name, func, func_test_cases, student_module)
        print_function_results(func_name, results, points)

        total_points += points
        if results["status"] == "PASS":
            earned_points += points

    percentage = (earned_points / total_points) * 100 if total_points > 0 else 0

    print()
    print("-" * 60)
    print(f"SCORE: {earned_points}/{total_points} points ({percentage:.1f}%)")

    return earned_points, total_points
