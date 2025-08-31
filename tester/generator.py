"""
Generic test case generator.

Takes test inputs and a reference implementation, generates JSON test cases.
"""

import json
from typing import Dict, List, Any, Callable


def safe_call_function(func: Callable, inputs: List[Any]) -> Dict[str, Any]:
    """Safely call a function and return result or expected error."""
    try:
        result = func(*inputs)
        return {"input": inputs, "expected_output": result}
    except Exception as e:
        return {"input": inputs, "expected_error": str(e)}


def generate_test_cases(
    test_inputs_dict: Dict[str, List[List[Any]]], reference_module, output_file: str
) -> None:
    """
    Generate test cases from inputs and reference implementation.

    Args:
        test_inputs_dict: Dict mapping function_name -> list of input lists
        reference_module: Module containing reference implementations
        output_file: Path to save generated test_cases.json
    """
    print("Generating test cases...")

    test_cases = {}
    total_cases = 0

    for func_name, input_lists in test_inputs_dict.items():
        print(f"- {func_name}...")

        # Get the reference function
        if not hasattr(reference_module, func_name):
            print(f"  Warning: {func_name} not found in reference module")
            continue

        func = getattr(reference_module, func_name)
        function_test_cases = []

        for inputs in input_lists:
            test_case = safe_call_function(func, inputs)
            function_test_cases.append(test_case)

        test_cases[func_name] = function_test_cases
        total_cases += len(function_test_cases)
        print(f"  Generated {len(function_test_cases)} test cases")

    # Save to JSON file
    with open(output_file, "w") as f:
        json.dump(test_cases, f, indent=2, default=str)

    print(f"Generated {total_cases} total test cases for {len(test_cases)} functions")
    print(f"Test cases saved to {output_file}")
