"""
Generic testing framework for homework assignments.
"""

from .generator import generate_test_cases
from .runner import run_tests, run_specific_functions

__all__ = ["generate_test_cases", "run_tests", "run_specific_functions"]
