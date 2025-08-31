"""
Part 1: From Scratch Implementation

Implement basic linear algebra operations using only Python's built-in data structures.
No external libraries allowed - build understanding from first principles!

Instructions:
- Use only Python's built-in data structures (lists, numbers)
- Do not import NumPy or any other libraries
- Implement each function completely
- Focus on understanding the mathematical concepts

To test your implementations:
    from test_linear_algebra import run_all_tests
    run_all_tests()
"""

import math
from typing import List


# =============================================================================
# 1.1 Vector Operations
# =============================================================================

def vector_add(v1: List[float], v2: List[float]) -> List[float]:
    """
    Add two vectors element-wise.
    
    Args:
        v1: First vector as a list of numbers
        v2: Second vector as a list of numbers
    
    Returns:
        List[float]: The sum v1 + v2
    
    Raises:
        ValueError: If vectors have different lengths
    
    Example:
        vector_add([1, 2, 3], [4, 5, 6]) -> [5, 7, 9]
    """
    # TODO: Implement this function
    pass


def scalar_multiply(scalar: float, vector: List[float]) -> List[float]:
    """
    Multiply a vector by a scalar.
    
    Args:
        scalar: The scalar multiplier
        vector: Vector as a list of numbers
    
    Returns:
        List[float]: The scaled vector
    
    Example:
        scalar_multiply(3, [1, 2, 3]) -> [3, 6, 9]
    """
    # TODO: Implement this function
    pass


def dot_product(v1: List[float], v2: List[float]) -> float:
    """
    Compute the dot product of two vectors.
    
    Args:
        v1: First vector as a list of numbers
        v2: Second vector as a list of numbers
    
    Returns:
        float: The dot product v1 * v2
    
    Raises:
        ValueError: If vectors have different lengths
    
    Example:
        dot_product([1, 2, 3], [4, 5, 6]) -> 32
    """
    # TODO: Implement this function
    pass


def vector_magnitude(vector: List[float]) -> float:
    """
    Compute the magnitude (length) of a vector.
    
    Args:
        vector: Vector as a list of numbers
    
    Returns:
        float: The magnitude of the vector
    
    Example:
        vector_magnitude([3, 4]) -> 5.0
    """
    # TODO: Implement this function
    pass


def normalize_vector(vector: List[float]) -> List[float]:
    """
    Normalize a vector to unit length.
    
    Args:
        vector: Vector as a list of numbers
    
    Returns:
        List[float]: The normalized vector
    
    Raises:
        ValueError: If the vector is the zero vector
    
    Example:
        normalize_vector([3, 4]) -> [0.6, 0.8]
    """
    # TODO: Implement this function
    pass


# =============================================================================
# 1.2 Matrix Operations
# =============================================================================

def matrix_add(m1: List[List[float]], m2: List[List[float]]) -> List[List[float]]:
    """
    Add two matrices element-wise.
    
    Args:
        m1: First matrix as a list of lists
        m2: Second matrix as a list of lists
    
    Returns:
        List[List[float]]: The sum m1 + m2
    
    Raises:
        ValueError: If matrices have different dimensions
    
    Example:
        matrix_add([[1, 2], [3, 4]], [[5, 6], [7, 8]]) -> [[6, 8], [10, 12]]
    """
    # TODO: Implement this function
    pass


def matrix_vector_multiply(matrix: List[List[float]], vector: List[float]) -> List[float]:
    """
    Multiply a matrix by a vector.
    
    Args:
        matrix: Matrix as a list of lists (rows)
        vector: Vector as a list of numbers
    
    Returns:
        List[float]: The product matrix * vector
    
    Raises:
        ValueError: If dimensions don't match for multiplication
    
    Example:
        matrix_vector_multiply([[1, 2], [3, 4]], [5, 6]) -> [17, 39]
    """
    # TODO: Implement this function
    pass


def matrix_multiply(m1: List[List[float]], m2: List[List[float]]) -> List[List[float]]:
    """
    Multiply two matrices.
    
    Args:
        m1: First matrix as a list of lists
        m2: Second matrix as a list of lists
    
    Returns:
        List[List[float]]: The product m1 * m2
    
    Raises:
        ValueError: If dimensions don't match for multiplication
    
    Example:
        matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]]) -> [[19, 22], [43, 50]]
    """
    # TODO: Implement this function
    pass


def matrix_transpose(matrix: List[List[float]]) -> List[List[float]]:
    """
    Compute the transpose of a matrix.
    
    Args:
        matrix: Matrix as a list of lists
    
    Returns:
        List[List[float]]: The transpose of the matrix
    
    Example:
        matrix_transpose([[1, 2, 3], [4, 5, 6]]) -> [[1, 4], [2, 5], [3, 6]]
    """
    # TODO: Implement this function
    pass


if __name__ == "__main__":
    print("Part 1: From Scratch Implementation")
    print("===================================")
    print()
    print("This module contains functions for implementing linear algebra operations")
    print("from scratch using only Python's built-in data structures.")
    print()
    print("Functions to implement:")
    print("  - vector_add")
    print("  - scalar_multiply") 
    print("  - dot_product")
    print("  - vector_magnitude")
    print("  - normalize_vector")
    print("  - matrix_add")
    print("  - matrix_vector_multiply")
    print("  - matrix_multiply")
    print("  - matrix_transpose")
    print()
    print("To test your implementations:")
    print("  from test_linear_algebra import run_all_tests")
    print("  run_all_tests()")