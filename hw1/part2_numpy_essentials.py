"""
Part 2: NumPy Essentials

Master fundamental NumPy operations for numerical computing and data manipulation.
Build fluency with array creation, indexing, broadcasting, and vectorized operations.

Instructions:
- Use NumPy functions and operations throughout
- Focus on understanding array broadcasting and vectorization
- Read the NumPy documentation when needed
- Some functions require combining multiple NumPy operations

To test your implementations:
    python test.py --part 2
    
Difficulty Legend: [EASY] (1 pt) | [MEDIUM] (2 pts) | [HARD] (3 pts)
"""

import numpy as np


# =============================================================================
# 2.1 Array Creation and Basic Operations
# =============================================================================

def create_zeros_array() -> np.ndarray:  # [EASY] - 1 point
    """
    Create a 3x3 matrix of zeros.
    
    Returns:
        np.ndarray: A 3x3 array filled with zeros
    
    Example:
        create_zeros_array() -> [[0. 0. 0.]
                                [0. 0. 0.]
                                [0. 0. 0.]]
    """
    # TODO: Implement this function
    pass


def create_scaled_identity_matrix(n: int, scalar: float) -> np.ndarray:  # [MEDIUM] - 2 points
    """
    Create an n*n identity matrix with scaled diagonal values.
    
    This requires creating an identity matrix first, then scaling the diagonal elements.
    Cannot be done with a single NumPy call.
    
    Args:
        n: Size of the square matrix
        scalar: Value to multiply the diagonal elements by
    
    Returns:
        np.ndarray: An n*n matrix with `scalar` on the diagonal, 0 elsewhere
    
    Example:
        create_scaled_identity_matrix(3, 5) -> [[5. 0. 0.]
                                               [0. 5. 0.]
                                               [0. 0. 5.]]
    """
    # TODO: Implement this function
    # Hint: You'll need to create an identity matrix first, then scale it
    pass


def create_range_matrix() -> np.ndarray:  # [EASY] Easy - 1 point
    """
    Create a 4x3 matrix filled with values from 0 to 11 in row-major order.
    
    This requires creating a range of numbers first, then reshaping them.
    Cannot be done with a single NumPy call.
    
    Returns:
        np.ndarray: A 4x3 matrix with values 0-11
    
    Example:
        create_range_matrix() -> [[ 0  1  2]
                                 [ 3  4  5]
                                 [ 6  7  8]
                                 [ 9 10 11]]
    """
    # TODO: Implement this function
    # Hint: Create a range of numbers first, then reshape to 4x3
    pass


# =============================================================================
# 2.2 Array Indexing and Slicing
# =============================================================================

def get_diagonal(arr: np.ndarray) -> np.ndarray:  # [EASY] Easy - 1 point
    """
    Get the main diagonal elements of a 2D array.
    
    Args:
        arr: A 2D numpy array
    
    Returns:
        np.ndarray: 1D array containing the diagonal elements
    
    Example:
        get_diagonal([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) -> [1, 5, 9]
    """
    # TODO: Implement this function
    pass


def get_submatrix(arr: np.ndarray, rows: slice, cols: slice) -> np.ndarray:  # [MEDIUM] Medium - 2 points
    """
    Extract a submatrix using row and column slices.
    
    Args:
        arr: A 2D numpy array
        rows: Slice object for row selection
        cols: Slice object for column selection
    
    Returns:
        np.ndarray: The extracted submatrix
    
    Example:
        arr = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        get_submatrix(arr, slice(0, 2), slice(1, 3)) -> [[2, 3], [5, 6]]
    """
    # TODO: Implement this function
    pass


# =============================================================================
# 2.3 Broadcasting and Vectorized Operations
# =============================================================================

def matrix_plus_row_vector(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:  # [MEDIUM] Medium - 2 points
    """
    Add a vector to each row of a matrix (broadcasting).
    
    Args:
        matrix: A 2D numpy array
        vector: A 1D numpy array (should broadcast to each row)
    
    Returns:
        np.ndarray: Matrix with vector added to each row
    
    Example:
        matrix = [[1, 2], [3, 4]]
        vector = [10, 20]
        Result: [[11, 22], [13, 24]]
    """
    # TODO: Implement this function
    pass


def matrix_plus_column_vector(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:  # [MEDIUM] Medium - 2 points
    """
    Add a vector to each column of a matrix (broadcasting).
    
    Args:
        matrix: A 2D numpy array
        vector: A 1D numpy array (should broadcast to each column)
    
    Returns:
        np.ndarray: Matrix with vector added to each column
    
    Example:
        matrix = [[1, 2], [3, 4]]
        vector = [10, 20]
        Result: [[11, 12], [23, 24]]
    """
    # TODO: Implement this function
    # Hint: You may need to reshape the vector for proper broadcasting
    pass


def conditional_values(matrix: np.ndarray, threshold: float) -> np.ndarray:  # [MEDIUM] Medium - 2 points
    """
    Return matrix elements where condition is true, 0 otherwise.
    
    Args:
        matrix: A numpy array
        threshold: Threshold value for comparison
    
    Returns:
        np.ndarray: Array with original values where > threshold, 0 elsewhere
    
    Example:
        conditional_values([[1, 5], [3, 2]], 2.5) -> [[0, 5], [3, 0]]
    """
    # TODO: Implement this function
    pass


# =============================================================================
# 2.4 Linear Algebra Operations
# =============================================================================

def numpy_matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:  # [EASY] Easy - 1 point
    """
    Multiply two matrices using NumPy.
    
    Args:
        A: First matrix
        B: Second matrix
    
    Returns:
        np.ndarray: The matrix product A * B
    
    Example:
        numpy_matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]]) -> [[19, 22], [43, 50]]
    """
    # TODO: Implement this function
    pass


# =============================================================================
# 2.5 Statistical Operations
# =============================================================================

def compute_row_means(data: np.ndarray) -> np.ndarray:  # [MEDIUM] Medium - 2 points
    """
    Compute the mean of each row.
    
    Args:
        data: A 2D numpy array
    
    Returns:
        np.ndarray: 1D array with the mean of each row
    
    Example:
        compute_row_means([[1, 2, 3], [4, 5, 6]]) -> [2.0, 5.0]
    """
    # TODO: Implement this function
    pass


def normalize_features(X: np.ndarray) -> np.ndarray:  # [HARD] Hard - 3 points
    """
    Normalize features (columns) to have mean=0 and std=1.
    
    This is a common preprocessing step in machine learning.
    Each column should be independently normalized.
    
    Args:
        X: A 2D numpy array where rows are samples, columns are features
    
    Returns:
        np.ndarray: Normalized array with mean=0, std=1 for each column
    
    Example:
        X = [[1, 2], [3, 4], [5, 6]]
        Result should have each column with mean~=0, std~=1
    """
    # TODO: Implement this function
    # Hint: Subtract mean and divide by standard deviation, column-wise
    pass


def find_max_indices(arr: np.ndarray) -> np.ndarray:  # [EASY] Easy - 1 point
    """
    Find the indices of maximum values along each row.
    
    Args:
        arr: A 2D numpy array
    
    Returns:
        np.ndarray: 1D array with the column index of max value in each row
    
    Example:
        find_max_indices([[1, 3, 2], [6, 4, 5]]) -> [1, 0]
    """
    # TODO: Implement this function
    pass


# =============================================================================
# 2.6 Array Manipulation
# =============================================================================

def stack_arrays(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:  # [EASY] Easy - 1 point
    """
    Stack two 1D arrays vertically to create a 2D array.
    
    Args:
        arr1: First 1D array
        arr2: Second 1D array
    
    Returns:
        np.ndarray: 2D array with arr1 as first row, arr2 as second row
    
    Example:
        stack_arrays([1, 2, 3], [4, 5, 6]) -> [[1, 2, 3], [4, 5, 6]]
    """
    # TODO: Implement this function
    pass


def filter_positive_values(arr: np.ndarray) -> np.ndarray:  # [EASY] Easy - 1 point
    """
    Extract only positive values from a 2D array using boolean masking.
    
    This demonstrates NumPy's powerful boolean indexing capabilities.
    
    Args:
        arr: A 2D numpy array
    
    Returns:
        np.ndarray: 1D array containing only positive values
    
    Example:
        filter_positive_values([[-1, 2], [3, -4]]) -> [2, 3]
    """
    # TODO: Implement this function
    # Hint: Create a boolean mask and use it for indexing
    pass


if __name__ == "__main__":
    print("Part 2: NumPy Essentials")
    print("========================")
    print()
    print("This module contains functions for learning essential NumPy operations.")
    print()
    print("Functions to implement (22 points total):")
    print("  [EASY] create_zeros_array (1 pt)")
    print("  [MEDIUM] create_scaled_identity_matrix (2 pts)")
    print("  [EASY] create_range_matrix (1 pt)")
    print("  [EASY] get_diagonal (1 pt)")
    print("  [MEDIUM] get_submatrix (2 pts)")
    print("  [MEDIUM] matrix_plus_row_vector (2 pts)")
    print("  [MEDIUM] matrix_plus_column_vector (2 pts)")
    print("  [MEDIUM] conditional_values (2 pts)")
    print("  [EASY] numpy_matrix_multiply (1 pt)")
    print("  [MEDIUM] compute_row_means (2 pts)")
    print("  [HARD] normalize_features (3 pts)")
    print("  [EASY] find_max_indices (1 pt)")
    print("  [EASY] stack_arrays (1 pt)")
    print("  [EASY] filter_positive_values (1 pt)")
    print()
    print("To test your implementations:")
    print("  python test.py --part 2")