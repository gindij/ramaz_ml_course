"""
HW2: K-Nearest Neighbors from Scratch

Implement the K-Nearest Neighbors algorithm from first principles to understand
the fundamental concepts of machine learning classification and regression.

Instructions:
- Use only Python's built-in data structures and math/statistics libraries
- Do not use scikit-learn, NumPy, or other ML libraries for the core algorithm
- Focus on understanding distance metrics, neighbor selection, and prediction
- Each function should be implemented completely with proper error handling

To test your implementations:
    python test.py
"""

import math
import statistics
from typing import List, Tuple, Any, Union


# =============================================================================
# Part 1: Distance Metrics
# =============================================================================


def euclidean_distance(
    point1: List[float], point2: List[float]
) -> float:  # [EASY] - 2 points
    """
    Calculate the Euclidean distance between two points.

    Args:
        point1: First point as a list of coordinates
        point2: Second point as a list of coordinates

    Returns:
        float: The Euclidean distance between the points

    Raises:
        ValueError: If points have different dimensions

    Example:
        euclidean_distance([0, 0], [3, 4]) -> 5.0
        euclidean_distance([1, 2, 3], [4, 6, 8]) -> 5.830951894845301
    """
    # TODO: Implement this function
    pass


def manhattan_distance(
    point1: List[float], point2: List[float]
) -> float:  # [EASY] - 2 points
    """
    Calculate the Manhattan (L1) distance between two points.

    Args:
        point1: First point as a list of coordinates
        point2: Second point as a list of coordinates

    Returns:
        float: The Manhattan distance between the points

    Raises:
        ValueError: If points have different dimensions

    Example:
        manhattan_distance([0, 0], [3, 4]) -> 7.0
        manhattan_distance([1, 2, 3], [4, 6, 8]) -> 12.0
    """
    # TODO: Implement this function
    pass


def minkowski_distance(
    point1: List[float], point2: List[float], p: float
) -> float:  # [MEDIUM] - 3 points
    """
    Calculate the Minkowski distance between two points.

    The Minkowski distance is a generalization where p=1 gives Manhattan distance
    and p=2 gives Euclidean distance.

    Args:
        point1: First point as a list of coordinates
        point2: Second point as a list of coordinates
        p: The order of the Minkowski distance (must be >= 1)

    Returns:
        float: The Minkowski distance between the points

    Raises:
        ValueError: If points have different dimensions or p < 1

    Example:
        minkowski_distance([0, 0], [3, 4], 2) -> 5.0  # Same as Euclidean
        minkowski_distance([0, 0], [3, 4], 1) -> 7.0  # Same as Manhattan
    """
    # TODO: Implement this function
    pass


# =============================================================================
# Part 2: Neighbor Finding
# =============================================================================


def find_k_nearest_neighbors(
    query_point: List[float],
    training_data: List[List[float]],
    training_labels: List[Any],
    k: int,
    distance_func: callable = euclidean_distance,
) -> List[Tuple[List[float], Any, float]]:  # [MEDIUM] - 4 points
    """
    Find the k nearest neighbors to a query point.

    Args:
        query_point: The point to find neighbors for
        training_data: List of training data points
        training_labels: List of labels corresponding to training data
        k: Number of nearest neighbors to find
        distance_func: Function to calculate distance (default: euclidean_distance)

    Returns:
        List of tuples: (neighbor_point, neighbor_label, distance) sorted by distance

    Raises:
        ValueError: If k <= 0, k > len(training_data), or mismatched data/labels lengths

    Example:
        query = [2, 2]
        data = [[1, 1], [3, 3], [5, 5], [1, 3]]
        labels = ['A', 'B', 'C', 'D']
        find_k_nearest_neighbors(query, data, labels, 2)
        -> [([1, 1], 'A', 1.414...), ([3, 3], 'B', 1.414...)]
    """
    # TODO: Implement this function
    pass


def find_neighbors_within_radius(
    query_point: List[float],
    training_data: List[List[float]],
    training_labels: List[Any],
    radius: float,
    distance_func: callable = euclidean_distance,
) -> List[Tuple[List[float], Any, float]]:  # [MEDIUM] - 3 points
    """
    Find all neighbors within a given radius of a query point.

    Args:
        query_point: The point to find neighbors for
        training_data: List of training data points
        training_labels: List of labels corresponding to training data
        radius: Maximum distance for neighbors
        distance_func: Function to calculate distance (default: euclidean_distance)

    Returns:
        List of tuples: (neighbor_point, neighbor_label, distance) sorted by distance

    Raises:
        ValueError: If radius <= 0 or mismatched data/labels lengths

    Example:
        query = [2, 2]
        data = [[1, 1], [3, 3], [5, 5], [1, 3]]
        labels = ['A', 'B', 'C', 'D']
        find_neighbors_within_radius(query, data, labels, 2.0)
        -> [([1, 1], 'A', 1.414...), ([3, 3], 'B', 1.414...), ([1, 3], 'D', 1.414...)]
    """
    # TODO: Implement this function
    pass


# =============================================================================
# Part 3: Classification
# =============================================================================


def majority_vote(labels: List[Any]) -> Any:  # [EASY] - 2 points
    """
    Determine the most common label using majority vote.

    Args:
        labels: List of labels from nearest neighbors

    Returns:
        The most frequent label (if tie, return any of the tied labels)

    Raises:
        ValueError: If labels list is empty

    Example:
        majority_vote(['A', 'B', 'A', 'C', 'A']) -> 'A'
        majority_vote([1, 2, 2, 3]) -> 2
    """
    # TODO: Implement this function
    pass


def weighted_majority_vote(
    neighbors: List[Tuple[List[float], Any, float]],
) -> Any:  # [MEDIUM] - 3 points
    """
    Determine the most common label using distance-weighted voting.

    Closer neighbors have more influence. Use inverse distance as weight.
    If distance is 0, use a large weight (e.g., 1000).

    Args:
        neighbors: List of (point, label, distance) tuples

    Returns:
        The label with highest weighted vote

    Raises:
        ValueError: If neighbors list is empty

    Example:
        neighbors = [([1, 1], 'A', 1.0), ([2, 2], 'B', 2.0), ([3, 3], 'A', 3.0)]
        weighted_majority_vote(neighbors) -> 'A'  # A gets weight 1.0 + 0.33 = 1.33
    """
    # TODO: Implement this function
    pass


def knn_classify(
    query_point: List[float],
    training_data: List[List[float]],
    training_labels: List[Any],
    k: int,
    distance_func: callable = euclidean_distance,
    weighted: bool = False,
) -> Any:  # [HARD] - 4 points
    """
    Classify a query point using K-Nearest Neighbors.

    Args:
        query_point: The point to classify
        training_data: List of training data points
        training_labels: List of labels corresponding to training data
        k: Number of nearest neighbors to consider
        distance_func: Function to calculate distance
        weighted: Whether to use distance-weighted voting

    Returns:
        Predicted label for the query point

    Raises:
        ValueError: If k <= 0 or other validation errors

    Example:
        query = [2.5, 2.5]
        data = [[1, 1], [3, 3], [1, 3], [3, 1]]
        labels = ['A', 'A', 'B', 'B']
        knn_classify(query, data, labels, 3) -> 'A' or 'B' (depends on distances)
    """
    # TODO: Implement this function
    pass


# =============================================================================
# Part 4: Regression
# =============================================================================


def mean_prediction(values: List[float]) -> float:  # [EASY] - 2 points
    """
    Calculate the mean of a list of values for regression prediction.

    Args:
        values: List of numeric values from nearest neighbors

    Returns:
        float: The mean of the values

    Raises:
        ValueError: If values list is empty

    Example:
        mean_prediction([1.0, 2.0, 3.0, 4.0]) -> 2.5
    """
    # TODO: Implement this function
    pass


def weighted_mean_prediction(
    neighbors: List[Tuple[List[float], float, float]],
) -> float:  # [MEDIUM] - 3 points
    """
    Calculate distance-weighted mean for regression prediction.

    Args:
        neighbors: List of (point, value, distance) tuples

    Returns:
        float: The weighted mean of the values

    Raises:
        ValueError: If neighbors list is empty

    Example:
        neighbors = [([1, 1], 10.0, 1.0), ([2, 2], 20.0, 2.0)]
        weighted_mean_prediction(neighbors) -> ~13.33 (closer point has more weight)
    """
    # TODO: Implement this function
    pass


def knn_regress(
    query_point: List[float],
    training_data: List[List[float]],
    training_values: List[float],
    k: int,
    distance_func: callable = euclidean_distance,
    weighted: bool = False,
) -> float:  # [HARD] - 4 points
    """
    Predict a continuous value using K-Nearest Neighbors regression.

    Args:
        query_point: The point to predict a value for
        training_data: List of training data points
        training_values: List of continuous values corresponding to training data
        k: Number of nearest neighbors to consider
        distance_func: Function to calculate distance
        weighted: Whether to use distance-weighted prediction

    Returns:
        float: Predicted continuous value

    Raises:
        ValueError: If k <= 0 or other validation errors

    Example:
        query = [2, 2]
        data = [[1, 1], [3, 3], [1, 3], [3, 1]]
        values = [10.0, 30.0, 20.0, 25.0]
        knn_regress(query, data, values, 2) -> ~22.5 (average of closest neighbors)
    """
    # TODO: Implement this function
    pass


# =============================================================================
# Part 5: Model Validation and Performance
# =============================================================================


def train_test_split(
    data: List[List[float]],
    labels: List[Any],
    test_size: float = 0.2,
    random_seed: int = None,
) -> Tuple[
    List[List[float]], List[List[float]], List[Any], List[Any]
]:  # [MEDIUM] - 3 points
    """
    Split data into training and testing sets.

    Args:
        data: List of data points
        labels: List of labels corresponding to data points
        test_size: Fraction of data to use for testing (0 < test_size < 1)
        random_seed: Seed for reproducible random splitting

    Returns:
        Tuple of (train_data, test_data, train_labels, test_labels)

    Raises:
        ValueError: If test_size not in (0, 1) or data/labels length mismatch

    Example:
        data = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
        labels = ['A', 'B', 'A', 'B', 'A']
        train_test_split(data, labels, 0.4) returns 4 train samples, 1 test sample
    """
    # TODO: Implement this function
    pass


def calculate_accuracy(
    true_labels: List[Any], predicted_labels: List[Any]
) -> float:  # [EASY] - 2 points
    """
    Calculate classification accuracy.

    Args:
        true_labels: Ground truth labels
        predicted_labels: Predicted labels

    Returns:
        float: Accuracy as a fraction between 0 and 1

    Raises:
        ValueError: If label lists have different lengths

    Example:
        calculate_accuracy(['A', 'B', 'A', 'B'], ['A', 'A', 'A', 'B']) -> 0.75
    """
    # TODO: Implement this function
    pass


def mean_squared_error(
    true_values: List[float], predicted_values: List[float]
) -> float:  # [EASY] - 2 points
    """
    Calculate mean squared error for regression.

    Args:
        true_values: Ground truth continuous values
        predicted_values: Predicted continuous values

    Returns:
        float: Mean squared error

    Raises:
        ValueError: If value lists have different lengths or are empty

    Example:
        mean_squared_error([1.0, 2.0, 3.0], [1.1, 2.2, 2.8]) -> 0.05
    """
    # TODO: Implement this function
    pass


def cross_validate_knn(
    data: List[List[float]],
    labels: List[Any],
    k_values: List[int],
    cv_folds: int = 5,
    distance_func: callable = euclidean_distance,
) -> List[Tuple[int, float]]:  # [HARD] - 5 points
    """
    Perform cross-validation to find the best k value for KNN.

    Args:
        data: List of data points
        labels: List of labels corresponding to data points
        k_values: List of k values to test
        cv_folds: Number of cross-validation folds
        distance_func: Function to calculate distance

    Returns:
        List of tuples: (k_value, average_accuracy) sorted by k_value

    Raises:
        ValueError: If cv_folds <= 1 or k_values contains invalid values

    Example:
        cross_validate_knn(data, labels, [1, 3, 5], cv_folds=3)
        -> [(1, 0.85), (3, 0.90), (5, 0.87)]
    """
    # TODO: Implement this function
    pass


if __name__ == "__main__":
    print("HW2: K-Nearest Neighbors from Scratch")
    print("=====================================")
    print()
    print("This module contains functions for implementing K-NN from first principles.")
    print()
    print("Functions to implement (40 points total):")
    print("  [EASY] euclidean_distance (2 pts)")
    print("  [EASY] manhattan_distance (2 pts)")
    print("  [MEDIUM] minkowski_distance (3 pts)")
    print("  [MEDIUM] find_k_nearest_neighbors (4 pts)")
    print("  [MEDIUM] find_neighbors_within_radius (3 pts)")
    print("  [EASY] majority_vote (2 pts)")
    print("  [MEDIUM] weighted_majority_vote (3 pts)")
    print("  [HARD] knn_classify (4 pts)")
    print("  [EASY] mean_prediction (2 pts)")
    print("  [MEDIUM] weighted_mean_prediction (3 pts)")
    print("  [HARD] knn_regress (4 pts)")
    print("  [MEDIUM] train_test_split (3 pts)")
    print("  [EASY] calculate_accuracy (2 pts)")
    print("  [EASY] mean_squared_error (2 pts)")
    print("  [HARD] cross_validate_knn (5 pts)")
    print()
    print("To test your implementations:")
    print("  python test.py")
