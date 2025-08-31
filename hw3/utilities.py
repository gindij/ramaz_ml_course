"""
HW3: Utilities - Shared functionality for Linear and Logistic Regression

This module contains common functions used by both linear and logistic regression
implementations. These utilities handle data preprocessing, evaluation metrics,
and cross-validation that apply to both regression types.

Instructions:
- Implement all functions completely
- These functions will be imported and used by linear_regression.py and logistic_regression.py
- Focus on making robust, reusable code

Functions to implement: 8 functions (20 points total)
"""

import numpy as np
from typing import Tuple, Optional


# =============================================================================
# Data Preprocessing (8 points)
# =============================================================================


def standardize_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features to have mean=0 and std=1.

    Args:
        X: Feature matrix of shape (n_samples, n_features)

    Returns:
        Tuple containing:
        - X_standardized: Standardized features
        - mean: Mean of each feature (for inverse transform)
        - std: Standard deviation of each feature (for inverse transform)

    Example:
        X = [[1, 2], [3, 4], [5, 6]]
        X_std, mean, std = standardize_features(X)
        # X_std should have mean ≈ 0, std ≈ 1 for each column
    """
    # TODO: Implement this function
    pass


def add_intercept(X: np.ndarray) -> np.ndarray:
    """
    Add intercept term (bias) to feature matrix.

    Args:
        X: Feature matrix of shape (n_samples, n_features)

    Returns:
        np.ndarray: Feature matrix with intercept column of ones

    Example:
        add_intercept([[1, 2], [3, 4]]) -> [[1, 1, 2], [1, 3, 4]]
    """
    # TODO: Implement this function
    pass


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets.

    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data for testing (default 0.2)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)

    Example:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    """
    # TODO: Implement this function
    pass


# =============================================================================
# Evaluation Metrics (4 points)
# =============================================================================


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.

    Args:
        y_true: True class labels (0 or 1)
        y_pred: Predicted class labels (0 or 1)

    Returns:
        float: Accuracy as fraction of correct predictions

    Example:
        accuracy = accuracy_score([0, 1, 1, 0], [0, 1, 0, 0])  # 0.75
    """
    # TODO: Implement this function
    pass


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Squared Error.

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        float: Mean squared error

    Example:
        mse = mean_squared_error([1, 2, 3], [1.1, 2.2, 2.9])
    """
    # TODO: Implement this function
    pass


# =============================================================================
# Optimization (5 points)
# =============================================================================


def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    gradient_func,
    cost_func,
    learning_rate: float = 0.01,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, List[float]]:
    """
    Generic gradient descent implementation for optimization.

    This function provides a general framework for gradient-based optimization
    that can be used for both linear and logistic regression. Students implement
    the gradient computation specific to each method.

    Args:
        X: Feature matrix with intercept (n_samples, n_features + 1)
        y: Target vector (n_samples,)
        gradient_func: Function that computes gradients given (X, y, theta)
        cost_func: Function that computes cost given (y_true, predictions)
        learning_rate: Step size for gradient descent
        max_iterations: Maximum number of iterations
        tolerance: Convergence threshold for cost function

    Returns:
        Tuple of:
        - theta: Learned parameters
        - costs: List of cost values during training

    Example:
        # For linear regression:
        theta, costs = gradient_descent(X, y, linear_gradient, mse_cost)

        # For logistic regression:
        theta, costs = gradient_descent(X, y, logistic_gradient, logistic_cost)
    """
    # Initialize parameters
    n_features = X.shape[1]
    theta = np.zeros(n_features)
    costs = []

    for i in range(max_iterations):
        # Compute cost for monitoring convergence
        predictions = X @ theta
        cost = cost_func(y, predictions)
        costs.append(cost)

        # Compute gradients using provided function
        gradients = gradient_func(X, y, theta)

        # Update parameters
        theta = theta - learning_rate * gradients

        # Check convergence
        if i > 0 and abs(costs[-2] - costs[-1]) < tolerance:
            break

    return theta, costs


# =============================================================================
# Cross-Validation (3 points)
# =============================================================================


def k_fold_split(
    n_samples: int, k: int = 5, random_state: Optional[int] = None
) -> list:
    """
    Generate k-fold cross-validation indices.

    Args:
        n_samples: Total number of samples
        k: Number of folds
        random_state: Random seed for reproducibility

    Returns:
        List of tuples (train_indices, val_indices) for each fold

    Example:
        folds = k_fold_split(100, k=5)
        # Returns 5 folds, each with ~80 train samples and ~20 val samples
    """
    # TODO: Implement this function
    pass


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)

    Returns:
        dict: Dictionary containing:
            - 'accuracy': Overall accuracy
            - 'precision': Precision for positive class
            - 'recall': Recall for positive class
            - 'f1_score': F1 score

    Example:
        metrics = compute_classification_metrics(y_true, y_pred)
        print(f"Accuracy: {metrics['accuracy']:.3f}")
    """
    # TODO: Implement this function
    pass


if __name__ == "__main__":
    print("HW3: Utilities")
    print("==============")
    print()
    print(
        "This module contains shared functionality for linear and logistic regression."
    )
    print()
    print("Functions to implement (20 points total):")
    print("  Data Preprocessing (8 points):")
    print("    - standardize_features (3 pts)")
    print("    - add_intercept (2 pts)")
    print("    - train_test_split (3 pts)")
    print("  Evaluation Metrics (4 points):")
    print("    - accuracy_score (2 pts)")
    print("    - mean_squared_error (2 pts)")
    print("  Optimization (5 points):")
    print("    - gradient_descent (5 pts) [PROVIDED IMPLEMENTATION]")
    print("  Cross-Validation (3 points):")
    print("    - k_fold_split (2 pts)")
    print("    - compute_classification_metrics (1 pt)")
    print()
    print("These functions will be imported by:")
    print("  - linear_regression.py")
    print("  - logistic_regression.py")
