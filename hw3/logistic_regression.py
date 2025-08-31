"""
HW3: Logistic Regression from Scratch

Implement logistic regression using gradient descent and maximum likelihood estimation.
This is part of a larger assignment that also includes linear regression.

Instructions:
- Use functions from utilities.py for shared functionality
- Implement gradient descent optimization (no analytical solution exists)
- Focus on binary classification (0/1 labels)
- Include proper probabilistic interpretation

To test your implementation:
    python test.py --part logistic

Functions to implement: 9 functions (30 points total)
"""

import numpy as np
from typing import Tuple, List
from utilities import (
    add_intercept,
    accuracy_score,
    compute_classification_metrics,
    gradient_descent,
)


# =============================================================================
# Activation Functions (4 points)
# =============================================================================


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Compute sigmoid activation function.

    σ(z) = 1 / (1 + e^(-z))

    Args:
        z: Linear combination of features (X @ theta)

    Returns:
        np.ndarray: Sigmoid probabilities between 0 and 1

    Example:
        sigmoid([0, 2, -2]) -> [0.5, 0.88, 0.12] (approximately)
    """
    # TODO: Implement this function
    # Hint: Use np.exp, handle numerical stability for large negative values
    pass


def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    """
    Compute derivative of sigmoid function.

    σ'(z) = σ(z) * (1 - σ(z))

    Args:
        z: Linear combination of features

    Returns:
        np.ndarray: Derivative values

    Example:
        sigmoid_derivative([0, 2, -2]) -> [0.25, 0.1, 0.1] (approximately)
    """
    # TODO: Implement this function
    # Hint: Use your sigmoid function
    pass


# =============================================================================
# Logistic Regression Core Implementation (15 points)
# =============================================================================


def compute_logistic_cost(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute logistic regression cost function (negative log-likelihood).

    Cost = -1/m * Σ[y*log(p) + (1-y)*log(1-p)]

    Args:
        y_true: True binary labels (0 or 1)
        y_prob: Predicted probabilities from sigmoid

    Returns:
        float: Logistic cost (cross-entropy loss)

    Example:
        cost = compute_logistic_cost([1, 0, 1], [0.9, 0.1, 0.8])
    """
    # TODO: Implement this function
    # Hint: Use np.log, add small epsilon to avoid log(0)
    pass


def logistic_cost_from_predictions(
    y_true: np.ndarray, predictions: np.ndarray
) -> float:
    """
    Wrapper to compute logistic cost from raw predictions (for gradient_descent).

    This function converts linear predictions to probabilities using sigmoid,
    then computes the logistic cost. Used by the generic gradient_descent function.

    Args:
        y_true: True binary labels (0 or 1)
        predictions: Raw predictions from X @ theta

    Returns:
        float: Logistic cost
    """
    # TODO: Implement this function
    # Hint: Apply sigmoid to predictions, then use compute_logistic_cost
    pass


def logistic_gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Compute gradient of logistic regression cost function with respect to theta.

    The cost function to differentiate is:
    J(θ) = -(1/m) * Σ[y^(i) * log(h_θ(x^(i))) + (1-y^(i)) * log(1-h_θ(x^(i)))]

    Where:
    - h_θ(x) = sigmoid(θ^T * x) = 1/(1 + e^(-θ^T * x))
    - m = number of training examples

    The gradient is:
    ∂J/∂θ = (1/m) * X^T * (sigmoid(X*θ) - y)

    Args:
        X: Feature matrix with intercept (n_samples, n_features + 1)
        y: Binary target vector (n_samples,)
        theta: Current parameters (n_features + 1,)

    Returns:
        np.ndarray: Gradient vector (n_features + 1,)

    Example:
        gradients = logistic_gradient(X_with_intercept, y, theta)
    """
    # TODO: Implement this function
    # Hint: predictions = sigmoid(X @ theta), then compute X.T @ (predictions - y) / m
    pass


def logistic_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, List[float]]:
    """
    Train logistic regression using gradient descent.

    This function uses the shared gradient_descent implementation from utilities
    with the logistic-specific gradient and cost functions.

    Args:
        X: Feature matrix with intercept (n_samples, n_features + 1)
        y: Binary target vector (n_samples,)
        learning_rate: Step size for gradient descent
        max_iterations: Maximum number of iterations
        tolerance: Convergence threshold for cost function

    Returns:
        Tuple of:
        - theta: Learned parameters
        - costs: List of cost values during training

    Example:
        theta, costs = logistic_gradient_descent(X, y, learning_rate=0.01)
    """
    # TODO: Implement this function
    # Hint: Use gradient_descent from utilities with logistic_gradient and logistic_cost_from_predictions
    pass


def predict_proba(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Predict class probabilities using logistic regression.

    Args:
        X: Feature matrix (with or without intercept)
        theta: Model parameters

    Returns:
        np.ndarray: Predicted probabilities for positive class

    Example:
        probabilities = predict_proba(X_test, theta)
    """
    # TODO: Implement this function
    # Hint: Add intercept if needed, then apply sigmoid to X @ theta
    pass


def predict_classes(
    X: np.ndarray, theta: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """
    Predict binary classes using logistic regression.

    Args:
        X: Feature matrix (with or without intercept)
        theta: Model parameters
        threshold: Decision threshold for classification

    Returns:
        np.ndarray: Predicted binary classes (0 or 1)

    Example:
        predictions = predict_classes(X_test, theta, threshold=0.5)
    """
    # TODO: Implement this function
    # Hint: Use predict_proba and apply threshold
    pass


# =============================================================================
# Model Evaluation (6 points)
# =============================================================================


def cross_validate_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    learning_rate: float = 0.01,
    max_iterations: int = 1000,
) -> Tuple[float, float]:
    """
    Perform k-fold cross-validation for logistic regression.

    Uses gradient descent to fit models and evaluates using accuracy.

    Args:
        X: Feature matrix (without intercept)
        y: Binary target vector
        k: Number of folds
        learning_rate: Learning rate for gradient descent
        max_iterations: Max iterations for gradient descent

    Returns:
        Tuple of (mean_accuracy, std_accuracy)

    Example:
        mean_acc, std_acc = cross_validate_logistic_regression(X, y, k=5)
    """
    # TODO: Implement this function
    # Hint: Use k_fold_split from utilities, logistic_gradient_descent, and accuracy_score
    pass


if __name__ == "__main__":
    print("HW3: Logistic Regression from Scratch")
    print("=====================================")
    print()
    print("This module implements logistic regression using gradient descent.")
    print("Part of a larger assignment including linear regression.")
    print()
    print("Functions to implement (30 points total):")
    print("  Activation Functions (4 points):")
    print("    - sigmoid (2 pts)")
    print("    - sigmoid_derivative (2 pts)")
    print("  Core Implementation (20 points):")
    print("    - compute_logistic_cost (3 pts)")
    print("    - logistic_cost_from_predictions (2 pts)")
    print("    - logistic_gradient (5 pts)")
    print("    - logistic_gradient_descent (3 pts)")
    print("    - predict_proba (4 pts)")
    print("    - predict_classes (3 pts)")
    print("  Model Evaluation (6 points):")
    print("    - cross_validate_logistic_regression (6 pts)")
    print()
    print("Shared functions in utilities.py:")
    print("  - standardize_features, add_intercept, train_test_split")
    print("  - accuracy_score, k_fold_split, compute_classification_metrics")
    print()
    print("To test your implementations:")
    print("  python test.py --part logistic")
