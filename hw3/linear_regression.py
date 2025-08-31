"""
HW3: Linear Regression from Scratch

Implement linear regression using gradient descent optimization.
This is part of a larger assignment that also includes logistic regression.

Instructions:
- Use functions from utilities.py for shared functionality
- Focus on gradient descent approach (students learn optimization)
- Implement both simple and multiple linear regression
- Include polynomial feature engineering

To test your implementation:
    python test.py --part linear

Functions to implement: 7 functions (22 points total)
"""

import numpy as np
from typing import Tuple, List
from utilities import add_intercept, mean_squared_error, gradient_descent


# =============================================================================
# Linear Regression Core Implementation (13 points)
# =============================================================================


def compute_cost(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Squared Error cost function for linear regression.

    This is a wrapper around the shared MSE function, scaled by 1/2 for
    mathematical convenience in derivatives (common in ML literature).

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        float: Mean squared error scaled by 1/2

    Example:
        cost = compute_cost([1, 2, 3], [1.1, 2.2, 2.9])  # Should be small
    """
    # TODO: Implement this function
    # Hint: Use mean_squared_error from utilities and scale by 1/2
    pass


def linear_gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Compute gradient of linear regression cost function with respect to theta.

    The cost function to differentiate is:
    J(θ) = (1/2m) * Σ(h_θ(x^(i)) - y^(i))^2

    Where:
    - h_θ(x) = θ^T * x (linear hypothesis)
    - m = number of training examples

    The gradient is:
    ∂J/∂θ = (1/m) * X^T * (X*θ - y)

    Args:
        X: Feature matrix with intercept (n_samples, n_features + 1)
        y: Target vector (n_samples,)
        theta: Current parameters (n_features + 1,)

    Returns:
        np.ndarray: Gradient vector (n_features + 1,)

    Example:
        gradients = linear_gradient(X_with_intercept, y, theta)
    """
    # TODO: Implement this function
    # Hint: predictions = X @ theta, then compute X.T @ (predictions - y) / m
    pass


def linear_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, List[float]]:
    """
    Train linear regression using gradient descent.

    This function uses the shared gradient_descent implementation from utilities
    with the linear-specific gradient and cost functions.

    Args:
        X: Feature matrix with intercept (n_samples, n_features + 1)
        y: Target vector (n_samples,)
        learning_rate: Step size for gradient descent
        max_iterations: Maximum number of iterations
        tolerance: Convergence threshold for cost function

    Returns:
        Tuple of:
        - theta: Learned parameters
        - costs: List of cost values during training

    Example:
        theta, costs = linear_gradient_descent(X_with_intercept, y)
    """
    # TODO: Implement this function
    # Hint: Use gradient_descent from utilities with linear_gradient and a cost function
    pass


def predict(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Make predictions using linear regression model.

    Args:
        X: Feature matrix (with or without intercept)
        theta: Model parameters

    Returns:
        np.ndarray: Predictions

    Example:
        predictions = predict(X_test, theta)
    """
    # TODO: Implement this function
    pass


# =============================================================================
# 3. Polynomial Features (8 points)
# =============================================================================


def polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """
    Generate polynomial features up to given degree.

    Args:
        X: Feature matrix (n_samples, n_features)
        degree: Maximum degree for polynomial features

    Returns:
        np.ndarray: Matrix with polynomial features

    Example:
        X = [[2], [3]]  # Single feature
        poly_features = polynomial_features(X, degree=2)
        # Result: [[2, 4], [3, 9]]  # [x, x^2]
    """
    # TODO: Implement this function
    pass


def fit_polynomial_regression(X: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """
    Fit polynomial regression model.

    Args:
        X: Feature matrix
        y: Target vector
        degree: Degree of polynomial

    Returns:
        np.ndarray: Fitted parameters

    Example:
        theta = fit_polynomial_regression(X, y, degree=3)
    """
    # TODO: Implement this function
    pass


# =============================================================================
# Model Evaluation (8 points)
# =============================================================================


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R-squared (coefficient of determination).
    R² = 1 - (SS_res / SS_tot)

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        float: R-squared value

    Example:
        r2 = r_squared(y_test, predictions)
    """
    # TODO: Implement this function
    pass


def cross_validate_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    learning_rate: float = 0.01,
    max_iterations: int = 1000,
) -> Tuple[float, float]:
    """
    Perform k-fold cross-validation for linear regression.

    Uses gradient descent to fit models and evaluates using R-squared.

    Args:
        X: Feature matrix (without intercept)
        y: Target vector
        k: Number of folds
        learning_rate: Learning rate for gradient descent
        max_iterations: Max iterations for gradient descent

    Returns:
        Tuple of (mean_r2_score, std_r2_score)

    Example:
        mean_r2, std_r2 = cross_validate_linear_regression(X, y, k=5)
    """
    # TODO: Implement this function
    # Hint: Use k_fold_split from utilities, linear_gradient_descent, and r_squared
    pass


if __name__ == "__main__":
    print("HW3: Linear Regression from Scratch")
    print("===================================")
    print()
    print("This module implements linear regression using analytical solutions.")
    print("Part of a larger assignment including logistic regression.")
    print()
    print("Functions to implement (22 points total):")
    print("  Core Implementation (13 points):")
    print("    - compute_cost (2 pts)")
    print("    - linear_gradient (4 pts)")
    print("    - linear_gradient_descent (3 pts)")
    print("    - predict (4 pts)")
    print("  Polynomial Features (6 points):")
    print("    - polynomial_features (3 pts)")
    print("    - fit_polynomial_regression (3 pts)")
    print("  Model Evaluation (3 points):")
    print("    - r_squared (2 pts)")
    print("    - cross_validate_linear_regression (1 pt)")
    print()
    print("Shared functions in utilities.py:")
    print("  - standardize_features, add_intercept, train_test_split")
    print("  - mean_squared_error, accuracy_score, k_fold_split")
    print()
    print("To test your implementations:")
    print("  python test.py --part linear")
