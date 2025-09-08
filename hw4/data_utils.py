"""
Data loading and preprocessing utilities.

This module provides functions for loading datasets, creating train/test splits,
and basic data preprocessing for the assignment.
"""

from typing import Tuple, Dict
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np


def load_assignment_dataset(
    dataset_name: str = "breast_cancer", random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dataset and create train/test split for the assignment.

    Args:
        dataset_name: Name of dataset to load ('breast_cancer')
        random_state: Random seed for reproducible splits

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if dataset_name == "breast_cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


def get_dataset_info(X: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    """
    Get basic information about the dataset.

    Args:
        X: Feature matrix
        y: Target vector

    Returns:
        dict: Dataset information including shape, classes, etc.
    """
    unique, counts = np.unique(y, return_counts=True)

    return {
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_classes": len(unique),
        "classes": unique,
        "class_counts": dict(zip(unique, counts)),
        "feature_range": {
            "min": X.min(axis=0),
            "max": X.max(axis=0),
            "mean": X.mean(axis=0),
        },
    }
