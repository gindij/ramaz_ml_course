"""
Plotting utilities for visualizing bias-variance tradeoff and model performance.

This module provides functions for creating complexity curves, ROC curves,
confusion matrices, and other visualizations.
"""

from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_complexity_curve(
    param_range: List,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    param_name: str = "max_depth",
    title: str = "Model Complexity Curve",
) -> plt.Figure:
    """
    Plot training vs validation scores across parameter values.

    Args:
        param_range: List of parameter values
        train_scores: Training scores for each parameter value
        val_scores: Validation scores for each parameter value
        param_name: Name of parameter being varied
        title: Plot title

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate means and stds
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # Plot curves
    ax.plot(param_range, train_mean, "o-", label="Training Score", color="blue")
    ax.fill_between(
        param_range,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.1,
        color="blue",
    )

    ax.plot(param_range, val_mean, "o-", label="Validation Score", color="red")
    ax.fill_between(
        param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color="red"
    )

    ax.set_xlabel(param_name)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_roc_curves(
    models_roc_data: Dict[str, tuple], title: str = "ROC Curves"
) -> plt.Figure:
    """
    Plot ROC curves for multiple models.

    Args:
        models_roc_data: Dict mapping model names to (fpr, tpr, auc) tuples
        title: Plot title

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = ["blue", "red", "green", "orange", "purple"]

    for i, (name, (fpr, tpr, auc)) in enumerate(models_roc_data.items()):
        color = colors[i % len(colors)]
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=color, linewidth=2)

    # Plot diagonal line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Classifier")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_confusion_matrices(
    models_cm: Dict[str, np.ndarray], figsize: tuple = (12, 4)
) -> plt.Figure:
    """
    Plot confusion matrices for multiple models side by side.

    Args:
        models_cm: Dict mapping model names to confusion matrices
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    n_models = len(models_cm)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)

    if n_models == 1:
        axes = [axes]

    for ax, (name, cm) in zip(axes, models_cm.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"{name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    return fig


def plot_bootstrap_variance(
    predictions_matrix: np.ndarray, sample_indices: Optional[List[int]] = None
) -> plt.Figure:
    """
    Visualize prediction variance across bootstrap samples.

    Args:
        predictions_matrix: Shape (n_trees, n_samples) prediction matrix
        sample_indices: Specific sample indices to highlight

    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Prediction variance for first 50 samples
    if sample_indices is None:
        sample_indices = list(range(min(50, predictions_matrix.shape[1])))

    prediction_variance = np.var(predictions_matrix, axis=0)
    ax1.plot(sample_indices, prediction_variance[sample_indices], "bo-")
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Prediction Variance")
    ax1.set_title("Bootstrap Prediction Variance by Sample")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Distribution of individual vs averaged predictions
    individual_preds = predictions_matrix[:, sample_indices[0]]
    averaged_pred = np.mean(individual_preds)

    ax2.hist(
        individual_preds,
        bins=20,
        alpha=0.7,
        label="Individual Trees",
        color="lightblue",
    )
    ax2.axvline(
        averaged_pred,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Average = {averaged_pred:.3f}",
    )
    ax2.set_xlabel("Prediction")
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"Prediction Distribution for Sample {sample_indices[0]}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_ensemble_size_comparison(
    n_estimators: List[int], train_scores: List[float], test_scores: List[float]
) -> plt.Figure:
    """
    Plot performance vs ensemble size.

    Args:
        n_estimators: List of ensemble sizes
        train_scores: Training scores
        test_scores: Test scores

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        n_estimators,
        train_scores,
        "o-",
        label="Training Score",
        color="blue",
        linewidth=2,
    )
    ax.plot(
        n_estimators, test_scores, "o-", label="Test Score", color="red", linewidth=2
    )

    ax.set_xlabel("Number of Estimators")
    ax.set_ylabel("Accuracy")
    ax.set_title("Random Forest Performance vs Ensemble Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    return fig
