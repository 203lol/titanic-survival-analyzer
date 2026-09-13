"""Evaluation utilities for Titanic survival models."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from titanic_analyzer.models import ModelResult


@dataclass
class EvaluationMetrics:
    """Store classification evaluation metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


def evaluate_model(result: ModelResult) -> EvaluationMetrics:
    """Calculate classification metrics for a trained model."""
    y_true = result.y_test
    y_pred = result.predictions

    probabilities = result.model.predict_proba(result.x_test)[:, 1]

    return EvaluationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        recall=float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        f1=float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        roc_auc=float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
    )


def compare_model_metrics(
    model_results: dict[str, ModelResult],
) -> pd.DataFrame:
    """Create a comparison table for multiple trained models."""
    rows = []

    for model_name, result in model_results.items():
        metrics = evaluate_model(result)

        rows.append(
            {
                "Model": model_name,
                "Accuracy": metrics.accuracy,
                "Precision": metrics.precision,
                "Recall": metrics.recall,
                "F1": metrics.f1,
                "ROC-AUC": metrics.roc_auc,
            }
        )

    comparison = pd.DataFrame(rows)

    return comparison.sort_values(
        by="F1",
        ascending=False,
    ).reset_index(drop=True)


def save_confusion_matrix(
    result: ModelResult,
    model_name: str,
    output_path: str | Path,
) -> Path:
    """Create and save a confusion matrix for a model."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()

    ConfusionMatrixDisplay.from_predictions(
        result.y_test,
        result.predictions,
        ax=ax,
    )

    ax.set_title(f"{model_name} Confusion Matrix")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path


def save_all_confusion_matrices(
    model_results: dict[str, ModelResult],
    output_directory: str | Path = "outputs/plots",
) -> list[Path]:
    """Save confusion matrices for all trained models."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    paths = []

    for model_name, result in model_results.items():
        filename = model_name.lower().replace(" ", "_").replace("-", "_")

        path = save_confusion_matrix(
            result,
            model_name,
            output_directory / f"confusion_matrix_{filename}.png",
        )

        paths.append(path)

    return paths
