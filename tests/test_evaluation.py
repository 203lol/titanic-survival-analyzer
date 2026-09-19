"""Tests for Titanic model evaluation."""

from pathlib import Path

import pandas as pd

from titanic_analyzer.evaluation import (
    compare_model_metrics,
    evaluate_model,
    save_confusion_matrix,
)
from titanic_analyzer.models import compare_models


def make_evaluation_dataframe() -> pd.DataFrame:
    """Create a small dataset for evaluation tests."""
    return pd.DataFrame(
        {
            "Age": [
                22,
                38,
                26,
                35,
                28,
                54,
                2,
                27,
                14,
                58,
                20,
                45,
                31,
                19,
                40,
                30,
            ],
            "Fare": [
                7.25,
                71.28,
                7.93,
                53.1,
                8.05,
                51.86,
                21.08,
                11.13,
                30.07,
                26.55,
                7.85,
                83.47,
                13.0,
                7.9,
                26.0,
                15.5,
            ],
            "FamilySize": [
                2,
                2,
                1,
                2,
                1,
                1,
                5,
                1,
                3,
                1,
                1,
                2,
                1,
                1,
                2,
                1,
            ],
            "IsAlone": [
                0,
                0,
                1,
                0,
                1,
                1,
                0,
                1,
                0,
                1,
                1,
                0,
                1,
                1,
                0,
                1,
            ],
            "FarePerPerson": [
                3.63,
                35.64,
                7.93,
                26.55,
                8.05,
                51.86,
                4.22,
                11.13,
                10.02,
                26.55,
                7.85,
                41.74,
                13.0,
                7.9,
                13.0,
                15.5,
            ],
            "Pclass": [
                3,
                1,
                3,
                1,
                3,
                1,
                3,
                3,
                2,
                1,
                3,
                1,
                2,
                3,
                2,
                3,
            ],
            "Sex": [
                "male",
                "female",
                "female",
                "female",
                "male",
                "male",
                "male",
                "female",
                "female",
                "female",
                "male",
                "female",
                "male",
                "male",
                "female",
                "male",
            ],
            "Embarked": [
                "S",
                "C",
                "S",
                "S",
                "S",
                "S",
                "S",
                "S",
                "C",
                "S",
                "S",
                "C",
                "S",
                "S",
                "C",
                "Q",
            ],
            "Title": [
                "Mr",
                "Mrs",
                "Miss",
                "Mrs",
                "Mr",
                "Mr",
                "Master",
                "Miss",
                "Miss",
                "Mrs",
                "Mr",
                "Mrs",
                "Mr",
                "Mr",
                "Mrs",
                "Mr",
            ],
            "Deck": [
                "Unknown",
                "C",
                "Unknown",
                "C",
                "Unknown",
                "E",
                "Unknown",
                "Unknown",
                "Unknown",
                "C",
                "Unknown",
                "B",
                "Unknown",
                "Unknown",
                "D",
                "Unknown",
            ],
            "Survived": [
                0,
                1,
                1,
                1,
                0,
                0,
                1,
                1,
                1,
                1,
                0,
                1,
                0,
                0,
                1,
                0,
            ],
        }
    )


def test_evaluate_model_returns_valid_metrics() -> None:
    dataframe = make_evaluation_dataframe()
    results = compare_models(dataframe)

    metrics = evaluate_model(results["Logistic Regression"])

    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.f1 <= 1.0
    assert 0.0 <= metrics.roc_auc <= 1.0


def test_compare_model_metrics_contains_all_models() -> None:
    dataframe = make_evaluation_dataframe()
    results = compare_models(dataframe)

    comparison = compare_model_metrics(results)

    assert len(comparison) == 3
    assert "Model" in comparison.columns
    assert "Accuracy" in comparison.columns
    assert "Precision" in comparison.columns
    assert "Recall" in comparison.columns
    assert "F1" in comparison.columns
    assert "ROC-AUC" in comparison.columns


def test_compare_model_metrics_is_sorted_by_f1() -> None:
    dataframe = make_evaluation_dataframe()
    results = compare_models(dataframe)

    comparison = compare_model_metrics(results)
    f1_scores = comparison["F1"].tolist()

    assert f1_scores == sorted(f1_scores, reverse=True)


def test_save_confusion_matrix_creates_file(
    tmp_path: Path,
) -> None:
    dataframe = make_evaluation_dataframe()
    results = compare_models(dataframe)

    output = tmp_path / "confusion_matrix.png"

    result = save_confusion_matrix(
        results["Logistic Regression"],
        "Logistic Regression",
        output,
    )

    assert result.exists()
    assert result.stat().st_size > 0