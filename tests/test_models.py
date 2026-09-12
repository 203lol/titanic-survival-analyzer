"""Tests for Titanic machine-learning models."""

import pandas as pd

from titanic_analyzer.models import (
    build_logistic_regression_pipeline,
    prepare_model_data,
    train_logistic_regression,
)


def make_model_dataframe() -> pd.DataFrame:
    """Create a small Titanic-like dataset for model tests."""
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
            ],
        }
    )


def test_prepare_model_data() -> None:
    dataframe = make_model_dataframe()

    features, target = prepare_model_data(dataframe)

    assert "Survived" not in features.columns
    assert len(features) == len(target)


def test_pipeline_contains_expected_steps() -> None:
    pipeline = build_logistic_regression_pipeline()

    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_train_logistic_regression_returns_accuracy() -> None:
    dataframe = make_model_dataframe()

    result = train_logistic_regression(
        dataframe,
        test_size=0.25,
        random_state=42,
    )

    assert 0.0 <= result.accuracy <= 1.0


def test_predictions_match_test_size() -> None:
    dataframe = make_model_dataframe()

    result = train_logistic_regression(
        dataframe,
        test_size=0.25,
        random_state=42,
    )

    assert len(result.predictions) == len(result.y_test)


def test_missing_model_column_raises_error() -> None:
    dataframe = make_model_dataframe().drop(columns=["Title"])

    try:
        prepare_model_data(dataframe)
    except ValueError as error:
        assert "Title" in str(error)
    else:
        raise AssertionError("Expected ValueError")
