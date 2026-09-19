"""Tests for saving and loading trained models."""

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from titanic_analyzer.models import train_logistic_regression
from titanic_analyzer.persistence import load_model, save_model


def make_model_dataframe() -> pd.DataFrame:
    """Create a small dataset for persistence tests."""
    return pd.DataFrame(
        {
            "Age": [22, 38, 26, 35, 28, 54, 2, 27],
            "Fare": [7.25, 71.28, 7.93, 53.1, 8.05, 51.86, 21.08, 11.13],
            "FamilySize": [2, 2, 1, 2, 1, 1, 5, 1],
            "IsAlone": [0, 0, 1, 0, 1, 1, 0, 1],
            "FarePerPerson": [
                3.63,
                35.64,
                7.93,
                26.55,
                8.05,
                51.86,
                4.22,
                11.13,
            ],
            "Pclass": [3, 1, 3, 1, 3, 1, 3, 3],
            "Sex": [
                "male",
                "female",
                "female",
                "female",
                "male",
                "male",
                "male",
                "female",
            ],
            "Embarked": ["S", "C", "S", "S", "S", "S", "S", "S"],
            "Title": ["Mr", "Mrs", "Miss", "Mrs", "Mr", "Mr", "Master", "Miss"],
            "Deck": [
                "Unknown",
                "C",
                "Unknown",
                "C",
                "Unknown",
                "E",
                "Unknown",
                "Unknown",
            ],
            "Survived": [0, 1, 1, 1, 0, 0, 1, 1],
        }
    )


def test_save_and_load_model(tmp_path) -> None:
    dataframe = make_model_dataframe()

    result = train_logistic_regression(
        dataframe,
        test_size=0.25,
        random_state=42,
    )

    model_path = tmp_path / "model.joblib"

    saved_path = save_model(result.model, model_path)
    loaded_model = load_model(saved_path)

    assert saved_path.exists()
    assert isinstance(loaded_model, Pipeline)


def test_load_missing_model_raises_error(tmp_path) -> None:
    model_path = tmp_path / "missing.joblib"

    with pytest.raises(
        FileNotFoundError,
        match="Model not found",
    ):
        load_model(model_path)