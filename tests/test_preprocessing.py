"""Tests for Titanic data preprocessing."""

import pandas as pd

from titanic_analyzer.preprocessing import (
    fill_missing_age,
    fill_missing_cabin,
    fill_missing_embarked,
    normalize_embarked,
    normalize_sex,
    preprocess_data,
)


def make_test_dataframe() -> pd.DataFrame:
    """Create a small Titanic-like DataFrame for preprocessing tests."""
    return pd.DataFrame(
        {
            "Age": [22.0, None, 35.0],
            "Cabin": ["C85", None, "E46"],
            "Embarked": ["s", None, "S"],
            "Sex": [" Male ", "FEMALE", "male"],
        }
    )


def test_fill_missing_age_removes_missing_values() -> None:
    dataframe = make_test_dataframe()

    result = fill_missing_age(dataframe)

    assert result["Age"].isna().sum() == 0


def test_fill_missing_age_uses_median() -> None:
    dataframe = make_test_dataframe()

    result = fill_missing_age(dataframe)

    expected_median = 28.5
    assert result.loc[1, "Age"] == expected_median


def test_fill_missing_cabin_uses_unknown() -> None:
    dataframe = make_test_dataframe()

    result = fill_missing_cabin(dataframe)

    assert result.loc[1, "Cabin"] == "Unknown"


def test_fill_missing_embarked_removes_missing_values() -> None:
    dataframe = make_test_dataframe()

    result = fill_missing_embarked(dataframe)

    assert result["Embarked"].isna().sum() == 0


def test_normalize_sex() -> None:
    dataframe = make_test_dataframe()

    result = normalize_sex(dataframe)

    assert list(result["Sex"]) == ["male", "female", "male"]


def test_normalize_embarked() -> None:
    dataframe = make_test_dataframe()

    result = normalize_embarked(dataframe)

    assert result.loc[0, "Embarked"] == "S"


def test_preprocess_data_does_not_modify_original() -> None:
    dataframe = make_test_dataframe()
    original = dataframe.copy(deep=True)

    preprocess_data(dataframe)

    pd.testing.assert_frame_equal(dataframe, original)


def test_preprocess_data_removes_expected_missing_values() -> None:
    dataframe = make_test_dataframe()

    result = preprocess_data(dataframe)

    assert result["Age"].isna().sum() == 0
    assert result["Cabin"].isna().sum() == 0
    assert result["Embarked"].isna().sum() == 0
