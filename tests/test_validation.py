"""Tests for Titanic dataset validation."""

import pandas as pd
import pytest

from titanic_analyzer.validation import (
    REQUIRED_COLUMNS,
    DatasetValidationError,
    validate_non_empty,
    validate_numeric_ranges,
    validate_passenger_classes,
    validate_required_columns,
    validate_sex_values,
    validate_survived_values,
    validate_titanic_data,
)


def make_valid_dataframe() -> pd.DataFrame:
    """Create a minimal valid Titanic DataFrame."""
    row = {column: 0 for column in REQUIRED_COLUMNS}

    row["Pclass"] = 3
    row["Name"] = "Test Passenger"
    row["Sex"] = "male"
    row["Ticket"] = "TEST"
    row["Cabin"] = "C1"
    row["Embarked"] = "S"

    return pd.DataFrame([row])


def test_required_columns_accept_valid_dataframe() -> None:
    dataframe = make_valid_dataframe()

    validate_required_columns(dataframe)


def test_missing_required_column_raises_error() -> None:
    dataframe = make_valid_dataframe().drop(columns=["Age"])

    with pytest.raises(
        DatasetValidationError,
        match="Age",
    ):
        validate_required_columns(dataframe)


def test_multiple_missing_columns_are_reported() -> None:
    dataframe = make_valid_dataframe().drop(columns=["Age", "Fare"])

    with pytest.raises(DatasetValidationError) as error:
        validate_required_columns(dataframe)

    message = str(error.value)

    assert "Age" in message
    assert "Fare" in message


def test_empty_dataframe_raises_error() -> None:
    dataframe = pd.DataFrame(columns=sorted(REQUIRED_COLUMNS))

    with pytest.raises(
        DatasetValidationError,
        match="contains no rows",
    ):
        validate_non_empty(dataframe)


def test_non_empty_dataframe_is_accepted() -> None:
    dataframe = make_valid_dataframe()

    validate_non_empty(dataframe)


def test_validate_titanic_data_accepts_valid_data() -> None:
    dataframe = make_valid_dataframe()

    validate_titanic_data(dataframe)


def test_valid_survived_values_are_accepted() -> None:
    dataframe = pd.DataFrame(
        {
            "Survived": [0, 1],
        }
    )

    validate_survived_values(dataframe)


def test_invalid_survived_value_raises_error() -> None:
    dataframe = pd.DataFrame(
        {
            "Survived": [0, 1, 2],
        }
    )

    with pytest.raises(
        DatasetValidationError,
        match="Survived",
    ):
        validate_survived_values(dataframe)


def test_valid_passenger_classes_are_accepted() -> None:
    dataframe = pd.DataFrame(
        {
            "Pclass": [1, 2, 3],
        }
    )

    validate_passenger_classes(dataframe)


def test_invalid_passenger_class_raises_error() -> None:
    dataframe = pd.DataFrame(
        {
            "Pclass": [1, 2, 3, 5],
        }
    )

    with pytest.raises(
        DatasetValidationError,
        match="Pclass",
    ):
        validate_passenger_classes(dataframe)


def test_valid_sex_values_are_accepted() -> None:
    dataframe = pd.DataFrame(
        {
            "Sex": ["male", "female"],
        }
    )

    validate_sex_values(dataframe)


def test_sex_validation_is_case_insensitive() -> None:
    dataframe = pd.DataFrame(
        {
            "Sex": ["MALE", "Female"],
        }
    )

    validate_sex_values(dataframe)


def test_invalid_sex_value_raises_error() -> None:
    dataframe = pd.DataFrame(
        {
            "Sex": ["male", "female", "unknown"],
        }
    )

    with pytest.raises(
        DatasetValidationError,
        match="Sex",
    ):
        validate_sex_values(dataframe)


def test_valid_numeric_ranges_are_accepted() -> None:
    dataframe = pd.DataFrame(
        {
            "Age": [0, 25, 80],
            "Fare": [0.0, 10.0, 100.0],
            "SibSp": [0, 1, 2],
            "Parch": [0, 1, 3],
        }
    )

    validate_numeric_ranges(dataframe)


def test_negative_age_raises_error() -> None:
    dataframe = pd.DataFrame(
        {
            "Age": [-5],
            "Fare": [10.0],
            "SibSp": [0],
            "Parch": [0],
        }
    )

    with pytest.raises(
        DatasetValidationError,
        match="Age",
    ):
        validate_numeric_ranges(dataframe)


def test_negative_fare_raises_error() -> None:
    dataframe = pd.DataFrame(
        {
            "Age": [25],
            "Fare": [-10.0],
            "SibSp": [0],
            "Parch": [0],
        }
    )

    with pytest.raises(
        DatasetValidationError,
        match="Fare",
    ):
        validate_numeric_ranges(dataframe)


def test_negative_sibsp_raises_error() -> None:
    dataframe = pd.DataFrame(
        {
            "Age": [25],
            "Fare": [10.0],
            "SibSp": [-1],
            "Parch": [0],
        }
    )

    with pytest.raises(
        DatasetValidationError,
        match="SibSp",
    ):
        validate_numeric_ranges(dataframe)


def test_negative_parch_raises_error() -> None:
    dataframe = pd.DataFrame(
        {
            "Age": [25],
            "Fare": [10.0],
            "SibSp": [0],
            "Parch": [-1],
        }
    )

    with pytest.raises(
        DatasetValidationError,
        match="Parch",
    ):
        validate_numeric_ranges(dataframe)