"""Tests for Titanic dataset validation."""

import pandas as pd
import pytest

from titanic_analyzer.validation import (
    REQUIRED_COLUMNS,
    DatasetValidationError,
    validate_non_empty,
    validate_required_columns,
    validate_titanic_data,
)


def make_valid_dataframe() -> pd.DataFrame:
    """Create a minimal structurally valid Titanic DataFrame."""
    row = {column: 0 for column in REQUIRED_COLUMNS}

    row["Name"] = "Test Passenger"
    row["Sex"] = "male"
    row["Ticket"] = "TEST"
    row["Cabin"] = "C1"
    row["Embarked"] = "S"

    return pd.DataFrame([row])


def test_required_columns_accept_valid_dataframe() -> None:
    """Validation should accept a DataFrame with all required columns."""
    dataframe = make_valid_dataframe()

    validate_required_columns(dataframe)


def test_missing_required_column_raises_error() -> None:
    """Validation should report missing required columns."""
    dataframe = make_valid_dataframe()
    dataframe = dataframe.drop(columns=["Age"])

    with pytest.raises(
        DatasetValidationError,
        match="Age",
    ):
        validate_required_columns(dataframe)


def test_multiple_missing_columns_are_reported() -> None:
    """Validation should report all missing required columns."""
    dataframe = make_valid_dataframe()
    dataframe = dataframe.drop(columns=["Age", "Fare"])

    with pytest.raises(DatasetValidationError) as error:
        validate_required_columns(dataframe)

    message = str(error.value)

    assert "Age" in message
    assert "Fare" in message


def test_empty_dataframe_raises_error() -> None:
    """Validation should reject a DataFrame with no rows."""
    dataframe = pd.DataFrame(columns=sorted(REQUIRED_COLUMNS))

    with pytest.raises(
        DatasetValidationError,
        match="contains no rows",
    ):
        validate_non_empty(dataframe)


def test_validate_titanic_data_accepts_valid_data() -> None:
    """Combined validation should accept valid Titanic data."""
    dataframe = make_valid_dataframe()

    validate_titanic_data(dataframe)
