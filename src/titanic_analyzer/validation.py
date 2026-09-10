"""Validation utilities for Titanic datasets."""

from collections.abc import Iterable

import pandas as pd

REQUIRED_COLUMNS = {
    "PassengerId",
    "Survived",
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin",
    "Embarked",
}


class DatasetValidationError(ValueError):
    """Raised when a Titanic dataset does not have the expected structure."""


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: Iterable[str] = REQUIRED_COLUMNS,
) -> None:
    """Validate that all required Titanic columns are present.

    Parameters
    ----------
    dataframe:
        The pandas DataFrame to validate.
    required_columns:
        Column names that must be present in the DataFrame.

    Raises
    ------
    DatasetValidationError
        If one or more required columns are missing.
    """
    required = set(required_columns)
    missing_columns = required.difference(dataframe.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise DatasetValidationError(f"Dataset is missing required columns: {missing}")


def validate_non_empty(dataframe: pd.DataFrame) -> None:
    """Validate that the dataset contains at least one row.

    Parameters
    ----------
    dataframe:
        The pandas DataFrame to validate.

    Raises
    ------
    DatasetValidationError
        If the DataFrame contains no rows.
    """
    if dataframe.empty:
        raise DatasetValidationError("Dataset contains no rows.")


def validate_titanic_data(dataframe: pd.DataFrame) -> None:
    """Run all basic structural validations for a Titanic dataset.

    Parameters
    ----------
    dataframe:
        The pandas DataFrame to validate.

    Raises
    ------
    DatasetValidationError
        If the dataset fails validation.
    """
    validate_non_empty(dataframe)
    validate_required_columns(dataframe)
