"""Validation for Titanic passenger data."""

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

VALID_SURVIVED_VALUES = {0, 1}
VALID_PASSENGER_CLASSES = {1, 2, 3}
VALID_SEX_VALUES = {"male", "female"}


class DatasetValidationError(ValueError):
    """Raised when the dataset fails validation."""


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: Iterable[str] = REQUIRED_COLUMNS,
) -> None:
    """Check that all required columns are present."""
    required = set(required_columns)
    missing_columns = required.difference(dataframe.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise DatasetValidationError(
            f"Dataset is missing required columns: {missing}"
        )


def validate_non_empty(dataframe: pd.DataFrame) -> None:
    """Check that the dataset is not empty."""
    if dataframe.empty:
        raise DatasetValidationError("Dataset contains no rows.")


def validate_survived_values(dataframe: pd.DataFrame) -> None:
    """Check values in the Survived column."""
    values = set(dataframe["Survived"].dropna().unique())
    invalid_values = values.difference(VALID_SURVIVED_VALUES)

    if invalid_values:
        raise DatasetValidationError(
            "Survived column must contain only 0 or 1."
        )


def validate_passenger_classes(dataframe: pd.DataFrame) -> None:
    """Check passenger class values."""
    values = set(dataframe["Pclass"].dropna().unique())
    invalid_values = values.difference(VALID_PASSENGER_CLASSES)

    if invalid_values:
        raise DatasetValidationError(
            "Pclass column must contain only 1, 2, or 3."
        )


def validate_sex_values(dataframe: pd.DataFrame) -> None:
    """Check values in the Sex column."""
    values = set(
        dataframe["Sex"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
    )
    invalid_values = values.difference(VALID_SEX_VALUES)

    if invalid_values:
        raise DatasetValidationError(
            "Sex column must contain only 'male' or 'female'."
        )


def validate_numeric_ranges(dataframe: pd.DataFrame) -> None:
    """Check numeric passenger values."""
    if (dataframe["Age"].dropna() < 0).any():
        raise DatasetValidationError("Age values cannot be negative.")

    if (dataframe["Fare"].dropna() < 0).any():
        raise DatasetValidationError("Fare values cannot be negative.")

    if (dataframe["SibSp"].dropna() < 0).any():
        raise DatasetValidationError("SibSp values cannot be negative.")

    if (dataframe["Parch"].dropna() < 0).any():
        raise DatasetValidationError("Parch values cannot be negative.")


def validate_titanic_data(dataframe: pd.DataFrame) -> None:
    """Run all dataset validation checks."""
    validate_non_empty(dataframe)
    validate_required_columns(dataframe)
    validate_survived_values(dataframe)
    validate_passenger_classes(dataframe)
    validate_sex_values(dataframe)
    validate_numeric_ranges(dataframe)