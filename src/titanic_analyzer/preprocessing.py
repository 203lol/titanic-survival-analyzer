"""Preprocessing for Titanic passenger data."""

import pandas as pd


def fill_missing_age(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Fill missing ages with the median age."""
    result = dataframe.copy()

    median_age = result["Age"].median()
    result["Age"] = result["Age"].fillna(median_age)

    return result


def fill_missing_embarked(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Fill missing embarkation ports with the most common value."""
    result = dataframe.copy()

    mode = result["Embarked"].mode()

    if not mode.empty:
        result["Embarked"] = result["Embarked"].fillna(mode.iloc[0])

    return result


def fill_missing_cabin(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Replace missing cabin values with Unknown."""
    result = dataframe.copy()

    result["Cabin"] = result["Cabin"].fillna("Unknown")

    return result


def normalize_sex(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize values in the Sex column."""
    result = dataframe.copy()

    result["Sex"] = result["Sex"].astype(str).str.strip().str.lower()

    return result


def normalize_embarked(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize values in the Embarked column."""
    result = dataframe.copy()

    result["Embarked"] = result["Embarked"].astype(str).str.strip().str.upper()

    return result


def preprocess_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply the preprocessing steps."""
    result = dataframe.copy()

    result = fill_missing_age(result)
    result = fill_missing_embarked(result)
    result = fill_missing_cabin(result)
    result = normalize_sex(result)
    result = normalize_embarked(result)

    return result
