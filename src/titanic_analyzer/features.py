"""Feature engineering for Titanic passenger data."""

import re

import numpy as np
import pandas as pd

AGE_BINS = [0, 12, 18, 35, 60, np.inf]
AGE_LABELS = [
    "Child",
    "Teen",
    "Young Adult",
    "Adult",
    "Senior",
]


def add_family_size(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add the passenger's family size."""
    result = dataframe.copy()
    result["FamilySize"] = result["SibSp"] + result["Parch"] + 1

    return result


def add_is_alone(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add whether the passenger travelled alone."""
    result = dataframe.copy()

    if "FamilySize" not in result.columns:
        result = add_family_size(result)

    result["IsAlone"] = (result["FamilySize"] == 1).astype(int)

    return result


def extract_title_from_name(name: str) -> str:
    """Extract the title from a passenger name."""
    match = re.search(r",\s*([^.]+)\.", name)

    if match is None:
        return "Unknown"

    title = match.group(1).strip()

    title_mapping = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
    }

    return title_mapping.get(title, title)


def add_title(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add passenger titles."""
    result = dataframe.copy()
    result["Title"] = result["Name"].astype(str).apply(extract_title_from_name)

    return result


def add_age_group(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add passenger age groups."""
    result = dataframe.copy()

    result["AgeGroup"] = pd.cut(
        result["Age"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=False,
    )

    return result


def extract_deck(cabin: object) -> str:
    """Extract the deck from a cabin value."""
    if pd.isna(cabin):
        return "Unknown"

    cabin_text = str(cabin).strip()

    if not cabin_text or cabin_text.lower() == "unknown":
        return "Unknown"

    return cabin_text[0].upper()


def add_deck(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add the passenger deck."""
    result = dataframe.copy()
    result["Deck"] = result["Cabin"].apply(extract_deck)

    return result


def add_fare_per_person(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add fare per family member."""
    result = dataframe.copy()

    if "FamilySize" not in result.columns:
        result = add_family_size(result)

    result["FarePerPerson"] = result["Fare"] / result["FamilySize"]

    return result


def engineer_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered features."""
    result = dataframe.copy()

    result = add_family_size(result)
    result = add_is_alone(result)
    result = add_title(result)
    result = add_age_group(result)
    result = add_deck(result)
    result = add_fare_per_person(result)

    return result
