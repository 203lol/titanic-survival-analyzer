"""Tests for Titanic feature engineering."""

import pandas as pd
import pytest

from titanic_analyzer.features import (
    add_age_group,
    add_deck,
    add_family_size,
    add_fare_per_person,
    add_is_alone,
    add_title,
    engineer_features,
    extract_deck,
    extract_title_from_name,
)


def make_feature_dataframe() -> pd.DataFrame:
    """Create sample data for feature engineering tests."""
    return pd.DataFrame(
        {
            "Name": [
                "Braund, Mr. Owen Harris",
                "Cumings, Mrs. John Bradley",
                "Heikkinen, Miss. Laina",
            ],
            "Age": [22.0, 38.0, 10.0],
            "SibSp": [1, 1, 0],
            "Parch": [0, 0, 2],
            "Fare": [14.0, 80.0, 30.0],
            "Cabin": ["Unknown", "C85", "E46"],
        }
    )


def test_add_family_size() -> None:
    dataframe = make_feature_dataframe()

    result = add_family_size(dataframe)

    assert list(result["FamilySize"]) == [2, 2, 3]


def test_add_is_alone() -> None:
    dataframe = make_feature_dataframe()
    dataframe.loc[0, "SibSp"] = 0

    result = add_is_alone(dataframe)

    assert result.loc[0, "IsAlone"] == 1
    assert result.loc[1, "IsAlone"] == 0


def test_single_passenger_is_marked_alone() -> None:
    dataframe = pd.DataFrame(
        {
            "SibSp": [0],
            "Parch": [0],
        }
    )

    result = add_is_alone(dataframe)

    assert result.loc[0, "IsAlone"] == 1
    assert result.loc[0, "FamilySize"] == 1


def test_family_passenger_is_not_marked_alone() -> None:
    dataframe = pd.DataFrame(
        {
            "SibSp": [1],
            "Parch": [2],
        }
    )

    result = add_is_alone(dataframe)

    assert result.loc[0, "IsAlone"] == 0
    assert result.loc[0, "FamilySize"] == 4


def test_extract_title_from_name() -> None:
    title = extract_title_from_name("Braund, Mr. Owen Harris")

    assert title == "Mr"


def test_extract_title_without_title_returns_unknown() -> None:
    title = extract_title_from_name("Passenger Without Title")

    assert title == "Unknown"


def test_add_title() -> None:
    dataframe = make_feature_dataframe()

    result = add_title(dataframe)

    assert list(result["Title"]) == ["Mr", "Mrs", "Miss"]


def test_title_is_extracted_from_name() -> None:
    dataframe = pd.DataFrame(
        {
            "Name": ["Braund, Mr. Owen Harris"],
        }
    )

    result = add_title(dataframe)

    assert result.loc[0, "Title"] == "Mr"


def test_add_age_group() -> None:
    dataframe = make_feature_dataframe()

    result = add_age_group(dataframe)

    assert str(result.loc[0, "AgeGroup"]) == "Young Adult"
    assert str(result.loc[1, "AgeGroup"]) == "Adult"
    assert str(result.loc[2, "AgeGroup"]) == "Child"


def test_extract_deck() -> None:
    assert extract_deck("C85") == "C"
    assert extract_deck("E46") == "E"
    assert extract_deck("Unknown") == "Unknown"
    assert extract_deck(None) == "Unknown"


def test_add_deck() -> None:
    dataframe = make_feature_dataframe()

    result = add_deck(dataframe)

    assert list(result["Deck"]) == [
        "Unknown",
        "C",
        "E",
    ]


def test_missing_cabin_produces_unknown_deck() -> None:
    dataframe = pd.DataFrame(
        {
            "Cabin": [None],
        }
    )

    result = add_deck(dataframe)

    assert result.loc[0, "Deck"] == "Unknown"


def test_add_fare_per_person() -> None:
    dataframe = make_feature_dataframe()

    result = add_fare_per_person(dataframe)

    assert result.loc[0, "FarePerPerson"] == pytest.approx(7.0)
    assert result.loc[1, "FarePerPerson"] == pytest.approx(40.0)
    assert result.loc[2, "FarePerPerson"] == pytest.approx(10.0)


def test_engineer_features_adds_expected_columns() -> None:
    dataframe = make_feature_dataframe()

    result = engineer_features(dataframe)

    expected_columns = {
        "FamilySize",
        "IsAlone",
        "Title",
        "AgeGroup",
        "Deck",
        "FarePerPerson",
    }

    assert expected_columns.issubset(result.columns)


def test_engineer_features_does_not_modify_original() -> None:
    dataframe = make_feature_dataframe()
    original = dataframe.copy(deep=True)

    engineer_features(dataframe)

    pd.testing.assert_frame_equal(
        dataframe,
        original,
    )
