"""Tests for individual passenger survival prediction."""

import pandas as pd
import pytest

from titanic_analyzer.models import train_logistic_regression
from titanic_analyzer.prediction import (
    Passenger,
    passenger_to_dataframe,
    predict_passenger_survival,
    validate_passenger,
)


def make_prediction_dataframe() -> pd.DataFrame:
    """Create a small Titanic-like dataset for prediction tests."""
    return pd.DataFrame(
        {
            "Age": [22, 38, 26, 35, 28, 54, 2, 27, 14, 58, 20, 45],
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
            "FamilySize": [2, 2, 1, 2, 1, 1, 5, 1, 3, 1, 1, 2],
            "IsAlone": [0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
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
            "Pclass": [3, 1, 3, 1, 3, 1, 3, 3, 2, 1, 3, 1],
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
            "Survived": [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
        }
    )


def test_passenger_to_dataframe() -> None:
    passenger = Passenger(
        pclass=3,
        sex="male",
        age=25,
        fare=15,
        sibsp=1,
        parch=0,
        embarked="S",
        title="Mr",
    )

    result = passenger_to_dataframe(passenger)

    assert len(result) == 1
    assert result.loc[0, "FamilySize"] == 2
    assert result.loc[0, "IsAlone"] == 0
    assert result.loc[0, "FarePerPerson"] == pytest.approx(7.5)


def test_alone_passenger_feature() -> None:
    passenger = Passenger(
        pclass=2,
        sex="female",
        age=30,
        fare=20,
    )

    result = passenger_to_dataframe(passenger)

    assert result.loc[0, "FamilySize"] == 1
    assert result.loc[0, "IsAlone"] == 1
    assert result.loc[0, "FarePerPerson"] == pytest.approx(20.0)


def test_passenger_defaults() -> None:
    passenger = Passenger(
        pclass=3,
        sex="male",
        age=25,
        fare=15,
    )

    assert passenger.sibsp == 0
    assert passenger.parch == 0
    assert passenger.embarked == "S"
    assert passenger.title == "Mr"
    assert passenger.deck == "Unknown"


def test_invalid_passenger_class() -> None:
    passenger = Passenger(
        pclass=4,
        sex="male",
        age=25,
        fare=10,
    )

    with pytest.raises(ValueError, match="Passenger class"):
        validate_passenger(passenger)


def test_invalid_sex_is_rejected() -> None:
    passenger = Passenger(
        pclass=3,
        sex="unknown",
        age=25,
        fare=10,
    )

    with pytest.raises(ValueError, match="Sex"):
        validate_passenger(passenger)


def test_negative_age_is_rejected() -> None:
    passenger = Passenger(
        pclass=3,
        sex="male",
        age=-1,
        fare=10,
    )

    with pytest.raises(ValueError, match="Age cannot"):
        validate_passenger(passenger)


def test_negative_fare_is_rejected() -> None:
    passenger = Passenger(
        pclass=3,
        sex="male",
        age=25,
        fare=-10,
    )

    with pytest.raises(ValueError, match="Fare cannot"):
        validate_passenger(passenger)


def test_negative_sibsp_is_rejected() -> None:
    passenger = Passenger(
        pclass=3,
        sex="male",
        age=25,
        fare=10,
        sibsp=-1,
    )

    with pytest.raises(ValueError, match="SibSp cannot"):
        validate_passenger(passenger)


def test_negative_parch_is_rejected() -> None:
    passenger = Passenger(
        pclass=3,
        sex="male",
        age=25,
        fare=10,
        parch=-1,
    )

    with pytest.raises(ValueError, match="Parch cannot"):
        validate_passenger(passenger)


def test_invalid_embarked_is_rejected() -> None:
    passenger = Passenger(
        pclass=3,
        sex="male",
        age=25,
        fare=10,
        embarked="X",
    )

    with pytest.raises(ValueError, match="Embarked"):
        validate_passenger(passenger)


def test_passenger_values_are_normalized() -> None:
    passenger = Passenger(
        pclass=2,
        sex=" Female ",
        age=30,
        fare=20,
        embarked=" c ",
        title=" Miss ",
        deck=" C ",
    )

    result = passenger_to_dataframe(passenger)

    assert result.loc[0, "Sex"] == "female"
    assert result.loc[0, "Embarked"] == "C"
    assert result.loc[0, "Title"] == "Miss"
    assert result.loc[0, "Deck"] == "C"


def test_predict_passenger_survival() -> None:
    dataframe = make_prediction_dataframe()

    result = train_logistic_regression(
        dataframe,
        test_size=0.25,
        random_state=42,
    )

    passenger = Passenger(
        pclass=3,
        sex="male",
        age=25,
        fare=15,
        embarked="S",
        title="Mr",
    )

    prediction = predict_passenger_survival(
        result.model,
        passenger,
    )

    assert isinstance(prediction.survived, bool)
    assert 0.0 <= prediction.survival_probability <= 1.0