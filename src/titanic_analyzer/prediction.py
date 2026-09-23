"""Passenger survival prediction."""

from dataclasses import dataclass

import pandas as pd
from sklearn.pipeline import Pipeline


@dataclass
class Passenger:
    """Passenger information used for prediction."""

    pclass: int
    sex: str
    age: float
    fare: float
    sibsp: int = 0
    parch: int = 0
    embarked: str = "S"
    title: str = "Mr"
    deck: str = "Unknown"


@dataclass
class PassengerPrediction:
    """Result of a passenger survival prediction."""

    survived: bool
    survival_probability: float


def validate_passenger(passenger: Passenger) -> None:
    """Check passenger input values."""
    if passenger.pclass not in {1, 2, 3}:
        raise ValueError("Passenger class must be 1, 2, or 3.")

    if passenger.sex.strip().lower() not in {"male", "female"}:
        raise ValueError("Sex must be 'male' or 'female'.")

    if passenger.age < 0:
        raise ValueError("Age cannot be negative.")

    if passenger.fare < 0:
        raise ValueError("Fare cannot be negative.")

    if passenger.sibsp < 0:
        raise ValueError("SibSp cannot be negative.")

    if passenger.parch < 0:
        raise ValueError("Parch cannot be negative.")

    if passenger.embarked.strip().upper() not in {"C", "Q", "S"}:
        raise ValueError("Embarked must be C, Q, or S.")


def passenger_to_dataframe(passenger: Passenger) -> pd.DataFrame:
    """Convert passenger data to model input."""
    validate_passenger(passenger)

    family_size = passenger.sibsp + passenger.parch + 1
    is_alone = int(family_size == 1)
    fare_per_person = passenger.fare / family_size

    return pd.DataFrame(
        [
            {
                "Age": float(passenger.age),
                "Fare": float(passenger.fare),
                "FamilySize": family_size,
                "IsAlone": is_alone,
                "FarePerPerson": fare_per_person,
                "Pclass": passenger.pclass,
                "Sex": passenger.sex.strip().lower(),
                "Embarked": passenger.embarked.strip().upper(),
                "Title": passenger.title.strip(),
                "Deck": passenger.deck.strip(),
            }
        ]
    )


def predict_passenger_survival(
    model: Pipeline,
    passenger: Passenger,
) -> PassengerPrediction:
    """Predict survival for one passenger."""
    passenger_data = passenger_to_dataframe(passenger)

    prediction = int(model.predict(passenger_data)[0])
    probability = float(model.predict_proba(passenger_data)[0, 1])

    return PassengerPrediction(
        survived=bool(prediction),
        survival_probability=probability,
    )
