"""Machine-learning models for Titanic survival prediction."""

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "Survived"

NUMERIC_FEATURES = [
    "Age",
    "Fare",
    "FamilySize",
    "IsAlone",
    "FarePerPerson",
]

CATEGORICAL_FEATURES = [
    "Pclass",
    "Sex",
    "Embarked",
    "Title",
    "Deck",
]


@dataclass
class ModelResult:
    """Store a trained model and its test performance."""

    model: Pipeline
    accuracy: float
    y_test: pd.Series
    predictions: pd.Series


def prepare_model_data(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix and target vector for survival prediction."""
    required_columns = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN])

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing model columns: {missing}")

    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    features = dataframe[feature_columns].copy()
    target = dataframe[TARGET_COLUMN].copy()

    return features, target


def build_logistic_regression_pipeline() -> Pipeline:
    """Build a preprocessing and Logistic Regression pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def train_logistic_regression(
    dataframe: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> ModelResult:
    """Train and evaluate a Logistic Regression survival model."""
    features, target = prepare_model_data(dataframe)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    pipeline = build_logistic_regression_pipeline()

    pipeline.fit(x_train, y_train)

    predictions_array = pipeline.predict(x_test)

    predictions = pd.Series(
        predictions_array,
        index=y_test.index,
        name="Prediction",
    )

    accuracy = accuracy_score(y_test, predictions)

    return ModelResult(
        model=pipeline,
        accuracy=float(accuracy),
        y_test=y_test,
        predictions=predictions,
    )
