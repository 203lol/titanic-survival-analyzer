"""Models for Titanic survival prediction."""

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

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
    """Store a trained model and its test results."""

    model: Pipeline
    accuracy: float
    x_test: pd.DataFrame
    y_test: pd.Series
    predictions: pd.Series


def prepare_model_data(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target for model training."""
    required_columns = set(
        NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
    )

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing model columns: {missing}")

    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    features = dataframe[feature_columns].copy()
    target = dataframe[TARGET_COLUMN].copy()

    return features, target


def build_preprocessor() -> ColumnTransformer:
    """Build the preprocessing steps used by the models."""
    return ColumnTransformer(
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


def build_logistic_regression_pipeline() -> Pipeline:
    """Build the Logistic Regression pipeline."""
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def build_decision_tree_pipeline() -> Pipeline:
    """Build the Decision Tree pipeline."""
    model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def build_random_forest_pipeline() -> Pipeline:
    """Build the Random Forest pipeline."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def train_model(
    dataframe: pd.DataFrame,
    pipeline: Pipeline,
    test_size: float = 0.2,
    random_state: int = 42,
) -> ModelResult:
    """Train a model and return its test results."""
    features, target = prepare_model_data(dataframe)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

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
        x_test=x_test,
        y_test=y_test,
        predictions=predictions,
    )


def train_logistic_regression(
    dataframe: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> ModelResult:
    """Train a Logistic Regression model."""
    return train_model(
        dataframe,
        build_logistic_regression_pipeline(),
        test_size=test_size,
        random_state=random_state,
    )


def train_decision_tree(
    dataframe: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> ModelResult:
    """Train a Decision Tree model."""
    return train_model(
        dataframe,
        build_decision_tree_pipeline(),
        test_size=test_size,
        random_state=random_state,
    )


def train_random_forest(
    dataframe: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> ModelResult:
    """Train a Random Forest model."""
    return train_model(
        dataframe,
        build_random_forest_pipeline(),
        test_size=test_size,
        random_state=random_state,
    )


def compare_models(
    dataframe: pd.DataFrame,
) -> dict[str, ModelResult]:
    """Train all available models."""
    return {
        "Logistic Regression": train_logistic_regression(dataframe),
        "Decision Tree": train_decision_tree(dataframe),
        "Random Forest": train_random_forest(dataframe),
    }