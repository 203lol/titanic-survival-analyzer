"""Titanic Survival Analyzer.

A Python package for analyzing Titanic passenger data
and predicting passenger survival.
"""

from titanic_analyzer.analysis import SurvivalAnalyzer
from titanic_analyzer.evaluation import (
    EvaluationMetrics,
    compare_model_metrics,
    evaluate_model,
    save_all_confusion_matrices,
    save_confusion_matrix,
)
from titanic_analyzer.features import (
    add_age_group,
    add_deck,
    add_family_size,
    add_fare_per_person,
    add_is_alone,
    add_title,
    engineer_features,
)
from titanic_analyzer.loader import load_titanic_data
from titanic_analyzer.models import (
    ModelResult,
    build_decision_tree_pipeline,
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
    compare_models,
    prepare_model_data,
    train_decision_tree,
    train_logistic_regression,
    train_model,
    train_random_forest,
)
from titanic_analyzer.prediction import (
    Passenger,
    PassengerPrediction,
    passenger_to_dataframe,
    predict_passenger_survival,
    validate_passenger,
)
from titanic_analyzer.preprocessing import (
    fill_missing_age,
    fill_missing_cabin,
    fill_missing_embarked,
    preprocess_data,
)
from titanic_analyzer.validation import (
    DatasetValidationError,
    validate_titanic_data,
)
from titanic_analyzer.visualization import (
    generate_all_plots,
    plot_age_distribution,
    plot_fare_distribution,
    plot_survival_by_class,
    plot_survival_by_family_size,
    plot_survival_by_sex,
)

__version__ = "0.1.0"

__all__ = [
    "DatasetValidationError",
    "EvaluationMetrics",
    "ModelResult",
    "Passenger",
    "PassengerPrediction",
    "SurvivalAnalyzer",
    "add_age_group",
    "add_deck",
    "add_family_size",
    "add_fare_per_person",
    "add_is_alone",
    "add_title",
    "build_decision_tree_pipeline",
    "build_logistic_regression_pipeline",
    "build_random_forest_pipeline",
    "compare_model_metrics",
    "compare_models",
    "engineer_features",
    "evaluate_model",
    "fill_missing_age",
    "fill_missing_cabin",
    "fill_missing_embarked",
    "generate_all_plots",
    "load_titanic_data",
    "passenger_to_dataframe",
    "plot_age_distribution",
    "plot_fare_distribution",
    "plot_survival_by_class",
    "plot_survival_by_family_size",
    "plot_survival_by_sex",
    "predict_passenger_survival",
    "prepare_model_data",
    "preprocess_data",
    "save_all_confusion_matrices",
    "save_confusion_matrix",
    "train_decision_tree",
    "train_logistic_regression",
    "train_model",
    "train_random_forest",
    "validate_passenger",
    "validate_titanic_data",
]
