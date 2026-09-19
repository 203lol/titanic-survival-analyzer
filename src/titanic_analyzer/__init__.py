"""Titanic Survival Analyzer"""

from titanic_analyzer.analysis import SurvivalAnalyzer
from titanic_analyzer.evaluation import (
    EvaluationMetrics,
    compare_model_metrics,
    evaluate_model,
    save_all_confusion_matrices,
)
from titanic_analyzer.features import engineer_features
from titanic_analyzer.loader import load_titanic_data
from titanic_analyzer.models import (
    ModelResult,
    compare_models,
    train_decision_tree,
    train_logistic_regression,
    train_random_forest,
)
from titanic_analyzer.persistence import load_model, save_model
from titanic_analyzer.prediction import (
    Passenger,
    PassengerPrediction,
    predict_passenger_survival,
)
from titanic_analyzer.preprocessing import preprocess_data
from titanic_analyzer.reporting import generate_reports
from titanic_analyzer.validation import (
    DatasetValidationError,
    validate_titanic_data,
)
from titanic_analyzer.visualization import generate_all_plots

__version__ = "0.1.0"

__all__ = [
    "DatasetValidationError",
    "EvaluationMetrics",
    "ModelResult",
    "Passenger",
    "PassengerPrediction",
    "SurvivalAnalyzer",
    "compare_model_metrics",
    "compare_models",
    "engineer_features",
    "evaluate_model",
    "generate_all_plots",
    "generate_reports",
    "load_model",
    "load_titanic_data",
    "predict_passenger_survival",
    "preprocess_data",
    "save_all_confusion_matrices",
    "save_model",
    "train_decision_tree",
    "train_logistic_regression",
    "train_random_forest",
    "validate_titanic_data",
]