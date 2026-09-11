"""Titanic Survival Analyzer.

A Python package for analyzing Titanic passenger data
and predicting passenger survival.
"""

from titanic_analyzer.analysis import SurvivalAnalyzer
from titanic_analyzer.loader import load_titanic_data
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

__version__ = "0.1.0"

__all__ = [
    "DatasetValidationError",
    "SurvivalAnalyzer",
    "fill_missing_age",
    "fill_missing_cabin",
    "fill_missing_embarked",
    "load_titanic_data",
    "preprocess_data",
    "validate_titanic_data",
]
