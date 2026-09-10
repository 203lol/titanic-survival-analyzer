"""Titanic Survival Analyzer.

A Python package for analyzing Titanic passenger data
and predicting passenger survival.
"""

from titanic_analyzer.loader import load_titanic_data
from titanic_analyzer.validation import (
    DatasetValidationError,
    validate_titanic_data,
)

__version__ = "0.1.0"

__all__ = [
    "DatasetValidationError",
    "load_titanic_data",
    "validate_titanic_data",
]
