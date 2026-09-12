"""Titanic Survival Analyzer.

A Python package for analyzing Titanic passenger data
and predicting passenger survival.
"""

from titanic_analyzer.analysis import SurvivalAnalyzer
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
    "SurvivalAnalyzer",
    "add_age_group",
    "add_deck",
    "add_family_size",
    "add_fare_per_person",
    "add_is_alone",
    "add_title",
    "engineer_features",
    "fill_missing_age",
    "fill_missing_cabin",
    "fill_missing_embarked",
    "generate_all_plots",
    "load_titanic_data",
    "plot_age_distribution",
    "plot_fare_distribution",
    "plot_survival_by_class",
    "plot_survival_by_family_size",
    "plot_survival_by_sex",
    "preprocess_data",
    "validate_titanic_data",
]
