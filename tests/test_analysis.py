"""Tests for Titanic survival analysis."""

import pandas as pd
import pytest

from titanic_analyzer.analysis import SurvivalAnalyzer


def make_analysis_dataframe() -> pd.DataFrame:
    """Create sample data for analysis tests."""
    return pd.DataFrame(
        {
            "Survived": [1, 0, 1, 0],
            "Sex": ["female", "male", "female", "male"],
            "Pclass": [1, 3, 2, 3],
            "Age": [25.0, 40.0, 10.0, 70.0],
            "Fare": [100.0, 10.0, 30.0, 8.0],
            "Embarked": ["S", "S", "C", "Q"],
        }
    )


def test_passenger_count() -> None:
    dataframe = make_analysis_dataframe()
    analyzer = SurvivalAnalyzer(dataframe)

    assert analyzer.passenger_count() == 4


def test_survivor_count() -> None:
    dataframe = make_analysis_dataframe()
    analyzer = SurvivalAnalyzer(dataframe)

    assert analyzer.survivor_count() == 2


def test_death_count() -> None:
    dataframe = make_analysis_dataframe()
    analyzer = SurvivalAnalyzer(dataframe)

    assert analyzer.death_count() == 2


def test_overall_survival_rate() -> None:
    dataframe = make_analysis_dataframe()
    analyzer = SurvivalAnalyzer(dataframe)

    assert analyzer.overall_survival_rate() == pytest.approx(50.0)


def test_survival_by_sex() -> None:
    dataframe = make_analysis_dataframe()
    analyzer = SurvivalAnalyzer(dataframe)

    result = analyzer.survival_by_sex()

    assert result["female"] == pytest.approx(100.0)
    assert result["male"] == pytest.approx(0.0)


def test_survival_by_class() -> None:
    dataframe = make_analysis_dataframe()
    analyzer = SurvivalAnalyzer(dataframe)

    result = analyzer.survival_by_class()

    assert result[1] == pytest.approx(100.0)
    assert result[2] == pytest.approx(100.0)
    assert result[3] == pytest.approx(0.0)


def test_survival_by_embarkation() -> None:
    dataframe = make_analysis_dataframe()
    analyzer = SurvivalAnalyzer(dataframe)

    result = analyzer.survival_by_embarkation()

    assert result["C"] == pytest.approx(100.0)
    assert result["Q"] == pytest.approx(0.0)
    assert result["S"] == pytest.approx(50.0)


def test_survival_by_age_group() -> None:
    dataframe = make_analysis_dataframe()
    analyzer = SurvivalAnalyzer(dataframe)

    result = analyzer.survival_by_age_group()

    assert result["Child"] == pytest.approx(100.0)
    assert result["Young Adult"] == pytest.approx(100.0)
    assert result["Adult"] == pytest.approx(0.0)
    assert result["Senior"] == pytest.approx(0.0)


def test_dataset_summary() -> None:
    dataframe = make_analysis_dataframe()
    analyzer = SurvivalAnalyzer(dataframe)

    summary = analyzer.dataset_summary()

    assert summary["passengers"] == 4
    assert summary["survivors"] == 2
    assert summary["deaths"] == 2
    assert summary["survival_rate"] == pytest.approx(50.0)


def test_fare_statistics() -> None:
    dataframe = make_analysis_dataframe()
    analyzer = SurvivalAnalyzer(dataframe)

    statistics = analyzer.fare_statistics()

    assert statistics["count"] == 4
    assert statistics["min"] == pytest.approx(8.0)
    assert statistics["max"] == pytest.approx(100.0)