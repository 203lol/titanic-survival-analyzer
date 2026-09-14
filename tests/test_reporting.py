"""Tests for Titanic report generation."""

import json
from pathlib import Path

import pandas as pd

from titanic_analyzer.models import compare_models
from titanic_analyzer.reporting import (
    generate_reports,
    save_analysis_report,
    save_model_comparison_csv,
    save_model_results_json,
)


def make_reporting_dataframe() -> pd.DataFrame:
    """Create a Titanic-like dataset for report tests."""
    return pd.DataFrame(
        {
            "Survived": [
                0,
                1,
                1,
                1,
                0,
                0,
                1,
                1,
                1,
                1,
                0,
                1,
                0,
                0,
                1,
                0,
            ],
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
                "male",
                "male",
                "female",
                "male",
            ],
            "Pclass": [
                3,
                1,
                3,
                1,
                3,
                1,
                3,
                3,
                2,
                1,
                3,
                1,
                2,
                3,
                2,
                3,
            ],
            "Age": [
                22,
                38,
                26,
                35,
                28,
                54,
                2,
                27,
                14,
                58,
                20,
                45,
                31,
                19,
                40,
                30,
            ],
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
                13.0,
                7.9,
                26.0,
                15.5,
            ],
            "FamilySize": [
                2,
                2,
                1,
                2,
                1,
                1,
                5,
                1,
                3,
                1,
                1,
                2,
                1,
                1,
                2,
                1,
            ],
            "IsAlone": [
                0,
                0,
                1,
                0,
                1,
                1,
                0,
                1,
                0,
                1,
                1,
                0,
                1,
                1,
                0,
                1,
            ],
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
                13.0,
                7.9,
                13.0,
                15.5,
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
                "S",
                "S",
                "C",
                "Q",
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
                "Mr",
                "Mr",
                "Mrs",
                "Mr",
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
                "Unknown",
                "Unknown",
                "D",
                "Unknown",
            ],
        }
    )


def test_save_analysis_report_creates_file(
    tmp_path: Path,
) -> None:
    dataframe = make_reporting_dataframe()
    output = tmp_path / "analysis_report.txt"

    result = save_analysis_report(
        dataframe,
        output,
    )

    assert result.exists()
    assert result.stat().st_size > 0

    content = result.read_text(encoding="utf-8")

    assert "Titanic Survival Analysis Report" in content
    assert "Passengers:" in content


def test_save_model_comparison_csv_creates_file(
    tmp_path: Path,
) -> None:
    dataframe = make_reporting_dataframe()
    model_results = compare_models(dataframe)

    output = tmp_path / "comparison.csv"

    result = save_model_comparison_csv(
        model_results,
        output,
    )

    assert result.exists()

    comparison = pd.read_csv(result)

    assert "Model" in comparison.columns
    assert "Accuracy" in comparison.columns
    assert len(comparison) == 3


def test_save_model_results_json_creates_valid_json(
    tmp_path: Path,
) -> None:
    dataframe = make_reporting_dataframe()
    model_results = compare_models(dataframe)

    output = tmp_path / "results.json"

    result = save_model_results_json(
        model_results,
        output,
    )

    assert result.exists()

    content = json.loads(result.read_text(encoding="utf-8"))

    assert "best_model_by_f1" in content
    assert "models" in content
    assert len(content["models"]) == 3


def test_generate_reports_creates_three_files(
    tmp_path: Path,
) -> None:
    dataframe = make_reporting_dataframe()
    model_results = compare_models(dataframe)

    paths = generate_reports(
        dataframe,
        model_results,
        tmp_path,
    )

    assert len(paths) == 3

    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0
