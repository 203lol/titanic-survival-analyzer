"""Tests for Titanic visualization utilities."""

from pathlib import Path

import pandas as pd
import pytest

from titanic_analyzer.visualization import (
    generate_all_plots,
    plot_age_distribution,
    plot_fare_distribution,
    plot_survival_by_class,
    plot_survival_by_family_size,
    plot_survival_by_sex,
)


def make_visualization_dataframe() -> pd.DataFrame:
    """Create a small Titanic-like DataFrame for visualization tests."""
    return pd.DataFrame(
        {
            "Survived": [1, 0, 1, 0, 1],
            "Sex": ["female", "male", "female", "male", "female"],
            "Pclass": [1, 3, 2, 3, 1],
            "Age": [22.0, 35.0, 10.0, 45.0, 28.0],
            "Fare": [70.0, 8.0, 25.0, 12.0, 50.0],
            "FamilySize": [2, 1, 3, 1, 2],
        }
    )


def test_plot_survival_by_sex_creates_file(tmp_path: Path) -> None:
    dataframe = make_visualization_dataframe()
    output = tmp_path / "survival_by_sex.png"

    result = plot_survival_by_sex(dataframe, output)

    assert result.exists()
    assert result.stat().st_size > 0


def test_plot_survival_by_class_creates_file(tmp_path: Path) -> None:
    dataframe = make_visualization_dataframe()
    output = tmp_path / "survival_by_class.png"

    result = plot_survival_by_class(dataframe, output)

    assert result.exists()
    assert result.stat().st_size > 0


def test_plot_age_distribution_creates_file(tmp_path: Path) -> None:
    dataframe = make_visualization_dataframe()
    output = tmp_path / "age_distribution.png"

    result = plot_age_distribution(dataframe, output)

    assert result.exists()
    assert result.stat().st_size > 0


def test_plot_fare_distribution_creates_file(tmp_path: Path) -> None:
    dataframe = make_visualization_dataframe()
    output = tmp_path / "fare_distribution.png"

    result = plot_fare_distribution(dataframe, output)

    assert result.exists()
    assert result.stat().st_size > 0


def test_plot_survival_by_family_size_creates_file(tmp_path: Path) -> None:
    dataframe = make_visualization_dataframe()
    output = tmp_path / "family_size.png"

    result = plot_survival_by_family_size(dataframe, output)

    assert result.exists()
    assert result.stat().st_size > 0


def test_family_size_plot_requires_feature(tmp_path: Path) -> None:
    dataframe = make_visualization_dataframe().drop(columns=["FamilySize"])

    with pytest.raises(ValueError, match="FamilySize"):
        plot_survival_by_family_size(
            dataframe,
            tmp_path / "family.png",
        )


def test_generate_all_plots_creates_five_files(tmp_path: Path) -> None:
    dataframe = make_visualization_dataframe()

    paths = generate_all_plots(dataframe, tmp_path)

    assert len(paths) == 5

    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0
