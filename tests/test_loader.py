"""Tests for Titanic dataset loading."""

from pathlib import Path

import pandas as pd
import pytest

from titanic_analyzer import load_titanic_data

DATA_PATH = Path("data/titanic.csv")


def test_load_titanic_data_returns_dataframe() -> None:
    dataframe = load_titanic_data(DATA_PATH)

    assert isinstance(dataframe, pd.DataFrame)


def test_load_titanic_data_contains_rows() -> None:
    dataframe = load_titanic_data(DATA_PATH)

    assert len(dataframe) > 0


def test_load_titanic_data_expected_shape() -> None:
    dataframe = load_titanic_data(DATA_PATH)

    assert dataframe.shape == (891, 12)


def test_missing_file_raises_file_not_found() -> None:
    with pytest.raises(
        FileNotFoundError,
        match="Dataset not found",
    ):
        load_titanic_data("data/does_not_exist.csv")


def test_directory_path_raises_value_error(tmp_path: Path) -> None:
    directory = tmp_path / "passengers.csv"
    directory.mkdir()

    with pytest.raises(
        ValueError,
        match="Dataset path is not a file",
    ):
        load_titanic_data(directory)


def test_non_csv_file_raises_value_error(tmp_path: Path) -> None:
    text_file = tmp_path / "passengers.txt"
    text_file.write_text(
        "example",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Expected a CSV file",
    ):
        load_titanic_data(text_file)