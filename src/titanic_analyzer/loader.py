"""Load Titanic passenger data."""

from pathlib import Path

import pandas as pd

from titanic_analyzer.validation import validate_titanic_data


def load_titanic_data(file_path: str | Path) -> pd.DataFrame:
    """Load and validate a Titanic CSV file."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if not path.is_file():
        raise ValueError(f"Dataset path is not a file: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV file, received: {path.suffix}")

    dataframe = pd.read_csv(path)
    validate_titanic_data(dataframe)

    return dataframe