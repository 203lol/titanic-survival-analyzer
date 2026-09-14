"""Functions for saving and loading trained models."""

from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline


def save_model(
    model: Pipeline,
    path: str | Path,
) -> Path:
    """Save a trained model."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_path)

    return output_path


def load_model(path: str | Path) -> Pipeline:
    """Load a trained model."""
    model_path = Path(path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    return joblib.load(model_path)
