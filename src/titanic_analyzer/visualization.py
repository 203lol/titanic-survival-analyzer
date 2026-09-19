"""Visualizations for Titanic passenger data."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _prepare_output_path(
    output_path: str | Path,
) -> Path:
    """Create the output directory if needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    return path


def plot_survival_by_sex(
    dataframe: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot survival rate by sex."""
    path = _prepare_output_path(output_path)

    survival_rates = dataframe.groupby("Sex")["Survived"].mean().mul(100)

    fig, ax = plt.subplots()

    survival_rates.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title("Titanic Survival Rate by Sex")
    ax.set_xlabel("Sex")
    ax.set_ylabel("Survival Rate (%)")
    ax.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path


def plot_survival_by_class(
    dataframe: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot survival rate by passenger class."""
    path = _prepare_output_path(output_path)

    survival_rates = (
        dataframe.groupby("Pclass")["Survived"]
        .mean()
        .mul(100)
        .sort_index()
    )

    fig, ax = plt.subplots()

    survival_rates.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title("Titanic Survival Rate by Passenger Class")
    ax.set_xlabel("Passenger Class")
    ax.set_ylabel("Survival Rate (%)")
    ax.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path


def plot_age_distribution(
    dataframe: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot the passenger age distribution."""
    path = _prepare_output_path(output_path)

    fig, ax = plt.subplots()

    ax.hist(
        dataframe["Age"].dropna(),
        bins=20,
        edgecolor="black",
    )

    ax.set_title("Titanic Passenger Age Distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Number of Passengers")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path


def plot_fare_distribution(
    dataframe: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot the passenger fare distribution."""
    path = _prepare_output_path(output_path)

    fig, ax = plt.subplots()

    ax.hist(
        dataframe["Fare"].dropna(),
        bins=30,
        edgecolor="black",
    )

    ax.set_title("Titanic Passenger Fare Distribution")
    ax.set_xlabel("Fare")
    ax.set_ylabel("Number of Passengers")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path


def plot_survival_by_family_size(
    dataframe: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot survival rate by family size."""
    path = _prepare_output_path(output_path)

    if "FamilySize" not in dataframe.columns:
        raise ValueError(
            "FamilySize column is required. "
            "Run engineer_features() before creating this plot."
        )

    survival_rates = (
        dataframe.groupby("FamilySize")["Survived"]
        .mean()
        .mul(100)
        .sort_index()
    )

    fig, ax = plt.subplots()

    survival_rates.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title("Titanic Survival Rate by Family Size")
    ax.set_xlabel("Family Size")
    ax.set_ylabel("Survival Rate (%)")
    ax.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path


def generate_all_plots(
    dataframe: pd.DataFrame,
    output_directory: str | Path = "outputs/plots",
) -> list[Path]:
    """Generate all Titanic plots."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    paths = [
        plot_survival_by_sex(
            dataframe,
            output_directory / "survival_by_sex.png",
        ),
        plot_survival_by_class(
            dataframe,
            output_directory / "survival_by_class.png",
        ),
        plot_age_distribution(
            dataframe,
            output_directory / "age_distribution.png",
        ),
        plot_fare_distribution(
            dataframe,
            output_directory / "fare_distribution.png",
        ),
        plot_survival_by_family_size(
            dataframe,
            output_directory / "survival_by_family_size.png",
        ),
    ]

    return paths