"""Report generation utilities for Titanic analysis results."""

import json
from pathlib import Path

import pandas as pd

from titanic_analyzer.analysis import SurvivalAnalyzer
from titanic_analyzer.evaluation import compare_model_metrics
from titanic_analyzer.models import ModelResult


def ensure_output_directory(
    output_directory: str | Path,
) -> Path:
    """Create and return the report output directory."""
    path = Path(output_directory)
    path.mkdir(parents=True, exist_ok=True)

    return path


def save_analysis_report(
    dataframe: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save a human-readable Titanic analysis report."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    analyzer = SurvivalAnalyzer(dataframe)
    summary = analyzer.dataset_summary()

    lines = [
        "Titanic Survival Analysis Report",
        "================================",
        "",
        "Dataset Summary",
        "---------------",
        f"Passengers: {summary['passengers']}",
        f"Survivors: {summary['survivors']}",
        f"Deaths: {summary['deaths']}",
        f"Overall survival rate: {summary['survival_rate']:.2f}%",
        "",
        "Survival by Sex",
        "---------------",
    ]

    for label, value in analyzer.survival_by_sex().items():
        lines.append(f"{label}: {value:.2f}%")

    lines.extend(
        [
            "",
            "Survival by Passenger Class",
            "---------------------------",
        ]
    )

    for label, value in analyzer.survival_by_class().items():
        lines.append(f"Class {label}: {value:.2f}%")

    lines.extend(
        [
            "",
            "Survival by Embarkation Port",
            "----------------------------",
        ]
    )

    for label, value in analyzer.survival_by_embarkation().items():
        lines.append(f"{label}: {value:.2f}%")

    lines.extend(
        [
            "",
            "Survival by Age Group",
            "---------------------",
        ]
    )

    for label, value in analyzer.survival_by_age_group().items():
        lines.append(f"{label}: {value:.2f}%")

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return path


def save_model_comparison_csv(
    model_results: dict[str, ModelResult],
    output_path: str | Path,
) -> Path:
    """Save model evaluation metrics as a CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    comparison = compare_model_metrics(model_results)

    comparison.to_csv(
        path,
        index=False,
    )

    return path


def save_model_results_json(
    model_results: dict[str, ModelResult],
    output_path: str | Path,
) -> Path:
    """Save model evaluation results as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    comparison = compare_model_metrics(model_results)

    models = {}

    for _, row in comparison.iterrows():
        models[row["Model"]] = {
            "accuracy": float(row["Accuracy"]),
            "precision": float(row["Precision"]),
            "recall": float(row["Recall"]),
            "f1": float(row["F1"]),
            "roc_auc": float(row["ROC-AUC"]),
        }

    report = {
        "best_model_by_f1": comparison.iloc[0]["Model"],
        "models": models,
    }

    path.write_text(
        json.dumps(
            report,
            indent=4,
        ),
        encoding="utf-8",
    )

    return path


def generate_reports(
    dataframe: pd.DataFrame,
    model_results: dict[str, ModelResult],
    output_directory: str | Path = "outputs/reports",
) -> list[Path]:
    """Generate all standard Titanic report files."""
    output_directory = ensure_output_directory(output_directory)

    report_paths = [
        save_analysis_report(
            dataframe,
            output_directory / "analysis_report.txt",
        ),
        save_model_comparison_csv(
            model_results,
            output_directory / "model_comparison.csv",
        ),
        save_model_results_json(
            model_results,
            output_directory / "model_results.json",
        ),
    ]

    return report_paths
