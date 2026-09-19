"""Tests for the Titanic command-line interface."""

from titanic_analyzer.cli import build_parser


def test_parser_accepts_summary_command() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "summary",
            "data/titanic.csv",
        ]
    )

    assert args.command == "summary"
    assert args.data == "data/titanic.csv"


def test_parser_accepts_analyze_command() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "analyze",
            "data/titanic.csv",
        ]
    )

    assert args.command == "analyze"
    assert args.data == "data/titanic.csv"


def test_parser_accepts_visualize_command() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "visualize",
            "data/titanic.csv",
        ]
    )

    assert args.command == "visualize"
    assert args.data == "data/titanic.csv"
    assert args.output == "outputs/plots"


def test_parser_accepts_train_command() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "train",
            "data/titanic.csv",
        ]
    )

    assert args.command == "train"
    assert args.data == "data/titanic.csv"
    assert args.output == "outputs/plots"
    assert args.save_confusion_matrices is False


def test_parser_accepts_predict_command() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "predict",
            "data/titanic.csv",
            "--pclass",
            "3",
            "--sex",
            "male",
            "--age",
            "25",
            "--fare",
            "15",
        ]
    )

    assert args.command == "predict"
    assert args.data == "data/titanic.csv"
    assert args.pclass == 3
    assert args.sex == "male"
    assert args.age == 25.0
    assert args.fare == 15.0


def test_predict_defaults() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "predict",
            "data/titanic.csv",
            "--pclass",
            "2",
            "--sex",
            "female",
            "--age",
            "30",
            "--fare",
            "20",
        ]
    )

    assert args.sibsp == 0
    assert args.parch == 0
    assert args.embarked == "S"
    assert args.title == "Mr"
    assert args.deck == "Unknown"
    assert args.model is None


def test_predict_accepts_saved_model_path() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "predict",
            "data/titanic.csv",
            "--model",
            "outputs/models/best_model.joblib",
            "--pclass",
            "3",
            "--sex",
            "male",
            "--age",
            "25",
            "--fare",
            "15",
        ]
    )

    assert args.command == "predict"
    assert args.model == "outputs/models/best_model.joblib"


def test_parser_accepts_report_command() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "report",
            "data/titanic.csv",
        ]
    )

    assert args.command == "report"
    assert args.data == "data/titanic.csv"
    assert args.output == "outputs/reports"


def test_report_accepts_custom_output_directory() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "report",
            "data/titanic.csv",
            "--output",
            "my_reports",
        ]
    )

    assert args.command == "report"
    assert args.data == "data/titanic.csv"
    assert args.output == "my_reports"


def test_parser_accepts_save_model_command() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "save-model",
            "data/titanic.csv",
        ]
    )

    assert args.command == "save-model"
    assert args.data == "data/titanic.csv"
    assert args.output == "outputs/models/best_model.joblib"