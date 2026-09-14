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


def test_parser_accepts_visualize_command() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "visualize",
            "data/titanic.csv",
        ]
    )

    assert args.command == "visualize"
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
    assert args.deck == "Unknown"
