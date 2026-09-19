"""Command-line interface for the Titanic Survival Analyzer."""

import argparse
from pathlib import Path

from titanic_analyzer import (
    Passenger,
    SurvivalAnalyzer,
    compare_model_metrics,
    compare_models,
    engineer_features,
    generate_all_plots,
    generate_reports,
    load_model,
    load_titanic_data,
    predict_passenger_survival,
    preprocess_data,
    save_all_confusion_matrices,
    save_model,
)


def load_prepared_data(data_path: str | Path):
    """Load and prepare the Titanic dataset."""
    dataframe = load_titanic_data(data_path)
    dataframe = preprocess_data(dataframe)
    dataframe = engineer_features(dataframe)

    return dataframe


def command_summary(args: argparse.Namespace) -> None:
    """Show a summary of the dataset."""
    dataframe = load_prepared_data(args.data)
    analyzer = SurvivalAnalyzer(dataframe)
    summary = analyzer.dataset_summary()

    print("Titanic Dataset Summary")
    print("=======================")
    print(f"Passengers: {summary['passengers']}")
    print(f"Survivors: {summary['survivors']}")
    print(f"Deaths: {summary['deaths']}")
    print(f"Survival rate: {summary['survival_rate']:.2f}%")


def command_analyze(args: argparse.Namespace) -> None:
    """Show survival statistics."""
    dataframe = load_prepared_data(args.data)
    analyzer = SurvivalAnalyzer(dataframe)

    print("Titanic Survival Analysis")
    print("=========================")
    print()

    print("Survival by Sex")
    print("----------------")
    for label, value in analyzer.survival_by_sex().items():
        print(f"{label}: {value:.2f}%")

    print()
    print("Survival by Passenger Class")
    print("---------------------------")
    for label, value in analyzer.survival_by_class().items():
        print(f"Class {label}: {value:.2f}%")

    print()
    print("Survival by Embarkation Port")
    print("----------------------------")
    for label, value in analyzer.survival_by_embarkation().items():
        print(f"{label}: {value:.2f}%")

    print()
    print("Survival by Age Group")
    print("---------------------")
    for label, value in analyzer.survival_by_age_group().items():
        print(f"{label}: {value:.2f}%")


def command_visualize(args: argparse.Namespace) -> None:
    """Generate visualization files."""
    dataframe = load_prepared_data(args.data)

    paths = generate_all_plots(
        dataframe,
        output_directory=args.output,
    )

    print("Generated visualizations:")
    for path in paths:
        print(f"- {path}")


def command_train(args: argparse.Namespace) -> None:
    """Train and compare the models."""
    dataframe = load_prepared_data(args.data)

    model_results = compare_models(dataframe)
    comparison = compare_model_metrics(model_results)

    print("Model Evaluation")
    print("================")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    best_model_name = comparison.iloc[0]["Model"]

    print()
    print(f"Best model by F1-score: {best_model_name}")

    if args.save_confusion_matrices:
        paths = save_all_confusion_matrices(
            model_results,
            output_directory=args.output,
        )

        print()
        print("Generated confusion matrices:")
        for path in paths:
            print(f"- {path}")


def command_predict(args: argparse.Namespace) -> None:
    """Predict survival for one passenger."""
    dataframe = load_prepared_data(args.data)

    if args.model:
        model = load_model(args.model)
        model_name = "Saved model"
    else:
        model_results = compare_models(dataframe)
        comparison = compare_model_metrics(model_results)

        model_name = comparison.iloc[0]["Model"]
        model = model_results[model_name].model

    passenger = Passenger(
        pclass=args.pclass,
        sex=args.sex,
        age=args.age,
        fare=args.fare,
        sibsp=args.sibsp,
        parch=args.parch,
        embarked=args.embarked,
        title=args.title,
        deck=args.deck,
    )

    prediction = predict_passenger_survival(
        model,
        passenger,
    )

    if prediction.survived:
        outcome = "Survived"
    else:
        outcome = "Did not survive"

    print("Passenger Survival Prediction")
    print("=============================")
    print(f"Model: {model_name}")
    print(f"Predicted outcome: {outcome}")
    print(f"Survival probability: {prediction.survival_probability:.2%}")


def command_report(args: argparse.Namespace) -> None:
    """Generate report files."""
    dataframe = load_prepared_data(args.data)
    model_results = compare_models(dataframe)

    paths = generate_reports(
        dataframe,
        model_results,
        output_directory=args.output,
    )

    print("Generated reports:")
    for path in paths:
        print(f"- {path}")


def command_save_model(args: argparse.Namespace) -> None:
    """Train and save the best model."""
    dataframe = load_prepared_data(args.data)

    model_results = compare_models(dataframe)
    comparison = compare_model_metrics(model_results)

    model_name = comparison.iloc[0]["Model"]
    model = model_results[model_name].model

    path = save_model(
        model,
        args.output,
    )

    print("Saved trained model:")
    print(f"- {path}")
    print(f"Model: {model_name}")


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="titanic_analyzer",
        description="Analyze Titanic passenger data and predict survival.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    summary_parser = subparsers.add_parser(
        "summary",
        help="Show a basic dataset summary.",
    )
    summary_parser.add_argument(
        "data",
        help="Path to the Titanic CSV file.",
    )
    summary_parser.set_defaults(func=command_summary)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run survival analysis.",
    )
    analyze_parser.add_argument(
        "data",
        help="Path to the Titanic CSV file.",
    )
    analyze_parser.set_defaults(func=command_analyze)

    visualize_parser = subparsers.add_parser(
        "visualize",
        help="Generate visualization files.",
    )
    visualize_parser.add_argument(
        "data",
        help="Path to the Titanic CSV file.",
    )
    visualize_parser.add_argument(
        "--output",
        default="outputs/plots",
        help="Directory where plots are saved.",
    )
    visualize_parser.set_defaults(func=command_visualize)

    train_parser = subparsers.add_parser(
        "train",
        help="Train and compare survival models.",
    )
    train_parser.add_argument(
        "data",
        help="Path to the Titanic CSV file.",
    )
    train_parser.add_argument(
        "--output",
        default="outputs/plots",
        help="Directory where evaluation plots are saved.",
    )
    train_parser.add_argument(
        "--save-confusion-matrices",
        action="store_true",
        help="Save a confusion matrix for each model.",
    )
    train_parser.set_defaults(func=command_train)

    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict survival for one passenger.",
    )
    predict_parser.add_argument(
        "data",
        help="Path to the Titanic CSV file.",
    )
    predict_parser.add_argument(
        "--model",
        help="Path to a saved model.",
    )
    predict_parser.add_argument(
        "--pclass",
        type=int,
        required=True,
        help="Passenger class: 1, 2, or 3.",
    )
    predict_parser.add_argument(
        "--sex",
        required=True,
        choices=["male", "female"],
        help="Passenger sex.",
    )
    predict_parser.add_argument(
        "--age",
        type=float,
        required=True,
        help="Passenger age.",
    )
    predict_parser.add_argument(
        "--fare",
        type=float,
        required=True,
        help="Passenger fare.",
    )
    predict_parser.add_argument(
        "--sibsp",
        type=int,
        default=0,
        help="Number of siblings/spouses aboard.",
    )
    predict_parser.add_argument(
        "--parch",
        type=int,
        default=0,
        help="Number of parents/children aboard.",
    )
    predict_parser.add_argument(
        "--embarked",
        default="S",
        choices=["C", "Q", "S"],
        help="Embarkation port.",
    )
    predict_parser.add_argument(
        "--title",
        default="Mr",
        help="Passenger title.",
    )
    predict_parser.add_argument(
        "--deck",
        default="Unknown",
        help="Passenger deck.",
    )
    predict_parser.set_defaults(func=command_predict)

    report_parser = subparsers.add_parser(
        "report",
        help="Generate analysis and model reports.",
    )
    report_parser.add_argument(
        "data",
        help="Path to the Titanic CSV file.",
    )
    report_parser.add_argument(
        "--output",
        default="outputs/reports",
        help="Directory where reports are saved.",
    )
    report_parser.set_defaults(func=command_report)

    save_model_parser = subparsers.add_parser(
        "save-model",
        help="Train and save the best model.",
    )
    save_model_parser.add_argument(
        "data",
        help="Path to the Titanic CSV file.",
    )
    save_model_parser.add_argument(
        "--output",
        default="outputs/models/best_model.joblib",
        help="Path where the model is saved.",
    )
    save_model_parser.set_defaults(func=command_save_model)

    return parser


def main() -> None:
    """Run the command-line interface."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))