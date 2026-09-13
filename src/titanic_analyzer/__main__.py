"""Command-line entry point for Titanic Survival Analyzer."""

from pathlib import Path

from titanic_analyzer import (
    Passenger,
    SurvivalAnalyzer,
    compare_model_metrics,
    compare_models,
    engineer_features,
    generate_all_plots,
    load_titanic_data,
    predict_passenger_survival,
    preprocess_data,
    save_all_confusion_matrices,
)


def print_percentage_series(title: str, series) -> None:
    """Print a percentage series in a readable format."""
    print(title)
    print("-" * len(title))

    for label, value in series.items():
        print(f"{label}: {value:.2f}%")

    print()


def main() -> None:
    """Run a basic Titanic survival analysis."""
    data_path = Path("data/titanic.csv")

    print("Titanic Survival Analyzer")
    print("=========================")
    print()

    try:
        dataframe = load_titanic_data(data_path)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")
        return

    cleaned = preprocess_data(dataframe)
    featured = engineer_features(cleaned)

    analyzer = SurvivalAnalyzer(featured)
    summary = analyzer.dataset_summary()

    print("Dataset Summary")
    print("---------------")
    print(f"Passengers: {summary['passengers']}")
    print(f"Survivors: {summary['survivors']}")
    print(f"Deaths: {summary['deaths']}")
    print(f"Overall survival rate: {summary['survival_rate']:.2f}%")
    print()

    print("Feature Engineering")
    print("-------------------")
    print("Added features:")
    print("FamilySize, IsAlone, Title, AgeGroup, Deck, FarePerPerson")
    print()

    print_percentage_series(
        "Survival by Sex",
        analyzer.survival_by_sex(),
    )

    print_percentage_series(
        "Survival by Passenger Class",
        analyzer.survival_by_class(),
    )

    print_percentage_series(
        "Survival by Embarkation Port",
        analyzer.survival_by_embarkation(),
    )

    print_percentage_series(
        "Survival by Age Group",
        analyzer.survival_by_age_group(),
    )

    model_results = compare_models(featured)

    comparison = compare_model_metrics(model_results)

    print("Model Evaluation")
    print("----------------")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )
    print()

    best_model_name = comparison.iloc[0]["Model"]
    best_model = model_results[best_model_name].model

    print(f"Best model by F1-score: {best_model_name}")
    print()

    example_passenger = Passenger(
        pclass=3,
        sex="male",
        age=25,
        fare=15.0,
        sibsp=0,
        parch=0,
        embarked="S",
        title="Mr",
        deck="Unknown",
    )

    prediction = predict_passenger_survival(
        best_model,
        example_passenger,
    )

    print("Example Passenger Prediction")
    print("----------------------------")

    outcome = "Survived" if prediction.survived else "Did not survive"

    print(f"Predicted outcome: {outcome}")
    print(f"Survival probability: {prediction.survival_probability:.2%}")
    print()

    confusion_paths = save_all_confusion_matrices(model_results)

    print("Confusion Matrices")
    print("------------------")

    for path in confusion_paths:
        print(f"- {path}")

    print()

    plot_paths = generate_all_plots(featured)

    print("Visualizations")
    print("--------------")
    print("Generated plot files:")

    for path in plot_paths:
        print(f"- {path}")

    print()


if __name__ == "__main__":
    main()
