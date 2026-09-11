"""Command-line entry point for Titanic Survival Analyzer."""

from pathlib import Path

from titanic_analyzer import load_titanic_data, preprocess_data


def main() -> None:
    """Run a basic demonstration of Titanic data preprocessing."""
    data_path = Path("data/titanic.csv")

    print("Titanic Survival Analyzer")
    print("Version 0.1.0")
    print()

    try:
        dataframe = load_titanic_data(data_path)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")
        return

    cleaned = preprocess_data(dataframe)

    print(f"Dataset loaded successfully: {data_path}")
    print(f"Passengers: {len(cleaned)}")
    print(f"Columns: {len(cleaned.columns)}")
    print()

    print("Missing values after preprocessing:")
    print(cleaned[["Age", "Cabin", "Embarked"]].isna().sum())


if __name__ == "__main__":
    main()