"""Command-line entry point for Titanic Survival Analyzer."""

from pathlib import Path

from titanic_analyzer import load_titanic_data


def main() -> None:
    """Run a basic demonstration of the Titanic Survival Analyzer."""
    data_path = Path("data/titanic.csv")

    print("Titanic Survival Analyzer")
    print("Version 0.1.0")
    print()

    try:
        dataframe = load_titanic_data(data_path)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")
        return

    print(f"Dataset loaded successfully: {data_path}")
    print(f"Passengers: {len(dataframe)}")
    print(f"Columns: {len(dataframe.columns)}")


if __name__ == "__main__":
    main()