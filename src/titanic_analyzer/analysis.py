"""Analysis functions for Titanic passenger data."""

import pandas as pd


class SurvivalAnalyzer:
    """Analyze survival patterns in the Titanic dataset."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe.copy()

    def passenger_count(self) -> int:
        """Return the number of passengers."""
        return len(self.dataframe)

    def survivor_count(self) -> int:
        """Return the number of survivors."""
        return int(self.dataframe["Survived"].sum())

    def death_count(self) -> int:
        """Return the number of deaths."""
        return int((self.dataframe["Survived"] == 0).sum())

    def overall_survival_rate(self) -> float:
        """Return the overall survival rate."""
        return float(self.dataframe["Survived"].mean() * 100)

    def survival_by_sex(self) -> pd.Series:
        """Calculate survival rate by sex."""
        return (
            self.dataframe.groupby("Sex")["Survived"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
        )

    def survival_by_class(self) -> pd.Series:
        """Calculate survival rate by passenger class."""
        return (
            self.dataframe.groupby("Pclass")["Survived"]
            .mean()
            .mul(100)
            .sort_index()
        )

    def survival_by_embarkation(self) -> pd.Series:
        """Calculate survival rate by embarkation port."""
        return (
            self.dataframe.groupby("Embarked")["Survived"]
            .mean()
            .mul(100)
            .sort_index()
        )

    def survival_by_age_group(self) -> pd.Series:
        """Calculate survival rate by age group."""
        age_groups = pd.cut(
            self.dataframe["Age"],
            bins=[0, 12, 18, 35, 60, float("inf")],
            labels=["Child", "Teen", "Young Adult", "Adult", "Senior"],
            right=False,
        )

        return (
            self.dataframe.assign(AgeGroup=age_groups)
            .groupby("AgeGroup", observed=False)["Survived"]
            .mean()
            .mul(100)
        )

    def fare_statistics(self) -> pd.Series:
        """Return fare statistics."""
        return self.dataframe["Fare"].describe()

    def dataset_summary(self) -> dict[str, float | int]:
        """Return a summary of the dataset."""
        return {
            "passengers": self.passenger_count(),
            "survivors": self.survivor_count(),
            "deaths": self.death_count(),
            "survival_rate": self.overall_survival_rate(),
        }