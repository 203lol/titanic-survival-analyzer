## Exploratory Survival Analysis

The `SurvivalAnalyzer` class provides exploratory statistics for understanding survival patterns in the Titanic dataset.

The analysis currently includes:

* Total number of passengers
* Number of survivors and deaths
* Overall survival rate
* Survival rate by sex
* Survival rate by passenger class
* Survival rate by embarkation port
* Survival rate by age group
* Descriptive statistics for passenger fares

Example:

```python
from titanic_analyzer import (
    SurvivalAnalyzer,
    load_titanic_data,
    preprocess_data,
)

data = load_titanic_data("data/titanic.csv")
clean_data = preprocess_data(data)

analyzer = SurvivalAnalyzer(clean_data)

print(analyzer.dataset_summary())
print(analyzer.survival_by_sex())
print(analyzer.survival_by_class())
print(analyzer.survival_by_embarkation())
print(analyzer.survival_by_age_group())
```

This analysis helps identify important relationships between passenger characteristics and survival outcomes.

## Feature Engineering

The package provides feature-engineering utilities that transform the original Titanic passenger attributes into additional features that can be used for further analysis and predictive modeling.

The following features are currently generated:

* `FamilySize` — total family size calculated from siblings/spouses, parents/children, and the passenger
* `IsAlone` — indicates whether the passenger was travelling without family
* `Title` — extracts titles such as `Mr`, `Mrs`, and `Miss` from passenger names
* `AgeGroup` — groups passengers into age categories such as Child, Teen, Young Adult, Adult, and Senior
* `Deck` — extracts the deck letter from the passenger's cabin information
* `FarePerPerson` — estimates the fare per family member

Example:

```python
from titanic_analyzer import (
    engineer_features,
    load_titanic_data,
    preprocess_data,
)

data = load_titanic_data("data/titanic.csv")
clean_data = preprocess_data(data)
featured_data = engineer_features(clean_data)

print(
    featured_data[
        [
            "FamilySize",
            "IsAlone",
            "Title",
            "AgeGroup",
            "Deck",
            "FarePerPerson",
        ]
    ].head()
)
```

Feature engineering provides additional information about family structure, passenger demographics, cabin location, and ticket cost that can later be used for survival prediction.

## Data Visualization

The package can generate visualizations of important patterns in the Titanic dataset. All plots are saved as PNG files so that they can be viewed later or included in reports.

Available visualizations include:

* Survival rate by sex
* Survival rate by passenger class
* Passenger age distribution
* Passenger fare distribution
* Survival rate by family size

Example:

```python
from titanic_analyzer import (
    engineer_features,
    generate_all_plots,
    load_titanic_data,
    preprocess_data,
)

data = load_titanic_data("data/titanic.csv")
data = preprocess_data(data)
data = engineer_features(data)

plot_paths = generate_all_plots(data)

for path in plot_paths:
    print(path)
```

By default, the generated plots are saved in:

```text
outputs/plots/
```

The generated PNG files can be used to visually explore relationships between passenger characteristics and survival outcomes.
