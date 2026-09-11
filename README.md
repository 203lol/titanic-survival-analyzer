## Exploratory Survival Analysis

The package provides exploratory statistics through the
`SurvivalAnalyzer` class.

Current analysis includes:

- Total passenger count
- Number of survivors and deaths
- Overall survival rate
- Survival rate by sex
- Survival rate by passenger class
- Survival rate by embarkation port
- Survival rate by age group
- Descriptive fare statistics

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