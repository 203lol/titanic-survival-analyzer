# Titanic Survival Analyzer

A Python toolkit for analyzing Titanic passenger data and predicting passenger survival.

The project includes data validation, preprocessing, feature engineering, exploratory analysis, visualization, machine-learning models, model evaluation, passenger prediction, report generation, and model persistence.

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

This analysis helps identify relationships between passenger characteristics and survival outcomes.

## Feature Engineering

The package provides feature-engineering utilities that transform the original Titanic passenger attributes into additional features for analysis and predictive modeling.

The following features are generated:

* `FamilySize` — total family size calculated from siblings/spouses, parents/children, and the passenger
* `IsAlone` — indicates whether the passenger was travelling without family
* `Title` — extracts titles such as `Mr`, `Mrs`, and `Miss` from passenger names
* `AgeGroup` — groups passengers into age categories
* `Deck` — extracts the deck letter from cabin information
* `FarePerPerson` — estimates the fare per family member

Example:

```python
from titanic_analyzer import (
    engineer_features,
    load_titanic_data,
    preprocess_data,
)

data = load_titanic_data("data/titanic.csv")
data = preprocess_data(data)
data = engineer_features(data)

print(
    data[
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

## Data Visualization

The package can generate visualizations of important patterns in the Titanic dataset.

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

By default, generated plots are saved in:

```text
outputs/plots/
```

## Logistic Regression Model

The package includes a Logistic Regression model for predicting Titanic passenger survival.

The machine-learning pipeline performs preprocessing and model training. Numeric features are standardized, while categorical features are converted using one-hot encoding.

The model uses the following features:

* `Age`
* `Fare`
* `FamilySize`
* `IsAlone`
* `FarePerPerson`
* `Pclass`
* `Sex`
* `Embarked`
* `Title`
* `Deck`

Example:

```python
from titanic_analyzer import (
    engineer_features,
    load_titanic_data,
    preprocess_data,
    train_logistic_regression,
)

data = load_titanic_data("data/titanic.csv")
data = preprocess_data(data)
data = engineer_features(data)

result = train_logistic_regression(data)

print(f"Accuracy: {result.accuracy:.3f}")
```

The dataset is divided into training and test subsets using a stratified split so that the survival-class distribution is preserved.

## Machine Learning Models

The package includes three supervised classification models:

* Logistic Regression
* Decision Tree
* Random Forest

All models use the same prepared feature set and train/test split so their performance can be compared consistently.

Example:

```python
from titanic_analyzer import (
    compare_models,
    engineer_features,
    load_titanic_data,
    preprocess_data,
)

data = load_titanic_data("data/titanic.csv")
data = preprocess_data(data)
data = engineer_features(data)

results = compare_models(data)

for model_name, result in results.items():
    print(f"{model_name}: {result.accuracy:.3f}")
```

The models can then be evaluated using several classification metrics.

## Model Evaluation

The classification models are evaluated using multiple metrics rather than accuracy alone.

The following metrics are reported:

* **Accuracy** — proportion of all predictions that are correct
* **Precision** — proportion of predicted survivors who actually survived
* **Recall** — proportion of actual survivors correctly identified
* **F1-score** — harmonic mean of precision and recall
* **ROC-AUC** — ability of the model to distinguish between survivors and non-survivors across different classification thresholds

Example:

```python
from titanic_analyzer import (
    compare_model_metrics,
    compare_models,
    engineer_features,
    load_titanic_data,
    preprocess_data,
)

data = load_titanic_data("data/titanic.csv")
data = preprocess_data(data)
data = engineer_features(data)

model_results = compare_models(data)
comparison = compare_model_metrics(model_results)

print(comparison)
```

Confusion matrices can also be generated for each classifier and saved in:

```text
outputs/plots/
```

The model with the highest F1-score is selected as the best model for prediction and model persistence.

## Individual Passenger Prediction

The package can use a trained survival model to predict the outcome for an individual passenger.

Passenger information is converted into the same feature structure used during model training, including family size, whether the passenger is travelling alone, and fare per person.

Example:

```python
from titanic_analyzer import (
    Passenger,
    compare_models,
    engineer_features,
    load_titanic_data,
    predict_passenger_survival,
    preprocess_data,
)

data = load_titanic_data("data/titanic.csv")
data = preprocess_data(data)
data = engineer_features(data)

model_results = compare_models(data)
model = model_results["Random Forest"].model

passenger = Passenger(
    pclass=3,
    sex="male",
    age=25,
    fare=15.0,
    embarked="S",
    title="Mr",
    deck="Unknown",
)

prediction = predict_passenger_survival(
    model,
    passenger,
)

print(prediction.survived)
print(prediction.survival_probability)
```

The result includes both the predicted survival class and the estimated survival probability.

## Command-Line Interface

Titanic Survival Analyzer provides a command-line interface for dataset inspection, analysis, visualization, model training, prediction, report generation, and model persistence.

Show the available commands:

```bash
uv run -m titanic_analyzer --help
```

### Dataset Summary

```bash
uv run -m titanic_analyzer summary data/titanic.csv
```

### Exploratory Analysis

```bash
uv run -m titanic_analyzer analyze data/titanic.csv
```

### Generate Visualizations

```bash
uv run -m titanic_analyzer visualize data/titanic.csv
```

Generated figures are saved to `outputs/plots/` by default.

A different output directory can be specified:

```bash
uv run -m titanic_analyzer visualize data/titanic.csv \
    --output outputs/custom_plots
```

### Train and Compare Models

```bash
uv run -m titanic_analyzer train data/titanic.csv
```

To additionally save confusion matrices:

```bash
uv run -m titanic_analyzer train data/titanic.csv \
    --save-confusion-matrices
```

### Predict an Individual Passenger

```bash
uv run -m titanic_analyzer predict data/titanic.csv \
    --pclass 3 \
    --sex male \
    --age 25 \
    --fare 15 \
    --sibsp 0 \
    --parch 0 \
    --embarked S \
    --title Mr \
    --deck Unknown
```

The prediction command reports the predicted survival outcome and estimated survival probability.

## Report Generation

Titanic Survival Analyzer can save analysis and model evaluation results as structured output files.

Generate all reports with:

```bash
uv run -m titanic_analyzer report data/titanic.csv
```

The generated files are saved by default in:

```text
outputs/reports/
```

Three report formats are generated:

* `analysis_report.txt` — human-readable exploratory survival analysis
* `model_comparison.csv` — model evaluation metrics for further analysis
* `model_results.json` — structured machine-learning results including the best model by F1-score

A custom output directory can be specified:

```bash
uv run -m titanic_analyzer report data/titanic.csv \
    --output outputs/custom_reports
```

The report-generation functions can also be used directly from Python:

```python
from titanic_analyzer import (
    compare_models,
    engineer_features,
    generate_reports,
    load_titanic_data,
    preprocess_data,
)

data = load_titanic_data("data/titanic.csv")
data = preprocess_data(data)
data = engineer_features(data)

model_results = compare_models(data)

paths = generate_reports(
    data,
    model_results,
)

for path in paths:
    print(path)
```

## Model Persistence

Trained models can be saved to disk and reused later for passenger predictions. This avoids training the models again each time a prediction is made.

### Save the Best Model

Train the available models, select the best-performing model by F1-score, and save it with:

```bash
uv run -m titanic_analyzer save-model data/titanic.csv
```

By default, the trained model is saved to:

```text
outputs/models/best_model.joblib
```

A different output path can be specified using `--output`:

```bash
uv run -m titanic_analyzer save-model data/titanic.csv \
    --output my_models/titanic_model.joblib
```

### Predict Using a Saved Model

Use `--model` with the `predict` command to load a previously saved model:

```bash
uv run -m titanic_analyzer predict data/titanic.csv \
    --model outputs/models/best_model.joblib \
    --pclass 3 \
    --sex male \
    --age 25 \
    --fare 15
```

When `--model` is provided, the saved model is loaded from disk and used for prediction instead of training the models again.

## Testing

Run the complete test suite with:

```bash
uv run pytest
```

Run the tests with coverage:

```bash
uv run pytest --cov=titanic_analyzer --cov-report=term-missing
```

## Code Quality

Check the project with Ruff:

```bash
uv run ruff check .
```

Check formatting with:

```bash
uv run ruff format --check .
```

Automatically format the project with:

```bash
uv run ruff format .
```
