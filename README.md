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
## Logistic Regression Model

The package includes a Logistic Regression model for predicting Titanic passenger survival.

The machine-learning pipeline performs both preprocessing and model training. Numeric features are standardized, while categorical features are converted using one-hot encoding.

The model currently uses the following features:

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

The package currently includes three supervised classification models for predicting Titanic passenger survival:

* Logistic Regression
* Decision Tree
* Random Forest

All models use the same prepared feature set and train/test split so that their performance can be compared consistently.

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

The model with the highest test accuracy can then be selected for further evaluation.

## Model Evaluation

The classification models are evaluated using multiple metrics rather than accuracy alone.

The following metrics are reported:

* **Accuracy** — proportion of all predictions that are correct
* **Precision** — proportion of predicted survivors who actually survived
* **Recall** — proportion of actual survivors correctly identified by the model
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

Confusion matrices are also generated for each classifier and saved as PNG files in:

```text
outputs/plots/
```

Using multiple metrics provides a more complete comparison of model performance than relying only on classification accuracy.

## Individual Passenger Prediction

The package can use a trained survival model to predict the outcome for an individual passenger.

Passenger information is converted into the same engineered feature structure used during model training, including family size, whether the passenger is travelling alone, and fare per person.

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

The returned result includes both the predicted survival class and the model's estimated survival probability.
