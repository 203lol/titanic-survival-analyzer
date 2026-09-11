# Titanic Survival Analyzer

Titanic Survival Analyzer is a Python package for exploring Titanic passenger
data and predicting passenger survival using machine-learning techniques.

The project is being developed as a final Python programming project.

## Planned Features

The project will eventually provide:

- Titanic passenger data loading and validation
- Data cleaning and preprocessing
- Exploratory survival analysis
- Feature engineering
- Data visualizations
- Machine-learning survival prediction
- Model comparison and evaluation
- Command-line interface
- Report generation

## Requirements

- Python 3.10 or newer
- uv

## Installation

Clone the repository and install the package in editable mode:

```bash
uv pip install -e .
## Data Preprocessing

The package includes a preprocessing pipeline for cleaning Titanic passenger
data before analysis.

Current preprocessing steps include:

- Filling missing `Age` values using the median age
- Filling missing `Embarked` values using the most frequent embarkation port
- Replacing missing `Cabin` values with `Unknown`
- Normalizing values in the `Sex` column
- Normalizing embarkation codes

Example:

```python
from titanic_analyzer import load_titanic_data, preprocess_data

data = load_titanic_data("data/titanic.csv")
clean_data = preprocess_data(data)

print(clean_data.isna().sum())