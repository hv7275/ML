# Insurance Charges Prediction

A machine learning project that predicts medical insurance charges using Random Forest regression.

This project performs end-to-end exploratory data analysis (EDA), data cleaning, feature engineering, and model training on medical insurance data to predict charges based on customer demographics and health factors.

## Project Overview

The notebook (`Part l/main.ipynb`) includes:

1. **Exploratory Data Analysis (EDA)**

   - Statistical summaries and data shape analysis
   - Missing values and duplicate detection
   - Distribution analysis (histograms, boxplots, countplots)
   - Correlation heatmap

2. **Data Cleaning & Processing**

   - Handling missing/duplicate values
   - Encoding categorical variables (`sex`, `smoker`) to numerical
   - One-Hot Encoding for `region` feature

3. **Feature Engineering**

   - BMI categorization (UnderWeight, NormalWeight, Overweight, Obesity)
   - Feature scaling using StandardScaler
   - Feature selection using Pearson correlation and Chi-square tests

4. **Model Training & Evaluation**
   - Train/test split (80/20)
   - Random Forest Regressor with 200 estimators
   - Metrics: R² Score, MAE, RMSE
   - Feature importance visualization

## Repository structure

- `requirements.txt` — Python dependencies for the project
- `Data/` — dataset folder
  - `Data.csv` — CSV dataset used by the notebook
- `Part l/` — contains Jupyter notebook(s)
  - `main.ipynb` — full ML pipeline: EDA → Data Cleaning → Feature Engineering → Model Training
- `.venv/` — (optional) virtual environment directory (gitignored)

## Quick setup (Windows PowerShell)

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
# Use PowerShell activation
.\.venv\Scripts\Activate.ps1
```

If you run into an execution policy error when activating, you can allow the script for the current session:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Start Jupyter and open the notebook:

```powershell
# Start a notebook server and open the main notebook
jupyter notebook "Part l\main.ipynb"
```

Or open Jupyter and navigate to the `Part l` folder in the browser.

## How to use the data

The dataset (`Data/Data.csv`) contains medical insurance information with the following features:

- `age` — age of the customer
- `sex` — gender (male/female)
- `bmi` — body mass index
- `children` — number of children
- `smoker` — smoking status (yes/no)
- `region` — geographic region (northwest, northeast, southeast, southwest)
- `charges` — annual medical charges (target variable)

In the notebook, the dataset is loaded at the beginning:

```python
import pandas as pd
df = pd.read_csv('Data/Data.csv')
print(df.shape)
```

If your working directory is different, provide the full path or adjust the relative path accordingly.

## Notes

- The folder name `Part l` contains a space — be careful when passing the path in terminals or scripts; quote the path or escape spaces.
- This README assumes Windows PowerShell usage. Commands should work on other shells with small adjustments.

## Dependencies

Key dependencies are:

- **pandas** — data manipulation
- **numpy** — numerical computing
- **scikit-learn** — machine learning (RandomForest, preprocessing, model evaluation)
- **matplotlib** & **seaborn** — data visualization
- **scipy** — statistical tests (Pearson correlation, Chi-square)
- **jupyter** — interactive notebook environment

See `requirements.txt` for the complete list. Install all with:

## License & contact

This repository does not specify a license. Add a `LICENSE` file if you want to open-source this project.

Last updated: 2025-11-18

Projects in this repository

- `Part l/` — Insurance charges prediction notebook and pipeline
- `Hosue-Prediction/` — California housing price prediction scripts and artifacts
- `nutrition/` — Nutrition analysis notebooks and datasets
