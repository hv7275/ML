# ML Notebooks — Project Overview

This repository contains several machine learning projects and notebooks used for teaching and experimentation. Each top-level folder contains one or more Jupyter notebooks that demonstrate data exploration, preprocessing, model training, and evaluation.

This README provides a quick map, setup instructions, and how to run the notebooks.

## 📁 Project Structure

- `requirements.txt` — project-wide Python dependencies
- `Data/` — datasets used by the notebooks (CSV files)

### Machine Learning Projects

- `Cal-House/` — California housing price prediction (`main.ipynb`, saved model `xgb_best.joblib`)
- `Ford Car Price/` — Ford used-car price prediction (`main.ipynb`)
- `Insurance/` — Insurance analysis notebook (`Insurance.ipynb`)
- `nutrition/` — Nutritional data analysis (`Nutieients.ipynb`)

### Algorithm Implementations

- `Decision Trees/` — Decision tree algorithm implementation (`main.ipynb`)
- `KNN/` — K-Nearest Neighbors algorithm (`main.ipynb`)
- `Logistic regression/` — Logistic regression experiments (`main.ipynb`)
- `Naive bayes/` — Naive Bayes classifier implementation (`main.ipynb`)

### Additional Notebooks

- `Part l/` — Insurance charges prediction notebook and pipeline (`main.ipynb`)
- `Part ll/` — Additional machine learning notebook (`main.ipynb`)
- `main.ipynb` — Root-level notebook

**Note:** Some folder names include spaces; when using them in terminal commands, wrap paths in quotes or escape spaces.

## 🚀 Quick Setup

### Linux / macOS

1. Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch Jupyter and open the notebook you want to run:

```bash
jupyter notebook
```

Or open a specific notebook directly:

```bash
jupyter notebook "Part l/main.ipynb"
```

### Windows

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Launch Jupyter:

```powershell
jupyter notebook
```

## 📓 How to Run a Notebook

1. Open the notebook in Jupyter Lab or Jupyter Notebook
2. Run cells sequentially (Cell → Run All, or Shift+Enter for individual cells)
3. **Important:** If a notebook reads files from `Data/`, ensure your working directory is the repository root, or adjust the paths accordingly
4. If you encounter a `ModuleNotFoundError`, activate the virtual environment and reinstall dependencies from `requirements.txt`

## 📊 Example Workflow

Most notebooks follow a similar workflow:

1. **Load Data** — Import datasets from the `Data/` folder
2. **Exploratory Data Analysis (EDA)** — Visualize and understand the data
3. **Preprocessing** — Handle missing values, encoding, scaling, feature engineering
4. **Train/Test Split** — Split data into training and testing sets
5. **Model Training** — Train machine learning models
6. **Evaluation** — Assess model performance using appropriate metrics

### Example: Insurance Charges Prediction

- **Path:** `Part l/main.ipynb`
- **Target:** Predict insurance `charges` (regression)
- **Workflow:** Load data → EDA → preprocess (encoding + scaling) → train/test split → train model → evaluate

## 💻 Dependencies

Key libraries used across notebooks (see `requirements.txt` for exact versions):

- **Data Handling:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn`, `xgboost`
- **Visualization:** `matplotlib`, `seaborn`
- **Notebook Environment:** `jupyter`, `ipykernel`
- **Additional:** `streamlit`, `fastapi` (for some projects)

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## 📝 Notes & Tips

- **File Paths:** Quote paths that contain spaces, e.g., `"Part l/main.ipynb"`
- **Virtual Environment:** Always activate your virtual environment before running notebooks
- **Working Directory:** Most notebooks expect to be run from the repository root
- **Individual READMEs:** Some folders have their own `README.md` files with model-specific notes (e.g., `Cal-House/README.md`, `Ford Car Price/README.md`, `nutrition/README.md`)

## 🔍 Available Models & Algorithms

This repository includes examples and implementations of:

- **Regression Models:** Linear Regression, XGBoost, Random Forest
- **Classification Models:** Logistic Regression, Naive Bayes, Decision Trees
- **Supervised Learning:** K-Nearest Neighbors (KNN)
- **Ensemble Methods:** Random Forest, XGBoost

## 📄 License

No license is specified. Add a `LICENSE` file if you intend to publish this repository publicly.

---

**Last updated:** January 2025
