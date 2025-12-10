df = pd.read_csv('Data/Data.csv')
# ML Notebooks — Project Overview

This repository contains several small machine-learning projects and notebooks used for teaching and experimentation. Each top-level folder contains one or more Jupyter notebooks that demonstrate data exploration, preprocessing, model training, and evaluation.

This README provides a quick map, setup instructions (Linux-focused), and how to run the notebooks.

## What's in this repo

- `requirements.txt` — project-wide Python dependencies
- `Data/` — datasets used by the notebooks (CSV files)
- `Logistic Regresion/`, `Logistic regression/` — logistic regression experiments (notebooks)
- `Part l/` — Insurance charges prediction notebook and pipeline (`main.ipynb`)
- `Cal-House/` — California housing prediction (`main.ipynb`, saved model `xgb_best.joblib`)
- `Ford Car Price/` — Ford used-car price prediction (`main.ipynb`)
- `Insurance/`, `nutrition/` — additional notebooks and analyses

Note: Some folder names include spaces; when using them in terminal commands, wrap paths in quotes or escape spaces.

## Quick setup (Linux / macOS)

1. Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch Jupyter and open the notebook you want to run (example: insurance notebook):

```bash
jupyter notebook "Part l/main.ipynb"
```

Or open `jupyter notebook` and navigate to the folder in your browser.

## How to run a specific notebook

- Open the notebook in Jupyter and run cells sequentially.
- If a notebook reads `Data/Data.csv`, ensure your working directory is the repository root or adjust the path.
- If you see a `ModuleNotFoundError`, activate the virtual environment and reinstall `requirements.txt`.

## Example: Insurance charges notebook

- Path: `Part l/main.ipynb`
- Workflow in the notebook: Load data → EDA → preprocess (encoding + scaling) → train/test split → train model → evaluate.
- Target column: `charges` (for regression notebooks) or `sex_female` where applicable for classification examples.

## Notes & tips

- File paths: quote paths that contain spaces, e.g. `"Part l/main.ipynb"`.
- Use `source .venv/bin/activate` on Linux/macOS; use the PowerShell activation on Windows.
- The repository includes examples with RandomForest, XGBoost, and LogisticRegression — check each folder's `README.md` (when present) for model-specific notes.

## Dependencies

Key libraries used across notebooks (see `requirements.txt` for exact versions):

- pandas, numpy — data handling
- scikit-learn — preprocessing, models, evaluation
- xgboost — boosted-tree models (used in `Cal-House`)
- matplotlib, seaborn — visualization
- jupyter — notebook environment

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## License

No license is specified. Add a `LICENSE` file if you intend to publish this repository publicly.

Last updated: 2025-12-10

---

If you'd like, I can:

- Make this README shorter or longer depending on the audience (readers vs contributors),
- Add a per-folder README summary for the main folders, or
- Create a small script to launch a selected notebook (e.g., `scripts/open_notebook.sh`).

Which would you like next?
import pandas as pd
