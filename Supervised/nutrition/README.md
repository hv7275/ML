````markdown
# Nutrition Analysis

This folder contains the nutrition analysis notebook, a trained model and a visualization used in the project.

Contents

- `Nutieients.ipynb` — Jupyter notebook with data exploration, preprocessing, and a simple calorie prediction demo.
- `calorie_model.pkl` — Trained calorie prediction model (pickle file) used by the notebook for quick inference.
- `Calories (kcal).png` — Visualization showing calorie distribution (used in the notebook/report).

Dataset

- The processed dataset used by the notebook is stored at the repository `Data` folder: `..\Data\Indian_Food_Nutrition_Processed.csv`.

Quick start (Windows PowerShell)

1. (Optional) Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies from the project root (if needed):

```powershell
pip install -r ..\requirements.txt
```

3. Start Jupyter and open the notebook:

```powershell
jupyter notebook "nutrition\Nutieients.ipynb"
```

Notes

- Notebook filename contains a typo (`Nutieients.ipynb`). The file is intentionally left with this name to avoid breaking existing references; rename it if you prefer and update any references.
- The notebook expects the processed CSV at `..\Data\Indian_Food_Nutrition_Processed.csv`. If you move the notebook, update the relative path accordingly.
- `calorie_model.pkl` is a simple demonstration model — retrain if you need production-quality predictions.

Last updated: 2025-11-18
````
