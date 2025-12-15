# ML Notebooks — Project Overview

Hands-on notebooks covering classical machine learning, ensemble methods, unsupervised learning, and basic NLP. Each folder includes one or more Jupyter notebooks that walk through data exploration, preprocessing, model training, and evaluation. Several projects also ship trained artifacts for quick inference demos.

## 📁 Project Map

- `requirements.txt` — shared Python dependencies
- `Data/` — CSV datasets consumed by many notebooks
- `Supervised/`
  - `Cal-House/` — California housing price regression (`main.ipynb`, `xgb_best.joblib`)
  - `Ford Car Price/` — used-car price regression (`main.ipynb`, `ford_rf_model.pkl`)
  - `Insurance/` — insurance cost analysis (`Insurance.ipynb`)
  - `Classification Project/` — heart-disease classifier with `app.py`, encoders, scaler, and `NaivBayes_Heart.pkl`
  - `nutrition/` — calorie prediction (`Nutieients.ipynb`, `calorie_model.pkl`)
  - Additional algorithm notebooks: `Decision Trees/`, `KNN/`, `Logistic regression/`, `Naive bayes/`, `Support Vector Machine (SVM)/`, `HyperParameter/`, `Ensamble Learning/`, `Part l/`, `Part ll/`
- `Unsupervised/`
  - `K Means Clustring/`, `DB SCAN/`, `PCA/` — clustering and dimensionality reduction notebooks with elbow/cluster plots
- `NLP/`
  - `Bag Of Words/` and `Porject/` — text classification experiments using bag-of-words features
- Root `main.ipynb` — general experimentation notebook

**Tip:** Some paths contain spaces; wrap them in quotes when launching (`jupyter notebook "Supervised/Ford Car Price/main.ipynb"`).

## 🚀 Quick Setup

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook
```

## 📓 How to Run

1) Activate the virtual environment.  
2) From the repo root, start Jupyter (`jupyter notebook`).  
3) Open a notebook and run cells top-to-bottom (Shift+Enter).  
4) If a notebook reads from `Data/`, keep the working directory at the repo root or update relative paths.  
5) For inference-ready projects, load the provided `.pkl`/`.joblib` artifacts as shown in their notebooks or companion `README.md` files.

## 🧭 Typical Workflow

Load data → EDA → preprocess (missing values, encoding, scaling) → train/test split → train model → evaluate (appropriate metrics, plots). Ensemble notebooks add cross-validation and hyperparameter tuning.

## 💻 Key Dependencies

- Data: `pandas`, `numpy`
- Modeling: `scikit-learn`, `xgboost`
- Visualization: `matplotlib`, `seaborn`
- Notebook runtime: `jupyter`, `ipykernel`
- Apps/serving (select projects): `streamlit`, `fastapi`

Install everything with `pip install -r requirements.txt`.

## 📝 Notes & Tips

- Quote paths with spaces when launching notebooks.  
- Keep notebooks under the repo root to avoid broken relative paths.  
- Some subfolders (e.g., `Cal-House/`, `Ford Car Price/`, `nutrition/`) include extra README details and saved models for quick reuse.  
- Models in `Classification Project/` expect the provided encoders/scalers; load them before inference.

## 📄 License

No license specified. Add a `LICENSE` file if you plan to publish or distribute.

---

**Last updated:** December 2025
