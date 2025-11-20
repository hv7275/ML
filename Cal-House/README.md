# Cal-House

Brief California housing price prediction project using XGBoost. This folder contains a Jupyter notebook and a trained model artifact.

Files

- `main.ipynb` — Notebook with EDA, preprocessing, model training, evaluation, and inference examples.
- `xgb_best.joblib` — Saved XGBoost model (trained) that can be loaded for inference.

Quick start

1. Install requirements (from repository root):

```powershell
pip install -r ..\requirements.txt
```

2. Open the notebook:

```powershell
jupyter notebook "Cal-House\main.ipynb"
```

3. Load the trained model in Python (example):

```python
import joblib
model = joblib.load('xgb_best.joblib')
# then call model.predict(X_new)
```

Notes

- The notebook documents the preprocessing steps required before inference. Ensure any new input follows the same feature ordering and scaling used in the notebook.
- If you want to retrain, run the cells in `main.ipynb` and save a new model artifact.

Contact

For questions or improvements, open an issue or contact the repo owner.
