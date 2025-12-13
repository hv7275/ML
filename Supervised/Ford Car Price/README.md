# Ford Car Price Prediction

This folder contains a notebook and a trained model for predicting used Ford car prices.

Contents

- `main.ipynb` — Jupyter notebook that performs data exploration, preprocessing, model training, and evaluation for Ford car price prediction.
- `ford_rf_model.pkl` — Serialized Random Forest model (trained and saved from the notebook).

Dataset

- The dataset used for this notebook is available in the repository at `Data/ford_car_price.csv`.

Quick start

1. Install dependencies listed in the repository `requirements.txt`:

```powershell
pip install -r requirements.txt
```

2. Open the notebook:

```powershell
jupyter notebook "Ford Car Price\main.ipynb"
```

Notes

- The notebook will load data from `Data/ford_car_price.csv` by default. If you move the dataset, update the notebook paths accordingly.
- `ford_rf_model.pkl` is provided for quick inference demonstrations. If you want to retrain the model, run the cells in `main.ipynb`.

Contact

- If you need changes or improvements to the notebook (feature engineering, model tuning, or evaluation), open an issue or contact the repository owner.

Last updated: 2025-11-19
