# ML (Project)

Short description

This repository contains a small machine learning project with data and a notebook stored in `Part l/main.ipynb`.

## Repository structure

- `requirements.txt` — Python dependencies for the project
- `Data/` — dataset folder
  - `Data.csv` — CSV dataset used by the notebook
- `Part l/` — contains Jupyter notebook(s)
  - `main.ipynb` — main notebook to run analyses and experiments
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

In the notebook the dataset is expected at `Data/Data.csv`. Example to load it in Python:

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

Primary dependencies are listed in `requirements.txt` (examples include pandas, scikit-learn, matplotlib, seaborn, numpy). Install them using the command above.

## License & contact

This repository does not specify a license. Add a `LICENSE` file if you want to open-source this project.
