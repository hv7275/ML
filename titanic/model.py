import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression

from xgboost import XGBClassifier

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_SPLITS = 5

# Paths
train_path = "/home/harsh/Desktop/Machine Learning/ML/titanic/train.csv"
test_path = "/home/harsh/Desktop/Machine Learning/ML/titanic/test.csv"

# Load Data
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# --- 1. Global Feature Engineering ---
# Concatenate to ensure frequency features (like Ticket counts) are accurate globally
train_df['is_train'] = 1
test_df['is_train'] = 0
test_df['Survived'] = np.nan 

full_df = pd.concat([train_df, test_df], ignore_index=True)

def process_data(df):
    df = df.copy()

    # Title Extraction
    df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
    df["Title"] = df["Title"].replace(
        ["Lady","Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],
        "Rare"
    )
    df["Title"] = df["Title"].replace({"Mlle":"Miss", "Ms":"Miss", "Mme":"Mrs"})

    # --- Smart Imputation (Crucial Step) ---
    # Impute Age based on Title median (better than global median)
    df["Age"] = df["Age"].fillna(df.groupby("Title")["Age"].transform("median"))
    
    # Impute missing Fare based on Pclass median
    df["Fare"] = df["Fare"].fillna(df.groupby("Pclass")["Fare"].transform("median"))

    # Feature Creation
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["Is_Alone"] = (df["FamilySize"] == 1).astype(int)

    # Ticket Frequency (Calculated on full dataset)
    df["Ticket_Count"] = df.groupby("Ticket")["Ticket"].transform("count")

    df["Fare_Per_Person"] = df["Fare"] / df["FamilySize"]
    
    # Interaction Terms (Now using imputed Age so we don't lose data)
    df["Age_Class"] = df["Age"] * df["Pclass"]
    df["Child"] = (df["Age"] < 14).astype(int)

    # Cleanup
    df.drop(columns=["Cabin","Ticket","Name","PassengerId"], inplace=True, errors="ignore")

    return df

full_df = process_data(full_df)

# Split back into Train and Test
train_df = full_df[full_df['is_train'] == 1].drop(columns=['is_train'])
test_df = full_df[full_df['is_train'] == 0].drop(columns=['is_train', 'Survived'])

X = train_df.drop(columns=["Survived"])
y = train_df["Survived"].values

# Identify columns
num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_cols = X.select_dtypes(include="object").columns.tolist()

# --- 2. Pipeline Setup ---

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")), # Safety net for any remaining NaNs
    ("scaler", RobustScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)) # Updated syntax
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

# --- 3. Model Definitions ---
# Slightly tuned hyperparameters
models = [
    LogisticRegression(
        C=1.2,
        max_iter=2000,
        solver="lbfgs",
        random_state=RANDOM_STATE
    ),
    RandomForestClassifier(
        n_estimators=600,
        max_depth=10,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE
    ),
    XGBClassifier(
        n_estimators=500,
        learning_rate=0.035,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        tree_method="hist", # Efficient for larger datasets
        random_state=RANDOM_STATE
    )
]

# --- 4. Training Loop (Stratified K-Fold) ---

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

oof_preds = np.zeros((len(X), len(models)))
test_preds = np.zeros((len(test_df), len(models)))

print(f"Starting Training on {len(models)} models with {N_SPLITS} folds...")

for i, model in enumerate(models):
    model_name = model.__class__.__name__
    fold_test_preds = np.zeros((len(test_df), N_SPLITS))
    
    print(f"Training {model_name}...", end=" ")
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_tr, y_tr)

        # Store Out-of-Fold predictions
        oof_preds[val_idx, i] = pipe.predict_proba(X_val)[:, 1]
        # Store Test predictions for this fold
        fold_test_preds[:, fold] = pipe.predict_proba(test_df)[:, 1]

    # Average predictions across folds for this model
    test_preds[:, i] = fold_test_preds.mean(axis=1)
    print("Done.")

# --- 5. Automated Ensemble Weighting ---
# Use Linear Regression (constrained to positive) to find optimal blending weights
print("Calculating optimal ensemble weights...")

meta_model = LinearRegression(fit_intercept=False, positive=True)
meta_model.fit(oof_preds, y)

weights = meta_model.coef_
weights /= weights.sum() # Normalize to sum to 1

print(f"Optimal Weights: LR={weights[0]:.3f}, RF={weights[1]:.3f}, XGB={weights[2]:.3f}")

# Calculate weighted probabilities
blend_val_probs = oof_preds @ weights

# --- 6. Threshold Tuning ---
thresholds = np.arange(0.40, 0.60, 0.001)
accs = [accuracy_score(y, (blend_val_probs >= t).astype(int)) for t in thresholds]

best_idx = np.argmax(accs)
best_thresh = thresholds[best_idx]
best_acc = accs[best_idx]

print(f"Best CV Accuracy: {best_acc:.4f} at threshold {best_thresh:.3f}")

# --- 7. Submission ---
blend_test_probs = test_preds @ weights
final_test_preds = (blend_test_probs >= best_thresh).astype(int)

# Reload original test csv just to get correct PassengerIds safely
submission = pd.DataFrame({
    "PassengerId": pd.read_csv(test_path)["PassengerId"],
    "Survived": final_test_preds
})

submission.to_csv("submission_refined.csv", index=False)
print("submission_refined.csv saved successfully.")