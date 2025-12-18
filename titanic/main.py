import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

RANDOM_STATE = 42
N_SPLITS = 5

# --------------------------------------------------
# Load data
# --------------------------------------------------
train_df = pd.read_csv("/home/harsh/Desktop/Machine Learning/ML/titanic/train.csv")
test_df = pd.read_csv("/home/harsh/Desktop/Machine Learning/ML/titanic/test.csv")

# --------------------------------------------------
# Feature Engineering (LAST-MILE SET)
# --------------------------------------------------
def feature_engineering(df):
    df = df.copy()

    # Title
    df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
    df["Title"] = df["Title"].replace(
        ["Lady","Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],
        "Rare"
    )
    df["Title"] = df["Title"].replace({"Mlle":"Miss","Ms":"Miss","Mme":"Mrs"})

    # Family
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["Is_Alone"] = (df["FamilySize"] == 1).astype(int)

    # Fare features
    df["Fare_Per_Person"] = df["Fare"] / df["FamilySize"]

    # Ticket groups
    df["Ticket_Count"] = df.groupby("Ticket")["Ticket"].transform("count")

    # Interactions
    df["Age_Class"] = df["Age"] * df["Pclass"]
    df["Child"] = (df["Age"] < 14).astype(int)

    df.drop(columns=["Cabin","Ticket","Name","PassengerId"], inplace=True, errors="ignore")

    return df


train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

X = train_df.drop(columns=["Survived"])
y = train_df["Survived"]

num_cols = X.select_dtypes(include=["int64","float64"]).columns
cat_cols = X.select_dtypes(include="object").columns

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

# --------------------------------------------------
# Strong Base Models ONLY
# --------------------------------------------------
models = [
    LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE),
    RandomForestClassifier(
        n_estimators=450,
        max_depth=9,
        n_jobs=-1,
        random_state=RANDOM_STATE
    ),
    XGBClassifier(
        n_estimators=400,
        learning_rate=0.045,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        tree_method="hist",
        use_label_encoder=False,
        random_state=RANDOM_STATE
    )
]

# --------------------------------------------------
# Blending (OOF)
# --------------------------------------------------
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

oof_preds = np.zeros((len(X), len(models)))
test_preds = np.zeros((len(test_df), len(models)))

for i, model in enumerate(models):
    fold_test_preds = np.zeros((len(test_df), N_SPLITS))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_tr, y_tr)

        oof_preds[val_idx, i] = pipe.predict_proba(X_val)[:, 1]
        fold_test_preds[:, fold] = pipe.predict_proba(test_df)[:, 1]

    test_preds[:, i] = fold_test_preds.mean(axis=1)

# --------------------------------------------------
# Weighted Blend (XGB favored)
# --------------------------------------------------
weights = np.array([0.28, 0.32, 0.40])  # LR, RF, XGB
blend_val_probs = np.dot(oof_preds, weights)

# --------------------------------------------------
# Threshold Search (LAST 0.5%)
# --------------------------------------------------
best_thresh = 0.5
best_acc = 0

for t in np.arange(0.45, 0.56, 0.01):
    acc = accuracy_score(y, (blend_val_probs >= t).astype(int))
    if acc > best_acc:
        best_acc = acc
        best_thresh = t

print(f"Best CV Accuracy: {best_acc:.4f} at threshold {best_thresh:.2f}")

# --------------------------------------------------
# Final Test Prediction
# --------------------------------------------------
blend_test_probs = np.dot(test_preds, weights)
final_test_preds = (blend_test_probs >= best_thresh).astype(int)

submission = pd.DataFrame({
    "PassengerId": pd.read_csv(
        "/home/harsh/Desktop/Machine Learning/ML/titanic/test.csv"
    )["PassengerId"],
    "Survived": final_test_preds
})

submission.to_csv("titanic/submission.csv", index=False)
print("submission.csv saved!")
