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
# Feature Engineering
# --------------------------------------------------
def feature_engineering(df):
    df = df.copy()

    df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
    df["Title"] = df["Title"].replace(
        ["Lady","Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],
        "Rare"
    )
    df["Title"] = df["Title"].replace({"Mlle":"Miss","Ms":"Miss","Mme":"Mrs"})

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["Is_Alone"] = (df["FamilySize"] == 1).astype(int)

    df["Fare_Per_Person"] = df["Fare"] / df["FamilySize"]

    df["Ticket_Count"] = df.groupby("Ticket")["Ticket"].transform("count")

    # High-signal interaction
    df["Age_Class"] = df["Age"] * df["Pclass"]

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
# Base models (strong only)
# --------------------------------------------------
models = [
    LogisticRegression(C=1, max_iter=1000, random_state=RANDOM_STATE),
    RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        n_jobs=-1,
        random_state=RANDOM_STATE
    ),
    XGBClassifier(
        n_estimators=350,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        use_label_encoder=False,
        tree_method="hist",
        random_state=RANDOM_STATE
    )
]

# --------------------------------------------------
# Blending (Out-of-Fold predictions)
# --------------------------------------------------
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

oof_preds = np.zeros((len(X), len(models)))
test_preds = np.zeros((len(test_df), len(models)))

for i, model in enumerate(models):
    fold_test_preds = np.zeros((len(test_df), N_SPLITS))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)

        oof_preds[val_idx, i] = pipe.predict_proba(X_val)[:, 1]
        fold_test_preds[:, fold] = pipe.predict_proba(test_df)[:, 1]

    test_preds[:, i] = fold_test_preds.mean(axis=1)

# --------------------------------------------------
# Meta-model (simple average works best)
# --------------------------------------------------
blend_val_preds = oof_preds.mean(axis=1)
blend_val_labels = (blend_val_preds >= 0.5).astype(int)

blend_acc = accuracy_score(y, blend_val_labels)
print(f"Blended CV Accuracy: {blend_acc:.4f}")

# --------------------------------------------------
# Final test prediction
# --------------------------------------------------
final_test_preds = (test_preds.mean(axis=1) >= 0.5).astype(int)

submission = pd.DataFrame({
    "PassengerId": pd.read_csv(
        "/home/harsh/Desktop/Machine Learning/ML/titanic/test.csv"
    )["PassengerId"],
    "Survived": final_test_preds
})

submission.to_csv("titanic/submission.csv", index=False)
print("submission.csv saved!")
