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

train_path = "/home/harsh/Desktop/Machine Learning/ML/titanic/train.csv"
test_path = "/home/harsh/Desktop/Machine Learning/ML/titanic/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

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

    df["Age_Class"] = df["Age"] * df["Pclass"]
    df["Child"] = (df["Age"] < 14).astype(int)

    df.drop(columns=["Cabin","Ticket","Name","PassengerId"], inplace=True, errors="ignore")

    return df

train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

X = train_df.drop(columns=["Survived"])
y = train_df["Survived"].values

num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_cols = X.select_dtypes(include="object").columns.tolist()

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
        tree_method="hist",
        random_state=RANDOM_STATE
    )
]

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

oof_preds = np.zeros((len(X), len(models)))
test_preds = np.zeros((len(test_df), len(models)))

for i, model in enumerate(models):
    fold_test_preds = np.zeros((len(test_df), N_SPLITS))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_tr, y_tr)

        oof_preds[val_idx, i] = pipe.predict_proba(X_val)[:, 1]
        fold_test_preds[:, fold] = pipe.predict_proba(test_df)[:, 1]

    test_preds[:, i] = fold_test_preds.mean(axis=1)

weights = np.array([0.25, 0.30, 0.45])
blend_val_probs = oof_preds @ weights

thresholds = np.arange(0.45, 0.56, 0.005)
accs = [accuracy_score(y, (blend_val_probs >= t).astype(int)) for t in thresholds]

best_idx = np.argmax(accs)
best_thresh = thresholds[best_idx]
best_acc = accs[best_idx]

print(f"Best CV Accuracy: {best_acc:.4f} at threshold {best_thresh:.3f}")

blend_test_probs = test_preds @ weights
final_test_preds = (blend_test_probs >= best_thresh).astype(int)

submission = pd.DataFrame({
    "PassengerId": pd.read_csv(test_path)["PassengerId"],
    "Survived": final_test_preds
})

submission.to_csv("submission.csv", index=False)
print("submission.csv saved")
