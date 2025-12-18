import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from xgboost import XGBClassifier

RANDOM_STATE = 42

# --------------------------------------------------
# Load data
# --------------------------------------------------
train_df = pd.read_csv("/home/harsh/Desktop/Machine Learning/ML/titanic/train.csv")
test_df = pd.read_csv("/home/harsh/Desktop/Machine Learning/ML/titanic/test.csv")

# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Title extraction
    df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
    df["Title"] = df["Title"].replace(
        ["Lady", "Countess", "Capt", "Col", "Don", "Dr",
         "Major", "Rev", "Sir", "Jonkheer", "Dona"], "Rare"
    )
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    # Family features
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["Is_Alone"] = (df["FamilySize"] == 1).astype(int)

    # Fare per person (strong signal)
    df["Fare_Per_Person"] = df["Fare"] / df["FamilySize"]

    # Ticket group size
    df["Ticket_Count"] = df.groupby("Ticket")["Ticket"].transform("count")

    # Drop unused columns
    df.drop(columns=["Cabin", "Ticket", "Name", "PassengerId"], inplace=True, errors="ignore")

    return df


train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

X = train_df.drop(columns=["Survived"])
y = train_df["Survived"]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include="object").columns

# --------------------------------------------------
# Preprocessing Pipeline
# --------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ]
)

# --------------------------------------------------
# Train / Validation split
# --------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

# --------------------------------------------------
# Models & Hyperparameters
# --------------------------------------------------
models = {
    "LR": (
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        {"model__C": [0.1, 1, 10]}
    ),
    "SVC": (
        SVC(probability=True, random_state=RANDOM_STATE),
        {"model__C": [1, 10], "model__kernel": ["rbf"]}
    ),
    "RF": (
        RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        {"model__n_estimators": [200, 400], "model__max_depth": [5, 10]}
    ),
    "XGB": (
        XGBClassifier(
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric="logloss",
            tree_method="hist",
            n_estimators=300
        ),
        {
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [3, 4, 5]
        }
    )
}

# --------------------------------------------------
# Train models
# --------------------------------------------------
best_estimators = []
val_scores = []

print("\nTraining models...\n")

for name, (model, params) in models.items():
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    grid = GridSearchCV(
        pipe,
        params,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    val_pred = best_model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)

    best_estimators.append((name, best_model))
    val_scores.append(val_acc)

    print(f"{name}")
    print(f"  Best CV Score : {grid.best_score_:.4f}")
    print(f"  Val Accuracy  : {val_acc:.4f}\n")

# --------------------------------------------------
# Weighted Soft Voting Ensemble
# --------------------------------------------------
ensemble = VotingClassifier(
    estimators=best_estimators,
    voting="soft",
    weights=val_scores,
    n_jobs=-1
)

ensemble.fit(X_train, y_train)

ensemble_acc = accuracy_score(y_val, ensemble.predict(X_val))
print(f"Final Ensemble Validation Accuracy: {ensemble_acc:.4f}")

# --------------------------------------------------
# Train on full data & predict test set
# --------------------------------------------------
ensemble.fit(X, y)
test_preds = ensemble.predict(test_df)

submission = pd.DataFrame({
    "PassengerId": pd.read_csv(
        "/home/harsh/Desktop/Machine Learning/ML/titanic/test.csv"
    )["PassengerId"],
    "Survived": test_preds
})

submission.to_csv("titanic/submission.csv", index=False)
print("\nsubmission.csv saved successfully!")
