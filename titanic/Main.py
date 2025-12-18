import numpy as np
import pandas as pd
import warnings
from scipy.stats import mode
from scipy.optimize import differential_evolution

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_SPLITS = 15  # Increased for more robust validation

train_path = "titanic/train.csv"
test_path = "titanic/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# ===== ULTRA FEATURE ENGINEERING =====

def get_family_survival_feature(df_train, df_test):
    df_train['is_train'] = 1
    df_test['is_train'] = 0
    df_test['Survived'] = np.nan
    
    full = pd.concat([df_train, df_test], ignore_index=True)
    full['Last_Name'] = full['Name'].apply(lambda x: str(x).split(',')[0].strip())
    full['Family_Survival'] = 0.5
    full['Family_Survival_Grp'] = 0.5

    # Family survival by Last Name + Fare
    for grp, grp_df in full.groupby(['Last_Name', 'Fare']):
        if len(grp_df) > 1:
            for ind, row in grp_df.iterrows():
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                
                if (smax == 1.0):
                    full.loc[full['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin == 0.0):
                    full.loc[full['PassengerId'] == passID, 'Family_Survival'] = 0

    # Ticket group survival
    for _, grp_df in full.groupby('Ticket'):
        if len(grp_df) > 1:
            for ind, row in grp_df.iterrows():
                if (row['Family_Survival'] == 0) or (row['Family_Survival'] == 0.5):
                    smax = grp_df.drop(ind)['Survived'].max()
                    smin = grp_df.drop(ind)['Survived'].min()
                    passID = row['PassengerId']
                    
                    if (smax == 1.0):
                        full.loc[full['PassengerId'] == passID, 'Family_Survival'] = 1
                    elif (smin == 0.0):
                        full.loc[full['PassengerId'] == passID, 'Family_Survival'] = 0
    
    # Group survival rate
    for _, grp_df in full.groupby('Ticket'):
        if len(grp_df) > 1:
            mean_survival = grp_df['Survived'].mean()
            for ind in grp_df.index:
                if pd.notna(mean_survival):
                    full.loc[ind, 'Family_Survival_Grp'] = mean_survival
                        
    return full

full_df = get_family_survival_feature(train_df, test_df)

def advanced_feature_engineering(df):
    df = df.copy()

    # === TITLES ===
    df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
    
    # More granular title mapping
    df["Title"] = df["Title"].replace(
        ["Lady","Countess","Dona"], "Royal_Female"
    )
    df["Title"] = df["Title"].replace(
        ["Capt","Col","Don","Major","Sir","Jonkheer"], "Royal_Male"
    )
    df["Title"] = df["Title"].replace(["Dr","Rev"], "Professional")
    df["Title"] = df["Title"].replace({"Mlle":"Miss", "Ms":"Miss", "Mme":"Mrs"})
    
    # === TICKET ===
    df['Ticket_Prefix'] = df['Ticket'].apply(lambda x: str(x).split()[0] if len(str(x).split()) > 1 else 'None')
    df['Ticket_Prefix'] = df['Ticket_Prefix'].replace(['LINE', 'SC/Paris', 'CA', 'PC', 'SOTON/OQ', 'STON/O2'], 'Rare')
    df['Ticket_Frequency'] = df.groupby('Ticket')['Ticket'].transform('count')
    df['Ticket_Shared'] = (df['Ticket_Frequency'] > 1).astype(int)
    
    # === CABIN ===
    df['Deck'] = df['Cabin'].apply(lambda x: str(x)[0] if pd.notna(x) else 'U')
    df['Deck'] = df['Deck'].replace(['T', 'G'], 'U')
    df['Has_Cabin'] = df['Cabin'].notna().astype(int)
    df['Cabin_Count'] = df['Cabin'].apply(lambda x: 0 if pd.isna(x) else len(str(x).split()))
    
    # Room number extraction
    def extract_room_number(cabin):
        if pd.isna(cabin):
            return -1
        try:
            numbers = ''.join(filter(str.isdigit, str(cabin).split()[0]))
            return int(numbers) if numbers else -1
        except:
            return -1
    
    df['Room_Number'] = df['Cabin'].apply(extract_room_number)
    df['Room_Side'] = df['Room_Number'].apply(lambda x: x % 2 if x > 0 else -1)  # Odd/Even side
    
    # === AGE IMPUTATION (Multi-level) ===
    df["Age"] = df["Age"].fillna(df.groupby(["Title", "Pclass", "Sex"])["Age"].transform("median"))
    df["Age"] = df["Age"].fillna(df.groupby(["Title", "Pclass"])["Age"].transform("median"))
    df["Age"] = df["Age"].fillna(df.groupby("Title")["Age"].transform("median"))
    df["Age"] = df["Age"].fillna(df["Age"].median())
    
    # === FARE IMPUTATION ===
    df["Fare"] = df["Fare"].fillna(df.groupby(["Pclass", "Embarked"])["Fare"].transform("median"))
    df["Fare"] = df["Fare"].fillna(df.groupby("Pclass")["Fare"].transform("median"))
    
    # === AGE FEATURES ===
    df['Age_Squared'] = df['Age'] ** 2
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 5, 12, 18, 25, 35, 50, 65, 80], labels=False)
    df['Is_Child'] = (df['Age'] <= 12).astype(int)
    df['Is_Young_Adult'] = ((df['Age'] > 18) & (df['Age'] <= 35)).astype(int)
    df['Is_Senior'] = (df['Age'] >= 60).astype(int)
    
    # === FARE FEATURES ===
    df['Fare_Squared'] = df['Fare'] ** 2
    df['Fare_Log'] = np.log1p(df['Fare'])
    df['FareBin'] = pd.qcut(df['Fare'], 8, duplicates='drop', labels=False)
    
    # === FAMILY FEATURES ===
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["Is_Alone"] = (df["FamilySize"] == 1).astype(int)
    df['Small_Family'] = ((df['FamilySize'] >= 2) & (df['FamilySize'] <= 4)).astype(int)
    df['Large_Family'] = (df['FamilySize'] >= 5).astype(int)
    
    df['Has_Spouse'] = (df['SibSp'] > 0).astype(int)
    df['Has_Children'] = (df['Parch'] > 0).astype(int)
    
    # Mother feature (enhanced)
    df["Is_Mother"] = ((df["Sex"] == "female") & (df["Parch"] > 0) & (df["Age"] > 18) & 
                       (df["Title"].isin(["Mrs", "Royal_Female"]))).astype(int)
    
    # === INTERACTION FEATURES ===
    df["Age_Class"] = df["Age"] * df["Pclass"]
    df["Fare_Class"] = df["Fare"] / (df["Pclass"] + 1)
    df["Fare_Per_Person"] = df["Fare"] / df["FamilySize"]
    df["Age_Fare"] = df["Age"] * df["Fare"]
    
    # Sex + Class combinations
    df["Sex_Pclass"] = df["Sex"].astype(str) + "_" + df["Pclass"].astype(str)
    df['Female_Class1'] = ((df['Sex'] == 'female') & (df['Pclass'] == 1)).astype(int)
    df['Female_Class2'] = ((df['Sex'] == 'female') & (df['Pclass'] == 2)).astype(int)
    df['Male_Class3'] = ((df['Sex'] == 'male') & (df['Pclass'] == 3)).astype(int)
    
    # Embarked + Class
    df['Embarked_Pclass'] = df['Embarked'].astype(str) + "_" + df['Pclass'].astype(str)
    
    # Complex features
    df['Rich_Woman'] = ((df['Sex'] == 'female') & (df['Pclass'] <= 2) & (df['Fare'] > 50)).astype(int)
    df['Poor_Man'] = ((df['Sex'] == 'male') & (df['Pclass'] == 3) & (df['Fare'] < 20)).astype(int)
    
    # Title + Fare interaction
    df['Title_Fare'] = df.groupby('Title')['Fare'].rank(pct=True)
    
    # === NAME LENGTH (social status indicator) ===
    df['Name_Length'] = df['Name'].apply(lambda x: len(str(x)))
    df['Name_Word_Count'] = df['Name'].apply(lambda x: len(str(x).split()))
    
    # Drop unused
    df.drop(columns=["Name", "PassengerId", "Ticket", "Cabin", "Last_Name"], 
            inplace=True, errors="ignore")
    
    return df

full_df = advanced_feature_engineering(full_df)

train_df = full_df[full_df['is_train'] == 1].drop(columns=['is_train'])
test_df = full_df[full_df['is_train'] == 0].drop(columns=['is_train', 'Survived'])

X = train_df.drop(columns=["Survived"])
y = train_df["Survived"].values

# Separate numeric and categorical
num_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
cat_cols = [col for col in X.columns if X[col].dtype == 'object']

print(f"Numeric features: {len(num_cols)}")
print(f"Categorical features: {len(cat_cols)}")

# === PREPROCESSING ===
numeric_transformer = Pipeline([
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

# === MODEL ENSEMBLE (Optimized Hyperparameters) ===
models = [
    RandomForestClassifier(
        n_estimators=1000, max_depth=8, min_samples_split=3, 
        min_samples_leaf=1, max_features='sqrt', class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    ExtraTreesClassifier(
        n_estimators=1000, max_depth=8, min_samples_split=3,
        min_samples_leaf=1, max_features='sqrt', class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    XGBClassifier(
        n_estimators=800, learning_rate=0.02, max_depth=5, 
        subsample=0.85, colsample_bytree=0.85, gamma=0.05,
        min_child_weight=2, reg_alpha=0.5, reg_lambda=1.0,
        eval_metric="logloss", scale_pos_weight=1,
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    lgb.LGBMClassifier(
        n_estimators=800, learning_rate=0.02, max_depth=6,
        num_leaves=31, subsample=0.85, colsample_bytree=0.85,
        min_child_samples=15, reg_alpha=0.5, reg_lambda=1.0,
        class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    ),
    CatBoostClassifier(
        iterations=1000, learning_rate=0.02, depth=7, 
        l2_leaf_reg=5, auto_class_weights='Balanced',
        verbose=0, random_state=RANDOM_STATE
    ),
    GradientBoostingClassifier(
        n_estimators=800, learning_rate=0.03, max_depth=5,
        subsample=0.85, min_samples_split=3, min_samples_leaf=2,
        random_state=RANDOM_STATE
    ),
    SVC(
        kernel='rbf', C=10.0, gamma='scale', 
        probability=True, class_weight='balanced', random_state=RANDOM_STATE
    ),
    LogisticRegression(
        C=0.1, penalty='l2', solver='liblinear', 
        class_weight='balanced', random_state=RANDOM_STATE, max_iter=1000
    ),
    KNeighborsClassifier(
        n_neighbors=20, weights='distance', metric='minkowski', p=2
    ),
    MLPClassifier(
        hidden_layer_sizes=(100, 50), activation='relu', alpha=0.01,
        learning_rate='adaptive', max_iter=1000, random_state=RANDOM_STATE
    )
]

# === ADVANCED STACKING ===
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

oof_preds = np.zeros((len(X), len(models)))
test_preds = np.zeros((len(test_df), len(models)))

print(f"\n{'='*60}")
print(f"Training {len(models)} models with {N_SPLITS}-fold CV")
print(f"{'='*60}\n")

for i, model in enumerate(models):
    model_name = model.__class__.__name__
    print(f"[{i+1}/{len(models)}] {model_name:<30}", end=" ")
    
    fold_test_preds = np.zeros((len(test_df), N_SPLITS))
    fold_scores = []
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_tr, y_tr)
        
        val_probs = pipe.predict_proba(X_val)[:, 1]
        oof_preds[val_idx, i] = val_probs
        fold_test_preds[:, fold] = pipe.predict_proba(test_df)[:, 1]
        
        fold_acc = accuracy_score(y_val, (val_probs >= 0.5).astype(int))
        fold_scores.append(fold_acc)
    
    test_preds[:, i] = fold_test_preds.mean(axis=1)
    
    cv_preds = (oof_preds[:, i] >= 0.5).astype(int)
    cv_acc = accuracy_score(y, cv_preds)
    cv_auc = roc_auc_score(y, oof_preds[:, i])
    
    print(f"CV Acc: {cv_acc:.5f} | AUC: {cv_auc:.5f}")

# === OPTIMIZE ENSEMBLE WEIGHTS ===
print(f"\n{'='*60}")
print("Optimizing Ensemble Weights...")
print(f"{'='*60}\n")

def ensemble_objective(weights):
    weights = np.abs(weights)
    weights /= weights.sum()
    blend = oof_preds @ weights
    thresholds = np.arange(0.4, 0.6, 0.01)
    best_acc = max([accuracy_score(y, (blend >= t).astype(int)) for t in thresholds])
    return -best_acc

# Optimize weights using differential evolution
bounds = [(0, 1)] * len(models)
result = differential_evolution(ensemble_objective, bounds, seed=RANDOM_STATE, maxiter=100, workers=-1)

optimal_weights = np.abs(result.x)
optimal_weights /= optimal_weights.sum()

print("Optimized Model Weights:")
model_names = ['RF', 'ET', 'XGB', 'LGBM', 'Cat', 'GB', 'SVC', 'LR', 'KNN', 'MLP']
for name, weight in zip(model_names, optimal_weights):
    if weight > 0.01:
        print(f"  {name:<6}: {weight:.4f}")

# === FIND OPTIMAL THRESHOLD ===
blend_val_probs = oof_preds @ optimal_weights

thresholds = np.arange(0.35, 0.65, 0.002)
accs = [accuracy_score(y, (blend_val_probs >= t).astype(int)) for t in thresholds]
best_thresh = thresholds[np.argmax(accs)]
best_acc = max(accs)

print(f"\nBest Threshold: {best_thresh:.4f}")
print(f"Best CV Accuracy: {best_acc:.5f}")
print(f"Best CV AUC: {roc_auc_score(y, blend_val_probs):.5f}")

# === FINAL PREDICTIONS ===
blend_test_probs = test_preds @ optimal_weights
final_test_preds = (blend_test_probs >= best_thresh).astype(int)

# Majority voting as backup
majority_preds = mode(np.array([(test_preds[:, i] >= 0.5).astype(int) for i in range(len(models))]), axis=0)[0].flatten()

# Combine weighted ensemble with majority voting
final_combined = np.where(
    (blend_test_probs >= 0.45) & (blend_test_probs <= 0.55),  # Uncertain region
    majority_preds,
    final_test_preds
)

submission = pd.DataFrame({
    "PassengerId": pd.read_csv(test_path)["PassengerId"],
    "Survived": final_combined
})

submission.to_csv("submission_advanced.csv", index=False)
print(f"\n{'='*60}")
print("✓ submission_advanced.csv saved!")
print(f"{'='*60}\n")