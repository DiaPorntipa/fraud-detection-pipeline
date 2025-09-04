from pathlib import Path
from eval import hybrid_eval
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from utils import drop_rule_based_rows


DATA_PATH = Path("data/processed.csv")  # from data_prep.py
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "fraud_rf.pkl"
METRICS_PATH = MODEL_DIR / "rf_metrics.json"
DATA_DIR = Path("data")
RISK_TYPES_PATH = DATA_DIR / "risk_types.json"


RANDOM_STATE = 42
TEST_SIZE = 0.20


def model_input_prep(df: pd.DataFrame):
    '''
    Prepare model input data by splitting into training and test sets,
    and applying rule-based filtering to the training set.
    '''
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=["is_fraud"]), df["is_fraud"],
        test_size=TEST_SIZE,
        stratify=df["is_fraud"],
        random_state=RANDOM_STATE
    )

    # Risk types sanity check on TRAIN
    with open(RISK_TYPES_PATH) as f:
        RISK_TYPES = set(json.load(f))
    assert y_train[~X_train["transac_type"].isin(RISK_TYPES)].sum() == 0, \
        "Found fraud in non-risk types—update RISK_TYPES or drop this assert."

    # TODO: Add option for training the entire dataset without using rule-based filtering
    X_train_risk, y_train_risk = drop_rule_based_rows(X_train, y_train)
    print(
        f"Training on {len(X_train_risk):,} risk rows (fraud rate: {100*y_train_risk.mean():.4f}%)")

    return X_train_risk, y_train_risk, X_test, y_test


def main():
    df = pd.read_csv(DATA_PATH)
    X_train_risk, y_train_risk, X_test, y_test = model_input_prep(df)

    # --- Build pipeline ---
    drop_cols = {"transac_type"}
    num_features = [c for c in X_train_risk.columns if c not in drop_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["transac_type"]),
            ("num", "passthrough", num_features),
        ],
        remainder="drop",
    )

    base_rf = RandomForestClassifier()
    pipe = Pipeline([("prep", pre), ("rf", base_rf)])

    # --- Train ---
    print("\n--- Training with RandomisedSearchCV ---")
    # RandomizedSearchCV (fast, imbalanced-friendly scoring)
    # TODO: Do more research for better distributions
    param_distributions = {
        "rf__n_estimators": [50, 100, 200],
        "rf__max_depth": [None, 8, 12, 20],
        "rf__min_samples_leaf": [1, 2, 4, 8],
        "rf__max_features": ["sqrt", "log2", 0.5, 0.8],
        "rf__class_weight": [None, "balanced_subsample"],
        "rf__bootstrap": [True],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions,
        n_iter=20,  # TODO: Increase iteration if time allows for better model performance
        scoring="average_precision",   # PR-AUC, defualt metric for fraud detection
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE,
        refit=True,  # refit on full TRAIN with best params
    )

    X_tune, y_tune = X_train_risk, y_train_risk
    search.fit(X_tune, y_tune)
    best_pipe = search.best_estimator_
    print("\nBest params:")
    for k in sorted(search.best_params_):
        print(f"  {k:28} = {search.best_params_[k]}")
    print("Best cross validation PR-AUC:", f"{search.best_score_:.4f}")

    # --- Evaluation and save ---
    hybrid_eval(best_pipe, X_test, y_test, metrics_path=METRICS_PATH)

    # Save model and risk types
    joblib.dump(best_pipe, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
