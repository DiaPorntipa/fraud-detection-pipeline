import json
from pathlib import Path

MODEL_DIR = Path("models")
RISK_TYPES_PATH = MODEL_DIR / "risk_types.json"


def drop_rule_based_rows(X, y):
    # load risk types
    with open(RISK_TYPES_PATH) as f:
        RISK_TYPES = set(json.load(f))

    # keep only risk types
    mask_risk = X["transac_type"].isin(RISK_TYPES)
    X_risk = X.loc[mask_risk].copy()
    y_risk = y.loc[mask_risk].copy()

    # drop flagged rows
    mask_flagged = X_risk["is_flagged_fraud"] == 1
    X_risk = X_risk.loc[~mask_flagged].drop(columns=["is_flagged_fraud"])
    y_risk = y_risk.loc[~mask_flagged]

    return X_risk, y_risk
