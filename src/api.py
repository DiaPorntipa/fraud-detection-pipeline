from pathlib import Path
from contextlib import asynccontextmanager
from typing import Literal
from src.features import feature_engineering
from src.fraud_db import init_db, save_fraud_rule, save_fraud_ml, list_frauds
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import json
from typing import Optional
import pandas as pd
import logging
log = logging.getLogger("uvicorn.error")


MODEL_DIR = Path("models")
DATA_DIR = Path("data")
pipe = joblib.load(MODEL_DIR / "fraud_best.pkl")
RISK = set(json.load(open(DATA_DIR / "risk_types.json")))
THRESHOLD = json.load(
    open(MODEL_DIR / "best_metrics.json")).get("threshold", 0.5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="Fraud API", lifespan=lifespan)


class Tx(BaseModel):
    '''
    Transaction input data model.
    '''
    time_ind: int
    transac_type: Literal["CASH_IN", "CASH_OUT",
                          "DEBIT", "PAYMENT", "TRANSFER"]
    amount: float
    src_acc: str
    src_bal: float
    src_new_bal: float
    dst_acc: str
    dst_bal: Optional[float] = 0.0
    dst_new_bal: Optional[float] = 0.0


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Select a subset of columns for parsing to the ML model.
    '''
    keep = [
        "transac_type", "amount", "src_bal", "dst_bal",
        "hour", "day_of_week", "flag_any_inconsistency", "amount_over_src"
    ]

    return df[keep]


@app.post("/predict")
def predict(tx: Tx):
    '''
    Predict fraud for a given transaction. Apply rule-based checks first, then ML model if needed. If fraud is detected, save to the database.
    '''
    tx_dict = tx.model_dump()

    # Rule-based prediction
    if tx_dict["amount"] > 200_000:
        save_fraud_rule(tx_dict)
        return {"fraud": 1, "reason": "rule_amount_cap"}
    # TODO: Add option for not using transac_type rule-based prediction
    # TODO (LOW): Add red flaf catching
    if tx_dict["transac_type"] not in RISK:
        return {"fraud": 0, "reason": "non_risk_type"}

    # Data preprocessing
    row = pd.DataFrame([tx_dict])
    row = feature_engineering(row)
    row = select_columns(row)

    # Model prediction
    proba = float(pipe.predict_proba(row)[:, 1][0])
    pred = int(proba >= THRESHOLD)

    if pred:
        save_fraud_ml(tx_dict, proba, THRESHOLD)
    return {"fraud": pred, "proba": proba, "threshold": THRESHOLD, "reason": "model"}


@app.get("/frauds")
def get_frauds():
    '''
    Get a list of all fraud cases.
    '''
    return list_frauds()
