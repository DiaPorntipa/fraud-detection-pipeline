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


MODELS = Path("models")
pipe = joblib.load(MODELS / "fraud_rf.pkl")
RISK = set(json.load(open(MODELS / "risk_types.json")))
THRESHOLD = json.load(open(MODELS / "rf_metrics.json")).get("threshold", 0.5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="Fraud API", lifespan=lifespan)


class Tx(BaseModel):
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
    keep = [
        "transac_type", "amount", "src_bal", "dst_bal",
        "hour", "day_of_week", "flag_any_inconsistency", "amount_over_src"
    ]

    return df[keep]


@app.post("/predict")
def predict(tx: Tx):
    tx_dict = tx.model_dump()

    # Rule-based prediction
    # TODO (LOW): Add red flaf catching
    if tx_dict["amount"] > 200_000:
        save_fraud_rule(tx_dict)
        return {"fraud": 1, "reason": "rule_amount_cap"}
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
        save_fraud_ml(tx_dict, proba, THRESHOLD, pred)
    return {"fraud": pred, "proba": proba, "threshold": THRESHOLD, "reason": "model"}


@app.get("/frauds")
def get_frauds():
    return list_frauds()
