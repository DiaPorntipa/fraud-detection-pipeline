from pathlib import Path
from contextlib import asynccontextmanager
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


class Tx(BaseModel):  # Tansaction
    transac_type: str
    amount: float
    src_bal: float
    dst_bal: Optional[float] = None
    hour: int
    day_of_week: int
    flag_any_inconsistency: int
    amount_over_src: float


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

    # Model prediction
    row = pd.DataFrame([tx_dict])
    proba = float(pipe.predict_proba(row)[:, 1][0])
    pred = int(proba >= THRESHOLD)

    if pred:
        save_fraud_ml(tx_dict, proba, THRESHOLD, pred)
    return {"fraud": pred, "proba": proba, "threshold": THRESHOLD, "reason": "model"}


@app.get("/frauds")
def get_frauds():
    return list_frauds()
