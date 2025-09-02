# TODO: Expand test suite
from fastapi.testclient import TestClient
import src.api as api

client = TestClient(api.app)


def _tx(**k):
    d = {
        "time_ind": 10,
        "transac_type": "TRANSFER",
        "amount": 100.0,
        "src_acc": "A",
        "src_bal": 1000.0,
        "src_new_bal": 900.0,
        "dst_acc": "B",
        "dst_bal": 0.0,
        "dst_new_bal": 100.0
    }
    d.update(k)
    return d


def test_predict_smoke():
    r = client.post("/predict", json=_tx())
    assert r.status_code == 200
    body = r.json()
    assert "fraud" in body and body["fraud"] in (0, 1)
    assert body["reason"] == "model"
    assert 0.0 <= body["proba"] <= 1.0
    assert body["threshold"] == api.THRESHOLD


def test_rule_amount_cap():
    r = client.post("/predict", json=_tx(amount=200_001))
    assert r.status_code == 200
    assert r.json() == {"fraud": 1, "reason": "rule_amount_cap"}


def test_rule_non_risk_type():
    r = client.post("/predict", json=_tx(transac_type="CASH_IN"))
    assert r.status_code == 200
    assert r.json() == {"fraud": 0, "reason": "non_risk_type"}


def test_frauds():
    r = client.get("/frauds")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    if body:
        item = body[0]
        assert {"id", "ts", "tx", "reason", "proba",
                "threshold", "status"} <= set(item.keys())
        assert isinstance(item["id"], int)
        assert isinstance(item["ts"], str)
        assert isinstance(item["tx"], dict)
        assert isinstance(item["reason"], str)
        assert (item["proba"] is None) or isinstance(
            item["proba"], (int, float))
        assert (item["threshold"] is None) or isinstance(
            item["threshold"], (int, float))
        assert item["status"] in ("NEW", "REVIEWED")
