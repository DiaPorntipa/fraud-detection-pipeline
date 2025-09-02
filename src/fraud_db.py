import sqlite3
import json
from pathlib import Path

DB_PATH = Path("db/frauds.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    with get_conn() as con:
        # TODO (LOW): Consider adding "CREATE INDEX IF NOT EXISTS idx_frauds_status_ts ON frauds(status, ts);"
        con.executescript("""
        CREATE TABLE IF NOT EXISTS frauds(
          id          INTEGER PRIMARY KEY AUTOINCREMENT,
          ts          TEXT    DEFAULT CURRENT_TIMESTAMP,
          tx_json     TEXT    NOT NULL,
          reason      TEXT    NOT NULL,
          proba       REAL,
          threshold   REAL,
          status      TEXT    NOT NULL DEFAULT 'NEW'
                    CHECK (status IN ('NEW','REVIEWED'))
        );
        """)
    print("Initialized SQLite DB at", DB_PATH)


def save_fraud_rule(tx: dict):
    with get_conn() as con:
        con.execute(
            "INSERT INTO frauds(tx_json, reason) VALUES(?,?)",
            (json.dumps(tx), "rule_amount_cap"),
        )


def save_fraud_ml(tx: dict, proba: float, thr: float):
    with get_conn() as con:
        con.execute(
            "INSERT INTO frauds(tx_json, reason, proba, threshold) VALUES(?,?,?,?)",
            (json.dumps(tx), "ml_model", proba, thr),
        )


def list_frauds(limit: int = 100):
    with get_conn() as con:
        rows = con.execute(
            "SELECT id, ts, tx_json, reason, proba, threshold, status "
            "FROM frauds ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

    return [
        {"id": r[0], "ts": r[1], "tx": json.loads(r[2]),
         "reason": r[3], "proba": r[4], "threshold": r[5], "status": r[6]}
        for r in rows
    ]
