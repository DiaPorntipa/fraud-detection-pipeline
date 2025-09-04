import json
import numpy as np
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, precision_recall_curve
)

from utils import drop_rule_based_rows


def hybrid_eval(pipe, X_test, y_test, metrics_path="metrics.json", min_precision=0.95):
    '''
    Evaluate the trained model on test data.
    The current state includes only model-based evaluation, but the goal is to extend it to hybrid evaluation (rule-based + ML).
    '''
    # TODO: Implement hybrid evaluation: apply rule-based checks (amount > 200000) and route
    #       non-risk types before computing metrics.
    X_test, y_test = drop_rule_based_rows(X_test, y_test)

    # Prediction confidence. For RF, it’s the fraction of trees voting fraud.
    y_score = pipe.predict_proba(X_test)[:, 1]

    # AUCs
    roc = roc_auc_score(y_test, y_score)
    ap = average_precision_score(y_test, y_score)

    # Recall @ fixed precision and operating threshold
    rec_at_p = None
    thr_star = None
    p, r, thr = precision_recall_curve(y_test, y_score)
    mask = p[:-1] >= min_precision  # align with thresholds
    if mask.any():
        idx = np.argmax(r[:-1][mask])  # max recall under the constraint
        thr_star = float(thr[mask][idx])
        y_pred = (y_score >= thr_star).astype(int)
    else:
        y_pred = (y_score >= 0.5).astype(int)  # fallback
    rec_at_p = float(r[:-1][mask].max()) if mask.any() else 0.0

    # Thresholded metrics at the chosen operating point
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )

    print("\n--- Test Metrics (model-only) ---")
    print(f"PR-AUC: {ap:.4f}   ROC-AUC: {roc:.4f}")
    print(f"Recall@P≥{int(min_precision*100)}%: {rec_at_p:.4f}")

    print(
        f"\nOperating threshold (P≥{int(min_precision*100)}%): {thr_star:.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nReport:\n", classification_report(
        y_test, y_pred, digits=4, zero_division=0))

    with open(metrics_path, "w") as f:
        json.dump(
            {
                "pr_auc": None if np.isnan(ap) else float(ap),
                "roc_auc": None if np.isnan(roc) else float(roc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "recall_at_p": rec_at_p,
                "min_precision": float(min_precision),
                "threshold": thr_star,
            },
            f,
            indent=2,
        )
    print(f"Saved metrics to: {metrics_path}")
