# app/inference.py
from __future__ import annotations

import logging, os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from requests.exceptions import RequestException, HTTPError, Timeout

# ────────────────────────────────────────────────────────────────
# Domain-specific errors so api.py can map them to HTTP
# ────────────────────────────────────────────────────────────────
class FeatureFetchError(RuntimeError):
    """Raised when online-feature retrieval from Feast fails."""
    pass


class InferenceError(RuntimeError):
    """Raised when the model cannot produce a prediction."""
    pass


logger = logging.getLogger(__name__)

FEATURE_ORDER = [
    "total_credit", "total_debit", "total_net_amt", "credit_mean", "debit_mean",
    "total_rate", "sd_credit", "sd_debit", "total_txn_count", "zero_txn",
    "volatility", "log_balance", "unique_source_inflow", "unique_source_outflow",
]


# ────────────────────────────────────────────────────────────────
# Business helpers (unchanged)
# ────────────────────────────────────────────────────────────────
def calculate_fico_score_micro(
    target_class: int,
    proba: float,
    min_fico: int,
    fico_max_class_0: int,
    fico_max_class_1: int,
    high_risky_max_bound: int,
    low_risky_lower_bound: int,
) -> int:
    if target_class == 0:
        return int(low_risky_lower_bound + proba * (fico_max_class_0 - low_risky_lower_bound))
    return int(min_fico + (1 - proba) * (high_risky_max_bound - min_fico))


def map_prediction_to_fico_score_micro(predict_prob: List[Tuple[int, float]]):
    min_fico, max_fico = 300, 850
    fico_max_class_0, fico_max_class_1 = 400, 150
    high_risky_max_bound = 450
    low_risky_lower_bound = 450

    best_score, best_class, max_probability = min_fico, 0, -1.0
    for label, proba in predict_prob:
        label = 1 if label is True else 0 if label is False else int(label)
        score = calculate_fico_score_micro(
            label, proba, min_fico,
            fico_max_class_0, fico_max_class_1,
            high_risky_max_bound, low_risky_lower_bound
        )
        if proba > max_probability:
            best_score, best_class, max_probability = score, label, proba
    return {"fico_score": best_score, "class": best_class}


# ────────────────────────────────────────────────────────────────
# The star of the show
# ────────────────────────────────────────────────────────────────
def run_inference(model, payload_from_client: Dict) -> Dict:
    FEAST_BASE_URL = os.getenv("FEAST_BASE_URL", "http://localhost:6567")

    # Normalise coop business quirk
    if payload_from_client.get("source_bank", "").lower() == "coop":
        payload_from_client["loan_type"] = "msme"

    feast_request = {
        "features": [f"txn_fv_new2:{f}" for f in FEATURE_ORDER],
        "entities": {"customerId": [payload_from_client["customerId"]]},
    }
    try:
        r = requests.post(f"{FEAST_BASE_URL}/get-online-features", json=feast_request, timeout=3)
        r.raise_for_status()  # HTTP 4xx/5xx → HTTPError :contentReference[oaicite:0]{index=0}
        meta, results = r.json()["metadata"], r.json()["results"]
    except (HTTPError, Timeout, RequestException) as e:  # all Requests errors inherit RequestException :contentReference[oaicite:1]{index=1}
        logger.error("Feast request failed: %s", e, exc_info=True)
        raise FeatureFetchError(str(e)) from e
    except (KeyError, ValueError) as e:
        # malformed JSON or missing keys
        logger.error("Unexpected Feast payload: %s", e, exc_info=True)
        raise FeatureFetchError("Invalid payload from Feast") from e

    try:
        values = [(v[0] if isinstance(v, list) else v) for r in results for v in r["values"]]
        feature_names = [c for c in meta["feature_names"] if c != "customerId"]
        df = (
            pd.DataFrame([values], columns=meta["feature_names"])  # shape mis-match → ValueError :contentReference[oaicite:2]{index=2}
            .drop(columns=["customerId"])
            .loc[:, FEATURE_ORDER]
        )
    except (ValueError, KeyError) as e:
        logger.error("Feature frame construction failed: %s", e, exc_info=True)
        raise FeatureFetchError("Feature frame construction failed") from e

    try:
        y_pred = model.predict(df)           # may raise if model not fitted :contentReference[oaicite:3]{index=3}
        y_pred_prob = model.predict_proba(df)[0]  # wrong classes shape → ValueError :contentReference[oaicite:4]{index=4}
    except Exception as e:
        logger.error("Model threw during predict: %s", e, exc_info=True)
        raise InferenceError(str(e)) from e

    try:
        coef_sum = sum(
            est.base_estimator.coef_ for est in model.calibrated_classifiers_
        ) / len(model.calibrated_classifiers_)
        logistic_reg_coeff = coef_sum[0]
        abs_coeff = np.abs(logistic_reg_coeff)
        feature_importance = abs_coeff / abs_coeff.sum()  # divide-by-zero guard – np handles but logs a warning :contentReference[oaicite:5]{index=5}
        fi_sorted = sorted(
            zip(feature_names, feature_importance.tolist()),
            key=lambda x: x[1], reverse=True
        )[:5]
        top_features = [{k: v} for k, v in fi_sorted]
    except Exception as e:
        logger.warning("Feature-importance calc failed: %s", e, exc_info=True)
        top_features = []  # keep going – not fatal

    fico_res = map_prediction_to_fico_score_micro(
        [(int(y_pred[0]), float(max(y_pred_prob)))]
    )

    return {
        "class_label": int(y_pred[0]),
        "predict_prob": y_pred_prob.tolist(),
        "feature_importance": top_features,
        "credit_score": str(int(round(float(fico_res["fico_score"]), 1))),
    }
