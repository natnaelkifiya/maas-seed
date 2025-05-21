# app/inference.py
from __future__ import annotations

import logging
import os
from typing import Dict, List, Tuple, Callable, Any

import numpy as np
import pandas as pd
import requests
from requests.exceptions import RequestException, HTTPError, Timeout

# ────────────────────────────────────────────────────────────────
# Domain‑specific errors so api.py can map them to HTTP
# ────────────────────────────────────────────────────────────────
class FeatureFetchError(RuntimeError):
    """Raised when online‑feature retrieval from Feast fails."""
    pass


class InferenceError(RuntimeError):
    """Raised when the model cannot produce a prediction."""
    pass


logger = logging.getLogger(__name__)

# KEEP FEATURE_ORDER IN SYNC WITH TRAINING TIME
FEATURE_ORDER =  ['age',
                'gender',
                'marital_status',
                'education_level',
                'household_size',
                'household_composition',
                'monthly_household_income',
                'dependents_education',
                'region',
                'migration_status',
                'employment_status',
                'employment_type',
                'digital_literacy',
                'access_to_extension',
                'cooperative_membership',
                'proximity_to_markets']


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
    """Map class + probability → FICO‑like 300‑850 score"""
    if target_class == 0:
        return int(low_risky_lower_bound + proba * (fico_max_class_0 - low_risky_lower_bound))
    return int(min_fico + (1 - proba) * (high_risky_max_bound - min_fico))


def map_prediction_to_fico_score_micro(predict_prob: List[Tuple[int, float]]):
    min_fico, max_fico = 300, 850  # noqa: F841  # reserved for future use
    fico_max_class_0, fico_max_class_1 = 400, 150
    high_risky_max_bound = 450
    low_risky_lower_bound = 450

    best_score, best_class, max_probability = min_fico, 0, -1.0
    for label, proba in predict_prob:
        label = 1 if label is True else 0 if label is False else int(label)
        score = calculate_fico_score_micro(
            label,
            proba,
            min_fico,
            fico_max_class_0,
            fico_max_class_1,
            high_risky_max_bound,
            low_risky_lower_bound,
        )
        if proba > max_probability:
            best_score, best_class, max_probability = score, label, proba
    return {"fico_score": best_score, "class": best_class}


# ────────────────────────────────────────────────────────────────
# Helper for ordinal encoding
# ────────────────────────────────────────────────────────────────

def _apply_ordinal_mapping(series: pd.Series, mapping: Any, default: float = np.nan) -> pd.Series:  # noqa: ANN001
    """Apply an ordinal mapping that can be a dict or a callable."""
    if callable(mapping):
        return series.apply(lambda x: mapping(x) if pd.notnull(x) else default)
    if isinstance(mapping, dict):
        return series.apply(lambda x: mapping.get(str(x).lower(), default) if pd.notnull(x) else default)
    raise TypeError("Mapping must be either a dict or a callable.")


# ────────────────────────────────────────────────────────────────
# The star of the show
# ────────────────────────────────────────────────────────────────

def run_inference(model: Dict[str, Any], payload_from_client: Dict) -> Dict:
    """Fetch online features, run the sklearn model, and post‑process output."""
    FEAST_BASE_URL = os.getenv("FEAST_BASE_URL", "http://localhost:6567")

    # ------------------------------------------------------
    # 1) Pull feature vector from Feast
    # ------------------------------------------------------
    feast_request = {
        "features": [f"demographic_fv:{f}" for f in FEATURE_ORDER],
        "entities": {"customerId": [payload_from_client["customerId"]]},
    }

    try:
        r = requests.post(f"{FEAST_BASE_URL}/get-online-features", json=feast_request, timeout=3)
        r.raise_for_status()  # HTTP 4xx/5xx → HTTPError
        meta, results = r.json()["metadata"], r.json()["results"]
    except (HTTPError, Timeout, RequestException) as e:
        logger.error("Feast request failed: %s", e, exc_info=True)
        raise FeatureFetchError(str(e)) from e
    except (KeyError, ValueError) as e:
        logger.error("Unexpected Feast payload: %s", e, exc_info=True)
        raise FeatureFetchError("Invalid payload from Feast") from e

    # ------------------------------------------------------
    # 2) Turn Feast response into a DataFrame ordered as at train time
    # ------------------------------------------------------
    try:
        values = [
            (v[0] if isinstance(v, list) else v)
            for res in results
            for v in res["values"]
        ]
        df = (
            pd.DataFrame([values], columns=meta["feature_names"])  # may raise ValueError if shape mismatches
            .drop(columns=["customerId"])
        )
        # Ensure deterministic column order to match training
        df = df.loc[:, FEATURE_ORDER]
    except (ValueError, KeyError) as e:
        logger.error("Feature frame construction failed: %s", e, exc_info=True)
        raise FeatureFetchError("Feature frame construction failed") from e

    # ------------------------------------------------------
    # 3) Unpack model bundle
    # ------------------------------------------------------
    try:
        linreg = model["model"]
        ordinal_mappings: Dict[str, Callable[[Any], Any] | Dict[str, Any]] = model["ordinal_mappings"]
        train_columns: List[str] = model["train_columns"]
    except Exception as e:
        logger.error("Model bundle missing expected keys: %s", e, exc_info=True)
        raise InferenceError("Corrupted model bundle") from e

    # ------------------------------------------------------
    # 4) Ordinal‑encode + align feature set exactly to train_columns
    # ------------------------------------------------------
    try:
        df_encoded = pd.DataFrame(index=df.index)

        for col in train_columns:
            if col in ordinal_mappings:
                mapping = ordinal_mappings[col]
                df_encoded[col] = _apply_ordinal_mapping(df[col], mapping, default=0)
            else:
                # Either numeric or unseen categorical – copy as‑is
                df_encoded[col] = df[col]

        # Any columns present at train time but missing here → fill with 0
        df_encoded = df_encoded.reindex(columns=train_columns, fill_value=0)
    except Exception as e:
        logger.error("Encoding failed: %s", e, exc_info=True)
        raise InferenceError("Feature encoding failed") from e

    # ------------------------------------------------------
    # 5) Predict
    # ------------------------------------------------------
    try:
        if hasattr(linreg, "predict_proba"):
            # Classification model
            y_pred_prob: np.ndarray = linreg.predict_proba(df_encoded)
            y_pred: np.ndarray = np.argmax(y_pred_prob, axis=1)
        else:
            # Regression model that outputs a risk score [0,1]
            preds_normalized: np.ndarray = linreg.predict(df_encoded)
            preds_normalized = np.clip(preds_normalized, 0, 1)
            y_pred_prob = np.column_stack([1 - preds_normalized, preds_normalized])
            y_pred = (preds_normalized >= 0.5).astype(int)
    except Exception as e:
        logger.error("Model threw during predict: %s", e, exc_info=True)
        raise InferenceError("Model prediction failed") from e

    # ------------------------------------------------------
    # 6) Feature Importance (top‑5) – best effort
    # ------------------------------------------------------
    try:
        coefs = getattr(linreg, "coef_", None)
        if coefs is not None:
            if isinstance(coefs, np.ndarray) and coefs.ndim > 1:
                coefs = coefs[0]  # binary classification shape (1, n_features)
            abs_coefs = np.abs(coefs)
            denom = abs_coefs.sum() or 1e-9
            importances = abs_coefs / denom
            fi_pairs = sorted(
                zip(train_columns, importances.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )[:5]
            top_features = [{k: float(v)} for k, v in fi_pairs]
        else:
            top_features = []
    except Exception as e:
        logger.warning("Feature‑importance calc failed: %s", e, exc_info=True)
        top_features = []

    # ------------------------------------------------------
    # 7) Map prediction → FICO‑style score
    # ------------------------------------------------------
    pred_prob_value = float(y_pred_prob[0][y_pred[0]])
    fico_res = map_prediction_to_fico_score_micro([(int(y_pred[0]), pred_prob_value)])

    # ------------------------------------------------------
    # 8) Build final response
    # ------------------------------------------------------
    return {
        "class_label": int(y_pred[0]),
        "predict_prob": y_pred_prob[0].tolist(),
        "feature_importance": top_features,
        "credit_score": str(int(round(float(fico_res["fico_score"]), 0))),
    }
