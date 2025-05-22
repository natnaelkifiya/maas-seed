from __future__ import annotations

import logging, os
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
from requests.exceptions import HTTPError, RequestException, Timeout
from scipy.special import expit                         # sigmoid

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────
# Domain-specific errors
# ────────────────────────────────────────────────────────────────
class FeatureFetchError(RuntimeError): ...
class InferenceError(RuntimeError):    ...

logger = logging.getLogger(__name__)

# KEEP FEATURE_ORDER IN SYNC WITH TRAINING TIME
FEATURE_ORDER = [
    "Poultry Training", "Poultry Sector", "Poultry Housing System",
    "Chicken Breed Size", "Comb Type", "Skin Color", "Place of Origin",
    "Health Management Plan", "Professionals", "Temperature (°C)",
    "Humidity (°F)", "Type of Feed", "Nutrient Balance", "Feed Source",
    "Waste Management", "Chicken Source", "Years of Experience",
    "Mortality Rate (%)", "Number of Chickens", "Number of Poultry Houses",
    "Proximity to Market", "Market Channels", "Production Period (weeks)",
    "Price per item (ETB)",
]

# ────────────────────────────────────────────────────────────────
# Main inference routine
# ────────────────────────────────────────────────────────────────
def _underlying_estimator(est) -> object:
    """Return the final estimator even if wrapped in a Pipeline."""
    if isinstance(est, Pipeline):
        return est.named_steps.get("model", est.steps[-1][1])
    return est


def run_inference(model: object, payload_from_client: Dict) -> Dict:
    """
    Fetch feature vector from Feast, run prediction with `model`,
    scale the score to 300-850, extract top-5 feature importances.
    """
    FEAST_BASE_URL = os.getenv("FEAST_BASE_URL", "http://localhost:6567")

    # ----------------------------------------------------------------
    # 1) Retrieve online features from Feast
    # ----------------------------------------------------------------
    feast_request = {
        "features": [f"poultry_fv:{f}" for f in FEATURE_ORDER],
        "entities": {"customerId": [payload_from_client["customerId"]]},
    }

    try:
        resp = requests.post(f"{FEAST_BASE_URL}/get-online-features",
                             json=feast_request, timeout=3)
        resp.raise_for_status()
        meta, results = resp.json()["metadata"], resp.json()["results"]
    except (HTTPError, Timeout, RequestException) as e:
        logger.error("Feast request failed: %s", e, exc_info=True)
        raise FeatureFetchError(str(e)) from e
    except (KeyError, ValueError) as e:
        logger.error("Unexpected Feast payload: %s", e, exc_info=True)
        raise FeatureFetchError("Invalid payload from Feast") from e

    # ----------------------------------------------------------------
    # 2) Build DataFrame in training order + type fixes
    # ----------------------------------------------------------------
    try:
        values = [
            (v[0] if isinstance(v, list) else v)
            for res in results for v in res["values"]
        ]
        df = (
            pd.DataFrame([values], columns=meta["feature_names"])
            .drop(columns=["customerId"])
            .loc[:, FEATURE_ORDER]
        )
    except (ValueError, KeyError) as e:
        logger.error("Feature frame construction failed: %s", e, exc_info=True)
        raise FeatureFetchError("Feature frame construction failed") from e

    # Cast columns exactly like the original helper ------------------
    if "Price per item (ETB)" in df.columns:
        df["Price per item (ETB)"] = df["Price per item (ETB)"].astype(str)

    if "Proximity to Market" in df.columns:
        df["Proximity to Market"] = df["Proximity to Market"].astype(int)

    # ----------------------------------------------------------------
    # 3) Prediction
    # ----------------------------------------------------------------
    try:
        raw_pred = model.predict(df)                # shape (1,) or (1,1)
        prob = float(expit(raw_pred)[0])            # sigmoid → [0,1]
        credit_score = int(round(300 + prob * 550)) # 300-850 scaling
    except Exception as e:
        logger.error("Model prediction failed: %s", e, exc_info=True)
        raise InferenceError("Model prediction failed") from e

    # ----------------------------------------------------------------
    # 4) Feature importance (top-5)
    # ----------------------------------------------------------------
    top_5: List[Dict[str, float]] = []
    try:
        est = _underlying_estimator(model)

        if isinstance(est, LinearRegression) and hasattr(est, "coef_"):
            weights = est.coef_.flatten()
            fi = dict(zip(df.columns, weights))
        elif isinstance(est, RandomForestRegressor) and hasattr(est, "feature_importances_"):
            weights = est.feature_importances_
            fi = dict(zip(df.columns, weights))
        else:
            fi = {}

        if fi:
            top_5 = [
                {feat: float(w)}
                for feat, w in sorted(fi.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
            ]
    except Exception as e:
        logger.warning("Feature-importance calc failed: %s", e, exc_info=True)

    # ----------------------------------------------------------------
    # 5) Build & return response
    # ----------------------------------------------------------------
    return {
        "credit_score": credit_score,
        "probability_positive": prob,
        "feature_importance": top_5,
    }