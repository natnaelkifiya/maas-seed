# app/inference.py  – run_inference only
from __future__ import annotations

import logging, os
from typing import Dict

import pandas as pd
import requests
from requests.exceptions import HTTPError, RequestException, Timeout
from scipy.special import expit                     # sigmoid

logger = logging.getLogger(__name__)

class FeatureFetchError(RuntimeError): ...
class InferenceError(RuntimeError):    ...

# KEEP FEATURE_ORDER IN SYNC WITH TRAINING
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
def run_inference(model: object, payload_from_client: Dict) -> Dict:
    """Fetch features from Feast, score with `model`, return 300-850 credit score."""
    FEAST_BASE_URL = os.getenv("FEAST_BASE_URL", "http://localhost:6567")

    # 1) Pull online features
    feast_req = {
        "features": [f"poultry_fv:{f}" for f in FEATURE_ORDER],
        "entities": {"customerId": [payload_from_client["customerId"]]},
    }
    try:
        resp = requests.post(f"{FEAST_BASE_URL}/get-online-features",
                             json=feast_req, timeout=3)
        resp.raise_for_status()
        meta, results = resp.json()["metadata"], resp.json()["results"]
    except (HTTPError, Timeout, RequestException) as e:
        logger.error("Feast request failed: %s", e, exc_info=True)
        raise FeatureFetchError(str(e)) from e
    except (KeyError, ValueError) as e:
        logger.error("Unexpected Feast payload: %s", e, exc_info=True)
        raise FeatureFetchError("Invalid payload from Feast") from e

    # 2) DataFrame (training order) + type fixes
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

    if "Price per item (ETB)" in df.columns:
        df["Price per item (ETB)"] = df["Price per item (ETB)"].astype(str)
    if "Proximity to Market" in df.columns:
        df["Proximity to Market"] = df["Proximity to Market"].astype(int)

    # 3) Predict
    try:
        raw_pred = model.predict(df)          # shape (1,) or (1,1)
        prob_pos = float(expit(raw_pred)[0])  # sigmoid -> [0–1]
        credit_score = int(round(300 + prob_pos * 550))
    except Exception as e:
        logger.error("Model prediction failed: %s", e, exc_info=True)
        raise InferenceError("Model prediction failed") from e

    # 4) Response (no feature importance)
    return {
        "credit_score": credit_score,
        "probability_positive": prob_pos,
    }
