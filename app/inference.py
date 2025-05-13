from typing import Dict, List, Tuple
import requests, pandas as pd, numpy as np
import os

FEATURE_ORDER = [
    "total_credit", "total_debit", "total_net_amt", "credit_mean", "debit_mean",
    "total_rate", "sd_credit", "sd_debit", "total_txn_count", "zero_txn",
    "volatility", "log_balance", "unique_source_inflow", "unique_source_outflow",
]

def calculate_fico_score_micro(
    target_class: int,
    proba: float,
    min_fico: int,
    fico_max_class_0: int,
    fico_max_class_1: int,
    high_risky_max_bound: int,
    low_risky_lower_bound: int,
) -> int:
    """
    Very simple linear mapping:
      • if predicted class == 0  (good)  → score slides between low_risky_lower_bound‑850
      • if predicted class == 1  (bad)   → score slides between 300‑high_risky_max_bound
    """
    if target_class == 0:
        # higher prob of GOOD → higher score
        return int(low_risky_lower_bound + proba * (fico_max_class_0 - low_risky_lower_bound))
    else:
        # higher prob of BAD → lower score
        return int(min_fico + (1 - proba) * (high_risky_max_bound - min_fico))
    

def map_prediction_to_fico_score_micro(predict_prob: List[Tuple[int, float]]):
    """Map prediction probabilities to a micro‑FICO score."""
    min_fico, max_fico = 300, 850
    fico_max_class_0, fico_max_class_1 = 400, 150
    high_risky_max_bound = 450   # ← comma removed
    low_risky_lower_bound  = 450

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


def run_inference(model, payload_from_client: Dict):
    FEAST_BASE_URL = os.getenv("FEAST_BASE_URL", "http://localhost:6567")

    
    if payload_from_client.get("source_bank", "").lower() == "coop":
        payload_from_client["loan_type"] = "msme" 
        
    # 1) pull features from Feast
    feast_request = {
        "features": [f"txn_fv_new2:{f}" for f in FEATURE_ORDER],
        "entities": {"customerId": [payload_from_client["customerId"]]},
    }
    resp = requests.post(f"{FEAST_BASE_URL}/get-online-features", json=feast_request)
    resp.raise_for_status()                     # raises if not 2xx

    meta, results = resp.json()["metadata"], resp.json()["results"]

    # 2) flatten values and build df
    values = [
        (v[0] if isinstance(v, list) else v)
        for r in results for v in r["values"]
    ]
    feature_names = [c for c in meta["feature_names"] if c != "customerId"]
    df = pd.DataFrame([values], columns=meta["feature_names"]).drop(columns=["customerId"])
    df = df[FEATURE_ORDER]      
    predictor_result = {}
    
    y_pred = model.predict(df)

    y_pred_prob = model.predict_proba(df)[0]

    coef_avg = 0

    for i in model.calibrated_classifiers_:
        coef_avg = coef_avg + i.base_estimator.coef_

    coef_avg = coef_avg/len(model.calibrated_classifiers_)

    coefficients_array = np.array(coef_avg)
    logistic_reg_coeff = coefficients_array[0]
    idx = np.argsort(np.abs(logistic_reg_coeff))[::-1]

    # Calculate the absolute values of the coefficients.
    abs_coefficients = np.abs(logistic_reg_coeff)
    # Calculate the sum of the absolute values of the coefficients.
    sum_abs_coefficients = np.sum(abs_coefficients)
    # Calculate the feature importance values.
    feature_importance = abs_coefficients / sum_abs_coefficients
    # Create feature-importaance pairs for feature importance
    feature_importance_list = list(
        zip(feature_names, feature_importance.tolist()))
    # dictionary comprehension to convert list of list into dictionary
    feature_importance_dict = {item[0]: item[1]
                                for item in feature_importance_list}
    # Sort the dictionary items based on values in descending order
    sorted_items = sorted(feature_importance_dict.items(),
                            key=lambda x: x[1], reverse=True)

    # Create a new dictionary with the top 5 features
    top_5_features = dict(sorted_items[:5])
    top_5_features = [{key: value}
                        for key, value in top_5_features.items()]

    predictor_result["class_label"] = y_pred[0]
    predictor_result["predict_prob"] = y_pred_prob.tolist()
    predictor_result["feature_importance"] = top_5_features
    
    max_f1_score = map_prediction_to_fico_score_micro(
        [
            (
                predictor_result["class_label"],
                max(predictor_result["predict_prob"]),
            )
        ]
        )


    predictor_result["credit_score"] = str(int(round(float(max_f1_score["fico_score"]), 1)))
    
    return predictor_result


