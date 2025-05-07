# import numpy as np
# from typing import Dict

# # The feature order expected by the model
# FEATURE_ORDER = [
#     "total_credit", "total_debit", "total_net_amt", "credit_mean", "debit_mean",
#     "total_rate", "sd_credit", "sd_debit", "total_txn_count", "zero_txn",
#     "volatility", "log_balance", "unique_source_inflow", "unique_source_outflow"
# ]

# def run_inference(model, data: Dict):
#     customer_id = data["customerId"]
#     loan_type = data["loan_type"]

#     # 🔧 Dummy feature values for now (simulate Feast output)
#     dummy_feature_values = {
#         "total_credit": 10000,
#         "total_debit": 9000,
#         "total_net_amt": 1000,
#         "credit_mean": 500,
#         "debit_mean": 450,
#         "total_rate": 0.65,
#         "sd_credit": 50.1,
#         "sd_debit": 47.8,
#         "total_txn_count": 42,
#         "zero_txn": 0,
#         "volatility": 0.15,
#         "log_balance": 7.0,
#         "unique_source_inflow": 4,
#         "unique_source_outflow": 3so upda
#     }

#     input_vector = [dummy_feature_values[feature] for feature in FEATURE_ORDER]
#     X = np.array([input_vector])
#     return model.predict(X)
# inference.py  (no Docker, no env vars)

# import logging
# import requests
# from requests.exceptions import RequestException
# import pandas as pd
# import numpy as np
# from typing import Dict


# def run_inference(model, data):
    
#     url = "http://localhost:6567/get-online-features"

#     data = {
#         "features": [
#                 "txn_fv_new2:total_credit",
#                 "txn_fv_new2:total_debit",
#                 "txn_fv_new2:total_net_amt",
#                 "txn_fv_new2:credit_mean",
#                 "txn_fv_new2:debit_mean",
#                 "txn_fv_new2:total_rate",
#                 "txn_fv_new2:sd_credit",
#                 "txn_fv_new2:sd_debit",
#                 "txn_fv_new2:total_txn_count",
#                 "txn_fv_new2:zero_txn",
#                 "txn_fv_new2:volatility",
#                 "txn_fv_new2:log_balance",
#                 "txn_fv_new2:unique_source_inflow",
#                 "txn_fv_new2:unique_source_outflow",
                
#         ],
#         "entities": {
#             "customerId": ['cust_test']
#         }
#     }

#     response = requests.post(url, json=data)
#     print(response.text)

#     if response.status_code == 200:

#         print("Request was successful. Response:")
#         response_json = response.json()

#         metadata = response_json['metadata']
#         feature_names = metadata['feature_names']
#         results = response_json['results']
#         values = [result['values'] for result in results]
#         df = pd.DataFrame([values], columns=feature_names).applymap(lambda x: x[0] if isinstance(x, list) else x)            

#         print(df.info())
#         predictor_result = {}
#         try:
#             del df['customerId']
#             del df['loan_type']
#             del df['loan_subtype']
#             del df['source_bank']
#             y_pred = model.predict(df)

#             y_pred_prob = model.predict_proba(df)[0]

#             coef_avg = 0

#             for i in model.calibrated_classifiers_:
#                 coef_avg = coef_avg + i.base_estimator.coef_

#             coef_avg = coef_avg/len(model.calibrated_classifiers_)

#             coefficients_array = np.array(coef_avg)
#             logistic_reg_coeff = coefficients_array[0]
#             idx = np.argsort(np.abs(logistic_reg_coeff))[::-1]

#             # Calculate the absolute values of the coefficients.
#             abs_coefficients = np.abs(logistic_reg_coeff)
#             # Calculate the sum of the absolute values of the coefficients.
#             sum_abs_coefficients = np.sum(abs_coefficients)
#             # Calculate the feature importance values.
#             feature_importance = abs_coefficients / sum_abs_coefficients
#             # Create feature-importaance pairs for feature importance
#             feature_importance_list = list(
#                 zip(feature_names, feature_importance.tolist()))
#             # dictionary comprehension to convert list of list into dictionary
#             feature_importance_dict = {item[0]: item[1]
#                                     for item in feature_importance_list}
#             # Sort the dictionary items based on values in descending order
#             sorted_items = sorted(feature_importance_dict.items(),
#                                 key=lambda x: x[1], reverse=True)

#             # Create a new dictionary with the top 5 features
#             top_5_features = dict(sorted_items[:5])
#             top_5_features = [{key: value}
#                             for key, value in top_5_features.items()]

#             predictor_result["class_label"] = y_pred[0]
#             predictor_result["predict_prob"] = y_pred_prob.tolist()
#             predictor_result["feature_importance"] = top_5_features

#             return predictor_result

#         except requests.exceptions.RequestException as e:
#             return {"error": "Failed to perform prediction on inference"}

from typing import Dict
import requests, pandas as pd, numpy as np

FEATURE_ORDER = [
    "total_credit", "total_debit", "total_net_amt", "credit_mean", "debit_mean",
    "total_rate", "sd_credit", "sd_debit", "total_txn_count", "zero_txn",
    "volatility", "log_balance", "unique_source_inflow", "unique_source_outflow",
]

def run_inference(model, payload_from_client: Dict):
    
    if payload_from_client.get("source_bank", "").lower() == "coop":
        payload_from_client["loan_type"] = "msme" 
        
    # 1) pull features from Feast
    feast_request = {
        "features": [f"txn_fv_new2:{f}" for f in FEATURE_ORDER],
        "entities": {"customerId": [payload_from_client["customerId"]]},
    }
    resp = requests.post("http://localhost:6567/get-online-features", json=feast_request)
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
    
    print(predictor_result)

    return predictor_result