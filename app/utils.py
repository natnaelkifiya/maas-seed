# app/utils.py  (new file – tiny helper)
from __future__ import annotations
import numpy as np
from typing import List, Dict, Any

def compute_top_features(
    estimator, train_columns: List[str], k: int = 5
) -> List[Dict[str, float]]:
    """
    Return a [{feature: weight}, …] list.
    Works for linear models (coef_) and RF-like models (feature_importances_).
    """
    if hasattr(estimator, "coef_"):
        weights = estimator.coef_.flatten()
    elif hasattr(estimator, "feature_importances_"):
        weights = estimator.feature_importances_
    else:                                   # not supported
        return []

    fi = {feat: w for feat, w in zip(train_columns, weights)}
    return [
        {f: float(w)}
        for f, w in sorted(fi.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]
    ]
