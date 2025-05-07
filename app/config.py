import os
from dataclasses import dataclass
from typing import Tuple

# ----------  Redis & misc constants  ----------
MODEL_DIR = os.getenv("MODEL_DIR", "./models")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
MODEL_CACHE_KEY = "latest_model"

# ----------  S3 fallback structure  ----------
@dataclass
class S3Fallback:
    bucket: str
    root_prefix: str          # common prefix before run‑IDs
    artifact_subpath: str     # ends with artifacts/lpo_model_named/model.pkl

# ----------  Model configuration  ----------
@dataclass
class ModelConfig:
    experiment_name: str
    artifact_path: str        # MLflow artifact path
    metric: str
    higher_is_better: bool
    s3_fallback: S3Fallback

ml_model_config = ModelConfig(
    experiment_name="ml_test_v5",
    artifact_path="txn_model_named",  # For logging in MLflow
    metric="f1_score",
    higher_is_better=True,
    s3_fallback=S3Fallback(
        bucket="cs-infernece-model-testdata",
        root_prefix="models/models/1/",  # <- this is the prefix S3 will scan
        artifact_subpath="Logistic_Regression_calibrated_model.joblib",  # <- match the filename exactly
    ),
)

