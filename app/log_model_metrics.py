"""
log_model_metrics.py
————————————
1. Download latest model artefact from S3
2. Log model + metrics to MLflow
3. Persist the metrics JSON back to S3

All three steps raise explicit domain-errors so callers can react.
"""

from __future__ import annotations

import json, logging, os, tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import boto3, botocore
import joblib, mlflow


# ────────────────────────────────────────────────────────────────
# 0)  Logging setup
# ────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ────────────────────────────────────────────────────────────────
# 1)  Custom domain errors
# ────────────────────────────────────────────────────────────────
class ConfigError(RuntimeError): ...
class S3DownloadError(RuntimeError): ...
class MLflowLogError(RuntimeError): ...
class S3UploadError(RuntimeError): ...


# ────────────────────────────────────────────────────────────────
# 2)  Configuration (env-vars take precedence over hard-codes)
# ────────────────────────────────────────────────────────────────
S3_URI: str = os.getenv(
    "MODEL_S3_URI",
    "s3://cs-infernece-model-testdata/models/models/1/Logistic_Regression_calibrated_model.joblib",
)
MLFLOW_EXPERIMENT: str = os.getenv("MLFLOW_EXPERIMENT", "ml_test_v6")
METRICS: Dict[str, float] = {
    "f1_score": float(os.getenv("F1_SCORE", 0.84)),
    "accuracy": float(os.getenv("ACCURACY", 0.89)),
}
METRICS_S3_KEY: str = os.getenv(
    "METRICS_S3_KEY", "models/models/1/model_metrics.json"
)
MLFLOW_TRACKING_URI: str | None = os.getenv("MLFLOW_TRACKING_URI")


# ────────────────────────────────────────────────────────────────
# 3)  Helper functions
# ────────────────────────────────────────────────────────────────
def _env_assert(var: str | None, name: str) -> str:
    if not var:
        raise ConfigError(f"Missing required env var: {name}")
    return var


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ConfigError("MODEL_S3_URI must start with s3://")
    bucket, key = uri[5:].split("/", 1)
    return bucket, key


def download_s3_file(s3, bucket: str, key: str) -> Path:
    dst = Path(tempfile.gettempdir(), Path(key).name)
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(dst))
        logger.info("Downloaded model from s3://%s/%s → %s", bucket, key, dst)
    except botocore.exceptions.BotoCoreError as e:
        logger.error("S3 download failed: %s", e, exc_info=True)
        raise S3DownloadError(str(e)) from e
    return dst


def log_to_mlflow(local_model: Path, metrics: Dict[str, float]) -> None:
    try:
        mlflow.set_tracking_uri(_env_assert(MLFLOW_TRACKING_URI, "MLFLOW_TRACKING_URI"))
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run():
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(local_model), artifact_path="txn_model_named")
        logger.info("Logged artefact + metrics to MLflow experiment '%s'", MLFLOW_EXPERIMENT)
    except Exception as e:  # mlflow throws a smorgasbord of exceptions
        logger.error("MLflow logging failed: %s", e, exc_info=True)
        raise MLflowLogError(str(e)) from e


def upload_metrics_json(s3, metrics: Dict[str, float], bucket: str, dst_key: str) -> None:
    tmp = Path(tempfile.gettempdir(), "model_metrics.json")
    tmp.write_text(json.dumps(metrics))
    try:
        s3.upload_file(str(tmp), bucket, dst_key)
        logger.info("Uploaded metrics to s3://%s/%s", bucket, dst_key)
    except botocore.exceptions.BotoCoreError as e:
        logger.error("S3 upload failed: %s", e, exc_info=True)
        raise S3UploadError(str(e)) from e


# ────────────────────────────────────────────────────────────────
# 4)  Orchestrator
# ────────────────────────────────────────────────────────────────
def main() -> None:
    bucket, key = parse_s3_uri(S3_URI)
    s3 = boto3.client("s3")

    # a) Download
    model_path = download_s3_file(s3, bucket, key)

    # b) Log to MLflow
    log_to_mlflow(model_path, METRICS)

    # c) Re-upload metrics JSON
    upload_metrics_json(s3, METRICS, bucket, METRICS_S3_KEY)


if __name__ == "__main__":  # pragma: no cover
    try:
        main()
    except (ConfigError, S3DownloadError, MLflowLogError, S3UploadError) as e:
        logger.critical("Job aborted: %s", e)
        raise
