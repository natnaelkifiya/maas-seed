"""
log_model_metrics.py
————————————
Utilities for
1. Downloading the latest model artefact from S3
2. Logging model + metrics to MLflow
3. Writing the metrics JSON back to S3
"""

from __future__ import annotations

import os, json, tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict

import boto3, mlflow, joblib


# ────────────────────────────────────────────────────────────────
# 1) Configuration
# ────────────────────────────────────────────────────────────────
S3_URI = "s3://cs-infernece-model-testdata/models/models/1/Logistic_Regression_calibrated_model.joblib"
MLFLOW_EXPERIMENT = "ml_test_v6"
METRICS: Dict[str, float] = {"f1_score": 0.84, "accuracy": 0.89}

# ────────────────────────────────────────────────────────────────
# 2) Helpers
# ────────────────────────────────────────────────────────────────
def parse_s3_uri(uri: str) -> tuple[str, str]:
    """
    Split s3://bucket/key into (bucket, key).
    """
    if not uri.startswith("s3://"):
        raise ValueError("URI must start with s3://")
    bucket, key = uri.replace("s3://", "", 1).split("/", 1)
    return bucket, key


def download_s3_file(s3_client, bucket: str, key: str, dst: Path) -> Path:
    """
    Download an S3 object to the given local path.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    s3_client.download_file(bucket, key, str(dst))
    return dst


def log_to_mlflow(local_model: Path, metrics: Dict[str, float]) -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run():
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(local_model), artifact_path="txn_model_named")
        print("Logged model and metrics to MLflow")


def upload_metrics_json(
    s3_client,
    metrics: Dict[str, float],
    bucket: str,
    dst_key: str = "models/models/1/model_metrics.json",
) -> None:
    """
    Write metrics dict to /tmp and upload to S3.
    """
    tmp_json = Path("/tmp/model_metrics.json")
    with tmp_json.open("w") as fp:
        json.dump(metrics, fp)
    s3_client.upload_file(str(tmp_json), bucket, dst_key)
    print(f"Saved metrics to S3: s3://{bucket}/{dst_key}")


# ────────────────────────────────────────────────────────────────
# 3) Orchestrator
# ────────────────────────────────────────────────────────────────
def main() -> None:
    s3 = boto3.client("s3")
    bucket, key = parse_s3_uri(S3_URI)

    # a) download model
    local_model_path = Path(tempfile.gettempdir()) / Path(key).name
    download_s3_file(s3, bucket, key, local_model_path)

    # b) log artefact + metrics to MLflow
    log_to_mlflow(local_model_path, METRICS)

    # c) re‑upload the metrics JSON
    upload_metrics_json(s3, METRICS, bucket)


if __name__ == "__main__":
    main()
