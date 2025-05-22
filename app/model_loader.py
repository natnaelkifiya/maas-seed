from __future__ import annotations

import importlib
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import boto3
import joblib

from .config import ml_model_config as cfg
from .utils import compute_top_features      # ← helper made earlier

log = logging.getLogger(__name__)
s3  = boto3.client("s3")

# ────────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────────
def _latest_s3_model(bucket: str, root_prefix: str, artifact_subpath: str) -> str:
    """
    Return the newest S3 key (by LastModified) that ends with `artifact_subpath`
    under `bucket/root_prefix/**`.
    """
    paginator = s3.get_paginator("list_objects_v2")
    newest_key, newest_time = None, datetime(1970, 1, 1, tzinfo=timezone.utc)

    for page in paginator.paginate(Bucket=bucket, Prefix=root_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(artifact_subpath) and obj["LastModified"] > newest_time:
                newest_key, newest_time = key, obj["LastModified"]

    if newest_key is None:
        raise FileNotFoundError("No matching model found in S3")
    return newest_key

# ────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────
def load_latest_model() -> Dict[str, Any]:
    """
    1. Download newest artefact from S3 into a temp-file
    2. Monkey-patch globals so joblib can unpickle custom funcs
    3. Load the model
    4. Attach `top_features` (cached list) so inference is cheap
    """
    fb  = cfg.s3_fallback
    key = _latest_s3_model(fb.bucket, fb.root_prefix, fb.artifact_subpath)
    log.info("Downloading model artefact: s3://%s/%s", fb.bucket, key)

    with tempfile.NamedTemporaryFile("wb+", delete=False) as tmp:
        s3.download_fileobj(fb.bucket, key, tmp)
        tmp.flush(); tmp.seek(0)
        model = joblib.load(tmp)

    # -----------------------------------------------------------------
    # Pre-compute and stash top-k feature importance (k=5)
    # -----------------------------------------------------------------
    try:
        model["top_features"] = compute_top_features(
            estimator=model["model"],
            train_columns=model["train_columns"],
            k=5,
        )
        log.info("Cached top-5 feature importance inside model bundle.")
    except Exception as e:
        log.warning("Failed to cache feature importance: %s", e, exc_info=True)

    return model
