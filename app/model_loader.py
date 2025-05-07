import os, tempfile, boto3, joblib
from datetime import datetime, timezone
from .config import ml_model_config as cfg

s3 = boto3.client("s3")

def _latest_s3_model(bucket: str, root_prefix: str, artifact_subpath: str):
    paginator = s3.get_paginator("list_objects_v2")
    newest_key, newest_time = None, datetime(1970,1,1,tzinfo=timezone.utc)

    for page in paginator.paginate(Bucket=bucket, Prefix=root_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(artifact_subpath):
                if obj["LastModified"] > newest_time:
                    newest_key = key
                    newest_time = obj["LastModified"]

    if newest_key is None:
        raise FileNotFoundError("No matching model found in S3")
    return newest_key

def load_latest_model():
    fb = cfg.s3_fallback
    key = _latest_s3_model(fb.bucket, fb.root_prefix, fb.artifact_subpath)
    print(f"Loading latest model from S3: s3://{fb.bucket}/{key}")
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        s3.download_fileobj(fb.bucket, key, tmp)
        model = joblib.load(tmp.name)
    return model