import os, tempfile, boto3, joblib, logging
from datetime import datetime, timezone
from .config import ml_model_config as cfg
import sys, importlib


log = logging.getLogger(__name__)
s3 = boto3.client("s3")


def _patch_pickle_globals():
    """
    Ensure functions referenced as '__main__.map_*' exist
    when we unpickle the model.
    """
    main_mod = sys.modules.get("__main__")
    if not main_mod:
        return

    # Import the real module that holds the functions
    helpers = importlib.import_module("app.mapping_asset_demo")

    # Copy every map_* symbol into __main__
    for name in dir(helpers):
        if name.startswith("map_"):
            setattr(main_mod, name, getattr(helpers, name))

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
    log.info("Downloading model artefact: s3://%s/%s", fb.bucket, key)
    
    _patch_pickle_globals() 

    # stream to a temp-file
    with tempfile.NamedTemporaryFile("wb+", delete=False) as tmp:
        s3.download_fileobj(fb.bucket, key, tmp)

        tmp.flush(); tmp.seek(0)
        model = joblib.load(tmp)

    return model