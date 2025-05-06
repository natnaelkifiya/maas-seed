import os
import pickle
from datetime import datetime
from .config import MODEL_DIR

def get_latest_model_path():
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]
    latest = max(models, key=lambda f: os.path.getmtime(os.path.join(MODEL_DIR, f)))
    return os.path.join(MODEL_DIR, latest)

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)
