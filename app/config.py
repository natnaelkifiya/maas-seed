import os

MODEL_DIR = os.getenv("MODEL_DIR", "./models")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
MODEL_CACHE_KEY = "latest_model"
