import redis
import pickle
from .config import REDIS_URL, MODEL_CACHE_KEY

redis_client = redis.Redis.from_url(REDIS_URL)

def cache_model(model):
    redis_client.set(MODEL_CACHE_KEY, pickle.dumps(model))

def get_cached_model():
    data = redis_client.get(MODEL_CACHE_KEY)
    if data:
        return pickle.loads(data)
    return None
