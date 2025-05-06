from fastapi import FastAPI
from . import api, model_loader, redis_cache

app = FastAPI()

@app.on_event("startup")
async def load_and_cache_model():
    path = model_loader.get_latest_model_path()
    model = model_loader.load_model(path)
    redis_cache.cache_model(model)

app.include_router(api.router)
