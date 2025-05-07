from fastapi import FastAPI
from . import api, model_loader, redis_cache

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    print("Loading model...")
    model = model_loader.load_latest_model()
    redis_cache.cache_model(model)
    print("Model loaded and cached successfully.")

app.include_router(api.router)
