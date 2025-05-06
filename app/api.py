from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from .redis_cache import get_cached_model
from .inference import run_inference
import asyncio
import logging

router = APIRouter()

class APIRequestData(BaseModel):
    source_bank: str
    customerId: str
    loan_type: str
    loan_subtype: str

RETRY_LIMIT = 3

@router.post("/transaction")
async def predict(data: APIRequestData, background_tasks: BackgroundTasks):
    model = get_cached_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    for attempt in range(RETRY_LIMIT):
        try:
            result = await asyncio.to_thread(run_inference, model, data.dict())
            return {"prediction": result[0]}
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(0.5 * (attempt + 1))  # exponential backoff

    raise HTTPException(status_code=500, detail="Inference failed after retries")
