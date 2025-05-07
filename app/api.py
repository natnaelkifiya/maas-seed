# api.py
from enum import Enum
import asyncio, logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from .redis_cache import get_cached_model
from .inference import run_inference

router = APIRouter()

# ────────────────────────────────────────────────────────────────
# 1) Enumerations of allowed categorical values
# ────────────────────────────────────────────────────────────────
class LoanType(str, Enum):
    msme = "msme"
    sme = "safee"
    retail = "ifb"

class SourceBank(str, Enum):
    coop = "coop"
    abyssinia = "zamzam"

# ────────────────────────────────────────────────────────────────
# 2) Request body schema with validation
# ────────────────────────────────────────────────────────────────
class APIRequestData(BaseModel):
    customerId: str = Field(..., min_length=1)
    loan_type: LoanType
    loan_subtype: str          # ⇠ keep free‑form or add another Enum later
    source_bank: SourceBank

# ────────────────────────────────────────────────────────────────
RETRY_LIMIT = 3

@router.post("/transaction")
async def predict(data: APIRequestData):
    model = get_cached_model()
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    loop = asyncio.get_running_loop()           # 3.7+

    for attempt in range(RETRY_LIMIT):
        try:
            result = await loop.run_in_executor(
                None,                           # default ThreadPoolExecutor
                run_inference, model, data.dict()
            )
            return {
                "credit_score": result["credit_score"],
                "feature_importance": result["feature_importance"],
                "customerId": data.customerId,
            }
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(0.5 * (attempt + 1))

    raise HTTPException(status_code=500, detail="Inference failed")
