import os

from fastapi.routing import APIRouter
from api.service.connectors.ollama import Ollama
from api.service.response import BetterJSONResponse

router = APIRouter(tags=["Chat"])
prefix = "/chat"

ollama_base_url = os.environ.get("OLLAMA_BASE_URL")
if not ollama_base_url:
    raise Exception("OLLAMA_BASE_URL environment variable not set.")
model = os.environ.get("OLLAMA_MODEL")
if not model:
    raise Exception("OLLAMA_MODEL environment variable not set.")

chatter = Ollama(base_url=ollama_base_url, model=model)

@router.get("/predict")
async def predict(message: str):
    res = chatter.get_response(message)

    return BetterJSONResponse(content=res.dict())


def setup(app):
    app.include_router(router, prefix=prefix)