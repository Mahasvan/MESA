from fastapi.routing import APIRouter

from api.service.connectors.cnn_model import CNNModel
from api.service.response import BetterJSONResponse

router = APIRouter(tags=["Tensorflow Models"])
prefix = "/cnn"

cnn_model = CNNModel()

@router.get("/predict")
async def predict(message: str):
    res = cnn_model.predict(message)
    return BetterJSONResponse(content=int(res))


def setup(app):
    app.include_router(router, prefix=prefix)
