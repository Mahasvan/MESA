from fastapi.routing import APIRouter

from api.service.connectors.bert_model import BertModel
from api.service.response import BetterJSONResponse

router = APIRouter(tags=["Tensorflow Models"])
prefix = "/bert"

bert_model = BertModel()


@router.get("/predict")
async def predict(message: str):
    res = bert_model.predict(message)
    print("Returning", int(res))
    return BetterJSONResponse(content=int(res))


def setup(app):
    app.include_router(router, prefix=prefix)
