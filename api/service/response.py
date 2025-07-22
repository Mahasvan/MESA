from fastapi.responses import JSONResponse

class BetterJSONResponse(JSONResponse):
    def __init__(self, content=None, status_code=200, headers=None, error=None):

        if content is not None:
            content = {
                "response": content
            }
        else:
            content = {}

        if error:
            content["error"] = str(error)

        super().__init__(content, status_code, headers)
