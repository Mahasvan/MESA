import os

from fastapi import APIRouter

from api.service.response import BetterJSONResponse

router = APIRouter(tags=["Meta"])
prefix = "/meta"

def _count_lines_helper(folder, exts=None):
    files = os.listdir(folder)
    lines = 0
    for file in files:
        file_path = os.path.join(folder, file)
        if os.path.isdir(file_path) and not file.startswith("."):
            lines += _count_lines_helper(os.path.join(folder, file), exts)
        else:
            if exts and not file.endswith(tuple(exts)):
                continue
            with open(file_path, "r") as f:
                lines += len(f.readlines())
    return lines

@router.get("/lines-of-code")
async def countlines():
    # count lines of code in the current working directory
    cwd = os.getcwd()
    lines = _count_lines_helper(cwd, exts=["py"])
    return BetterJSONResponse(content={"lines": lines})

def setup(app):
    app.include_router(router, prefix=prefix)