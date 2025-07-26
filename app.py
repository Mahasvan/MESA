import importlib.util
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.service import shell

if not os.environ.get("IS_CONTAINER", False):
    shell.print_yellow_message("Running on bare metal. Loading environment variables from .env file...")
    # docker-compose is configured to set IS_CONTAINER environment variable.
    # if this variable is not set, the app is running on bare metal.
    # so, we will load the environment variables from the `.env ` file.
    from dotenv import load_dotenv
    load_dotenv()
    shell.print_cyan_message("Environment variables loaded successfully.")

with open("version.txt", "r") as f:
    VERSION = f.read().strip()

app = FastAPI(
    title="Message Sentiment Analyzer",
    description="Made for STIRS2023_MESA",
    swagger_ui_parameters={"syntaxHighlight": {"theme": "obsidian"}},
    version=VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"ping": "I am alive!"}


# load API Routers
routes = [x.rstrip(".py") for x in os.listdir("api/route") if x.endswith(".py") and not x.startswith("_")]

print(shell.format_bold("Loading API routes..."))
for route in routes:
    shell.print_cyan_message(f"Loading {route}...")
    try:
        importlib.util.spec_from_file_location(route, f"api/route/{route}.py")
        module = importlib.import_module(f"api.route.{route}")
        module.setup(app)
        shell.print_green_message("Success!")
    except Exception as e:
        shell.print_red_message(f"Failed:")
        print(e)

"""
Environment Variables:
OLLAMA_MODEL = gemma3:1b (or whatever is in .env)
OLLAMA_BASE_URL = http://vader.mahasvan.int:11434
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
