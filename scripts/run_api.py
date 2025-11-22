"""Run the FastAPI game controller locally.

Usage:
    python scripts/run_api.py

Then open http://127.0.0.1:8000/docs for the interactive API docs.
"""

import uvicorn


if __name__ == "__main__":
    uvicorn.run("services.api:app", host="127.0.0.1", port=8000, log_level="info")
