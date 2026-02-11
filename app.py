# Top-level runner for the FastAPI app

import uvicorn
from app.app import app as fastapi_app

if __name__ == "__main__":
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, reload=True)
