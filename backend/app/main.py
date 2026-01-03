from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.api import endpoints
from app.core.stream_processor import stream_processor
import os

app = FastAPI(title="ChainWarner API", description="Supply Chain Risk Monitoring Platform")

# Configure CORS
origins = ["*"] # Allow all for simplicity in demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static and Templates
if os.path.isdir("app/static"):
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Include API Router
app.include_router(endpoints.router, prefix="/api/v1")

@app.get("/")
async def root(request: Request):
    """Serve the frontend dashboard"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.on_event("shutdown")
async def _shutdown():
    await stream_processor.aclose()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
