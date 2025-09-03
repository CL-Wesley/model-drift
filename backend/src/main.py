"""
Unified FastAPI application for both Data Drift and Model Drift
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers
from shared.upload import router as upload_router
from data_drift.routes.upload import router as data_drift_upload_router
from model_drift.routes.upload import router as model_drift_upload_router

app = FastAPI(
    title="Unified Drift Detection API",
    version="1.0.0",
    description="Single backend for Data Drift and Model Drift services",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root health check
@app.get("/")
async def root():
    return {
        "message": "Unified Drift Detection API", 
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "ok", 
        "message": "Server is running.",
        "services": ["data-drift", "model-drift", "unified-upload"]
    }

# Mount routers
app.include_router(upload_router)           # /api/v1/upload
app.include_router(data_drift_upload_router)  # /api/v1/data-drift  
app.include_router(model_drift_upload_router) # /api/v1/model-drift
