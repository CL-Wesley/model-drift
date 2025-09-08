"""
Unified FastAPI application for both Data Drift and Model Drift
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers
from .shared.upload import router as upload_router
from .data_drift.routes.upload import router as data_drift_upload_router
from .data_drift.routes.dashboard_new import router as data_drift_dashboard_router
from .data_drift.routes.class_imbalance import router as data_drift_class_imbalance_router
from .data_drift.routes.statistical import router as data_drift_statistical_router
from .data_drift.routes.feature_analysis import router as data_drift_feature_analysis_router
from .model_drift.routes.upload import router as model_drift_upload_router

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
"""
Mount upload, data-drift, and model-drift routers (each router defines its own path prefix)
"""
app.include_router(upload_router)
app.include_router(data_drift_upload_router)
app.include_router(data_drift_dashboard_router)
app.include_router(data_drift_class_imbalance_router)
app.include_router(data_drift_statistical_router)
app.include_router(data_drift_feature_analysis_router)
app.include_router(model_drift_upload_router)
