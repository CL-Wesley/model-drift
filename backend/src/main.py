"""
Unified FastAPI application for both Data Drift and Model Drift
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import routers with error handling
try:
    from .shared.upload import router as upload_router
    logger.info("✅ Imported shared upload router")
except ImportError as e:
    logger.error(f"❌ Failed to import shared upload router: {e}")
    upload_router = None

try:
    from .shared.s3_endpoints import router as s3_router
    logger.info("✅ Imported S3 endpoints router")
    # Log the routes in s3_router for debugging
    if hasattr(s3_router, 'routes'):
        logger.info(f"S3 Router has {len(s3_router.routes)} routes:")
        for route in s3_router.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                logger.info(f"  - {list(route.methods)} {route.path}")
except ImportError as e:
    logger.error(f"❌ Failed to import S3 router: {e}")
    s3_router = None

try:
    from .data_drift.routes.upload import router as data_drift_upload_router
    logger.info("✅ Imported data drift upload router")
except ImportError as e:
    logger.error(f"❌ Failed to import data drift upload router: {e}")
    data_drift_upload_router = None

try:
    from .data_drift.routes.dashboard_new import router as data_drift_dashboard_router
    logger.info("✅ Imported data drift dashboard router")
except ImportError as e:
    logger.error(f"❌ Failed to import data drift dashboard router: {e}")
    data_drift_dashboard_router = None

try:
    from .data_drift.routes.class_imbalance import router as data_drift_class_imbalance_router
    logger.info("✅ Imported data drift class imbalance router")
except ImportError as e:
    logger.error(f"❌ Failed to import data drift class imbalance router: {e}")
    data_drift_class_imbalance_router = None

try:
    from .data_drift.routes.statistical import router as data_drift_statistical_router
    logger.info("✅ Imported data drift statistical router")
except ImportError as e:
    logger.error(f"❌ Failed to import data drift statistical router: {e}")
    data_drift_statistical_router = None

try:
    from .data_drift.routes.feature_analysis import router as data_drift_feature_analysis_router
    logger.info("✅ Imported data drift feature analysis router")
except ImportError as e:
    logger.error(f"❌ Failed to import data drift feature analysis router: {e}")
    data_drift_feature_analysis_router = None

try:
    from .model_drift.routes.upload import router as model_drift_upload_router
    logger.info("✅ Imported model drift upload router")
except ImportError as e:
    logger.error(f"❌ Failed to import model drift upload router: {e}")
    model_drift_upload_router = None

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

# Mount routers with better organization and error handling
logger.info("🚀 Mounting routers...")

# Shared services (no prefix - they define their own)
if upload_router:
    app.include_router(upload_router)
    logger.info("✅ Mounted shared upload router")

# S3 services with specific prefix to avoid conflicts
if s3_router:
    app.include_router(s3_router, prefix="/api/v1/s3", tags=["S3 Services"])
    logger.info("✅ Mounted S3 router at /api/v1/s3")

# Data Drift services
if data_drift_upload_router:
    app.include_router(data_drift_upload_router)
    logger.info("✅ Mounted data drift upload router")

if data_drift_dashboard_router:
    app.include_router(data_drift_dashboard_router)
    logger.info("✅ Mounted data drift dashboard router")

if data_drift_class_imbalance_router:
    app.include_router(data_drift_class_imbalance_router)
    logger.info("✅ Mounted data drift class imbalance router")

if data_drift_statistical_router:
    app.include_router(data_drift_statistical_router)
    logger.info("✅ Mounted data drift statistical router")

if data_drift_feature_analysis_router:
    app.include_router(data_drift_feature_analysis_router)
    logger.info("✅ Mounted data drift feature analysis router")

# Model Drift services
if model_drift_upload_router:
    app.include_router(model_drift_upload_router)
    logger.info("✅ Mounted model drift upload router")

logger.info("🎉 All routers mounted successfully!")

# Debug endpoint to show all registered routes
@app.get("/api/v1/debug/routes")
async def debug_routes():
    """Debug endpoint to show all registered routes"""
    routes = []
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            routes.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": getattr(route, 'name', 'Unknown')
            })
    return {
        "total_routes": len(routes),
        "routes": sorted(routes, key=lambda x: x['path'])
    }

# Startup event to log all routes
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Application startup complete!")
    logger.info("📋 Registered routes:")
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            logger.info(f"  {list(route.methods)} {route.path}")