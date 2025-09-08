from fastapi import APIRouter, UploadFile, File, Form
from ..services.drift_service import run_data_drift

router = APIRouter(prefix="/api/v1/data-drift", tags=["data-drift"])

@router.post("/upload")
async def data_drift_upload(
    reference_data: UploadFile = File(...),
    current_data: UploadFile = File(...),
    low_threshold: float = Form(0.05),
    medium_threshold: float = Form(0.15),
    high_threshold: float = Form(0.25)
):
    """
    Data drift specific upload endpoint
    """
    result = await run_data_drift(reference_data, current_data, low_threshold, medium_threshold, high_threshold)
    return result

@router.get("/health")
async def data_drift_health():
    """Health check for data drift service"""
    return {"status": "ok", "service": "data-drift"}
