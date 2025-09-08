from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import pandas as pd
from ..services.data_drift_analysis import DataDriftAnalysis
from .upload import session_storage

router = APIRouter(prefix="/data-drift", tags=["data-drift"])

def get_session_data(session_id: str) -> Dict[str, pd.DataFrame]:
    """Get session data from storage"""
    if session_id not in session_storage:
        raise HTTPException(status_code=404, detail="Session not found. Please upload data first.")
    return session_storage[session_id]

@router.get("/feature-analysis/{session_id}")
async def get_feature_analysis(session_id: str):
    """
    Get detailed feature analysis for uploaded datasets
    
    Args:
        session_id: Session identifier for uploaded data
        
    Returns:
        Feature analysis results with deep dive into individual features
    """
    try:
        # Get session data
        data = get_session_data(session_id)
        reference_df = data.get("reference")
        current_df = data.get("current")
        
        if reference_df is None or current_df is None:
            raise HTTPException(status_code=400, detail="Both reference and current datasets are required")
        
        # Initialize analysis service
        analysis_service = DataDriftAnalysis()
        
        # Perform feature deep dive analysis
        result = await analysis_service.analyze_feature_deep_dive(reference_df, current_df)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature analysis failed: {str(e)}")
