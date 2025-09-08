from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional, Dict, Any
import io
import pandas as pd
import json

# Import the service functions
from ..data_drift.services.drift_service import run_data_drift
from ..model_drift.services.enhanced_model_service import enhanced_model_service
from ..model_drift.models.analysis_config import AnalysisConfiguration, ModelType, DriftThresholds
from pydantic import ValidationError

router = APIRouter(prefix="/api/v1", tags=["upload"])

@router.post("/upload")
async def unified_upload(
    reference_data: UploadFile = File(..., description="Reference dataset CSV file"),
    current_data: UploadFile = File(..., description="Current dataset CSV file"),
    model_file: Optional[UploadFile] = File(None, description="Optional: Trained model pickle file"),
    low_threshold: float = Form(0.05, description="Low drift threshold"),
    medium_threshold: float = Form(0.15, description="Medium drift threshold"),
    high_threshold: float = Form(0.25, description="High drift threshold")
):
    """
    Legacy unified upload endpoint (backward compatibility)
    
    - **reference_data**: CSV file containing the reference/baseline dataset
    - **current_data**: CSV file containing the current dataset to compare against reference
    - **model_file**: Optional pickle or joblib file containing trained model (triggers model drift analysis)
    - **low_threshold**: Threshold for low drift detection (default: 0.05)
    - **medium_threshold**: Threshold for medium drift detection (default: 0.15)
    - **high_threshold**: Threshold for high drift detection (default: 0.25)
    
    Returns combined analysis results from both data drift and model drift (if model provided)
    """
    from model_drift.services.model_service import run_model_drift
    
    try:
        # Validate file types
        if not reference_data.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Reference data must be a CSV file")
        if not current_data.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Current data must be a CSV file")
        if model_file and not (model_file.filename.lower().endswith(('.pkl', '.pickle', '.joblib', '.pkl.joblib'))):
            raise HTTPException(status_code=400, detail="Model file must be a pickle (.pkl, .pickle) or joblib (.joblib) file")

        # Validate thresholds
        if not (0 < low_threshold < medium_threshold < high_threshold < 1):
            raise HTTPException(status_code=400, detail="Thresholds must be: 0 < low < medium < high < 1")

        # Always run data drift analysis
        print("Running data drift analysis...")
        data_drift_result = await run_data_drift(
            reference_data, 
            current_data, 
            low_threshold, 
            medium_threshold, 
            high_threshold
        )

        # Run model drift analysis if model file is provided
        model_drift_result = None
        if model_file:
            print("Running model drift analysis...")
            model_drift_result = await run_model_drift(reference_data, current_data, model_file)

        # Create comprehensive response
        response = {
            "upload_info": {
                "reference_file": reference_data.filename,
                "current_file": current_data.filename,
                "model_file": model_file.filename if model_file else None,
                "timestamp": pd.Timestamp.now().isoformat()
            },
            "analysis_results": {
                "data_drift": data_drift_result,
                "model_drift": model_drift_result
            },
            "summary": {
                "data_drift_detected": get_drift_status(data_drift_result),
                "model_drift_detected": get_drift_status(model_drift_result) if model_drift_result else False,
                "analysis_type": "combined" if model_file else "data_only"
            }
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/analyze")
async def unified_analyze(
    reference_data: UploadFile = File(..., description="Reference dataset CSV file"),
    current_data: UploadFile = File(..., description="Current dataset CSV file"),
    model_file: Optional[UploadFile] = File(None, description="Optional: Trained model file"),
    config: Optional[str] = Form(None, description="JSON configuration for model drift analysis")
):
    """
    Enhanced unified analysis endpoint with JSON configuration
    
    - **reference_data**: CSV file containing the reference/baseline dataset
    - **current_data**: CSV file containing the current dataset to compare
    - **model_file**: Optional trained model file (enables model drift analysis)
    - **config**: JSON configuration string for model drift analysis
    
    Config JSON format (required if model_file provided):
    {
        "analysis_name": "My Analysis",
        "description": "Optional description",
        "model_type": "classification",
        "selected_metrics": ["accuracy", "precision", "recall"],
        "statistical_test": "mcnemar",
        "low_threshold": 0.05,
        "medium_threshold": 0.15,
        "high_threshold": 0.25
    }
    """
    from model_drift.services.model_service import run_model_drift
    
    try:
        # Default values
        low_threshold = 0.05
        medium_threshold = 0.15 
        high_threshold = 0.25
        analysis_name = "Drift Analysis"
        description = ""
        
        # Parse config if provided
        parsed_config = None
        if config:
            try:
                config_dict = json.loads(config)
                # Use values from config
                analysis_name = config_dict.get("analysis_name", analysis_name)
                description = config_dict.get("description", description)
                low_threshold = config_dict.get("low_threshold", low_threshold)
                medium_threshold = config_dict.get("medium_threshold", medium_threshold)
                high_threshold = config_dict.get("high_threshold", high_threshold)
                
                # Store parsed config for model drift
                parsed_config = config_dict
            except (json.JSONDecodeError, ValueError) as e:
                raise HTTPException(status_code=400, detail=f"Invalid config JSON: {str(e)}")
        
        # Validate file types
        if not reference_data.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Reference data must be a CSV file")
        if not current_data.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Current data must be a CSV file")
        if model_file and not (model_file.filename.lower().endswith(('.pkl', '.pickle', '.joblib', '.pkl.joblib'))):
            raise HTTPException(status_code=400, detail="Model file must be a pickle (.pkl, .pickle) or joblib (.joblib) file")

        # Validate thresholds
        if not (0 < low_threshold < medium_threshold < high_threshold < 1):
            raise HTTPException(status_code=400, detail="Thresholds must be: 0 < low < medium < high < 1")

        # Read files into memory to avoid file stream exhaustion
        print("Reading files into memory...")
        reference_bytes = await reference_data.read()
        current_bytes = await current_data.read()
        model_bytes = None
        if model_file:
            model_bytes = await model_file.read()
        
        # Create temporary UploadFile-like class for reuse
        class TempUploadFile:
            def __init__(self, file_obj, filename):
                self.file = file_obj
                self.filename = filename
                self.content_type = "text/csv" if filename.endswith('.csv') else "application/octet-stream"
            
            async def read(self):
                self.file.seek(0)
                return self.file.read()
            
            def __getattr__(self, name):
                return getattr(self.file, name)
        
        # Create file-like objects for data drift analysis
        reference_file_for_drift = io.BytesIO(reference_bytes)
        reference_file_for_drift.name = reference_data.filename
        
        current_file_for_drift = io.BytesIO(current_bytes)
        current_file_for_drift.name = current_data.filename
        
        temp_ref_file = TempUploadFile(reference_file_for_drift, reference_data.filename)
        temp_curr_file = TempUploadFile(current_file_for_drift, current_data.filename)

        # Always run data drift analysis
        print("Running data drift analysis...")
        data_drift_result = await run_data_drift(
            temp_ref_file, 
            temp_curr_file, 
            low_threshold, 
            medium_threshold, 
            high_threshold
        )

        # Run model drift analysis if model file is provided
        model_drift_result = None
        if model_file and parsed_config:
            print("Running enhanced model drift analysis with configuration...")
            
            # Validate required config fields for model drift
            required_fields = ["model_type", "selected_metrics", "statistical_test"]
            missing_fields = [field for field in required_fields if field not in parsed_config]
            if missing_fields:
                raise HTTPException(status_code=400, detail=f"Missing required config fields for model drift: {missing_fields}")
            
            # Create configuration object for model drift
            try:
                model_type_enum = ModelType(parsed_config["model_type"].lower())
                
                full_config = AnalysisConfiguration(
                    analysis_name=analysis_name,
                    description=description,
                    model_type=model_type_enum,
                    selected_metrics=parsed_config["selected_metrics"],
                    statistical_test=parsed_config["statistical_test"],
                    drift_thresholds=DriftThresholds(
                        low_threshold=low_threshold,
                        medium_threshold=medium_threshold,
                        high_threshold=high_threshold
                    )
                )
                
                # Run enhanced model drift analysis
                # Create fresh file objects for model drift from bytes
                reference_file_for_model = io.BytesIO(reference_bytes)
                current_file_for_model = io.BytesIO(current_bytes)
                model_file_for_model = io.BytesIO(model_bytes)
                
                # Set filenames for the file objects
                reference_file_for_model.name = reference_data.filename
                current_file_for_model.name = current_data.filename
                model_file_for_model.name = model_file.filename
                
                # Create temporary UploadFile objects for model drift
                temp_ref_file_model = TempUploadFile(reference_file_for_model, reference_data.filename)
                temp_curr_file_model = TempUploadFile(current_file_for_model, current_data.filename)
                temp_model_file = TempUploadFile(model_file_for_model, model_file.filename)
                
                model_drift_result = await enhanced_model_service.run_configured_analysis(
                    temp_ref_file_model, temp_curr_file_model, temp_model_file, full_config
                )
                
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid model_type: {parsed_config['model_type']}. Must be 'classification' or 'regression'")
            except ValidationError as e:
                raise HTTPException(status_code=400, detail=f"Configuration validation error: {str(e)}")
                
        elif model_file and not parsed_config:
            print("Running basic model drift analysis (no configuration provided)...")
            # Create fresh file objects for basic model drift from bytes
            reference_file_for_model = io.BytesIO(reference_bytes)
            current_file_for_model = io.BytesIO(current_bytes)
            model_file_for_model = io.BytesIO(model_bytes)
            
            # Set filenames for the file objects
            reference_file_for_model.name = reference_data.filename
            current_file_for_model.name = current_data.filename
            model_file_for_model.name = model_file.filename
            
            # Create temporary UploadFile objects for basic model drift
            temp_ref_file_model = TempUploadFile(reference_file_for_model, reference_data.filename)
            temp_curr_file_model = TempUploadFile(current_file_for_model, current_data.filename)
            temp_model_file = TempUploadFile(model_file_for_model, model_file.filename)
            
            # Fallback to basic model drift analysis
            model_drift_result = await run_model_drift(temp_ref_file_model, temp_curr_file_model, temp_model_file)

        # Create comprehensive response
        response = {
            "upload_info": {
                "reference_file": reference_data.filename,
                "current_file": current_data.filename,
                "model_file": model_file.filename if model_file else None,
                "timestamp": pd.Timestamp.now().isoformat(),
                "analysis_name": analysis_name,
                "description": description
            },
            "analysis_results": {
                "data_drift": data_drift_result,
                "model_drift": model_drift_result.dict() if hasattr(model_drift_result, 'dict') else model_drift_result
            },
            "summary": {
                "data_drift_detected": get_drift_status(data_drift_result),
                "model_drift_detected": get_enhanced_drift_status(model_drift_result) if model_drift_result else False,
                "analysis_type": "enhanced_combined" if (model_file and parsed_config) else ("combined" if model_file else "data_only"),
                "configuration_used": bool(model_file and parsed_config)
            }
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def get_drift_status(result: Optional[Dict[str, Any]]) -> bool:
    """Helper function to extract drift detection status from analysis results"""
    if not result:
        return False
    
    # Check if there's an error
    if "error" in result:
        return False
    
    # Check for explicit drift_detected field
    if "drift_detected" in result:
        return result["drift_detected"]
    
    # Check drift severity for data drift results
    if "drift_metrics" in result and "drift_severity" in result["drift_metrics"]:
        severity = result["drift_metrics"]["drift_severity"]
        return severity in ["Medium", "High"]
    
    return False

def get_enhanced_drift_status(result) -> bool:
    """Helper function to extract drift detection status from enhanced analysis results"""
    if not result:
        return False
        
    # Handle enhanced analysis results
    if hasattr(result, 'success'):
        if not result.success:
            return False
        
        if hasattr(result, 'analysis_results') and result.analysis_results:
            if hasattr(result.analysis_results, 'drift_metrics') and result.analysis_results.drift_metrics:
                return result.analysis_results.drift_metrics.drift_detected
    
    # Handle dictionary results
    if isinstance(result, dict):
        return get_drift_status(result)
        
    return False

@router.get("/health")
async def health_check():
    """Health check endpoint for the unified upload service"""
    return {
        "status": "healthy",
        "service": "unified-upload",
        "version": "2.0.0",
        "endpoints": {
            "upload": "/api/v1/upload",
            "analyze": "/api/v1/analyze",
            "data_drift": "/api/v1/data-drift/upload", 
            "model_drift": "/api/v1/model-drift/upload"
        }
    }
