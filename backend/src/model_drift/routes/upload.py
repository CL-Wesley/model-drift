from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Body
from fastapi.encoders import jsonable_encoder
import numpy as np
import math
from typing import Optional, List
import json
import io
import pickle
import joblib
from datetime import datetime
from ..services.model_service import run_model_drift
from ..services.enhanced_model_service import enhanced_model_service
from ..services.analysis.performance_comparison_service import performance_comparison_service
from ..services.analysis.degradation_metrics_service import degradation_metrics_service
from ..services.analysis.statistical_significance_service import statistical_significance_service
from ..models.analysis_config import AnalysisConfiguration, ModelType, DriftThresholds
from pydantic import ValidationError, BaseModel
from ...shared.session_manager import get_session_manager
from ...shared.ai_explanation_service import ai_explanation_service

# Session-to-UploadFile Adapter Classes
class SessionUploadFile:
    """Adapter class to convert session data back to UploadFile-like object"""
    def __init__(self, content: bytes, filename: str, content_type: str = "text/csv"):
        self.content = content
        self.filename = filename
        self.content_type = content_type
        self._bytes_io = io.BytesIO(content)
    
    async def read(self) -> bytes:
        self._bytes_io.seek(0)
        return self._bytes_io.read()
    
    def seek(self, offset: int, whence: int = 0):
        return self._bytes_io.seek(offset, whence)
    
    def tell(self) -> int:
        return self._bytes_io.tell()

def create_upload_files_from_session(session_id: str):
    """Convert session data back to UploadFile-like objects for existing services"""
    session_manager = get_session_manager()
    if not session_manager.session_exists(session_id):
        return None, None, None
    
    session_data = session_manager.get_model_drift_format(session_id)
    if not session_data:
        return None, None, None
    
    # Convert DataFrames back to CSV bytes
    reference_csv = session_data["reference_df"].to_csv(index=False).encode('utf-8')
    current_csv = session_data["current_df"].to_csv(index=False).encode('utf-8')
    
    # Create UploadFile-like objects
    reference_file = SessionUploadFile(
        content=reference_csv, 
        filename=session_data["reference_filename"],
        content_type="text/csv"
    )
    
    current_file = SessionUploadFile(
        content=current_csv,
        filename=session_data["current_filename"], 
        content_type="text/csv"
    )
    
    model_file = None
    if session_data["model_file_content"]:
        model_file = SessionUploadFile(
            content=session_data["model_file_content"],
            filename=session_data["model_filename"],
            content_type="application/octet-stream"
        )
    
    return reference_file, current_file, model_file, session_data.get("config")

def clean_float_values(obj):
    """Recursively clean non-finite float values from nested data structures"""
    if isinstance(obj, dict):
        return {k: clean_float_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_float_values(item) for item in obj]
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            return None  # or 0, or "N/A" depending on your preference
        return float(obj)
    elif isinstance(obj, np.generic):
        # Handle other numpy types
        item = obj.item()
        return clean_float_values(item)
    else:
        return obj

def serialize_response(result):
    """Serialize response with proper handling of numpy types and non-finite values"""
    cleaned_result = clean_float_values(result)
    return jsonable_encoder(
        cleaned_result,
        custom_encoder={np.generic: lambda x: x.item()}
    )

class AnalysisConfig(BaseModel):
    analysis_name: str
    description: str = ""
    model_type: str
    selected_metrics: List[str]
    statistical_test: str
    low_threshold: float = 0.05
    medium_threshold: float = 0.15
    high_threshold: float = 0.25

router = APIRouter(prefix="/api/v1/model-drift", tags=["model-drift"])

@router.post("/upload")
async def model_drift_upload(
    reference_data: UploadFile = File(...),
    current_data: UploadFile = File(...),
    model_file: UploadFile = File(...)
):
    """
    Legacy model drift upload endpoint (backward compatibility)
    Requires both datasets and a model file
    """
    result = await run_model_drift(reference_data, current_data, model_file)
    return result

@router.post("/analyze")
async def model_drift_analyze(
    reference_data: UploadFile = File(..., description="Reference dataset CSV file"),
    current_data: UploadFile = File(..., description="Current dataset CSV file"), 
    model_file: UploadFile = File(..., description="Trained model file (.pkl, .pickle, .joblib)"),
    config: str = Form(..., description="JSON configuration string")
):
    """
    Enhanced model drift analysis endpoint with JSON configuration
    
    For the config parameter, send a JSON string like:
    {"analysis_name": "My Analysis", "model_type": "classification", "selected_metrics": ["accuracy", "precision"], "statistical_test": "mcnemar"}
    """
    try:
        # Parse config from JSON string
        try:
            config_dict = json.loads(config)
            parsed_config = AnalysisConfig(**config_dict)
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid config JSON: {str(e)}")
        
        # Validate file types
        if not reference_data.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Reference data must be a CSV file")
        if not current_data.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Current data must be a CSV file")
        if not (model_file.filename.lower().endswith(('.pkl', '.pickle', '.joblib', '.pkl.joblib'))):
            raise HTTPException(status_code=400, detail="Model file must be a pickle (.pkl, .pickle) or joblib (.joblib) file")

        # Validate model type
        try:
            model_type_enum = ModelType(parsed_config.model_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid model_type: {parsed_config.model_type}. Must be 'classification' or 'regression'")

        # Create full configuration object
        try:
            full_config = AnalysisConfiguration(
                analysis_name=parsed_config.analysis_name,
                description=parsed_config.description,
                model_type=model_type_enum,
                selected_metrics=parsed_config.selected_metrics,
                statistical_test=parsed_config.statistical_test,
                drift_thresholds=DriftThresholds(
                    low_threshold=parsed_config.low_threshold,
                    medium_threshold=parsed_config.medium_threshold,
                    high_threshold=parsed_config.high_threshold
                )
            )
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=f"Configuration validation error: {str(e)}")

        # Run enhanced analysis
        result = await enhanced_model_service.run_configured_analysis(
            reference_data, current_data, model_file, full_config
        )
        
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/metrics")
async def get_available_metrics():
    """Get available performance metrics for each model type"""
    return {
        "classification": [
            {"id": "accuracy", "label": "Accuracy", "description": "Overall prediction accuracy"},
            {"id": "precision", "label": "Precision", "description": "True positives / (True positives + False positives)"},
            {"id": "recall", "label": "Recall (Sensitivity)", "description": "True positives / (True positives + False negatives)"},
            {"id": "f1_score", "label": "F1-Score", "description": "Harmonic mean of precision and recall"},
            {"id": "specificity", "label": "Specificity", "description": "True negatives / (True negatives + False positives)"},
            {"id": "roc_auc", "label": "ROC AUC", "description": "Area under ROC curve"},
            {"id": "pr_auc", "label": "PR AUC", "description": "Area under precision-recall curve"},
            {"id": "cohen_kappa", "label": "Cohen's Kappa", "description": "Inter-rater reliability metric"},
            {"id": "mcc", "label": "Matthews Correlation Coefficient", "description": "Correlation between predictions and actual"}
        ],
        "regression": [
            {"id": "mse", "label": "Mean Squared Error (MSE)", "description": "Average squared prediction errors"},
            {"id": "rmse", "label": "Root Mean Squared Error (RMSE)", "description": "Square root of MSE"},
            {"id": "mae", "label": "Mean Absolute Error (MAE)", "description": "Average absolute prediction errors"},
            {"id": "r2", "label": "R-squared (R²)", "description": "Coefficient of determination"},
            {"id": "adjusted_r2", "label": "Adjusted R-squared", "description": "R² adjusted for number of predictors"},
            {"id": "mape", "label": "Mean Absolute Percentage Error (MAPE)", "description": "Average absolute percentage errors"},
            {"id": "explained_variance", "label": "Explained Variance Score", "description": "Proportion of variance explained"},
            {"id": "max_error", "label": "Max Error", "description": "Maximum residual error"}
        ]
    }

@router.get("/tests")
async def get_available_tests():
    """Get available statistical tests for each model type"""
    return {
        "classification": [
            {
                "id": "mcnemar",
                "label": "McNemar's Test",
                "description": "Compares paired categorical data for classification models",
                "complexity": "Simple",
                "category": "Non-parametric"
            },
            {
                "id": "delong",
                "label": "DeLong Test", 
                "description": "Compares ROC curves for statistical significance",
                "complexity": "Moderate",
                "category": "ROC-based"
            },
            {
                "id": "five_two_cv",
                "label": "5×2 Cross-Validation F-Test",
                "description": "Robust cross-validation based comparison",
                "complexity": "Complex",
                "category": "Cross-validation"
            },
            {
                "id": "bootstrap_confidence",
                "label": "Bootstrap Confidence Intervals",
                "description": "Non-parametric confidence interval estimation",
                "complexity": "Moderate",
                "category": "Resampling"
            },
            {
                "id": "paired_ttest",
                "label": "Paired t-Test",
                "description": "Classical statistical test for paired samples",
                "complexity": "Simple",
                "category": "Parametric"
            }
        ],
        "regression": [
            {
                "id": "five_two_cv",
                "label": "5×2 Cross-Validation F-Test", 
                "description": "Robust cross-validation based comparison",
                "complexity": "Complex",
                "category": "Cross-validation"
            },
            {
                "id": "bootstrap_confidence",
                "label": "Bootstrap Confidence Intervals",
                "description": "Non-parametric confidence interval estimation", 
                "complexity": "Moderate",
                "category": "Resampling"
            },
            {
                "id": "diebold_mariano",
                "label": "Diebold-Mariano Test",
                "description": "Compares predictive accuracy of forecasting models",
                "complexity": "Moderate", 
                "category": "Time series"
            },
            {
                "id": "paired_ttest",
                "label": "Paired t-Test",
                "description": "Classical statistical test for paired samples",
                "complexity": "Simple",
                "category": "Parametric"
            }
        ]
    }

@router.post("/performance-comparison")
async def performance_comparison_analysis(
    reference_data: UploadFile = File(..., description="Reference dataset CSV file"),
    current_data: UploadFile = File(..., description="Current dataset CSV file"), 
    model_file: UploadFile = File(..., description="Trained model file (.pkl, .pickle, .joblib)"),
    config: str = Form(..., description="JSON configuration string")
):
    """
    Performance Comparison Analysis (Tab 1)
    
    Returns comprehensive performance comparison between reference and current model performance
    """
    try:
        # Parse config from JSON string
        try:
            config_dict = json.loads(config)
            parsed_config = AnalysisConfig(**config_dict)
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid config JSON: {str(e)}")
        
        # Validate file types
        if not reference_data.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Reference data must be a CSV file")
        if not current_data.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Current data must be a CSV file")
        if not (model_file.filename.lower().endswith(('.pkl', '.pickle', '.joblib', '.pkl.joblib'))):
            raise HTTPException(status_code=400, detail="Model file must be a pickle (.pkl, .pickle) or joblib (.joblib) file")

        # Validate model type
        try:
            model_type_enum = ModelType(parsed_config.model_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid model_type: {parsed_config.model_type}. Must be 'classification' or 'regression'")

        # Create full configuration object
        try:
            full_config = AnalysisConfiguration(
                analysis_name=parsed_config.analysis_name,
                description=parsed_config.description,
                model_type=model_type_enum,
                selected_metrics=parsed_config.selected_metrics,
                statistical_test=parsed_config.statistical_test,
                drift_thresholds=DriftThresholds(
                    low_threshold=parsed_config.low_threshold,
                    medium_threshold=parsed_config.medium_threshold,
                    high_threshold=parsed_config.high_threshold
                )
            )
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=f"Configuration validation error: {str(e)}")

        # Run performance comparison analysis only
        result = await enhanced_model_service.run_performance_comparison_analysis(
            reference_data, current_data, model_file, full_config
        )
        
        return serialize_response(result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance comparison analysis failed: {str(e)}")

@router.post("/degradation-metrics")
async def degradation_metrics_analysis(
    reference_data: UploadFile = File(..., description="Reference dataset CSV file"),
    current_data: UploadFile = File(..., description="Current dataset CSV file"), 
    model_file: UploadFile = File(..., description="Trained model file (.pkl, .pickle, .joblib)"),
    config: str = Form(..., description="JSON configuration string")
):
    """
    Degradation Metrics Analysis (Tab 2)
    
    Returns comprehensive degradation analysis with 3 sub-tabs:
    - Model Disagreement
    - Confidence Analysis  
    - Feature Importance Drift
    """
    try:
        # Parse config from JSON string
        try:
            config_dict = json.loads(config)
            parsed_config = AnalysisConfig(**config_dict)
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid config JSON: {str(e)}")
        
        # Validate file types
        if not reference_data.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Reference data must be a CSV file")
        if not current_data.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Current data must be a CSV file")
        if not (model_file.filename.lower().endswith(('.pkl', '.pickle', '.joblib', '.pkl.joblib'))):
            raise HTTPException(status_code=400, detail="Model file must be a pickle (.pkl, .pickle) or joblib (.joblib) file")

        # Validate model type
        try:
            model_type_enum = ModelType(parsed_config.model_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid model_type: {parsed_config.model_type}. Must be 'classification' or 'regression'")

        # Create full configuration object
        try:
            full_config = AnalysisConfiguration(
                analysis_name=parsed_config.analysis_name,
                description=parsed_config.description,
                model_type=model_type_enum,
                selected_metrics=parsed_config.selected_metrics,
                statistical_test=parsed_config.statistical_test,
                drift_thresholds=DriftThresholds(
                    low_threshold=parsed_config.low_threshold,
                    medium_threshold=parsed_config.medium_threshold,
                    high_threshold=parsed_config.high_threshold
                )
            )
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=f"Configuration validation error: {str(e)}")

        # Run degradation metrics analysis only
        result = await enhanced_model_service.run_degradation_metrics_analysis(
            reference_data, current_data, model_file, full_config
        )
        
        return serialize_response(result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Degradation metrics analysis failed: {str(e)}")

@router.post("/statistical-significance")
async def statistical_significance_analysis(
    reference_data: UploadFile = File(..., description="Reference dataset CSV file"),
    current_data: UploadFile = File(..., description="Current dataset CSV file"), 
    model_file: UploadFile = File(..., description="Trained model file (.pkl, .pickle, .joblib)"),
    config: str = Form(..., description="JSON configuration string")
):
    """
    Statistical Significance Analysis (Tab 3)
    
    Returns comprehensive statistical significance testing with:
    - Hypothesis testing
    - Effect size analysis
    - Power analysis
    - Multiple comparisons correction
    """
    try:
        # Parse config from JSON string
        try:
            config_dict = json.loads(config)
            parsed_config = AnalysisConfig(**config_dict)
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid config JSON: {str(e)}")
        
        # Validate file types
        if not reference_data.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Reference data must be a CSV file")
        if not current_data.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Current data must be a CSV file")
        if not (model_file.filename.lower().endswith(('.pkl', '.pickle', '.joblib', '.pkl.joblib'))):
            raise HTTPException(status_code=400, detail="Model file must be a pickle (.pkl, .pickle) or joblib (.joblib) file")

        # Validate model type
        try:
            model_type_enum = ModelType(parsed_config.model_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid model_type: {parsed_config.model_type}. Must be 'classification' or 'regression'")

        # Create full configuration object
        try:
            full_config = AnalysisConfiguration(
                analysis_name=parsed_config.analysis_name,
                description=parsed_config.description,
                model_type=model_type_enum,
                selected_metrics=parsed_config.selected_metrics,
                statistical_test=parsed_config.statistical_test,
                drift_thresholds=DriftThresholds(
                    low_threshold=parsed_config.low_threshold,
                    medium_threshold=parsed_config.medium_threshold,
                    high_threshold=parsed_config.high_threshold
                )
            )
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=f"Configuration validation error: {str(e)}")

        # Run statistical significance analysis only
        result = await enhanced_model_service.run_statistical_significance_analysis(
            reference_data, current_data, model_file, full_config
        )
        
        return serialize_response(result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Statistical significance analysis failed: {str(e)}")

# Clean Session-Based Model Drift Endpoints (Using Adapter Pattern)

@router.get("/session/performance-comparison/{session_id}")
async def session_performance_comparison(session_id: str):
    """Performance Comparison Analysis using session data with adapter pattern"""
    try:
        # Convert session data to UploadFile-like objects
        reference_file, current_file, model_file, config = create_upload_files_from_session(session_id)
        
        if not reference_file or not current_file or not model_file:
            raise HTTPException(status_code=404, detail="Session not found or does not contain required files")
        
        # Use configuration from session or default
        if config:
            # Handle statistical_test as either string or list (take first item if list)
            statistical_test = config.get("statistical_test", "mcnemar")
            if isinstance(statistical_test, list):
                statistical_test = statistical_test[0] if statistical_test else "mcnemar"
            
            parsed_config = AnalysisConfig(
                analysis_name=config.get("analysis_name", "Performance Comparison"),
                description=config.get("description", "Session-based performance comparison analysis"),
                model_type=config.get("model_type", "classification"),
                selected_metrics=config.get("selected_metrics", ["accuracy", "precision", "recall"]),
                statistical_test=statistical_test,
                low_threshold=config.get("low_threshold", 0.05),
                medium_threshold=config.get("medium_threshold", 0.15),
                high_threshold=config.get("high_threshold", 0.25)
            )
        else:
            parsed_config = AnalysisConfig(
                analysis_name="Performance Comparison",
                description="Session-based performance comparison analysis",
                model_type="classification",
                selected_metrics=["accuracy", "precision", "recall"],
                statistical_test="mcnemar",
                low_threshold=0.05,
                medium_threshold=0.15,
                high_threshold=0.25
            )
        
        # Create full configuration for enhanced service
        model_type_enum = ModelType(parsed_config.model_type.lower())
        full_config = AnalysisConfiguration(
            analysis_name=parsed_config.analysis_name,
            description=parsed_config.description,
            model_type=model_type_enum,
            selected_metrics=parsed_config.selected_metrics,
            statistical_test=parsed_config.statistical_test,
            drift_thresholds=DriftThresholds(
                low_threshold=parsed_config.low_threshold,
                medium_threshold=parsed_config.medium_threshold,
                high_threshold=parsed_config.high_threshold
            )
        )
        
        # Use existing enhanced model service (preserves all your complex logic)
        result = await enhanced_model_service.run_performance_comparison_analysis(
            reference_file, current_file, model_file, full_config
        )
        
        serialized_result = serialize_response(result)

        # Generate AI explanation for the performance comparison analysis
        try:
            ai_explanation = ai_explanation_service.generate_explanation(
                analysis_data=serialized_result, 
                analysis_type="model_performance"
            )
            serialized_result["llm_response"] = ai_explanation
        except Exception as e:
            print(f"Warning: AI explanation failed: {e}")
            # Continue without AI explanation
            serialized_result["llm_response"] = {
                "summary": "Model performance comparison analysis completed successfully.",
                "detailed_explanation": "Performance comparison between reference and current model predictions has been completed. AI explanations are temporarily unavailable.",
                "key_takeaways": [
                    "Performance comparison analysis completed successfully",
                    "Review performance metrics for degradation patterns",
                    "AI explanations will return when service is restored"
                ]
            }
        
        return serialized_result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance comparison analysis failed: {str(e)}")

@router.get("/session/degradation-metrics/{session_id}")
async def session_degradation_metrics(session_id: str):
    """Degradation Metrics Analysis using session data with adapter pattern"""
    try:
        # Convert session data to UploadFile-like objects
        reference_file, current_file, model_file, config = create_upload_files_from_session(session_id)
        
        if not reference_file or not current_file or not model_file:
            raise HTTPException(status_code=404, detail="Session not found or does not contain required files")
        
        # Use configuration from session or default
        if config:
            # Handle statistical_test as either string or list (take first item if list)
            statistical_test = config.get("statistical_test", "mcnemar")
            if isinstance(statistical_test, list):
                statistical_test = statistical_test[0] if statistical_test else "mcnemar"
            
            parsed_config = AnalysisConfig(
                analysis_name=config.get("analysis_name", "Degradation Metrics"),
                description=config.get("description", "Session-based degradation metrics analysis"),
                model_type=config.get("model_type", "classification"),
                selected_metrics=config.get("selected_metrics", ["accuracy", "precision", "recall"]),
                statistical_test=statistical_test,
                low_threshold=config.get("low_threshold", 0.05),
                medium_threshold=config.get("medium_threshold", 0.15),
                high_threshold=config.get("high_threshold", 0.25)
            )
        else:
            parsed_config = AnalysisConfig(
                analysis_name="Degradation Metrics",
                description="Session-based degradation metrics analysis",
                model_type="classification",
                selected_metrics=["accuracy", "precision", "recall"],
                statistical_test="mcnemar",
                low_threshold=0.05,
                medium_threshold=0.15,
                high_threshold=0.25
            )
        
        # Create full configuration for enhanced service
        model_type_enum = ModelType(parsed_config.model_type.lower())
        full_config = AnalysisConfiguration(
            analysis_name=parsed_config.analysis_name,
            description=parsed_config.description,
            model_type=model_type_enum,
            selected_metrics=parsed_config.selected_metrics,
            statistical_test=parsed_config.statistical_test,
            drift_thresholds=DriftThresholds(
                low_threshold=parsed_config.low_threshold,
                medium_threshold=parsed_config.medium_threshold,
                high_threshold=parsed_config.high_threshold
            )
        )
        
        # Use existing enhanced model service (preserves all your complex logic)
        result = await enhanced_model_service.run_degradation_metrics_analysis(
            reference_file, current_file, model_file, full_config
        )
        
        serialized_result = serialize_response(result)

        # Generate AI explanation for the degradation metrics analysis
        try:
            ai_explanation = ai_explanation_service.generate_explanation(
                analysis_data=serialized_result, 
                analysis_type="degradation_metrics"
            )
            serialized_result["llm_response"] = ai_explanation
        except Exception as e:
            print(f"Warning: AI explanation failed: {e}")
            # Continue without AI explanation
            serialized_result["llm_response"] = {
                "summary": "Model degradation metrics analysis completed successfully.",
                "detailed_explanation": "Comprehensive degradation analysis including model disagreement and confidence patterns has been completed. AI explanations are temporarily unavailable.",
                "key_takeaways": [
                    "Degradation metrics analysis completed successfully",
                    "Review model disagreement and confidence trends",
                    "AI explanations will return when service is restored"
                ]
            }
        
        return serialized_result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Degradation metrics analysis failed: {str(e)}")

@router.get("/session/statistical-significance/{session_id}")
async def session_statistical_significance(session_id: str):
    """Statistical Significance Analysis using session data with adapter pattern"""
    try:
        # Convert session data to UploadFile-like objects
        reference_file, current_file, model_file, config = create_upload_files_from_session(session_id)
        
        if not reference_file or not current_file or not model_file:
            raise HTTPException(status_code=404, detail="Session not found or does not contain required files")
        
        # Use configuration from session or default
        if config:
            # Handle statistical_test as either string or list (take first item if list)
            statistical_test = config.get("statistical_test", "mcnemar")
            if isinstance(statistical_test, list):
                statistical_test = statistical_test[0] if statistical_test else "mcnemar"
            
            parsed_config = AnalysisConfig(
                analysis_name=config.get("analysis_name", "Statistical Significance"),
                description=config.get("description", "Session-based statistical significance analysis"),
                model_type=config.get("model_type", "classification"),
                selected_metrics=config.get("selected_metrics", ["accuracy", "precision", "recall"]),
                statistical_test=statistical_test,
                low_threshold=config.get("low_threshold", 0.05),
                medium_threshold=config.get("medium_threshold", 0.15),
                high_threshold=config.get("high_threshold", 0.25)
            )
        else:
            parsed_config = AnalysisConfig(
                analysis_name="Statistical Significance",
                description="Session-based statistical significance analysis",
                model_type="classification",
                selected_metrics=["accuracy", "precision", "recall"],
                statistical_test="mcnemar",
                low_threshold=0.05,
                medium_threshold=0.15,
                high_threshold=0.25
            )
        
        # Create full configuration for enhanced service
        model_type_enum = ModelType(parsed_config.model_type.lower())
        full_config = AnalysisConfiguration(
            analysis_name=parsed_config.analysis_name,
            description=parsed_config.description,
            model_type=model_type_enum,
            selected_metrics=parsed_config.selected_metrics,
            statistical_test=parsed_config.statistical_test,
            drift_thresholds=DriftThresholds(
                low_threshold=parsed_config.low_threshold,
                medium_threshold=parsed_config.medium_threshold,
                high_threshold=parsed_config.high_threshold
            )
        )
        
        # Use existing enhanced model service (preserves all your complex logic)
        result = await enhanced_model_service.run_statistical_significance_analysis(
            reference_file, current_file, model_file, full_config
        )
        
        serialized_result = serialize_response(result)

        # Generate AI explanation for the statistical significance analysis
        try:
            ai_explanation = ai_explanation_service.generate_explanation(
                analysis_data=serialized_result, 
                analysis_type="statistical_significance"
            )
            serialized_result["llm_response"] = ai_explanation
        except Exception as e:
            print(f"Warning: AI explanation failed: {e}")
            # Continue without AI explanation
            serialized_result["llm_response"] = {
                "summary": "Statistical significance analysis completed successfully.",
                "detailed_explanation": "Statistical testing of model performance changes has been completed with hypothesis testing and effect size analysis. AI explanations are temporarily unavailable.",
                "key_takeaways": [
                    "Statistical significance analysis completed successfully",
                    "Review statistical test results and confidence intervals",
                    "AI explanations will return when service is restored"
                ]
            }
        
        return serialized_result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Statistical significance analysis failed: {str(e)}")

@router.get("/health")
async def model_drift_health():
    """Health check for model drift service"""
    return {"status": "ok", "service": "model-drift"}
