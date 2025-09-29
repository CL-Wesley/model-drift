"""
Pydantic models for S3-based stateless data loading
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
import json


class LoadDataRequest(BaseModel):
    """Request model for loading data and models from S3"""
    reference_url: str = Field(..., description="S3 URL of the reference/baseline dataset CSV")
    current_url: str = Field(..., description="S3 URL of the current dataset CSV")
    model_url: Optional[str] = Field(None, description="Optional S3 URL of the trained model file (pickle/joblib)")
    target_column: Optional[str] = Field(None, description="Optional target column name for classification analysis")
    config: Optional[Dict[str, Any]] = Field(None, description="Optional configuration dictionary for analysis")
    
    @validator('reference_url', 'current_url')
    def validate_urls(cls, v):
        if not v or not v.strip():
            raise ValueError("URL cannot be empty")
        return v.strip()
    
    @validator('model_url')
    def validate_model_url(cls, v):
        if v is not None and not v.strip():
            raise ValueError("Model URL cannot be empty string")
        return v.strip() if v else None
    
    @validator('target_column')
    def validate_target_column(cls, v):
        if v is not None and not v.strip():
            raise ValueError("Target column cannot be empty string")
        return v.strip() if v else None


class LoadDataResponse(BaseModel):
    """Response model for successful data loading"""
    status: str = "success"
    message: str
    reference_dataset: Dict[str, Any]
    current_dataset: Dict[str, Any]
    model_loaded: bool = False
    model_info: Optional[Dict[str, Any]] = None
    target_column: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    common_columns: List[str]
    validation_status: str = "passed"


class S3FileMetadata(BaseModel):
    """Model for S3 file metadata"""
    files: List[Dict[str, Any]]
    models: List[Dict[str, Any]]


class AnalysisRequest(BaseModel):
    """Request model for analysis endpoints (replaces session_id)"""
    reference_url: str = Field(..., description="S3 URL of the reference dataset")
    current_url: str = Field(..., description="S3 URL of the current dataset") 
    model_url: Optional[str] = Field(None, description="Optional S3 URL of the model")
    target_column: Optional[str] = Field(None, description="Optional target column name")
    config: Optional[Dict[str, Any]] = Field(None, description="Optional configuration")
    
    @validator('reference_url', 'current_url')
    def validate_urls(cls, v):
        if not v or not v.strip():
            raise ValueError("URL cannot be empty")
        return v.strip()


class ModelDriftAnalysisRequest(BaseModel):
    """Request model for model drift analysis endpoints"""
    reference_url: str = Field(..., description="S3 URL of the reference dataset CSV")
    current_url: str = Field(..., description="S3 URL of the current dataset CSV") 
    model_url: str = Field(..., description="S3 URL of the trained model file (required for model drift)")
    target_column: Optional[str] = Field(None, description="Target column name for predictions")
    analysis_config: Optional[Dict[str, Any]] = Field(None, description="Analysis configuration (thresholds, metrics, etc.)")
    
    @validator('reference_url', 'current_url', 'model_url')
    def validate_urls(cls, v):
        if not v or not v.strip():
            raise ValueError("URL cannot be empty")
        return v.strip()