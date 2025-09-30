"""
S3-based stateless endpoints for data discovery and loading
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging
from datetime import datetime
import io

import math
import numpy as np
import pandas as pd
from fastapi.encoders import jsonable_encoder
from .s3_utils import get_s3_file_metadata, load_s3_csv, load_s3_model, validate_dataframe, validate_target_column
from .models import LoadDataRequest, LoadDataResponse, S3FileMetadata, AnalysisRequest, ModelDriftAnalysisRequest
from ..model_drift.routes.upload import (
    create_ai_summary_for_performance_comparison,
    create_ai_summary_for_degradation_metrics,
    create_ai_summary_for_statistical_significance,
    serialize_response
)
from ..model_drift.services.enhanced_model_service import enhanced_model_service
from ..model_drift.models.analysis_config import AnalysisConfiguration, ModelType, DriftThresholds
from .ai_explanation_service import ai_explanation_service
from ..model_drift.services.analysis.performance_comparison_service import performance_comparison_service
from ..model_drift.services.analysis.degradation_metrics_service import degradation_metrics_service  
from ..model_drift.services.analysis.statistical_significance_service import statistical_significance_service

logger = logging.getLogger(__name__)

router = APIRouter()

router = APIRouter()

# ===== MODEL WRAPPING UTILITIES =====

def apply_consistent_preprocessing(ref_df, curr_df, feature_columns, target_column, preprocessing_info=None):
    """
    Apply consistent preprocessing to both reference and current datasets
    
    Args:
        ref_df: Reference DataFrame
        curr_df: Current DataFrame
        feature_columns: List of expected feature columns
        target_column: Target column name
        preprocessing_info: Preprocessing information from model wrapper
        
    Returns:
        Tuple of (X_ref, y_ref, X_curr, y_curr, preprocessing_metadata)
    """
    # Check if model was likely trained on raw categorical columns (no preprocessing)
    # This is a heuristic - if feature_columns contains known categorical columns, assume
    # the model expects raw categorical data
    categorical_columns = []
    for col in ref_df.columns:
        if col != target_column and ref_df[col].dtype == 'object':
            categorical_columns.append(col)
    
    # If any of our categorical columns are in feature_columns, the model likely
    # expects raw categorical data without one-hot encoding
    raw_categorical_model = False
    if feature_columns:
        for col in categorical_columns:
            if col in feature_columns:
                raw_categorical_model = True
                logger.info(f"Detected model trained on raw categorical features: {col} found in feature_columns")
                break
    
    # If we detect the model expects raw categorical features, use a simplified preprocessing approach
    if raw_categorical_model:
        logger.info("Using simplified preprocessing to maintain original column structure")
        
        # For reference data
        if feature_columns:
            X_ref = ref_df[feature_columns].copy()
        else:
            X_ref = ref_df.drop(columns=[target_column]) if target_column in ref_df.columns else ref_df.copy()
            
        y_ref = ref_df[target_column] if target_column in ref_df.columns else None
        
        # For current data
        if feature_columns:
            X_curr = curr_df[feature_columns].copy()
        else:
            X_curr = curr_df.drop(columns=[target_column]) if target_column in curr_df.columns else curr_df.copy()
            
        y_curr = curr_df[target_column] if target_column in curr_df.columns else None
        
        # Simple metadata
        preprocessing_metadata = {
            "categorical_columns": categorical_columns,
            "encoded_columns": [],
            "original_columns": list(X_ref.columns),
            "raw_categorical_model": True
        }
        
        # Handle categorical columns with label encoding (most models expect this)
        from sklearn.preprocessing import LabelEncoder
        label_encoders = {}
        
        for col in categorical_columns:
            if col in X_ref.columns:
                logger.info(f"Label encoding categorical column: {col}")
                
                # Create label encoder and fit on combined data to ensure consistency
                le = LabelEncoder()
                combined_values = pd.concat([X_ref[col].astype(str), X_curr[col].astype(str)])
                le.fit(combined_values.dropna())
                
                # Transform both datasets
                X_ref[col] = le.transform(X_ref[col].astype(str))
                X_curr[col] = le.transform(X_curr[col].astype(str))
                
                label_encoders[col] = le
                logger.info(f"Encoded {col} with {len(le.classes_)} unique values: {le.classes_[:5]}...")
        
        # Basic type conversion for numeric columns
        for col in X_ref.columns:
            if col not in categorical_columns:
                try:
                    X_ref[col] = pd.to_numeric(X_ref[col], errors='coerce')
                    X_curr[col] = pd.to_numeric(X_curr[col], errors='coerce')
                except:
                    pass
        
        # Fill any NaN values
        X_ref = X_ref.fillna(0)
        X_curr = X_curr.fillna(0)
        
        preprocessing_metadata["label_encoders"] = label_encoders
        
        logger.info(f"Raw categorical preprocessing complete. Shape: Ref {X_ref.shape}, Curr {X_curr.shape}")
        logger.info(f"Data types: {dict(X_ref.dtypes.value_counts())}")
        
        return X_ref, y_ref, X_curr, y_curr, preprocessing_metadata
    
    # Otherwise, use our standard one-hot encoding approach
    else:
        logger.info("Using one-hot encoding preprocessing for model trained on encoded features")
        
        # Prepare reference data
        X_ref, y_ref, ref_metadata = prepare_data_for_model(
            ref_df, feature_columns, target_column, preprocessing_info
        )
        
        # For current data, we need to apply the SAME preprocessing as reference
        # This ensures consistent feature encoding between datasets
        X_curr, y_curr, curr_metadata = prepare_data_for_model(
            curr_df, feature_columns, target_column, preprocessing_info
        )
        
        # Ensure both datasets have the same columns after preprocessing
        ref_columns = set(X_ref.columns)
        curr_columns = set(X_curr.columns)
        
        if ref_columns != curr_columns:
            logger.warning(f"Column mismatch between datasets. Ref: {len(ref_columns)}, Curr: {len(curr_columns)}")
            
            # Get the union of all columns and add missing columns as zeros
            all_columns = sorted(ref_columns.union(curr_columns))
            
            for col in all_columns:
                if col not in X_ref.columns:
                    X_ref[col] = 0
                if col not in X_curr.columns:
                    X_curr[col] = 0
            
            # Reorder columns to match
            X_ref = X_ref[all_columns]
            X_curr = X_curr[all_columns]
            
            logger.info(f"Aligned datasets to {len(all_columns)} columns")
        
        return X_ref, y_ref, X_curr, y_curr, ref_metadata

def extract_model_from_wrapper(wrapped_model):
    """
    Extract the actual model from a wrapped model, handling both wrapped and unwrapped cases
    
    Args:
        wrapped_model: Either a wrapped model dict or direct model object
    
    Returns:
        Tuple of (model, metadata)
    """
    if isinstance(wrapped_model, dict) and "model" in wrapped_model:
        # This is a wrapped model
        logger.info("Found wrapped model with metadata")
        return wrapped_model["model"], {
            "feature_columns": wrapped_model.get("feature_columns", []),
            "target_column": wrapped_model.get("target_column"),
            "model_type": wrapped_model.get("model_type"),
            "preprocessing_info": wrapped_model.get("preprocessing_info", {}),
            "wrapper_version": wrapped_model.get("wrapper_version")
        }
    else:
        # This is an unwrapped model - try to extract what we can
        logger.info("Found unwrapped model, attempting to extract feature information")
        
        feature_columns = []
        
        # Try to extract feature names from sklearn models
        if hasattr(wrapped_model, 'feature_names_in_'):
            feature_columns = list(wrapped_model.feature_names_in_)
            logger.info(f"Extracted {len(feature_columns)} feature names from sklearn model: {feature_columns[:5]}...")
        elif hasattr(wrapped_model, 'feature_importances_'):
            # For models with feature importance, we can at least get the count
            n_features = len(wrapped_model.feature_importances_)
            feature_columns = [f"feature_{i}" for i in range(n_features)]
            logger.info(f"Model has {n_features} features but no feature names available")
        
        return wrapped_model, {
            "feature_columns": feature_columns,
            "target_column": None,
            "model_type": str(type(wrapped_model).__name__),
            "preprocessing_info": {},
            "wrapper_version": None
        }


def prepare_data_for_model(df, feature_columns, target_column, preprocessing_info=None):
    """
    Prepare DataFrame for model prediction, handling data type conversion and categorical encoding
    
    Args:
        df: Input DataFrame
        feature_columns: List of expected feature columns
        target_column: Target column name (optional)
        preprocessing_info: Dict with preprocessing information from model wrapper
    
    Returns:
        Tuple of (X, y, preprocessing_metadata) where y is None if target_column not provided
    """
    try:
        # If feature columns are specified, use only those
        if feature_columns:
            # Check if all required features are present
            missing_features = set(feature_columns) - set(df.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            X = df[feature_columns].copy()
        else:
            # If no feature columns specified, use all except target
            if target_column and target_column in df.columns:
                X = df.drop(columns=[target_column])
            else:
                X = df.copy()
        
        # Handle target column
        y = None
        if target_column and target_column in df.columns:
            y = df[target_column]
        
        # Track preprocessing metadata for consistency
        preprocessing_metadata = {
            "categorical_columns": [],
            "encoded_columns": [],
            "original_columns": list(X.columns)
        }
        
        # First pass: identify categorical and numeric columns
        categorical_columns = []
        for col in X.columns:
            if X[col].dtype == 'object':
                # Try to convert to numeric first
                numeric_series = pd.to_numeric(X[col], errors='coerce')
                # If conversion creates too many NaNs, treat as categorical
                if numeric_series.isna().sum() / len(numeric_series) > 0.5:
                    categorical_columns.append(col)
                    logger.info(f"Column {col} identified as categorical")
                else:
                    X[col] = numeric_series
                    logger.info(f"Column {col} converted to numeric")
        
        # Handle categorical columns with one-hot encoding
        if categorical_columns:
            logger.info(f"One-hot encoding categorical columns: {categorical_columns}")
            preprocessing_metadata["categorical_columns"] = categorical_columns
            
            # Store original values for reference
            for col in categorical_columns:
                preprocessing_metadata[f"{col}_unique_values"] = list(X[col].unique())
            
            # Apply one-hot encoding
            X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=False, dummy_na=True)
            preprocessing_metadata["encoded_columns"] = [col for col in X_encoded.columns if col not in X.columns]
            
            logger.info(f"Original shape: {X.shape}, Encoded shape: {X_encoded.shape}")
            logger.info(f"New encoded columns: {preprocessing_metadata['encoded_columns'][:5]}...")  # Show first 5
            
            X = X_encoded
        
        # Final data type conversion and validation
        for col in X.columns:
            if X[col].dtype == 'object':
                # Try one more time to convert any remaining object columns
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except:
                    pass
        
        # Fill any remaining NaN values
        X = X.fillna(0)
        
        logger.info(f"Final prepared data shape: {X.shape}")
        logger.info(f"Data types: {dict(X.dtypes.value_counts())}")
        
        return X, y, preprocessing_metadata
        
    except Exception as e:
        logger.error(f"Error preparing data for model: {e}")
        raise HTTPException(
            status_code=400, 
            detail=f"Data preparation failed: {str(e)}. This might be due to categorical data that needs preprocessing."
        )


# ===== ENDPOINTS =====

@router.get("/s3/files/{project_id}", response_model=S3FileMetadata)
async def get_s3_files(project_id: str):
    """
    Get list of files and models available in S3 for a specific project
    
    Args:
        project_id: The project ID to fetch files for
        
    Returns:
        Dictionary containing files and models lists with metadata
    """
    try:
        logger.info(f"Fetching S3 files for project: {project_id}")
        result = get_s3_file_metadata(project_id)
        logger.info(f"Found {len(result.get('files', []))} files and {len(result.get('models', []))} models")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching S3 files for project {project_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch S3 files: {str(e)}")


@router.post("/load", response_model=LoadDataResponse)
async def load_data(payload: LoadDataRequest):
    """
    Load datasets and optional model from S3 URLs for analysis
    
    Args:
        payload: LoadDataRequest containing S3 URLs and configuration
        
    Returns:
        LoadDataResponse with loaded data information and validation status
    """
    try:
        logger.info(f"Loading data from URLs: ref={payload.reference_url}, curr={payload.current_url}")
        
        # Load datasets from S3
        reference_df = load_s3_csv(payload.reference_url)
        current_df = load_s3_csv(payload.current_url)
        
        # Validate datasets
        validate_dataframe(reference_df, "Reference")
        validate_dataframe(current_df, "Current")
        
        # Find common columns
        ref_columns = set(reference_df.columns)
        curr_columns = set(current_df.columns)
        common_columns = list(ref_columns.intersection(curr_columns))
        
        if not common_columns:
            raise HTTPException(
                status_code=400, 
                detail="No common columns found between reference and current datasets"
            )
        
        # Validate target column if provided
        if payload.target_column:
            validate_target_column(reference_df, payload.target_column, "reference")
            validate_target_column(current_df, payload.target_column, "current")
        
        # Load model if provided
        model = None
        model_info = None
        model_loaded = False
        
        if payload.model_url:
            try:
                model = load_s3_model(payload.model_url)
                model_loaded = True
                model_info = {
                    "type": str(type(model).__name__),
                    "url": payload.model_url,
                    "loaded_successfully": True
                }
                logger.info(f"Model loaded successfully: {type(model).__name__}")
            except Exception as e:
                logger.warning(f"Failed to load model from {payload.model_url}: {e}")
                model_info = {
                    "type": "unknown",
                    "url": payload.model_url,
                    "loaded_successfully": False,
                    "error": str(e)
                }
        
        # Prepare response
        response = LoadDataResponse(
            message=f"Successfully loaded datasets with {len(common_columns)} common columns",
            reference_dataset={
                "shape": reference_df.shape,
                "columns": list(reference_df.columns),
                "dtypes": {col: str(dtype) for col, dtype in reference_df.dtypes.items()},
                "url": payload.reference_url
            },
            current_dataset={
                "shape": current_df.shape,
                "columns": list(current_df.columns),
                "dtypes": {col: str(dtype) for col, dtype in current_df.dtypes.items()},
                "url": payload.current_url
            },
            model_loaded=model_loaded,
            model_info=model_info,
            target_column=payload.target_column,
            config=payload.config,
            common_columns=common_columns
        )
        
        logger.info(f"Data loading completed successfully. Ref: {reference_df.shape}, Curr: {current_df.shape}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise HTTPException(status_code=500, detail=f"Data loading failed: {str(e)}")


# ===== MODEL DRIFT S3 ENDPOINTS =====
# These load data directly from S3 URLs and perform model drift analysis

@router.post("/model-drift/performance-comparison", 
            summary="S3-based Performance Comparison Analysis",
            description="Loads data directly from S3 URLs and performs model performance comparison analysis")
async def s3_model_drift_performance_comparison(request: ModelDriftAnalysisRequest):
    """S3-based Performance Comparison Analysis - loads data directly from S3 URLs"""
    try:
        # Load data and model directly from S3 URLs
        reference_df = load_s3_csv(request.reference_url)
        current_df = load_s3_csv(request.current_url)
        wrapped_model = load_s3_model(request.model_url)
        
        # Extract model and metadata from wrapper
        model, model_metadata = extract_model_from_wrapper(wrapped_model)
        
        # Validate datasets
        validate_dataframe(reference_df, "Reference")
        validate_dataframe(current_df, "Current")
        
        # Determine target column (use from model metadata if available, then request, then last column)
        target_column = (
            model_metadata.get("target_column") or 
            request.target_column or 
            reference_df.columns[-1]
        )
        
        # Prepare data using enhanced preprocessing for consistent processing
        try:
            X_ref, y_true_ref, X_curr, y_true_curr, preprocessing_metadata = apply_consistent_preprocessing(
                reference_df, 
                current_df,
                model_metadata.get("feature_columns"), 
                target_column,
                model_metadata.get("preprocessing_info")
            )
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Data preparation failed: {str(e)}. Please ensure your data is properly formatted and matches the model's expected input."
            )
        
        # Generate predictions with error handling
        try:
            pred_ref = model.predict(X_ref)
            pred_curr = model.predict(X_curr)
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Model prediction failed: {str(e)}. This might be due to data format mismatch or categorical variables."
            )
        
        # Get prediction probabilities if available
        pred_ref_proba = None
        pred_curr_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                pred_ref_proba = model.predict_proba(X_ref)
                pred_curr_proba = model.predict_proba(X_curr)
            except Exception as e:
                logger.warning(f"Could not get prediction probabilities: {e}")
        
        # Get analysis configuration
        config = request.analysis_config or {}
        
        # Run performance comparison analysis
        result = performance_comparison_service.analyze_performance_comparison(
            y_true=y_true_ref,  # Use reference ground truth as baseline
            pred_ref=pred_ref,
            pred_curr=pred_curr,
            pred_ref_proba=pred_ref_proba,
            pred_curr_proba=pred_curr_proba,
            X=X_ref,  # Pass reference features
            model_ref=model,
            model_curr=model
        )
        
        # Add metadata including model wrapper info
        result["analysis_metadata"] = {
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "reference_dataset_size": len(reference_df),
            "current_dataset_size": len(current_df),
            "model_type": model_metadata.get("model_type", str(type(model).__name__)),
            "model_wrapper_version": model_metadata.get("wrapper_version"),
            "feature_columns": model_metadata.get("feature_columns", []),
            "target_column": target_column,
            "analysis_config": config,
            "data_sources": {
                "reference_url": request.reference_url,
                "current_url": request.current_url,
                "model_url": request.model_url
            }
        }
        
        # Serialize response
        serialized_result = serialize_response(result)

        # Generate AI explanation
        try:
            ai_summary_payload = create_ai_summary_for_performance_comparison(serialized_result)
            ai_explanation = ai_explanation_service.generate_explanation(
                analysis_data=ai_summary_payload, 
                analysis_type="model_performance"
            )
            serialized_result["llm_response"] = ai_explanation
        except Exception as e:
            logger.warning(f"AI explanation failed: {e}")
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
        logger.error(f"Model drift performance comparison failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Performance comparison analysis failed: {str(e)}")


@router.post("/model-drift/degradation-metrics",
            summary="S3-based Degradation Metrics Analysis", 
            description="Loads data directly from S3 URLs and performs model degradation metrics analysis")
async def s3_model_drift_degradation_metrics(request: ModelDriftAnalysisRequest):
    """S3-based Degradation Metrics Analysis - loads data directly from S3 URLs"""
    try:
        # Load data and model directly from S3 URLs
        reference_df = load_s3_csv(request.reference_url)
        current_df = load_s3_csv(request.current_url)
        wrapped_model = load_s3_model(request.model_url)
        
        # Extract model and metadata from wrapper
        model, model_metadata = extract_model_from_wrapper(wrapped_model)
        
        # Validate datasets
        validate_dataframe(reference_df, "Reference")
        validate_dataframe(current_df, "Current")
        
        # Determine target column (use from model metadata if available, then request, then last column)
        target_column = (
            model_metadata.get("target_column") or 
            request.target_column or 
            reference_df.columns[-1]
        )
        
        # Prepare data using enhanced preprocessing for consistent processing
        try:
            X_ref, y_true_ref, X_curr, y_true_curr, preprocessing_metadata = apply_consistent_preprocessing(
                reference_df, 
                current_df,
                model_metadata.get("feature_columns"), 
                target_column,
                model_metadata.get("preprocessing_info")
            )
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Data preparation failed: {str(e)}. Please ensure your data is properly formatted and matches the model's expected input."
            )
        
        # Generate predictions with enhanced error handling and format consistency
        try:
            # Get binary predictions
            pred_ref = model.predict(X_ref)
            pred_curr = model.predict(X_curr)
            
            # Ensure predictions are in consistent format
            pred_ref = np.asarray(pred_ref).flatten()
            pred_curr = np.asarray(pred_curr).flatten()
            
            logger.info(f"Binary predictions generated. Shapes: ref={pred_ref.shape}, curr={pred_curr.shape}")
            
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Model prediction failed: {str(e)}. This might be due to data format mismatch or categorical variables."
            )
        
        # Get prediction probabilities with enhanced handling
        pred_ref_proba = None
        pred_curr_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                pred_ref_proba_raw = model.predict_proba(X_ref)
                pred_curr_proba_raw = model.predict_proba(X_curr)
                
                # Handle different probability formats
                if len(pred_ref_proba_raw.shape) > 1 and pred_ref_proba_raw.shape[1] > 1:
                    if pred_ref_proba_raw.shape[1] == 2:
                        # Binary classification - use positive class probability
                        pred_ref_proba = pred_ref_proba_raw[:, 1]
                        pred_curr_proba = pred_curr_proba_raw[:, 1]
                        logger.info(f"Binary classification detected - using positive class probabilities")
                    else:
                        # Multi-class - use maximum probability for confidence analysis
                        pred_ref_proba = np.max(pred_ref_proba_raw, axis=1)
                        pred_curr_proba = np.max(pred_curr_proba_raw, axis=1)
                        logger.info(f"Multi-class classification detected - using max probabilities")
                else:
                    # Single probability per prediction
                    pred_ref_proba = pred_ref_proba_raw.flatten()
                    pred_curr_proba = pred_curr_proba_raw.flatten()
                
                # Ensure probabilities are in valid range [0, 1]
                pred_ref_proba = np.clip(pred_ref_proba, 0, 1)
                pred_curr_proba = np.clip(pred_curr_proba, 0, 1)
                
                # Validate probability quality
                ref_min, ref_max = np.min(pred_ref_proba), np.max(pred_ref_proba)
                curr_min, curr_max = np.min(pred_curr_proba), np.max(pred_curr_proba)
                
                logger.info(f"Probability ranges: ref=[{ref_min:.6f}, {ref_max:.6f}], curr=[{curr_min:.6f}, {curr_max:.6f}]")
                
                # Check for extreme probability distributions
                if ref_max - ref_min < 0.01 or curr_max - curr_min < 0.01:
                    logger.warning("Detected very narrow probability distribution - model may not be well calibrated")
                
                if ref_min < 1e-10 or curr_min < 1e-10:
                    logger.warning("Detected extremely small probabilities - may cause numerical issues")
                    # Set minimum threshold to avoid numerical problems
                    pred_ref_proba = np.maximum(pred_ref_proba, 1e-10)
                    pred_curr_proba = np.maximum(pred_curr_proba, 1e-10)
                    
            except Exception as e:
                logger.warning(f"Could not get prediction probabilities: {e}")
                pred_ref_proba = None
                pred_curr_proba = None

        # Final data validation before analysis
        logger.info(f"Data validation - X_ref shape: {X_ref.shape}, X_curr shape: {X_curr.shape}")
        logger.info(f"Predictions - pred_ref shape: {pred_ref.shape}, pred_curr shape: {pred_curr.shape}")
        if pred_ref_proba is not None:
            logger.info(f"Probabilities - pred_ref_proba shape: {pred_ref_proba.shape}, pred_curr_proba shape: {pred_curr_proba.shape}")
        
        # Ensure we have ground truth for analysis
        if y_true_curr is None:
            logger.warning("No current ground truth available, using reference ground truth")
            y_true_curr = y_true_ref

        # Get analysis configuration
        config = request.analysis_config or {}
        
        # Run degradation metrics analysis
        # Note: For S3 analysis, we compare the same model's performance on different datasets
        # rather than comparing two different models
        result = degradation_metrics_service.analyze_degradation_metrics(
            y_true=y_true_ref,  # Use reference ground truth for evaluation
            pred_ref=pred_ref,
            pred_curr=pred_curr,
            pred_ref_proba=pred_ref_proba,
            pred_curr_proba=pred_curr_proba,
            X_ref=X_ref,
            y_ref=y_true_ref,
            X_curr=X_curr,
            y_curr=y_true_curr,
            model_ref=model,
            model_curr=model,  # Same model, different datasets
            feature_names=list(X_ref.columns) if hasattr(X_ref, 'columns') else None
        )        # Add metadata including model wrapper info
        result["analysis_metadata"] = {
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "reference_dataset_size": len(reference_df),
            "current_dataset_size": len(current_df),
            "model_type": model_metadata.get("model_type", str(type(model).__name__)),
            "model_wrapper_version": model_metadata.get("wrapper_version"),
            "feature_columns": model_metadata.get("feature_columns", []),
            "target_column": target_column,
            "analysis_config": config,
            "data_sources": {
                "reference_url": request.reference_url,
                "current_url": request.current_url,
                "model_url": request.model_url
            }
        }
        
        # Serialize response
        serialized_result = serialize_response(result)

        # Generate AI explanation
        try:
            ai_summary_payload = create_ai_summary_for_degradation_metrics(serialized_result)
            ai_explanation = ai_explanation_service.generate_explanation(
                analysis_data=ai_summary_payload, 
                analysis_type="degradation_metrics"
            )
            serialized_result["llm_response"] = ai_explanation
        except Exception as e:
            logger.warning(f"AI explanation failed: {e}")
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
        logger.error(f"Model drift degradation metrics failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Degradation metrics analysis failed: {str(e)}")


@router.post("/model-drift/statistical-significance",
            summary="S3-based Statistical Significance Analysis",
            description="Loads data directly from S3 URLs and performs statistical significance analysis")
async def s3_model_drift_statistical_significance(request: ModelDriftAnalysisRequest):
    """S3-based Statistical Significance Analysis - loads data directly from S3 URLs"""
    try:
        # Load data and model directly from S3 URLs
        reference_df = load_s3_csv(request.reference_url)
        current_df = load_s3_csv(request.current_url)
        wrapped_model = load_s3_model(request.model_url)
        
        # Extract model and metadata from wrapper
        model, model_metadata = extract_model_from_wrapper(wrapped_model)
        
        # Validate datasets
        validate_dataframe(reference_df, "Reference")
        validate_dataframe(current_df, "Current")
        
        # Determine target column (use from model metadata if available, then request, then last column)
        target_column = (
            model_metadata.get("target_column") or 
            request.target_column or 
            reference_df.columns[-1]
        )
        
        # Prepare data using enhanced preprocessing for consistent processing
        try:
            X_ref, y_true_ref, X_curr, y_true_curr, preprocessing_metadata = apply_consistent_preprocessing(
                reference_df, 
                current_df,
                model_metadata.get("feature_columns"), 
                target_column,
                model_metadata.get("preprocessing_info")
            )
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Data preparation failed: {str(e)}. Please ensure your data is properly formatted and matches the model's expected input."
            )
        
        # Generate predictions with error handling
        try:
            pred_ref = model.predict(X_ref)
            pred_curr = model.predict(X_curr)
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Model prediction failed: {str(e)}. This might be due to data format mismatch or categorical variables."
            )
        
        # Get prediction probabilities if available
        pred_ref_proba = None
        pred_curr_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                pred_ref_proba = model.predict_proba(X_ref)
                pred_curr_proba = model.predict_proba(X_curr)
            except Exception as e:
                logger.warning(f"Could not get prediction probabilities: {e}")
        
        # Get analysis configuration
        config = request.analysis_config or {}
        
        # Run statistical significance analysis
        result = statistical_significance_service.analyze_statistical_significance(
            y_true=y_true_ref,  # Use reference ground truth
            pred_ref=pred_ref,
            pred_curr=pred_curr,
            pred_ref_proba=pred_ref_proba,
            pred_curr_proba=pred_curr_proba,
            X=X_ref,
            model_ref=model,
            model_curr=model,
            alpha=config.get("alpha", 0.05)
        )
        
        # Add metadata including model wrapper info
        result["analysis_metadata"] = {
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "reference_dataset_size": len(reference_df),
            "current_dataset_size": len(current_df),
            "model_type": model_metadata.get("model_type", str(type(model).__name__)),
            "model_wrapper_version": model_metadata.get("wrapper_version"),
            "feature_columns": model_metadata.get("feature_columns", []),
            "target_column": target_column,
            "analysis_config": config,
            "data_sources": {
                "reference_url": request.reference_url,
                "current_url": request.current_url,
                "model_url": request.model_url
            }
        }
        
        # Serialize response
        serialized_result = serialize_response(result)

        # Generate AI explanation
        try:
            ai_summary_payload = create_ai_summary_for_statistical_significance(serialized_result)
            ai_explanation = ai_explanation_service.generate_explanation(
                analysis_data=ai_summary_payload, 
                analysis_type="statistical_significance"
            )
            serialized_result["llm_response"] = ai_explanation
        except Exception as e:
            logger.warning(f"AI explanation failed: {e}")
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
        logger.error(f"Model drift statistical significance failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Statistical significance analysis failed: {str(e)}")


@router.get("/model-drift/metrics",
           summary="Get Available Metrics",
           description="Get available performance metrics for each model type - S3 compatible")
async def get_available_metrics():
    """Get available performance metrics for each model type - S3 compatible"""
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


@router.get("/model-drift/tests",
           summary="Get Available Statistical Tests",
           description="Get available statistical tests for each model type - S3 compatible")
async def get_available_tests():
    """Get available statistical tests for each model type - S3 compatible"""
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

@router.get("/health",
           summary="Health Check",
           description="Health check endpoint for S3 Data Loading Service")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "S3 Data Loading Service"}