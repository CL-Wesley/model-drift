from fastapi import APIRouter, HTTPException
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency, entropy
from .upload import get_session_storage
from ...shared.session_manager import get_session_manager

router = APIRouter(prefix="/data-drift", tags=["Data Drift - Dashboard"])

def get_session_data(session_id: str):
    """Get session data with fallback logic: unified session first, then individual storage"""
    # Try unified session manager first (priority)
    unified_session_manager = get_session_manager()
    if unified_session_manager.session_exists(session_id):
        return unified_session_manager.get_data_drift_format(session_id)
    
    # Fallback to individual Data Drift session storage
    individual_storage = get_session_storage()
    if session_id in individual_storage:
        # Convert individual storage format to expected format
        individual_data = individual_storage[session_id]
        return {
            "reference_df": individual_data["reference"],
            "current_df": individual_data["current"],
            "reference_filename": individual_data.get("reference_filename", ""),
            "current_filename": individual_data.get("current_filename", ""),
            "reference_shape": individual_data.get("reference_shape", (0, 0)),
            "current_shape": individual_data.get("current_shape", (0, 0)),
            "common_columns": list(set(individual_data["reference"].columns) & set(individual_data["current"].columns)),
            "upload_timestamp": individual_data.get("upload_timestamp", "")
        }
    
    return None

def compute_kl_divergence(p, q):
    """Compute KL divergence between two probability distributions"""
    # Add small constant to avoid log(0)
    p = np.array(p) + 1e-10
    q = np.array(q) + 1e-10
    return entropy(p, q)

@router.get("/dashboard/{session_id}")
async def get_drift_dashboard(session_id: str):
    """
    Get drift dashboard analysis for uploaded datasets
    
    Args:
        session_id: Session identifier for uploaded data
        
    Returns:
        Dashboard analysis results
    """
    try:
        # Get session data with fallback logic
        session_data = get_session_data(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found. Please upload data first.")
        
        reference_df = session_data["reference_df"]
        current_df = session_data["current_df"]

        drifted_features = []
        feature_analysis_list = []

        # Define bins for numeric columns (using same bins for ref and curr)
        numeric_bins = {
            col: np.histogram_bin_edges(reference_df[col].dropna(), bins='auto')
            for col in reference_df.select_dtypes(include=["int64", "float64"]).columns
        }

        for col in reference_df.columns:
            if col not in current_df.columns:
                continue  # Skip columns that don't exist in current dataset
                
            if reference_df[col].dtype in ["int64", "float64"]:
                # KS test for drift
                ref_vals = reference_df[col].dropna()
                curr_vals = current_df[col].dropna()
                
                if len(ref_vals) == 0 or len(curr_vals) == 0:
                    continue
                    
                stat, p_value = ks_2samp(ref_vals, curr_vals)
                drift_status = "high" if p_value < 0.01 else "medium" if p_value < 0.05 else "low"
                drift_score = abs(stat) * 5

                # Compute histograms for ref and current with same bins
                bins = numeric_bins[col]
                ref_hist, _ = np.histogram(ref_vals, bins=bins, density=True)
                curr_hist, _ = np.histogram(curr_vals, bins=bins, density=True)

                kl_divergence = compute_kl_divergence(ref_hist, curr_hist)

                # Convert histograms to list of frequencies in percentages
                distribution_ref = (ref_hist * 100).tolist()
                distribution_current = (curr_hist * 100).tolist()

                # Create labels for bins like "640-660"
                bin_labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]

                feature_analysis_list.append({
                    "feature": col,
                    "drift_score": drift_score,
                    "kl_divergence": round(kl_divergence, 3),
                    "status": drift_status,
                    "p_value": p_value,
                    "distribution_ref": distribution_ref,
                    "distribution_current": distribution_current,
                    "bin_labels": bin_labels
                })

            else:
                # Categorical variables
                try:
                    # Create contingency table
                    crosstab = pd.crosstab(reference_df[col], current_df[col])
                    chi2, p_value, _, _ = chi2_contingency(crosstab)
                    drift_status = "high" if p_value < 0.01 else "medium" if p_value < 0.05 else "low"
                    drift_score = chi2 / 10

                    distribution_ref = reference_df[col].value_counts().to_dict()
                    distribution_current = current_df[col].value_counts().to_dict()

                    feature_analysis_list.append({
                        "feature": col,
                        "drift_score": drift_score,
                        "status": drift_status,
                        "p_value": p_value,
                        "distribution_ref": distribution_ref,
                        "distribution_current": distribution_current
                    })
                except Exception as e:
                    # Handle cases where categorical crosstab fails
                    continue

            if drift_status in ["high", "medium"]:
                drifted_features.append(col)

        if len(feature_analysis_list) == 0:
            raise HTTPException(status_code=400, detail="No features could be analyzed")

        total_features = len(feature_analysis_list)
        high_drift_features = sum(1 for f in feature_analysis_list if f.get("status") == "high")
        medium_drift_features = sum(1 for f in feature_analysis_list if f.get("status") == "medium")
        overall_drift_score = sum(f["drift_score"] for f in feature_analysis_list) / total_features

        overall_status = (
            "high" if overall_drift_score > 3
            else "medium" if overall_drift_score > 1
            else "low"
        )

        top_features = ", ".join(drifted_features[:2]) if drifted_features else "no significant features"
        executive_summary = (
            f"Analysis shows {overall_status}-level drift primarily driven by {top_features}. "
            "The model performance may be impacted and retraining should be considered within the next quarter."
        )

        data_quality_score = reference_df.notnull().mean().mean()

        return {
            "status": "success",
            "data": {
                "high_drift_features": high_drift_features,
                "medium_drift_features": medium_drift_features,
                "data_quality_score": float(data_quality_score),
                "total_features": total_features,
                "overall_drift_score": round(overall_drift_score, 2),
                "executive_summary": executive_summary,
                "overall_status": overall_status,
                "analysis_timestamp": datetime.utcnow().strftime("%d/%m/%Y"),
                "feature_analysis": feature_analysis_list,
                "recommendations": [
                    "Monitor features with high drift closely",
                    "Retrain model if drift persists",
                ]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dashboard analysis failed: {str(e)}")
