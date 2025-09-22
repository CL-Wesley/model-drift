from fastapi import APIRouter, HTTPException
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency, entropy
from .upload import get_session_storage
from ...shared.session_manager import get_session_manager
from ...shared.ai_explanation_service import ai_explanation_service

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

def create_ai_summary_for_dashboard(analysis_data: dict) -> dict:
    """
    Summarizes the detailed dashboard analysis into a compact format suitable for an LLM prompt.
    This prevents token limit issues by sending only high-level insights instead of raw data.
    """
    # Extract top-level KPIs
    summary = {
        "overall_status": analysis_data.get("overall_status"),
        "overall_drift_score": round(analysis_data.get("overall_drift_score", 0), 2),
        "total_features_analyzed": analysis_data.get("total_features"),
        "high_drift_features_count": analysis_data.get("high_drift_features"),
        "medium_drift_features_count": analysis_data.get("medium_drift_features"),
        "data_quality_score": round(analysis_data.get("data_quality_score", 0), 2),
        "executive_summary": analysis_data.get("executive_summary", "")
    }

    # Extract info for ONLY the top N most drifted features
    # This is the most important step to reduce token count
    feature_analysis = analysis_data.get("feature_analysis", [])
    
    # Sort features by drift score to find the most impactful ones
    sorted_features = sorted(feature_analysis, key=lambda x: x.get('drift_score', 0), reverse=True)
    
    top_n = 5  # Limit to top 5 features to keep prompt manageable
    top_drifted_features_summary = []

    for feature in sorted_features[:top_n]:
        feature_summary = {
            "feature_name": feature.get("feature"),
            "drift_status": feature.get("status"),
            "drift_score": round(feature.get("drift_score", 0), 2),
            "feature_type": feature.get("feature_type", "unknown")
        }
        
        # Add a simple change description without raw distribution data
        if feature.get("status") in ["high", "critical"]:
            if feature.get("feature_type") == "numerical":
                ref_mean = feature.get("ref_mean", 0)
                curr_mean = feature.get("curr_mean", 0)
                if ref_mean != 0:
                    pct_change = round(((curr_mean - ref_mean) / ref_mean) * 100, 1)
                    feature_summary["change_description"] = f"Mean changed by {pct_change}%"
            elif feature.get("feature_type") == "categorical":
                feature_summary["change_description"] = "Category distribution has shifted significantly"
        
        top_drifted_features_summary.append(feature_summary)

    summary["top_drifted_features"] = top_drifted_features_summary
    
    # Add recommendations from the original analysis
    summary["recommendations"] = analysis_data.get("recommendations", [])
    
    return summary

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

        # Only analyze common columns
        common_columns = list(set(reference_df.columns) & set(current_df.columns))
        
        if not common_columns:
            raise HTTPException(status_code=400, detail="No common columns found between datasets")

        for col in common_columns:
            if reference_df[col].dtype in ["int64", "float64"]:
                # Numerical features - use KS test
                ref_vals = reference_df[col].dropna()
                curr_vals = current_df[col].dropna()
                
                if len(ref_vals) == 0 or len(curr_vals) == 0:
                    continue
                    
                ks_stat, p_value = ks_2samp(ref_vals, curr_vals)
                
                # Unified severity classification based on p-value
                if p_value < 0.01:
                    drift_status = "high"
                elif p_value < 0.05:
                    drift_status = "medium"
                else:
                    drift_status = "low"
                
                drift_score = abs(ks_stat) * 5  # Scale for display

                # Compute histograms for visualization
                bins = np.histogram_bin_edges(ref_vals, bins='auto')
                ref_hist, _ = np.histogram(ref_vals, bins=bins, density=True)
                curr_hist, _ = np.histogram(curr_vals, bins=bins, density=True)
                
                distribution_ref = (ref_hist * 100).tolist()
                distribution_current = (curr_hist * 100).tolist()
                bin_labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]
                
                feature_analysis_list.append({
                    "feature": col,
                    "feature_type": "numerical",
                    "drift_score": drift_score,
                    "status": drift_status,
                    "p_value": p_value,
                    "ks_statistic": float(ks_stat),
                    "distribution_ref": distribution_ref,
                    "distribution_current": distribution_current,
                    "bin_labels": bin_labels
                })
                
            else:
                # Categorical features - use Chi-square test
                try:
                    ref_counts = reference_df[col].value_counts()
                    curr_counts = current_df[col].value_counts()
                    
                    # Align categories
                    all_cats = ref_counts.index.union(curr_counts.index)
                    ref_aligned = ref_counts.reindex(all_cats, fill_value=0)
                    curr_aligned = curr_counts.reindex(all_cats, fill_value=0)
                    
                    if len(all_cats) > 1 and (ref_aligned > 0).sum() > 0 and (curr_aligned > 0).sum() > 0:
                        contingency_table = np.array([ref_aligned.values, curr_aligned.values])
                        chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
                    else:
                        chi2_stat, p_value = 0.0, 1.0
                    
                    # Unified severity classification based on p-value
                    if p_value < 0.01:
                        drift_status = "high"
                    elif p_value < 0.05:
                        drift_status = "medium"
                    else:
                        drift_status = "low"
                        
                    drift_score = chi2_stat / 10  # Scale for display
                    
                    distribution_ref = ref_counts.to_dict()
                    distribution_current = curr_counts.to_dict()
                    
                    feature_analysis_list.append({
                        "feature": col,
                        "feature_type": "categorical", 
                        "drift_score": drift_score,
                        "status": drift_status,
                        "p_value": p_value,
                        "chi2_statistic": float(chi2_stat),
                        "distribution_ref": distribution_ref,
                        "distribution_current": distribution_current
                    })
                    
                except Exception as e:
                    continue
                    
            if drift_status in ["high", "medium"]:
                drifted_features.append(col)

        if len(feature_analysis_list) == 0:
            raise HTTPException(status_code=400, detail="No features could be analyzed")

        # Calculate overall metrics using consistent approach
        total_features = len(feature_analysis_list)
        high_drift_features = sum(1 for f in feature_analysis_list if f.get("status") == "high")
        medium_drift_features = sum(1 for f in feature_analysis_list if f.get("status") == "medium")
        low_drift_features = sum(1 for f in feature_analysis_list if f.get("status") == "low")
        
        # Overall drift score as average
        overall_drift_score = sum(f["drift_score"] for f in feature_analysis_list) / total_features
        
        # Overall status based on feature counts
        if high_drift_features > total_features * 0.3:  # >30% high drift features
            overall_status = "high"
        elif medium_drift_features + high_drift_features > total_features * 0.5:  # >50% medium+ drift
            overall_status = "medium"
        else:
            overall_status = "low"

        top_features = ", ".join(drifted_features[:2]) if drifted_features else "no significant features"
        executive_summary = (
            f"Analysis shows {overall_status}-level drift primarily driven by {top_features}. "
            "The model performance may be impacted and retraining should be considered within the next quarter."
        )

        data_quality_score = reference_df.notnull().mean().mean()

        result = {
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

        # Generate AI explanation for the dashboard analysis
        try:
            # *** NEW STEP: Create the summary FIRST ***
            ai_summary_payload = create_ai_summary_for_dashboard(result["data"])

            ai_explanation = ai_explanation_service.generate_explanation(
                # *** CHANGE: Send the summary, NOT the full result["data"] ***
                analysis_data=ai_summary_payload, 
                analysis_type="data_drift_dashboard"
            )
            result["llm_response"] = ai_explanation
        except Exception as e:
            print(f"Warning: AI explanation failed: {e}")
            # Continue without AI explanation
            result["llm_response"] = {
                "summary": "Data drift dashboard analysis completed successfully.",
                "detailed_explanation": "Your comprehensive data drift dashboard has been generated, showing drift patterns across all features. AI explanations are temporarily unavailable.",
                "key_takeaways": [
                    "Dashboard analysis completed successfully",
                    "Review drift scores and feature patterns",
                    "AI explanations will return when service is restored"
                ]
            }

        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dashboard analysis failed: {str(e)}")
