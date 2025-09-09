import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException
from scipy.stats import ks_2samp, entropy, chi2_contingency
from datetime import datetime
from .upload import get_session_storage
from ...shared.session_manager import get_session_manager

router = APIRouter(
    prefix="/data-drift",
    tags=["Data Drift - Statistical Reports"]
)

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

def psi(ref, curr, bins=10):
    """Population Stability Index (simplified)"""
    try:
        if len(ref) == 0 or len(curr) == 0:
            return 0.0
        ref_hist, bin_edges = np.histogram(ref, bins=bins)
        curr_hist, _ = np.histogram(curr, bins=bin_edges)
        ref_pct = ref_hist / max(np.sum(ref_hist), 1)
        curr_pct = curr_hist / max(np.sum(curr_hist), 1)
        curr_pct = np.where(curr_pct == 0, 0.0001, curr_pct)
        ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
        return float(np.sum((ref_pct - curr_pct) * np.log(ref_pct / curr_pct)))
    except:
        return 0.0

@router.get("/statistical-reports/{session_id}")
async def get_statistical_reports(session_id: str):
    """
    Get statistical reports analysis for uploaded datasets
    
    Args:
        session_id: Session identifier for uploaded data
        
    Returns:
        Statistical reports analysis results
    """
    try:
        # Get session data with fallback logic
        session_data = get_session_data(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found. Please upload data first.")
        
        reference_df = session_data["reference_df"]
        current_df = session_data["current_df"]

        feature_analysis_list = []
        ks_tests = []
        chi_tests = []
        total_drift_score = 0

        # Only analyze common columns
        common_columns = list(set(reference_df.columns) & set(current_df.columns))
        
        if not common_columns:
            raise HTTPException(status_code=400, detail="No common columns found between datasets")

        for col in common_columns:
            dtype = "numerical" if reference_df[col].dtype in ["int64", "float64"] else "categorical"
            missing_ref = int(reference_df[col].isna().sum())
            missing_curr = int(current_df[col].isna().sum())
            drift_score = 0.0
            kl_divergence = 0.0
            psi_value = 0.0
            ks_statistic = 0.0
            p_value = 1.0

            if dtype == "numerical":
                try:
                    ref_vals = reference_df[col].dropna()
                    curr_vals = current_df[col].dropna()
                    
                    if len(ref_vals) == 0 or len(curr_vals) == 0:
                        continue
                        
                    # Safer drift score calculation
                    ref_std = ref_vals.std()
                    if ref_std > 0:
                        drift_score = abs(ref_vals.mean() - curr_vals.mean()) / ref_std
                    else:
                        drift_score = 0.0
                    
                    # Safer KL divergence calculation
                    try:
                        ref_hist, bins = np.histogram(ref_vals, bins=10, density=True)
                        curr_hist, _ = np.histogram(curr_vals, bins=bins, density=True)
                        kl_divergence = float(entropy(ref_hist + 1e-6, curr_hist + 1e-6))
                    except:
                        kl_divergence = 0.0
                    
                    psi_value = psi(ref_vals, curr_vals)
                    ks_statistic, p_value = ks_2samp(ref_vals, curr_vals)
                    
                    ks_tests.append({
                        "feature": col,
                        "ks_statistic": float(ks_statistic),
                        "p_value": float(p_value),
                        "result": "Significant" if p_value < 0.05 else "Not Significant"
                    })
                    
                    feature_stats = {
                        "feature": col,
                        "data_type": dtype,
                        "ref_mean": float(ref_vals.mean()),
                        "ref_std": float(ref_vals.std()),
                        "ref_min": float(ref_vals.min()),
                        "ref_max": float(ref_vals.max()),
                        "curr_mean": float(curr_vals.mean()),
                        "curr_std": float(curr_vals.std()),
                        "curr_min": float(curr_vals.min()),
                        "curr_max": float(curr_vals.max()),
                        "missing_values_ref": missing_ref,
                        "missing_values_current": missing_curr,
                        "drift_score": float(drift_score),
                        "kl_divergence": float(kl_divergence),
                        "psi": float(psi_value),
                        "ks_statistic": float(ks_statistic),
                        "p_value": float(p_value),
                        "status": "low" if drift_score < 0.5 else "medium" if drift_score < 1.5 else "high"
                    }
                except Exception as e:
                    # Skip problematic numerical features
                    continue

            else:
                # Categorical features
                try:
                    ref_counts = reference_df[col].value_counts(normalize=True).to_dict()
                    curr_counts = current_df[col].value_counts(normalize=True).to_dict()
                    all_keys = set(ref_counts.keys()).union(curr_counts.keys())
                    drift_score = sum(abs(ref_counts.get(k,0) - curr_counts.get(k,0)) for k in all_keys)
                    
                    # Chi-Square test - safer implementation
                    categories = list(all_keys)
                    ref_vals = [reference_df[col].value_counts().get(k,0) for k in categories]
                    curr_vals = [current_df[col].value_counts().get(k,0) for k in categories]
                    
                    if sum(ref_vals) > 0 and sum(curr_vals) > 0 and len(categories) > 1:
                        chi2_stat, chi_p, _, _ = chi2_contingency([ref_vals, curr_vals])
                        chi_tests.append({
                            "feature": col,
                            "chi_square": float(chi2_stat),
                            "p_value": float(chi_p),
                            "result": "Significant" if chi_p < 0.05 else "Not Significant"
                        })
                    
                    feature_stats = {
                        "feature": col,
                        "data_type": dtype,
                        "ref_counts": {str(k): float(v) for k,v in ref_counts.items()},
                        "curr_counts": {str(k): float(v) for k,v in curr_counts.items()},
                        "missing_values_ref": missing_ref,
                        "missing_values_current": missing_curr,
                        "drift_score": float(drift_score),
                        "kl_divergence": float(kl_divergence),
                        "psi": float(psi_value),
                        "ks_statistic": float(ks_statistic),
                        "p_value": float(p_value),
                        "status": "low" if drift_score < 0.1 else "medium" if drift_score < 0.3 else "high"
                    }
                except Exception as e:
                    # Skip problematic categorical features
                    continue

            total_drift_score += drift_score
            feature_analysis_list.append(feature_stats)

        if len(feature_analysis_list) == 0:
            raise HTTPException(status_code=400, detail="No features could be analyzed")

        overall_drift_score = total_drift_score / len(feature_analysis_list)
        overall_status = "low" if overall_drift_score < 0.5 else "medium" if overall_drift_score < 1.5 else "high"
        
        # Safer data quality score calculation
        total_cells = current_df.shape[0] * current_df.shape[1]
        missing_cells = sum(f.get("missing_values_current", 0) for f in feature_analysis_list)
        data_quality_score = 1 - (missing_cells / max(total_cells, 1))

        # Dynamic executive summary
        count_high = sum(1 for f in feature_analysis_list if f["status"]=="high")
        count_medium = sum(1 for f in feature_analysis_list if f["status"]=="medium")
        count_low = sum(1 for f in feature_analysis_list if f["status"]=="low")
        executive_summary = (
            f"Analyzed {len(feature_analysis_list)} features: "
            f"{count_high} high drift, {count_medium} medium drift, {count_low} low drift. "
            f"Overall drift status: {overall_status.upper()}."
        )

        # Correlation analysis - safer implementation
        correlation_analysis = []
        numerical_cols = [c for c in common_columns if reference_df[c].dtype in ["int64","float64"] and current_df[c].dtype in ["int64","float64"]]
        for i in range(len(numerical_cols)):
            for j in range(i+1, min(len(numerical_cols), i+21)):  # Limit to prevent too many correlations
                f1, f2 = numerical_cols[i], numerical_cols[j]
                try:
                    ref_corr = reference_df[f1].corr(reference_df[f2])
                    curr_corr = current_df[f1].corr(current_df[f2])
                    if pd.notna(ref_corr) and pd.notna(curr_corr):
                        correlation_analysis.append({
                            "feature1": f1,
                            "feature2": f2,
                            "correlation": float(ref_corr),
                            "drift_correlation": float(curr_corr),
                            "correlation_change": float(curr_corr - ref_corr)
                        })
                except:
                    continue

        return {
            "status": "success",
            "data": {
                "feature_analysis": feature_analysis_list,
                "ks_tests": ks_tests,
                "chi_tests": chi_tests,
                "correlation_analysis": correlation_analysis,
                "total_features": len(feature_analysis_list),
                "overall_drift_score": round(overall_drift_score, 3),
                "overall_status": overall_status,
                "data_quality_score": round(data_quality_score, 3),
                "executive_summary": executive_summary,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "summary_stats": {
                    "high_drift_features": count_high,
                    "medium_drift_features": count_medium,
                    "low_drift_features": count_low,
                    "significant_ks_tests": len([t for t in ks_tests if t["result"] == "Significant"]),
                    "significant_chi_tests": len([t for t in chi_tests if t["result"] == "Significant"])
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Statistical reports analysis failed: {str(e)}")
