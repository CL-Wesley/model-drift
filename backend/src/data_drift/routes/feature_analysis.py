import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException
from scipy.stats import ks_2samp, chi2_contingency, entropy
from datetime import datetime
from .upload import get_session_storage
from ...shared.session_manager import get_session_manager

router = APIRouter(
    prefix="/data-drift",
    tags=["Data Drift - Feature Analysis"]
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

def calc_change(ref, curr):
    if ref == 0:
        return 0.0
    return round(((curr - ref) / ref) * 100, 1)

def population_stability_index(ref, curr, bins=10):
    """Calculate PSI between two distributions"""
    try:
        if len(ref) == 0 or len(curr) == 0:
            return 0.0
        ref_percents, _ = np.histogram(ref, bins=bins)
        curr_percents, _ = np.histogram(curr, bins=bins)

        ref_percents = ref_percents / max(len(ref), 1)
        curr_percents = curr_percents / max(len(curr), 1)

        psi_vals = []
        for i in range(len(ref_percents)):
            if ref_percents[i] == 0 or curr_percents[i] == 0:
                continue
            psi_vals.append((ref_percents[i] - curr_percents[i]) *
                            np.log(ref_percents[i] / curr_percents[i]))

        return round(np.sum(psi_vals), 4)
    except:
        return 0.0

@router.get("/feature-analysis/{session_id}")
async def get_feature_analysis(session_id: str):
    """
    Get feature analysis for uploaded datasets
    
    Args:
        session_id: Session identifier for uploaded data
        
    Returns:
        Feature analysis results
    """
    try:
        # Use the new session data retrieval with fallback logic
        session_data = get_session_data(session_id)
        if session_data is None:
            raise HTTPException(status_code=404, detail="Session not found. Please upload data first.")
        
        # Extract DataFrames from session data
        reference_df = session_data["reference_df"]
        curr_df = session_data["current_df"]

        feature_analysis_list = []

        # Only analyze common columns
        common_columns = session_data.get("common_columns", 
            list(set(reference_df.columns) & set(curr_df.columns))
        )
        
        if not common_columns:
            raise HTTPException(status_code=400, detail="No common columns found between datasets")

        # Rest of your analysis code remains the same...
        for col in common_columns:
            dtype = "numerical" if reference_df[col].dtype in ["int64", "float64"] else "categorical"

            missing_ref = int(reference_df[col].isna().sum())
            missing_curr = int(curr_df[col].isna().sum())

            drift_score = 0.0
            p_value = 1.0
            ks_stat = 0.0
            kl_div = None
            psi_val = None
            histogram_ref = None
            histogram_current = None

            if dtype == "numerical":
                try:
                    ref_vals = reference_df[col].dropna()
                    curr_vals = curr_df[col].dropna()

                    if len(ref_vals) == 0 or len(curr_vals) == 0:
                        continue

                    ks_stat, p_value = ks_2samp(ref_vals, curr_vals)
                    drift_score = float(abs(ks_stat) * 5)

                    # Safe KL divergence calculation
                    try:
                        ref_hist_density, bins = np.histogram(ref_vals, bins=10, density=True)
                        curr_hist_density, _ = np.histogram(curr_vals, bins=bins, density=True)
                        kl_div = float(entropy(ref_hist_density + 1e-8, curr_hist_density + 1e-8))
                    except:
                        kl_div = 0.0

                    psi_val = float(population_stability_index(ref_vals, curr_vals))

                    # Create histogram data
                    try:
                        ref_hist, bins = np.histogram(ref_vals, bins=10)
                        curr_hist, _ = np.histogram(curr_vals, bins=bins)
                        histogram_ref = [{"bin": float(bins[i]), "count": int(ref_hist[i])} for i in range(len(ref_hist))]
                        histogram_current = [{"bin": float(bins[i]), "count": int(curr_hist[i])} for i in range(len(curr_hist))]
                    except:
                        histogram_ref = []
                        histogram_current = []

                    distribution_ref = {
                        "mean": float(ref_vals.mean()),
                        "std": float(ref_vals.std()),
                        "min": float(ref_vals.min()),
                        "max": float(ref_vals.max()),
                        "q25": float(ref_vals.quantile(0.25)),
                        "q50": float(ref_vals.quantile(0.5)),
                        "q75": float(ref_vals.quantile(0.75)),
                    }
                    distribution_curr = {
                        "mean": float(curr_vals.mean()),
                        "std": float(curr_vals.std()),
                        "min": float(curr_vals.min()),
                        "max": float(curr_vals.max()),
                        "q25": float(curr_vals.quantile(0.25)),
                        "q50": float(curr_vals.quantile(0.5)),
                        "q75": float(curr_vals.quantile(0.75)),
                    }

                    summary_stats = {
                        "mean": {
                            "reference": float(ref_vals.mean()),
                            "current": float(curr_vals.mean()),
                            "change": calc_change(ref_vals.mean(), curr_vals.mean()),
                        },
                        "std": {
                            "reference": float(ref_vals.std()),
                            "current": float(curr_vals.std()),
                            "change": calc_change(ref_vals.std(), curr_vals.std()),
                        },
                        "min": {
                            "reference": float(ref_vals.min()),
                            "current": float(curr_vals.min()),
                            "change": calc_change(ref_vals.min(), curr_vals.min()),
                        },
                        "max": {
                            "reference": float(ref_vals.max()),
                            "current": float(curr_vals.max()),
                            "change": calc_change(ref_vals.max(), curr_vals.max()),
                        },
                        "q25": {
                            "reference": float(ref_vals.quantile(0.25)),
                            "current": float(curr_vals.quantile(0.25)),
                            "change": calc_change(ref_vals.quantile(0.25), curr_vals.quantile(0.25)),
                        },
                        "q50": {
                            "reference": float(ref_vals.quantile(0.5)),
                            "current": float(curr_vals.quantile(0.5)),
                            "change": calc_change(ref_vals.quantile(0.5), curr_vals.quantile(0.5)),
                        },
                        "q75": {
                            "reference": float(ref_vals.quantile(0.75)),
                            "current": float(curr_vals.quantile(0.75)),
                            "change": calc_change(ref_vals.quantile(0.75), curr_vals.quantile(0.75)),
                        },
                    }
                except Exception as e:
                    # Skip problematic numerical features
                    continue

            else:
                # Categorical features
                try:
                    crosstab = pd.crosstab(reference_df[col], curr_df[col])
                    chi2, p_value, _, _ = chi2_contingency(crosstab)
                    drift_score = float(chi2 / 10)
                    distribution_ref = {
                        "counts": {str(k): int(v) for k, v in reference_df[col].value_counts().to_dict().items()}
                    }
                    distribution_curr = {
                        "counts": {str(k): int(v) for k, v in curr_df[col].value_counts().to_dict().items()}
                    }
                    summary_stats = {}
                except Exception as e:
                    # Skip problematic categorical features
                    continue

            drift_status = "high" if p_value < 0.01 else "medium" if p_value < 0.05 else "low"

            feature_analysis_list.append({
                "feature": col,
                "status": drift_status,
                "data_type": dtype,

                "drift_score": float(drift_score),
                "kl_divergence": kl_div,
                "psi": psi_val,
                "ks_statistic": float(ks_stat),
                "p_value": float(p_value),

                "histogram_ref": histogram_ref,
                "histogram_current": histogram_current,

                "missing_values_ref": missing_ref,
                "missing_values_current": missing_curr,
                "data_type_label": "Continuous" if dtype == "numerical" else "Categorical",
                "statistical_test": "Significant" if p_value < 0.05 else "Not Significant",

                "summary_stats": summary_stats,
                "distribution_ref": distribution_ref,
                "distribution_current": distribution_curr,
            })

        if len(feature_analysis_list) == 0:
            raise HTTPException(status_code=400, detail="No features could be analyzed")

        return {
            "status": "success",
            "data": {
                "features": feature_analysis_list,
                "analysis_timestamp": datetime.utcnow().isoformat(),
            },
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature analysis failed: {str(e)}")