from fastapi import APIRouter, Query
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency, entropy
import os

router = APIRouter(prefix="/api/v1/data-drift", tags=["Data Drift - Dashboard"])

def compute_kl_divergence(p, q):
    # Add small constant to avoid log(0)
    p = np.array(p) + 1e-10
    q = np.array(q) + 1e-10
    return entropy(p, q)

@router.get("/dashboard")
async def get_drift_dashboard(
    reference_filename: str = Query(...),
    current_filename: str = Query(...)
):
    ref_path = os.path.join("uploads", reference_filename)
    curr_path = os.path.join("uploads", current_filename)
    
    ref_df = pd.read_csv(ref_path)
    curr_df = pd.read_csv(curr_path)

    drifted_features = []
    feature_analysis_list = []

    # Define bins for numeric columns (using same bins for ref and curr)
    numeric_bins = {
        col: np.histogram_bin_edges(ref_df[col].dropna(), bins='auto')
        for col in ref_df.select_dtypes(include=["int64", "float64"]).columns
    }

    for col in ref_df.columns:
        if ref_df[col].dtype in ["int64", "float64"]:
            # KS test for drift
            stat, p_value = ks_2samp(ref_df[col].dropna(), curr_df[col].dropna())
            drift_status = "high" if p_value < 0.01 else "medium" if p_value < 0.05 else "low"
            drift_score = abs(stat) * 5

            # Compute histograms for ref and current with same bins
            bins = numeric_bins[col]
            ref_hist, _ = np.histogram(ref_df[col].dropna(), bins=bins, density=True)
            curr_hist, _ = np.histogram(curr_df[col].dropna(), bins=bins, density=True)

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
            chi2, p_value, _, _ = chi2_contingency(pd.crosstab(ref_df[col], curr_df[col]))
            drift_status = "high" if p_value < 0.01 else "medium" if p_value < 0.05 else "low"
            drift_score = chi2 / 10

            distribution_ref = ref_df[col].value_counts().to_dict()
            distribution_current = curr_df[col].value_counts().to_dict()

            feature_analysis_list.append({
                "feature": col,
                "drift_score": drift_score,
                "status": drift_status,
                "p_value": p_value,
                "distribution_ref": distribution_ref,
                "distribution_current": distribution_current
            })

        if drift_status in ["high", "medium"]:
            drifted_features.append(col)

    total_features = len(ref_df.columns)
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

    data_quality_score = ref_df.notnull().mean().mean()

    return {
        "status": "success",
        "data": {
            "high_drift_features": high_drift_features,
            "medium_drift_features": medium_drift_features,
            "data_quality_score": data_quality_score,
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
