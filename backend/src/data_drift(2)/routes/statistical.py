import os
import pandas as pd
import numpy as np
from fastapi import APIRouter
from scipy.stats import ks_2samp, entropy, chi2_contingency
from datetime import datetime

router = APIRouter(
    prefix="/api/v1/data-drift",
    tags=["Data Drift - Statistical Reports"]
)

def psi(ref, curr, bins=10):
    """Population Stability Index (simplified)"""
    ref_hist, bin_edges = np.histogram(ref, bins=bins)
    curr_hist, _ = np.histogram(curr, bins=bin_edges)
    ref_pct = ref_hist / np.sum(ref_hist)
    curr_pct = curr_hist / np.sum(curr_hist)
    curr_pct = np.where(curr_pct == 0, 0.0001, curr_pct)
    ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
    return float(np.sum((ref_pct - curr_pct) * np.log(ref_pct / curr_pct)))

@router.get("/statistical-reports")
async def get_statistical_reports():
    ref_path = os.path.join("uploads", "latest_reference.csv")
    curr_path = os.path.join("uploads", "latest_current.csv")
    
    ref_df = pd.read_csv(ref_path)
    curr_df = pd.read_csv(curr_path)

    feature_analysis_list = []
    ks_tests = []
    chi_tests = []
    total_drift_score = 0

    for col in ref_df.columns:
        dtype = "numerical" if ref_df[col].dtype in ["int64", "float64"] else "categorical"
        missing_ref = int(ref_df[col].isna().sum())
        missing_curr = int(curr_df[col].isna().sum())
        drift_score = 0.0
        kl_divergence = 0.0
        psi_value = 0.0
        ks_statistic = 0.0
        p_value = 1.0

        if dtype == "numerical":
            drift_score = abs(ref_df[col].mean() - curr_df[col].mean()) / (ref_df[col].std() + 1e-6)
            kl_divergence = float(entropy(
                np.histogram(ref_df[col], bins=10, density=True)[0] + 1e-6,
                np.histogram(curr_df[col], bins=10, density=True)[0] + 1e-6
            ))
            psi_value = psi(ref_df[col].dropna(), curr_df[col].dropna())
            ks_statistic, p_value = ks_2samp(ref_df[col].dropna(), curr_df[col].dropna())
            
            ks_tests.append({
                "feature": col,
                "ks_statistic": float(ks_statistic),
                "p_value": float(p_value),
                "result": "Significant" if p_value < 0.05 else "Not Significant"
            })
            
            feature_stats = {
                "feature": col,
                "data_type": dtype,
                "ref_mean": float(ref_df[col].mean()),
                "ref_std": float(ref_df[col].std()),
                "ref_min": float(ref_df[col].min()),
                "ref_max": float(ref_df[col].max()),
                "curr_mean": float(curr_df[col].mean()),
                "curr_std": float(curr_df[col].std()),
                "curr_min": float(curr_df[col].min()),
                "curr_max": float(curr_df[col].max()),
                "missing_values_ref": missing_ref,
                "missing_values_current": missing_curr,
                "drift_score": float(drift_score),
                "kl_divergence": float(kl_divergence),
                "psi": float(psi_value),
                "ks_statistic": float(ks_statistic),
                "p_value": float(p_value),
                "status": "low" if drift_score < 0.5 else "medium" if drift_score < 1.5 else "high"
            }

        else:
            ref_counts = ref_df[col].value_counts(normalize=True).to_dict()
            curr_counts = curr_df[col].value_counts(normalize=True).to_dict()
            all_keys = set(ref_counts.keys()).union(curr_counts.keys())
            drift_score = sum(abs(ref_counts.get(k,0) - curr_counts.get(k,0)) for k in all_keys)
            
            # Chi-Square test
            categories = list(all_keys)
            ref_vals = [ref_counts.get(k,0) for k in categories]
            curr_vals = [curr_counts.get(k,0) for k in categories]
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

        total_drift_score += drift_score
        feature_analysis_list.append(feature_stats)

    overall_drift_score = total_drift_score / len(ref_df.columns)
    overall_status = "low" if overall_drift_score < 0.5 else "medium" if overall_drift_score < 1.5 else "high"
    data_quality_score = 1 - sum(f["missing_values_current"] for f in feature_analysis_list) / (curr_df.shape[0]*curr_df.shape[1])

    # Fully dynamic executive summary
    count_high = sum(1 for f in feature_analysis_list if f["status"]=="high")
    count_medium = sum(1 for f in feature_analysis_list if f["status"]=="medium")
    count_low = sum(1 for f in feature_analysis_list if f["status"]=="low")
    executive_summary = (
        f"Analyzed {len(ref_df.columns)} features: "
        f"{count_high} high drift, {count_medium} medium drift, {count_low} low drift. "
        f"Overall drift status: {overall_status.upper()}."
    )

    # Correlation analysis
    correlation_analysis = []
    numerical_cols = [c for c in ref_df.columns if ref_df[c].dtype in ["int64","float64"]]
    for i in range(len(numerical_cols)):
        for j in range(i+1, len(numerical_cols)):
            f1, f2 = numerical_cols[i], numerical_cols[j]
            correlation_analysis.append({
                "feature1": f1,
                "feature2": f2,
                "correlation": float(ref_df[f1].corr(ref_df[f2])),
                "drift_correlation": float(curr_df[f1].corr(curr_df[f2]))
            })

    return {
        "status": "success",
        "data": {
            "feature_analysis": feature_analysis_list,
            "ks_tests": ks_tests,
            "chi_tests": chi_tests,
            "correlation_analysis": correlation_analysis,
            "total_features": len(ref_df.columns),
            "overall_drift_score": overall_drift_score,
            "overall_status": overall_status,
            "data_quality_score": data_quality_score,
            "executive_summary": executive_summary,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
    }
