import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException
from scipy.stats import ks_2samp, entropy, chi2_contingency
from datetime import datetime
import numpy as np
from ...shared.ai_explanation_service import ai_explanation_service
from ...shared.models import AnalysisRequest
from ...shared.s3_utils import load_s3_csv, validate_dataframe

router = APIRouter(
    prefix="/data-drift",
    tags=["Data Drift - Statistical Reports"]
)


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

def create_ai_summary_for_statistical_analysis(analysis_data: dict) -> dict:
    """
    Summarizes the detailed statistical analysis into a compact format suitable for an LLM prompt.
    """
    summary = {
        "total_features": analysis_data.get("total_features", 0),
        "overall_drift_score": analysis_data.get("overall_drift_score"),
        "overall_status": analysis_data.get("overall_status"),
        "data_quality_score": analysis_data.get("data_quality_score"),
        "executive_summary": analysis_data.get("executive_summary", "")
    }

    # Add summary statistics instead of full feature arrays
    summary_stats = analysis_data.get("summary_stats", {})
    summary.update({
        "high_drift_features": summary_stats.get("high_drift_features", 0),
        "medium_drift_features": summary_stats.get("medium_drift_features", 0), 
        "low_drift_features": summary_stats.get("low_drift_features", 0),
        "significant_ks_tests": summary_stats.get("significant_ks_tests", 0),
        "significant_chi_tests": summary_stats.get("significant_chi_tests", 0)
    })

    # Add top 5 most drifted features only
    feature_analysis = analysis_data.get("feature_analysis", [])
    sorted_features = sorted(feature_analysis, key=lambda x: x.get('drift_score', 0), reverse=True)
    
    top_drifted_features = []
    for feature in sorted_features[:5]:
        top_drifted_features.append({
            "feature": feature.get("feature"),
            "data_type": feature.get("data_type"),
            "drift_score": round(feature.get("drift_score", 0), 3),
            "status": feature.get("status"),
            "p_value": round(feature.get("p_value", 1), 4),
            "ks_statistic": round(feature.get("ks_statistic", 0), 3)
        })
    
    summary["top_drifted_features"] = top_drifted_features
    
    return summary

@router.post("/statistical-reports")
async def get_statistical_reports(request: AnalysisRequest):
    """
    Get statistical reports analysis for datasets loaded from S3
    
    Args:
        request: AnalysisRequest containing S3 URLs and configuration
        
    Returns:
        Statistical reports analysis results
    """
    try:
        # Load data from S3 URLs
        reference_df = load_s3_csv(request.reference_url)
        current_df = load_s3_csv(request.current_url)
        
        # Validate datasets
        validate_dataframe(reference_df, "Reference")
        validate_dataframe(current_df, "Current")

        feature_analysis_list = []
        ks_tests = []
        chi_tests = []

        # Only analyze common columns
        common_columns = list(set(reference_df.columns) & set(current_df.columns))
        
        if not common_columns:
            raise HTTPException(status_code=400, detail="No common columns found between datasets")

        for col in common_columns:
            try:
                ref_series = reference_df[col].dropna()
                curr_series = current_df[col].dropna()
                
                # Skip if insufficient data
                if len(ref_series) == 0 or len(curr_series) == 0:
                    continue
                
                dtype = "numerical" if reference_df[col].dtype in ["int64", "float64"] else "categorical"
                missing_ref = int(reference_df[col].isna().sum())
                missing_curr = int(current_df[col].isna().sum())
                
                if dtype == "numerical":
                    # Use same approach as dashboard - KS test with p-value based severity
                    ks_stat, p_value = ks_2samp(ref_series, curr_series)
                    
                    # Unified severity classification
                    if p_value < 0.01:
                        status = "high"
                    elif p_value < 0.05:
                        status = "medium"
                    else:
                        status = "low"
                    
                    drift_score = abs(ks_stat) * 5  # Same scaling as dashboard
                    
                    feature_stats = {
                        "feature": col,
                        "data_type": dtype,
                        "ref_mean": float(ref_series.mean()),
                        "ref_std": float(ref_series.std()),
                        "ref_min": float(ref_series.min()),
                        "ref_max": float(ref_series.max()),
                        "curr_mean": float(curr_series.mean()),
                        "curr_std": float(curr_series.std()),
                        "curr_min": float(curr_series.min()),
                        "curr_max": float(curr_series.max()),
                        "missing_values_ref": missing_ref,
                        "missing_values_current": missing_curr,
                        "drift_score": drift_score,
                        "ks_statistic": float(ks_stat),
                        "p_value": float(p_value),
                        "status": status
                    }
                    
                    ks_tests.append({
                        "feature": col,
                        "ks_statistic": float(ks_stat),
                        "p_value": float(p_value),
                        "result": "Significant" if p_value < 0.05 else "Not Significant"
                    })
                
                else:  # categorical
                    # Use same approach as dashboard - Chi-square with p-value based severity
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
                    
                    # Unified severity classification
                    if p_value < 0.01:
                        status = "high"
                    elif p_value < 0.05:
                        status = "medium"
                    else:
                        status = "low"
                    
                    drift_score = chi2_stat / 10  # Same scaling as dashboard
                    
                    feature_stats = {
                        "feature": col,
                        "data_type": dtype,
                        "ref_unique_values": len(ref_counts),
                        "curr_unique_values": len(curr_counts),
                        "ref_mode": ref_series.mode().iloc[0] if len(ref_series.mode()) > 0 else None,
                        "curr_mode": curr_series.mode().iloc[0] if len(curr_series.mode()) > 0 else None,
                        "missing_values_ref": missing_ref,
                        "missing_values_current": missing_curr,
                        "drift_score": drift_score,
                        "chi2_statistic": float(chi2_stat),
                        "p_value": float(p_value),
                        "status": status
                    }
                    
                    chi_tests.append({
                        "feature": col,
                        "chi2_statistic": float(chi2_stat),
                        "p_value": float(p_value),
                        "result": "Significant" if p_value < 0.05 else "Not Significant"
                    })
                
                feature_analysis_list.append(feature_stats)
                
            except Exception as e:
                # Skip problematic features but log for debugging
                print(f"Warning: Could not analyze feature {col}: {e}")
                continue

        if len(feature_analysis_list) == 0:
            raise HTTPException(status_code=400, detail="No features could be analyzed")

        # Use same overall calculation approach as dashboard
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
        
        # Safer data quality score calculation
        total_cells = current_df.shape[0] * current_df.shape[1]
        missing_cells = sum(f.get("missing_values_current", 0) for f in feature_analysis_list)
        data_quality_score = 1 - (missing_cells / max(total_cells, 1))

        # Dynamic executive summary using consistent metrics
        count_high = high_drift_features
        count_medium = medium_drift_features  
        count_low = low_drift_features
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

        result = {
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

        # Generate AI explanation for the analysis results
        try:
            ai_summary_payload = create_ai_summary_for_statistical_analysis(result["data"])
            ai_explanation = ai_explanation_service.generate_explanation(
                analysis_data=ai_summary_payload, 
                analysis_type="statistical_analysis"
            )
            result["llm_response"] = ai_explanation
        except Exception as e:
            print(f"Warning: AI explanation failed: {e}")
            # Continue without AI explanation
            result["llm_response"] = {
                "summary": "Statistical drift analysis completed successfully.",
                "detailed_explanation": "The statistical analysis has been completed using various drift detection methods including KS tests and chi-square tests. AI explanations are temporarily unavailable.",
                "key_takeaways": [
                    "Statistical analysis completed",
                    "Review drift metrics for insights", 
                    "AI explanations will return when service is restored"
                ]
            }

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Statistical reports analysis failed: {str(e)}")
