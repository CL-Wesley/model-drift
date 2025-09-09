import pandas as pd
from fastapi import APIRouter, HTTPException
from datetime import datetime
from scipy.stats import chi2_contingency, ks_2samp
import numpy as np
import json
import tempfile
import os
from .upload import get_session_storage
from ...shared.session_manager import get_session_manager

router = APIRouter(
    prefix="/data-drift/class-imbalance",
    tags=["Data Drift - Class Imbalance"]
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

def gini(counts):
    array = np.array(list(counts.values()))
    if array.sum() == 0:
        return 0.0
    array = np.sort(array)
    n = array.size
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * array)) / (n * array.sum()))

def shannon_entropy(counts):
    probs = np.array(list(counts.values())) / sum(counts.values())
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))

def effective_number_of_classes(counts):
    probs = np.array(list(counts.values())) / sum(counts.values())
    return float(1.0 / np.sum(probs ** 2))

def class_balance_index(counts):
    probs = np.array(list(counts.values())) / sum(counts.values())
    n_classes = len(counts)
    return float((np.prod(probs ** probs)) * n_classes) if n_classes > 1 else 1.0

@router.get("/analysis/{session_id}")
async def get_class_imbalance_analysis(session_id: str):
    """
    Get class imbalance analysis for uploaded datasets
    
    Args:
        session_id: Session identifier for uploaded data
        
    Returns:
        Class imbalance analysis results
    """
    try:
        # Get session data with fallback logic
        session_data = get_session_data(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found. Please upload data first.")
        
        reference_df = session_data["reference_df"]
        current_df = session_data["current_df"]

        # Use the last column as target column (assumes target is last column)
        target_col = reference_df.columns[-1]

        if target_col not in current_df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_col}' not found in current dataset")

        # Class counts
        class_counts_ref = reference_df[target_col].value_counts().to_dict()
        class_counts_curr = current_df[target_col].value_counts().to_dict()

        total_ref = int(reference_df.shape[0])
        total_curr = int(current_df.shape[0])

        class_percent_ref = {k: round(v / total_ref * 100, 2) for k, v in class_counts_ref.items()}
        class_percent_curr = {k: round(v / total_curr * 100, 2) for k, v in class_counts_curr.items()}

        # Imbalance Ratio - handle division by zero
        max_count = max(class_counts_curr.values()) if class_counts_curr else 1
        min_count = min(class_counts_curr.values()) if class_counts_curr else 1
        imbalance_ratio = float(max_count / max(min_count, 1))  # Prevent division by zero

        # Severity Level
        if imbalance_ratio < 2:
            severity = "Low"
        elif imbalance_ratio < 5:
            severity = "Medium"
        else:
            severity = "High"

        # Gini Coefficient
        gini_ref = gini(class_counts_ref)
        gini_curr = gini(class_counts_curr)

        # Shannon Entropy
        entropy_ref = shannon_entropy(class_counts_ref)
        entropy_curr = shannon_entropy(class_counts_curr)

        # Effective Number of Classes
        enc_ref = effective_number_of_classes(class_counts_ref)
        enc_curr = effective_number_of_classes(class_counts_curr)

        # Class Balance Index
        cbi_ref = class_balance_index(class_counts_ref)
        cbi_curr = class_balance_index(class_counts_curr)

        # Chi-square test for class distribution
        all_classes = list(set(class_counts_ref.keys()).union(set(class_counts_curr.keys())))
        ref_vals = [class_counts_ref.get(c, 0) for c in all_classes]
        curr_vals = [class_counts_curr.get(c, 0) for c in all_classes]
        
        try:
            chi2_stat, p_value, dof, _ = chi2_contingency([ref_vals, curr_vals])
            chi2_stat = float(chi2_stat)
            p_value = float(p_value)
            dof = int(dof)
            chi_significance = "Highly Significant" if p_value < 0.01 else ("Significant" if p_value < 0.05 else "Not Significant")
        except Exception as e:
            chi2_stat, p_value, dof, chi_significance = 0.0, 1.0, 0, "Not Significant"

        # KS test for numeric columns (excluding target)
        ks_results = {}
        numeric_cols = reference_df.select_dtypes(include=np.number).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        for col in numeric_cols:
            if col in current_df.columns:
                try:
                    ks_stat, ks_p = ks_2samp(reference_df[col].dropna(), current_df[col].dropna())
                    ks_results[col] = {
                        "ks_statistic": float(ks_stat),
                        "p_value": float(ks_p),
                        "interpretation": "Significant" if ks_p < 0.05 else "Not Significant"
                    }
                except Exception as e:
                    ks_results[col] = {
                        "ks_statistic": 0.0,
                        "p_value": 1.0,
                        "interpretation": "Not Significant"
                    }

        # Imbalance Trend Over Time (simulate trend data for now)
        # In a real implementation, you'd store this in a database
        trend_data = [
            {"timestamp": datetime.utcnow().isoformat(), "imbalance_ratio": imbalance_ratio}
        ]

        return {
            "status": "success",
            "data": {
                "overall_imbalance_score": imbalance_ratio,
                "severity_level": severity,
                "total_samples": {
                    "reference": total_ref,
                    "current": total_curr
                },
                "class_counts": {
                    "reference": {str(k): int(v) for k, v in class_counts_ref.items()},
                    "current": {str(k): int(v) for k, v in class_counts_curr.items()}
                },
                "class_percentages": {
                    "reference": class_percent_ref,
                    "current": class_percent_curr
                },
                "imbalance_metrics": {
                    "imbalance_ratio": imbalance_ratio,
                    "gini_coefficient": {"reference": gini_ref, "current": gini_curr},
                    "shannon_entropy": {"reference": entropy_ref, "current": entropy_curr},
                    "effective_number_of_classes": {"reference": enc_ref, "current": enc_curr},
                    "class_balance_index": {"reference": cbi_ref, "current": cbi_curr},
                    "imbalance_trend_over_time": trend_data,
                    "chi_square_test": {
                        "statistic": chi2_stat,
                        "p_value": p_value,
                        "degrees_of_freedom": dof,
                        "interpretation": chi_significance
                    }
                },
                "statistical_significance": {
                    "ks_test": ks_results
                },
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Class imbalance analysis failed: {str(e)}")
