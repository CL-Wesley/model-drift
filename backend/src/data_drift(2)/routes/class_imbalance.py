import os
import pandas as pd
from fastapi import APIRouter
from datetime import datetime
from scipy.stats import chi2_contingency, ks_2samp
import numpy as np
import json

router = APIRouter(
    prefix="/api/v1/data-drift/class-imbalance",
    tags=["Data Drift - Class Imbalance"]
)

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

@router.get("/analysis")
async def get_class_imbalance_analysis():
    # Read latest CSVs
    ref_path = os.path.join("uploads", "latest_reference.csv")
    curr_path = os.path.join("uploads", "latest_current.csv")
    
    ref_df = pd.read_csv(ref_path)
    curr_df = pd.read_csv(curr_path)

    target_col = ref_df.columns[-1]

    # Class counts
    class_counts_ref = ref_df[target_col].value_counts().to_dict()
    class_counts_curr = curr_df[target_col].value_counts().to_dict()

    total_ref = int(ref_df.shape[0])
    total_curr = int(curr_df.shape[0])

    class_percent_ref = {k: round(v / total_ref * 100, 2) for k, v in class_counts_ref.items()}
    class_percent_curr = {k: round(v / total_curr * 100, 2) for k, v in class_counts_curr.items()}

    # Imbalance Ratio
    imbalance_ratio = float(max(class_counts_curr.values()) / min(class_counts_curr.values()))

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
    chi2_stat, p_value, dof, _ = chi2_contingency([ref_vals, curr_vals])
    chi2_stat = float(chi2_stat)
    p_value = float(p_value)
    dof = int(dof)
    chi_significance = "Highly Significant" if p_value < 0.01 else ("Significant" if p_value < 0.05 else "Not Significant")

    # KS test for numeric columns (excluding target)
    ks_results = {}
    numeric_cols = ref_df.select_dtypes(include=np.number).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    for col in numeric_cols:
        ks_stat, ks_p = ks_2samp(ref_df[col], curr_df[col])
        ks_results[col] = {
            "ks_statistic": float(ks_stat),
            "p_value": float(ks_p),
            "interpretation": "Significant" if ks_p < 0.05 else "Not Significant"
        }

    # Imbalance Trend Over Time (last 4 entries)
    trend_file = os.path.join("uploads", "imbalance_trend.json")
    if os.path.exists(trend_file):
        with open(trend_file, "r") as f:
            trend_data = json.load(f)
    else:
        trend_data = []

    trend_data.append({"timestamp": datetime.utcnow().isoformat(), "imbalance_ratio": imbalance_ratio})
    trend_data = trend_data[-4:]

    with open(trend_file, "w") as f:
        json.dump(trend_data, f)

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
                # ✅ move chi-square here for FE
                "chi_square_test": {
                    "statistic": chi2_stat,
                    "p_value": p_value,
                    "degrees_of_freedom": dof,
                    "interpretation": chi_significance
                }
            },
            # Keep KS test separately if needed
            "statistical_significance": {
                "ks_test": ks_results
            },
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
    }

