import os
import pandas as pd
import numpy as np
from fastapi import APIRouter, FastAPI
from fastapi.staticfiles import StaticFiles
from scipy.stats import ks_2samp, chi2_contingency, entropy
from datetime import datetime

app = FastAPI()

# Mount the uploads directory as static files for images or other assets
app.mount("/static", StaticFiles(directory="uploads"), name="static")

router = APIRouter(
    prefix="/api/v1/data-drift",
    tags=["Data Drift - Feature Analysis"]
)

def calc_change(ref, curr):
    if ref == 0:
        return 0.0
    return round(((curr - ref) / ref) * 100, 1)

def population_stability_index(ref, curr, bins=10):
    """Calculate PSI between two distributions"""
    ref_percents, _ = np.histogram(ref, bins=bins)
    curr_percents, _ = np.histogram(curr, bins=bins)

    ref_percents = ref_percents / len(ref)
    curr_percents = curr_percents / len(curr)

    psi_vals = []
    for i in range(len(ref_percents)):
        if ref_percents[i] == 0 or curr_percents[i] == 0:
            continue
        psi_vals.append((ref_percents[i] - curr_percents[i]) *
                        np.log(ref_percents[i] / curr_percents[i]))

    return round(np.sum(psi_vals), 4)

@router.get("/feature-analysis")
async def get_feature_analysis():
    ref_path = os.path.join("uploads", "latest_reference.csv")
    curr_path = os.path.join("uploads", "latest_current.csv")

    ref_df = pd.read_csv(ref_path)
    curr_df = pd.read_csv(curr_path)

    feature_analysis_list = []

    for col in ref_df.columns:
        dtype = "numerical" if ref_df[col].dtype in ["int64", "float64"] else "categorical"

        missing_ref = int(ref_df[col].isna().sum())
        missing_curr = int(curr_df[col].isna().sum())

        drift_score = 0.0
        p_value = 1.0
        ks_stat = 0.0
        kl_div = None
        psi_val = None
        histogram_ref = None
        histogram_current = None

        if dtype == "numerical":
            ref_vals = ref_df[col].dropna()
            curr_vals = curr_df[col].dropna()

            ks_stat, p_value = ks_2samp(ref_vals, curr_vals)
            drift_score = float(abs(ks_stat) * 5)

            ref_hist_density, bins = np.histogram(ref_vals, bins=10, density=True)
            curr_hist_density, _ = np.histogram(curr_vals, bins=bins, density=True)
            kl_div = float(entropy(ref_hist_density + 1e-8, curr_hist_density + 1e-8))

            psi_val = float(population_stability_index(ref_vals, curr_vals))

            ref_hist, bins = np.histogram(ref_vals, bins=10)
            curr_hist, _ = np.histogram(curr_vals, bins=bins)
            histogram_ref = [{"bin": float(bins[i]), "count": int(ref_hist[i])} for i in range(len(ref_hist))]
            histogram_current = [{"bin": float(bins[i]), "count": int(curr_hist[i])} for i in range(len(curr_hist))]

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

        else:
            chi2, p_value, _, _ = chi2_contingency(pd.crosstab(ref_df[col], curr_df[col]))
            drift_score = float(chi2 / 10)
            distribution_ref = {
                "counts": {str(k): int(v) for k, v in ref_df[col].value_counts().to_dict().items()}
            }
            distribution_curr = {
                "counts": {str(k): int(v) for k, v in curr_df[col].value_counts().to_dict().items()}
            }
            summary_stats = {}

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

    return {
        "status": "success",
        "data": {
            "features": feature_analysis_list,
            "analysis_timestamp": datetime.utcnow().isoformat(),
        },
    }

