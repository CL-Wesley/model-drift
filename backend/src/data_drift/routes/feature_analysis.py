import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException
from scipy.stats import ks_2samp, chi2_contingency, entropy
from datetime import datetime
from .upload import get_session_storage
from ...shared.session_manager import get_session_manager
from ...shared.ai_explanation_service import ai_explanation_service

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
    """Calculate percentage change between reference and current values"""
    if ref == 0:
        return 0.0
    return round(((curr - ref) / ref) * 100, 1)

def generate_feature_analysis_insights(feature_results, overall_drift_score):
    """Generate human-readable insights for feature analysis"""
    drifted_features = [f for f in feature_results if f['drift_detected']]
    high_drift_features = [f for f in feature_results if f['drift_severity'] in ['High', 'Critical']]
    
    if len(drifted_features) == 0:
        return {
            "summary": "No significant feature drift detected across the dataset. All features maintain stable distributions.",
            "recommendations": [
                "Continue regular monitoring of feature distributions",
                "Current model should perform reliably on new data",
                "Focus on other aspects of model performance monitoring"
            ]
        }
    
    drift_percentage = (len(drifted_features) / len(feature_results)) * 100
    
    if drift_percentage > 50:
        severity_text = "widespread drift"
    elif drift_percentage > 25:
        severity_text = "moderate drift"
    else:
        severity_text = "limited drift"
    
    summary = f"Feature drift analysis detected {severity_text} affecting {len(drifted_features)} out of {len(feature_results)} features ({drift_percentage:.1f}%). "
    
    if high_drift_features:
        top_drifted = sorted(high_drift_features, key=lambda x: x['drift_score'], reverse=True)[:3]
        feature_names = [f"'{f['feature_name']}'" for f in top_drifted]
        summary += f"Features with highest drift: {', '.join(feature_names)}. "
    
    summary += f"Overall dataset drift score: {overall_drift_score:.3f}."
    
    recommendations = []
    if drift_percentage > 50:
        recommendations.extend([
            "Investigate data collection process for systematic changes",
            "Consider retraining the model with recent data",
            "Implement feature-level monitoring and alerting",
            "Review data preprocessing and feature engineering steps"
        ])
    elif drift_percentage > 25:
        recommendations.extend([
            "Monitor model performance metrics closely",
            "Consider incremental model updates if performance degrades",
            "Investigate root causes of drift in most affected features"
        ])
    else:
        recommendations.extend([
            "Continue monitoring drifted features more frequently",
            "Current model should perform adequately",
            "Consider minor threshold adjustments if needed"
        ])
    
    return {
        "summary": summary,
        "recommendations": recommendations
    }

def calculate_feature_impact_score(feature_result):
    """Calculate business impact score for a feature based on drift characteristics"""
    base_score = feature_result['drift_score']
    
    # Adjust based on statistical significance
    if feature_result['statistical_test']['p_value'] < 0.001:
        significance_multiplier = 1.3
    elif feature_result['statistical_test']['p_value'] < 0.01:
        significance_multiplier = 1.2
    elif feature_result['statistical_test']['p_value'] < 0.05:
        significance_multiplier = 1.1
    else:
        significance_multiplier = 0.8
    
    # Adjust based on feature type (categorical features might have more business impact)
    type_multiplier = 1.2 if feature_result['feature_type'] == 'categorical' else 1.0
    
    impact_score = base_score * significance_multiplier * type_multiplier
    return min(1.0, impact_score)  # Cap at 1.0

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
    Get comprehensive feature drift analysis for uploaded datasets
    
    Args:
        session_id: Session identifier for uploaded data
        
    Returns:
        Enhanced feature analysis results with insights and recommendations
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

        # Track overall drift metrics
        total_drift_score = 0.0
        significant_drifts = 0

        # Analyze each feature
        for col in common_columns:
            feature_type = "numerical" if reference_df[col].dtype in ["int64", "float64"] else "categorical"

            missing_ref = int(reference_df[col].isna().sum())
            missing_curr = int(curr_df[col].isna().sum())

            drift_score = 0.0
            p_value = 1.0
            ks_stat = 0.0
            kl_div = None
            psi_val = None
            histogram_ref = None
            histogram_current = None
            summary_stats = {}
            distribution_ref = {}
            distribution_curr = {}

            if feature_type == "numerical":
                try:
                    ref_vals = reference_df[col].dropna()
                    curr_vals = curr_df[col].dropna()

                    if len(ref_vals) == 0 or len(curr_vals) == 0:
                        continue

                    ks_stat, p_value = ks_2samp(ref_vals, curr_vals)
                    drift_score = float(abs(ks_stat))

                    # Safe KL divergence calculation
                    try:
                        ref_hist_density, bins = np.histogram(ref_vals, bins=10, density=True)
                        curr_hist_density, _ = np.histogram(curr_vals, bins=bins, density=True)
                        kl_div = float(entropy(ref_hist_density + 1e-8, curr_hist_density + 1e-8))
                    except:
                        kl_div = 0.0

                    psi_val = float(population_stability_index(ref_vals, curr_vals))

                    # Create histogram data for visualization
                    try:
                        ref_hist, bins = np.histogram(ref_vals, bins=10)
                        curr_hist, _ = np.histogram(curr_vals, bins=bins)
                        histogram_ref = [{"bin": f"{bins[i]:.2f}-{bins[i+1]:.2f}", "count": int(ref_hist[i])} for i in range(len(ref_hist))]
                        histogram_current = [{"bin": f"{bins[i]:.2f}-{bins[i+1]:.2f}", "count": int(curr_hist[i])} for i in range(len(curr_hist))]
                    except:
                        histogram_ref = []
                        histogram_current = []

                    # Enhanced summary statistics with change calculations
                    summary_stats = {
                        "mean": {
                            "reference": round(float(ref_vals.mean()), 4),
                            "current": round(float(curr_vals.mean()), 4),
                            "change_percent": calc_change(ref_vals.mean(), curr_vals.mean()),
                            "absolute_change": round(float(curr_vals.mean() - ref_vals.mean()), 4)
                        },
                        "std": {
                            "reference": round(float(ref_vals.std()), 4),
                            "current": round(float(curr_vals.std()), 4),
                            "change_percent": calc_change(ref_vals.std(), curr_vals.std()),
                            "absolute_change": round(float(curr_vals.std() - ref_vals.std()), 4)
                        },
                        "min": {
                            "reference": round(float(ref_vals.min()), 4),
                            "current": round(float(curr_vals.min()), 4),
                            "change_percent": calc_change(ref_vals.min(), curr_vals.min()),
                            "absolute_change": round(float(curr_vals.min() - ref_vals.min()), 4)
                        },
                        "max": {
                            "reference": round(float(ref_vals.max()), 4),
                            "current": round(float(curr_vals.max()), 4),
                            "change_percent": calc_change(ref_vals.max(), curr_vals.max()),
                            "absolute_change": round(float(curr_vals.max() - ref_vals.max()), 4)
                        },
                        "median": {
                            "reference": round(float(ref_vals.median()), 4),
                            "current": round(float(curr_vals.median()), 4),
                            "change_percent": calc_change(ref_vals.median(), curr_vals.median()),
                            "absolute_change": round(float(curr_vals.median() - ref_vals.median()), 4)
                        }
                    }

                    distribution_ref = summary_stats
                    distribution_curr = summary_stats

                except Exception as e:
                    # Skip problematic numerical features
                    continue

            else:
                # Enhanced categorical analysis
                try:
                    ref_counts = reference_df[col].value_counts()
                    curr_counts = curr_df[col].value_counts()
                    
                    # Get all unique categories
                    all_categories = list(set(ref_counts.index).union(set(curr_counts.index)))
                    
                    # Prepare contingency table
                    ref_vals = [ref_counts.get(cat, 0) for cat in all_categories]
                    curr_vals = [curr_counts.get(cat, 0) for cat in all_categories]
                    
                    if sum(ref_vals) > 0 and sum(curr_vals) > 0:
                        chi2, p_value, _, _ = chi2_contingency([ref_vals, curr_vals])
                        drift_score = float(chi2 / max(chi2, 100))  # Normalize chi2 score
                    else:
                        chi2, p_value, drift_score = 0.0, 1.0, 0.0
                    
                    # Create category distribution comparison
                    category_comparison = []
                    for cat in all_categories:
                        ref_count = ref_counts.get(cat, 0)
                        curr_count = curr_counts.get(cat, 0)
                        ref_pct = (ref_count / len(reference_df)) * 100
                        curr_pct = (curr_count / len(curr_df)) * 100
                        
                        category_comparison.append({
                            "category": str(cat),
                            "reference_count": int(ref_count),
                            "current_count": int(curr_count),
                            "reference_percent": round(ref_pct, 2),
                            "current_percent": round(curr_pct, 2),
                            "change_percent": round(curr_pct - ref_pct, 2)
                        })
                    
                    summary_stats = {
                        "unique_categories": {
                            "reference": len(ref_counts),
                            "current": len(curr_counts),
                            "change": len(curr_counts) - len(ref_counts)
                        },
                        "category_distribution": category_comparison
                    }

                    distribution_ref = {"counts": ref_counts.to_dict()}
                    distribution_curr = {"counts": curr_counts.to_dict()}

                except Exception as e:
                    # Skip problematic categorical features
                    continue

            # Determine drift severity
            if p_value < 0.001:
                drift_severity = "Critical"
                significant_drifts += 1
            elif p_value < 0.01:
                drift_severity = "High" 
                significant_drifts += 1
            elif p_value < 0.05:
                drift_severity = "Medium"
                significant_drifts += 1
            else:
                drift_severity = "Low"

            drift_detected = p_value < 0.05
            total_drift_score += drift_score

            # Calculate business impact score
            feature_result_temp = {
                'drift_score': drift_score,
                'statistical_test': {'p_value': p_value},
                'feature_type': feature_type
            }
            impact_score = calculate_feature_impact_score(feature_result_temp)

            feature_analysis_list.append({
                "feature_name": col,
                "feature_type": feature_type,
                "drift_detected": drift_detected,
                "drift_severity": drift_severity,
                "drift_score": round(drift_score, 4),
                "business_impact_score": round(impact_score, 4),
                
                # Statistical tests
                "statistical_test": {
                    "test_name": "Kolmogorov-Smirnov" if feature_type == "numerical" else "Chi-Square",
                    "statistic": float(ks_stat) if feature_type == "numerical" else float(chi2) if 'chi2' in locals() else 0.0,
                    "p_value": float(p_value),
                    "interpretation": "Significant" if p_value < 0.05 else "Not Significant"
                },
                
                # Drift metrics
                "drift_metrics": {
                    "kl_divergence": kl_div,
                    "psi": psi_val,
                    "missing_values_change": missing_curr - missing_ref
                },
                
                # Data quality
                "data_quality": {
                    "missing_values": {
                        "reference": missing_ref,
                        "current": missing_curr,
                        "change": missing_curr - missing_ref,
                        "reference_percent": round((missing_ref / len(reference_df)) * 100, 2),
                        "current_percent": round((missing_curr / len(curr_df)) * 100, 2)
                    }
                },
                
                # Enhanced statistics and distributions
                "summary_statistics": summary_stats,
                "distribution_comparison": {
                    "reference": distribution_ref,
                    "current": distribution_curr
                },
                
                # Visualization data
                "visualization_data": {
                    "histogram_reference": histogram_ref,
                    "histogram_current": histogram_current
                }
            })

        if len(feature_analysis_list) == 0:
            raise HTTPException(status_code=400, detail="No features could be analyzed")

        # Calculate overall drift metrics
        overall_drift_score = total_drift_score / len(feature_analysis_list)
        drift_percentage = (significant_drifts / len(feature_analysis_list)) * 100

        # Generate insights and recommendations
        insights = generate_feature_analysis_insights(feature_analysis_list, overall_drift_score)

        # Enhanced response with executive summary
        result = {
            "status": "success",
            "data": {
                # Executive Summary
                "executive_summary": {
                    "total_features_analyzed": len(feature_analysis_list),
                    "features_with_drift": significant_drifts,
                    "drift_percentage": round(drift_percentage, 1),
                    "overall_drift_score": round(overall_drift_score, 4),
                    "highest_drift_feature": max(feature_analysis_list, key=lambda x: x['drift_score'])['feature_name'] if feature_analysis_list else None
                },
                
                # Detailed feature analysis
                "feature_analysis": feature_analysis_list,
                
                # Insights and recommendations
                "insights": {
                    "summary_text": insights["summary"],
                    "recommendations": insights["recommendations"],
                    "risk_level": "High" if drift_percentage > 50 else "Medium" if drift_percentage > 25 else "Low"
                },
                
                # Metadata
                "analysis_metadata": {
                    "analysis_timestamp": datetime.utcnow().isoformat(),
                    "reference_dataset_size": len(reference_df),
                    "current_dataset_size": len(curr_df),
                    "common_features": len(common_columns)
                }
            }
        }

        # Generate AI explanation for the feature analysis
        try:
            ai_explanation = ai_explanation_service.generate_explanation(
                analysis_data=result["data"], 
                analysis_type="feature_analysis"
            )
            result["llm_response"] = ai_explanation
        except Exception as e:
            print(f"Warning: AI explanation failed: {e}")
            # Continue without AI explanation
            result["llm_response"] = {
                "summary": "Feature analysis completed successfully.",
                "detailed_explanation": "Detailed feature-level drift analysis has been completed, showing individual feature changes and importance rankings. AI explanations are temporarily unavailable.",
                "key_takeaways": [
                    "Feature analysis completed successfully", 
                    "Check individual feature drift patterns",
                    "AI explanations will return when service is restored"
                ]
            }

        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature analysis failed: {str(e)}")