"""
Performance Comparison Service - Tab 1 Analysis
Orchestrates comprehensive performance metrics comparison between reference and current models
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from ..core.metrics_calculation_service import metrics_calculation_service
from ..core.statistical_tests_service import statistical_tests_service
from ..core.effect_size_service import effect_size_service
from ..advanced.psi_service import psi_service
import warnings
warnings.filterwarnings('ignore')

class PerformanceComparisonService:
    """Service for comprehensive performance comparison analysis (Tab 1)"""
    
    def __init__(self):
        pass
    
    def analyze_performance_comparison(self, 
                                     y_true: np.ndarray,
                                     pred_ref: np.ndarray, 
                                     pred_curr: np.ndarray,
                                     pred_ref_proba: Optional[np.ndarray] = None,
                                     pred_curr_proba: Optional[np.ndarray] = None,
                                     X: Optional[np.ndarray] = None,
                                     model_ref = None,
                                     model_curr = None) -> Dict[str, Any]:
        """
        Perform comprehensive performance comparison analysis
        
        Args:
            y_true: True labels
            pred_ref: Reference model predictions
            pred_curr: Current model predictions
            pred_ref_proba: Reference model probabilities (optional)
            pred_curr_proba: Current model probabilities (optional)
            X: Feature matrix (optional, for advanced tests)
            model_ref: Reference model object (optional)
            model_curr: Current model object (optional)
            
        Returns:
            Dictionary with complete performance comparison analysis
        """
        try:
            analysis_results = {
                "analysis_type": "performance_comparison",
                "summary": {},
                "metrics_comparison": {},
                "statistical_tests": {},
                "effect_size_analysis": {},
                "prediction_drift": {},
                "overall_assessment": {}
            }
            
            # 1. Calculate comprehensive metrics for both models
            ref_metrics = metrics_calculation_service.classification_metrics(
                y_true, pred_ref, pred_ref_proba
            )
            curr_metrics = metrics_calculation_service.classification_metrics(
                y_true, pred_curr, pred_curr_proba
            )
            
            if "error" in ref_metrics or "error" in curr_metrics:
                return {"error": "Failed to calculate basic metrics"}
            
            # 2. Calculate metrics differences and drift analysis
            metrics_diff = metrics_calculation_service.calculate_metrics_difference(
                ref_metrics, curr_metrics
            )
            
            # 3. Performance degradation analysis
            degradation_analysis = metrics_calculation_service.performance_degradation_analysis(
                ref_metrics, curr_metrics
            )
            
            # 4. Statistical significance testing
            statistical_results = statistical_tests_service.run_all_tests(
                y_true, pred_ref, pred_curr, pred_ref_proba, pred_curr_proba, X, model_ref, model_curr
            )
            
            # 5. Effect size analysis
            ref_accuracy_array = np.array([ref_metrics["accuracy"]])
            curr_accuracy_array = np.array([curr_metrics["accuracy"]])
            effect_analysis = effect_size_service.comprehensive_effect_analysis(
                ref_accuracy_array, curr_accuracy_array, "Reference Model", "Current Model"
            )
            
            # 6. Prediction distribution drift (PSI)
            prediction_drift = {}
            if pred_ref_proba is not None and pred_curr_proba is not None:
                prediction_drift = psi_service.calculate_prediction_psi(pred_ref_proba, pred_curr_proba)
            
            # 7. Overall performance assessment
            overall_assessment = self._generate_overall_assessment(
                metrics_diff, degradation_analysis, statistical_results, effect_analysis, prediction_drift
            )
            
            # Compile results
            analysis_results.update({
                "summary": {
                    "reference_samples": len(y_true),
                    "current_samples": len(y_true),
                    "metrics_analyzed": len([k for k in metrics_diff.keys() if k != "drift_summary"])
                },
                "metrics_comparison": {
                    "reference_metrics": ref_metrics,
                    "current_metrics": curr_metrics,
                    "metrics_differences": metrics_diff,
                    "degradation_analysis": degradation_analysis
                },
                "statistical_tests": statistical_results,
                "effect_size_analysis": effect_analysis,
                "prediction_drift": prediction_drift,
                "overall_assessment": overall_assessment
            })
            
            return analysis_results
            
        except Exception as e:
            return {"error": f"Performance comparison analysis failed: {str(e)}"}
    
    def _generate_overall_assessment(self, metrics_diff: Dict, degradation_analysis: Dict,
                                   statistical_results: Dict, effect_analysis: Dict,
                                   prediction_drift: Dict) -> Dict[str, Any]:
        """Generate overall assessment of model performance comparison"""
        
        try:
            assessment = {
                "drift_detected": False,
                "drift_severity": "Minimal",
                "key_findings": [],
                "recommendations": [],
                "risk_level": "Low"
            }
            
            # Check for drift indicators
            drift_indicators = 0
            
            # 1. Metrics degradation check
            if "drift_summary" in metrics_diff:
                drift_severity = metrics_diff["drift_summary"]["drift_severity"]
                if drift_severity in ["High", "Medium"]:
                    drift_indicators += 1
                    assessment["key_findings"].append(f"Performance metrics show {drift_severity.lower()} drift")
            
            # 2. Statistical significance check
            if "summary" in statistical_results:
                significant_tests = statistical_results["summary"]["significant_tests"]
                if significant_tests > 0:
                    drift_indicators += 1
                    assessment["key_findings"].append(f"{significant_tests} statistical tests show significant differences")
            
            # 3. Effect size check
            if "overall_assessment" in effect_analysis:
                effect_magnitude = effect_analysis["overall_assessment"]["magnitude"]
                if effect_magnitude in ["Medium", "Large"]:
                    drift_indicators += 1
                    assessment["key_findings"].append(f"{effect_magnitude} practical effect size detected")
            
            # 4. Prediction drift check
            if "psi" in prediction_drift:
                psi_level = prediction_drift["interpretation"]["level"]
                if psi_level in ["High", "Medium"]:
                    drift_indicators += 1
                    assessment["key_findings"].append(f"{psi_level} prediction distribution drift (PSI)")
            
            # 5. Critical degradation check
            if "summary" in degradation_analysis:
                critical_degradations = degradation_analysis["summary"]["critical_degradations"]
                if critical_degradations > 0:
                    drift_indicators += 2  # Weight this more heavily
                    assessment["key_findings"].append(f"{critical_degradations} critical performance degradations")
            
            # Determine overall drift status
            if drift_indicators >= 3:
                assessment["drift_detected"] = True
                assessment["drift_severity"] = "High"
                assessment["risk_level"] = "High"
            elif drift_indicators >= 2:
                assessment["drift_detected"] = True
                assessment["drift_severity"] = "Medium"
                assessment["risk_level"] = "Medium"
            elif drift_indicators >= 1:
                assessment["drift_detected"] = True
                assessment["drift_severity"] = "Low"
                assessment["risk_level"] = "Low"
            
            # Generate recommendations
            if assessment["drift_detected"]:
                if assessment["drift_severity"] == "High":
                    assessment["recommendations"].extend([
                        "Immediate model retraining recommended",
                        "Investigate root causes of performance degradation",
                        "Consider emergency rollback to previous model version"
                    ])
                elif assessment["drift_severity"] == "Medium":
                    assessment["recommendations"].extend([
                        "Schedule model retraining within next cycle",
                        "Monitor performance closely",
                        "Investigate data quality and distribution changes"
                    ])
                else:  # Low
                    assessment["recommendations"].extend([
                        "Continue monitoring model performance",
                        "Prepare for potential retraining",
                        "Document observed changes for trend analysis"
                    ])
            else:
                assessment["recommendations"].append("Model performance remains stable - continue regular monitoring")
            
            # Add no findings message if applicable
            if not assessment["key_findings"]:
                assessment["key_findings"].append("No significant performance changes detected")
            
            return assessment
            
        except Exception as e:
            return {"error": f"Overall assessment generation failed: {str(e)}"}
    
    def generate_executive_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary for performance comparison"""
        
        try:
            if "error" in analysis_results:
                return analysis_results
            
            overall_assessment = analysis_results["overall_assessment"]
            metrics_diff = analysis_results["metrics_comparison"]["metrics_differences"]
            
            # Key performance changes
            key_changes = []
            if "drift_summary" in metrics_diff:
                for metric, diff_data in metrics_diff.items():
                    if metric == "drift_summary" or "error" in str(diff_data):
                        continue
                    
                    if isinstance(diff_data, dict) and "relative_difference" in diff_data:
                        rel_change = diff_data["relative_difference"]
                        if abs(rel_change) >= 5:  # 5% threshold
                            direction = "improved" if rel_change > 0 else "degraded"
                            key_changes.append({
                                "metric": metric.replace("_", " ").title(),
                                "change_percentage": f"{rel_change:+.1f}%",
                                "direction": direction
                            })
            
            # Sort by magnitude of change
            key_changes.sort(key=lambda x: abs(float(x["change_percentage"].replace("%", "").replace("+", ""))), reverse=True)
            
            summary = {
                "model_drift_detected": overall_assessment["drift_detected"],
                "drift_severity": overall_assessment["drift_severity"],
                "risk_level": overall_assessment["risk_level"],
                "key_performance_changes": key_changes[:5],  # Top 5 changes
                "primary_concerns": overall_assessment["key_findings"][:3],  # Top 3 findings
                "immediate_actions": overall_assessment["recommendations"][:2],  # Top 2 recommendations
                "detailed_analysis_available": True
            }
            
            return summary
            
        except Exception as e:
            return {"error": f"Executive summary generation failed: {str(e)}"}

# Service instance
performance_comparison_service = PerformanceComparisonService()
