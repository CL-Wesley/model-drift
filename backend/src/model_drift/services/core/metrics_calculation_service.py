"""
Metrics Calculation Service - Implements all performance metrics
Following research specifications for comprehensive model evaluation
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, log_loss,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from typing import Dict, Any, Union, Optional
import warnings
warnings.filterwarnings('ignore')

class MetricsCalculationService:
    """Service for calculating comprehensive performance metrics for both classification and regression"""
    
    def __init__(self):
        pass
    
    def classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels (binary)
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary with all classification metrics
        """
        try:
            metrics = {}
            
            # Basic Classification Metrics
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            metrics["precision"] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
            metrics["recall"] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
            metrics["f1_score"] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
            
            # Precision, Recall, F1 for each class
            precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
            recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
            f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
            
            metrics["precision_per_class"] = [float(p) for p in precision_per_class]
            metrics["recall_per_class"] = [float(r) for r in recall_per_class]
            metrics["f1_score_per_class"] = [float(f) for f in f1_per_class]
            
            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics["confusion_matrix"] = cm.tolist()
            
            # True Positives, False Positives, True Negatives, False Negatives
            if len(np.unique(y_true)) == 2:  # Binary classification
                tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, len(y_true))
                metrics["true_positives"] = int(tp)
                metrics["false_positives"] = int(fp)
                metrics["true_negatives"] = int(tn)
                metrics["false_negatives"] = int(fn)
                
                # Sensitivity and Specificity
                metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
                
                # Positive and Negative Predictive Value
                metrics["positive_predictive_value"] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                metrics["negative_predictive_value"] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
                
                # Matthews Correlation Coefficient
                denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                metrics["matthews_correlation"] = float((tp * tn - fp * fn) / denominator) if denominator > 0 else 0.0
            
            # Probabilistic Metrics (if probabilities provided)
            if y_pred_proba is not None:
                try:
                    # ROC AUC
                    if len(np.unique(y_true)) == 2:  # Binary classification
                        metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba))
                    else:  # Multi-class
                        metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted'))
                    
                    # Average Precision (PR AUC)
                    metrics["average_precision"] = float(average_precision_score(y_true, y_pred_proba, average='weighted'))
                    
                    # Log Loss
                    metrics["log_loss"] = float(log_loss(y_true, y_pred_proba))
                    
                except Exception as e:
                    metrics["probabilistic_metrics_error"] = f"Could not calculate probabilistic metrics: {str(e)}"
            
            # Classification Report
            class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            metrics["classification_report"] = class_report
            
            return metrics
            
        except Exception as e:
            return {"error": f"Classification metrics calculation failed: {str(e)}"}
    
    def regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Calculate comprehensive regression metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with all regression metrics
        """
        try:
            metrics = {}
            
            # Basic Regression Metrics
            metrics["mean_squared_error"] = float(mean_squared_error(y_true, y_pred))
            metrics["root_mean_squared_error"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            metrics["mean_absolute_error"] = float(mean_absolute_error(y_true, y_pred))
            metrics["r2_score"] = float(r2_score(y_true, y_pred))
            
            # Mean Absolute Percentage Error
            try:
                metrics["mean_absolute_percentage_error"] = float(mean_absolute_percentage_error(y_true, y_pred))
            except:
                # Manual MAPE calculation if sklearn version doesn't support it
                mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
                metrics["mean_absolute_percentage_error"] = float(mape)
            
            # Mean Squared Logarithmic Error (if all values are positive)
            if np.all(y_true >= 0) and np.all(y_pred >= 0):
                msle = np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)
                metrics["mean_squared_logarithmic_error"] = float(msle)
            
            # Additional metrics
            residuals = y_true - y_pred
            metrics["mean_residual"] = float(np.mean(residuals))
            metrics["std_residual"] = float(np.std(residuals))
            metrics["max_residual"] = float(np.max(np.abs(residuals)))
            
            # Adjusted R²
            n = len(y_true)
            p = 1  # Assuming simple regression, adjust if needed
            adj_r2 = 1 - (1 - metrics["r2_score"]) * (n - 1) / (n - p - 1)
            metrics["adjusted_r2"] = float(adj_r2)
            
            return metrics
            
        except Exception as e:
            return {"error": f"Regression metrics calculation failed: {str(e)}"}
    
    def calculate_metrics_difference(self, metrics_ref: Dict[str, Any], 
                                   metrics_curr: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate the difference between reference and current model metrics
        
        Args:
            metrics_ref: Reference model metrics
            metrics_curr: Current model metrics
            
        Returns:
            Dictionary with metric differences and drift analysis
        """
        try:
            differences = {}
            
            # Skip non-numeric keys
            skip_keys = ['confusion_matrix', 'classification_report', 'error', 
                        'precision_per_class', 'recall_per_class', 'f1_score_per_class']
            
            for key in metrics_ref:
                if key in skip_keys or key not in metrics_curr:
                    continue
                
                if isinstance(metrics_ref[key], (int, float)) and isinstance(metrics_curr[key], (int, float)):
                    ref_val = float(metrics_ref[key])
                    curr_val = float(metrics_curr[key])
                    
                    absolute_diff = curr_val - ref_val
                    relative_diff = (absolute_diff / ref_val * 100) if ref_val != 0 else 0.0
                    
                    differences[key] = {
                        "reference": ref_val,
                        "current": curr_val,
                        "absolute_difference": absolute_diff,
                        "relative_difference": relative_diff,
                        "drift_magnitude": abs(absolute_diff)
                    }
            
            # Calculate overall drift assessment
            drift_scores = [abs(diff["relative_difference"]) for diff in differences.values()]
            if drift_scores:
                avg_drift = np.mean(drift_scores)
                max_drift = np.max(drift_scores)
                
                # Determine drift severity
                if max_drift >= 20:  # 20% change
                    severity = "High"
                elif max_drift >= 10:  # 10% change
                    severity = "Medium"
                elif max_drift >= 5:   # 5% change
                    severity = "Low"
                else:
                    severity = "Minimal"
                
                differences["drift_summary"] = {
                    "average_relative_drift": avg_drift,
                    "maximum_relative_drift": max_drift,
                    "drift_severity": severity,
                    "metrics_analyzed": len(differences) - 1  # Subtract 1 for drift_summary itself
                }
            
            return differences
            
        except Exception as e:
            return {"error": f"Metrics difference calculation failed: {str(e)}"}
    
    def performance_degradation_analysis(self, metrics_ref: Dict[str, Any], 
                                       metrics_curr: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze performance degradation between models
        
        Args:
            metrics_ref: Reference model metrics
            metrics_curr: Current model metrics
            
        Returns:
            Dictionary with degradation analysis
        """
        try:
            analysis = {
                "degraded_metrics": [],
                "improved_metrics": [],
                "stable_metrics": [],
                "critical_degradations": []
            }
            
            # Define critical metrics for different model types
            critical_metrics = {
                "classification": ["accuracy", "f1_score", "roc_auc", "precision", "recall"],
                "regression": ["r2_score", "mean_squared_error", "mean_absolute_error"]
            }
            
            # Determine model type based on available metrics
            model_type = "classification" if "accuracy" in metrics_ref else "regression"
            
            differences = self.calculate_metrics_difference(metrics_ref, metrics_curr)
            
            for metric, diff_data in differences.items():
                if metric == "drift_summary" or "error" in diff_data:
                    continue
                
                relative_change = diff_data["relative_difference"]
                
                # Classify metric change
                if abs(relative_change) < 2:  # Less than 2% change
                    analysis["stable_metrics"].append({
                        "metric": metric,
                        "change": relative_change
                    })
                elif relative_change < 0:  # Performance degraded
                    degradation_info = {
                        "metric": metric,
                        "change": relative_change,
                        "severity": "Critical" if abs(relative_change) > 10 else "Moderate"
                    }
                    analysis["degraded_metrics"].append(degradation_info)
                    
                    # Check if this is a critical metric
                    if metric in critical_metrics.get(model_type, []):
                        analysis["critical_degradations"].append(degradation_info)
                        
                else:  # Performance improved
                    analysis["improved_metrics"].append({
                        "metric": metric,
                        "change": relative_change
                    })
            
            # Overall assessment
            total_metrics = len(analysis["degraded_metrics"]) + len(analysis["improved_metrics"]) + len(analysis["stable_metrics"])
            degraded_count = len(analysis["degraded_metrics"])
            critical_count = len(analysis["critical_degradations"])
            
            if critical_count > 0:
                overall_status = "Critical Degradation"
            elif degraded_count > total_metrics * 0.5:
                overall_status = "Significant Degradation"
            elif degraded_count > 0:
                overall_status = "Minor Degradation"
            else:
                overall_status = "Stable or Improved"
            
            analysis["summary"] = {
                "overall_status": overall_status,
                "total_metrics": total_metrics,
                "degraded_count": degraded_count,
                "improved_count": len(analysis["improved_metrics"]),
                "stable_count": len(analysis["stable_metrics"]),
                "critical_degradations": critical_count
            }
            
            return analysis
            
        except Exception as e:
            return {"error": f"Performance degradation analysis failed: {str(e)}"}

# Service instance
metrics_calculation_service = MetricsCalculationService()
