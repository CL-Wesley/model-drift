// Core data types for the Model Drift Detection Platform

export interface Dataset {
    id: string;
    name: string;
    rows: number;
    columns: string[];
    dataTypes: Record<string, string>;
    uploadDate: Date;
    size: number;
    preview?: DataPoint[];
}

export interface DataPoint {
    [key: string]: any;
}

export interface ModelInfo {
    id: string;
    name: string;
    type: string;
    features: string[];
    accuracy?: number;
    created_date: string;
    feature_importance?: Record<string, number>;
    format: 'pkl' | 'onnx';
    size: number;
}

export interface FeatureAnalysis {
    feature: string;
    drift_score: number;
    status: 'low' | 'medium' | 'high';
    kl_divergence: number;
    psi: number;
    ks_statistic: number;
    p_value: number;
    data_type: 'numerical' | 'categorical';
    missing_values_ref: number;
    missing_values_current: number;
    distribution_ref?: DistributionStats;
    distribution_current?: DistributionStats;
}

export interface DistributionStats {
    mean?: number;
    std?: number;
    min?: number;
    max?: number;
    q25?: number;
    q50?: number;
    q75?: number;
    histogram?: HistogramBin[];
    categories?: CategoryCount[];
}

export interface HistogramBin {
    bin_start: number;
    bin_end: number;
    count: number;
    frequency: number;
}

export interface CategoryCount {
    category: string;
    count: number;
    frequency: number;
}

export interface DriftAnalysisResults {
    id: string;
    overall_drift_score: number;
    overall_status: 'low' | 'medium' | 'high';
    total_features: number;
    high_drift_features: number;
    medium_drift_features: number;
    low_drift_features: number;
    data_quality_score: number;
    model_compatibility_status: 'compatible' | 'warning' | 'incompatible';
    analysis_timestamp: Date;
    feature_analysis: FeatureAnalysis[];
    recommendations?: string[];
    executive_summary?: string;
}

export interface AnalysisConfiguration {
    analysis_name: string;
    description?: string;
    drift_thresholds: {
        low_threshold: number;
        medium_threshold: number;
        high_threshold: number;
    };
    selected_features?: string[];
    excluded_features?: string[];
    statistical_tests: string[];
}

export interface UploadProgress {
    reference_dataset?: number;
    current_dataset?: number;
    model?: number;
}

export interface AnalysisStatus {
    status: 'pending' | 'processing' | 'completed' | 'error';
    progress: number;
    message?: string;
    error?: string;
}

export interface ExportOptions {
    format: 'pdf' | 'csv' | 'html' | 'json';
    include_charts: boolean;
    include_detailed_stats: boolean;
    include_recommendations: boolean;
}

export interface AlertConfiguration {
    enabled: boolean;
    email_notifications: string[];
    webhook_url?: string;
    drift_threshold: number;
    monitoring_schedule: 'daily' | 'weekly' | 'monthly';
}

export type TabType =
    | 'upload'
    | 'dashboard'
    | 'feature-analysis'
    | 'statistical-reports'
    | 'model-insights'
    | 'export-alerts';

export interface Theme {
    primary: string;
    secondary: string;
    success: string;
    warning: string;
    error: string;
    background: string;
    surface: string;
    text: {
        primary: string;
        secondary: string;
    };
}
