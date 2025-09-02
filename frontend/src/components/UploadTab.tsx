import React, { useState, useCallback } from 'react';
import {
    Box,
    Paper,
    Typography,
    Grid,
    Button,
    Card,
    CardContent,
    LinearProgress,
    Chip,
    Alert,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    TextField,
    Switch,
    FormControlLabel,
    Divider,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Tooltip,
    FormGroup,
    RadioGroup,
    Radio,
    FormLabel,
    Collapse,
    Checkbox,
} from '@mui/material';
import {
    CloudUpload,
    CheckCircle,
    PlayArrow,
    ModelTraining,
    DatasetLinked,
    Analytics,
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';

import { Dataset, ModelInfo, AnalysisConfiguration } from '../types';
import { mockReferenceDataset, mockCurrentDataset, mockModelInfo } from '../data/mockData';

interface UploadState {
    referenceDataset?: Dataset;
    currentDataset?: Dataset;
    model?: ModelInfo;
    uploadProgress: {
        reference?: number;
        current?: number;
        model?: number;
    };
    modelType?: 'classification' | 'regression';
    selectedMetrics: string[];
    statisticalTest?: string;
    isModelProcessing: boolean;
    modelProcessingStage?: 'uploading' | 'analyzing' | 'extracting-metadata' | 'complete';
    analysisName: string;
    validationErrors: string[];
}

// Performance Metrics Configuration
const classificationMetrics = [
    { id: 'accuracy', label: 'Accuracy', description: 'Overall prediction accuracy', default: true },
    { id: 'precision', label: 'Precision', description: 'True positives / (True positives + False positives)', default: true },
    { id: 'recall', label: 'Recall (Sensitivity)', description: 'True positives / (True positives + False negatives)', default: true },
    { id: 'f1_score', label: 'F1-Score', description: 'Harmonic mean of precision and recall', default: true },
    { id: 'specificity', label: 'Specificity', description: 'True negatives / (True negatives + False positives)', default: false },
    { id: 'roc_auc', label: 'ROC AUC', description: 'Area under ROC curve', default: true },
    { id: 'pr_auc', label: 'PR AUC', description: 'Area under precision-recall curve', default: false },
    { id: 'cohen_kappa', label: 'Cohen\'s Kappa', description: 'Inter-rater reliability metric', default: false },
    { id: 'mcc', label: 'Matthews Correlation Coefficient', description: 'Correlation between predictions and actual', default: false }
];

const regressionMetrics = [
    { id: 'mse', label: 'Mean Squared Error (MSE)', description: 'Average squared prediction errors', default: true },
    { id: 'rmse', label: 'Root Mean Squared Error (RMSE)', description: 'Square root of MSE', default: true },
    { id: 'mae', label: 'Mean Absolute Error (MAE)', description: 'Average absolute prediction errors', default: true },
    { id: 'r2', label: 'R-squared (R²)', description: 'Coefficient of determination', default: true },
    { id: 'adjusted_r2', label: 'Adjusted R-squared', description: 'R² adjusted for number of predictors', default: false },
    { id: 'mape', label: 'Mean Absolute Percentage Error (MAPE)', description: 'Average absolute percentage errors', default: false },
    { id: 'explained_variance', label: 'Explained Variance Score', description: 'Proportion of variance explained', default: false },
    { id: 'max_error', label: 'Max Error', description: 'Maximum residual error', default: false }
];

const statisticalTests = [
    {
        id: 'mcnemar',
        label: 'McNemar\'s Test',
        description: 'Compares paired categorical data for classification models',
        applicableTo: ['classification'],
        tooltip: 'Best for comparing two classification models on the same dataset',
        recommended: true,
        complexity: 'Simple',
        assumptions: ['Paired data', 'Binary outcomes', 'Same test set'],
        category: 'Non-parametric'
    },
    {
        id: 'delong',
        label: 'DeLong Test',
        description: 'Compares ROC curves for statistical significance',
        applicableTo: ['classification'],
        tooltip: 'Specifically designed for comparing AUC values between models',
        recommended: true,
        complexity: 'Moderate',
        assumptions: ['ROC curves available', 'Binary classification'],
        category: 'ROC-based'
    },
    {
        id: 'five_two_cv',
        label: '5×2 Cross-Validation F-Test',
        description: 'Robust cross-validation based comparison',
        applicableTo: ['classification', 'regression'],
        tooltip: 'Reduces Type I error compared to simple t-test',
        recommended: false,
        complexity: 'Complex',
        assumptions: ['Sufficient data for CV', 'Independent folds'],
        category: 'Cross-validation'
    },
    {
        id: 'bootstrap_confidence',
        label: 'Bootstrap Confidence Intervals',
        description: 'Non-parametric confidence interval estimation',
        applicableTo: ['classification', 'regression'],
        tooltip: 'Does not assume normal distribution of performance differences',
        recommended: true,
        complexity: 'Moderate',
        assumptions: ['Large enough sample size', 'Resampling validity'],
        category: 'Resampling'
    },
    {
        id: 'diebold_mariano',
        label: 'Diebold-Mariano Test',
        description: 'Compares predictive accuracy of forecasting models',
        applicableTo: ['regression'],
        tooltip: 'Popular for time series and regression model comparisons',
        recommended: true,
        complexity: 'Moderate',
        assumptions: ['Time series data', 'Stationary differences'],
        category: 'Time series'
    },
    {
        id: 'paired_ttest',
        label: 'Paired t-Test',
        description: 'Classical statistical test for paired samples',
        applicableTo: ['classification', 'regression'],
        tooltip: 'Assumes normal distribution of performance differences',
        recommended: false,
        complexity: 'Simple',
        assumptions: ['Normal distribution', 'Paired observations'],
        category: 'Parametric'
    }
];

const UploadTab: React.FC = () => {
    const [uploadState, setUploadState] = useState<UploadState>({
        uploadProgress: {},
        selectedMetrics: [],
        isModelProcessing: false,
        analysisName: 'Drift Analysis - ' + new Date().toLocaleDateString(),
        validationErrors: []
    });

    const [config, setConfig] = useState<AnalysisConfiguration>({
        analysis_name: 'Drift Analysis - ' + new Date().toLocaleDateString(),
        description: '',
        drift_thresholds: {
            low_threshold: 1.0,
            medium_threshold: 2.0,
            high_threshold: 3.0,
        },
        statistical_tests: ['ks_test', 'chi_square', 'psi'],
    });

    const [showPreview, setShowPreview] = useState<{
        reference: boolean;
        current: boolean;
    }>({
        reference: false,
        current: false,
    });

    // Mock file upload handlers
    const onReferenceDatasetDrop = useCallback((acceptedFiles: File[]) => {
        const file = acceptedFiles[0];
        if (file) {
            // Simulate upload progress
            setUploadState(prev => ({
                ...prev,
                uploadProgress: { ...prev.uploadProgress, reference: 0 }
            }));

            // Simulate progressive upload
            const progressInterval = setInterval(() => {
                setUploadState(prev => {
                    const currentProgress = prev.uploadProgress.reference || 0;
                    if (currentProgress >= 100) {
                        clearInterval(progressInterval);
                        return {
                            ...prev,
                            referenceDataset: { ...mockReferenceDataset, name: file.name },
                            uploadProgress: { ...prev.uploadProgress, reference: 100 }
                        };
                    }
                    return {
                        ...prev,
                        uploadProgress: { ...prev.uploadProgress, reference: currentProgress + 10 }
                    };
                });
            }, 200);
        }
    }, []);

    const onCurrentDatasetDrop = useCallback((acceptedFiles: File[]) => {
        const file = acceptedFiles[0];
        if (file) {
            setUploadState(prev => ({
                ...prev,
                uploadProgress: { ...prev.uploadProgress, current: 0 }
            }));

            const progressInterval = setInterval(() => {
                setUploadState(prev => {
                    const currentProgress = prev.uploadProgress.current || 0;
                    if (currentProgress >= 100) {
                        clearInterval(progressInterval);
                        return {
                            ...prev,
                            currentDataset: { ...mockCurrentDataset, name: file.name },
                            uploadProgress: { ...prev.uploadProgress, current: 100 }
                        };
                    }
                    return {
                        ...prev,
                        uploadProgress: { ...prev.uploadProgress, current: currentProgress + 10 }
                    };
                });
            }, 200);
        }
    }, []);

    const onModelDrop = useCallback((acceptedFiles: File[]) => {
        const file = acceptedFiles[0];
        if (file) {
            // Start upload progress
            setUploadState(prev => ({
                ...prev,
                uploadProgress: { ...prev.uploadProgress, model: 0 },
                isModelProcessing: true,
                modelProcessingStage: 'uploading'
            }));

            // Simulate upload progress
            const progressInterval = setInterval(() => {
                setUploadState(prev => {
                    const currentProgress = prev.uploadProgress.model || 0;
                    if (currentProgress >= 100) {
                        clearInterval(progressInterval);
                        // Start model processing after upload completes
                        setUploadState(prev => ({
                            ...prev,
                            model: { ...mockModelInfo, name: file.name },
                            uploadProgress: { ...prev.uploadProgress, model: 100 },
                            modelProcessingStage: 'analyzing'
                        }));

                        // Simulate model analysis stages
                        setTimeout(() => {
                            setUploadState(prev => ({
                                ...prev,
                                modelProcessingStage: 'extracting-metadata'
                            }));
                        }, 1000);

                        setTimeout(() => {
                            setUploadState(prev => ({
                                ...prev,
                                isModelProcessing: false,
                                modelProcessingStage: 'complete',
                                modelType: 'classification', // Auto-detect or default
                                selectedMetrics: classificationMetrics.filter(m => m.default).map(m => m.id)
                            }));
                        }, 3000);

                        return prev;
                    }
                    return {
                        ...prev,
                        uploadProgress: { ...prev.uploadProgress, model: currentProgress + 10 }
                    };
                });
            }, 200);
        }
    }, []);

    const {
        getRootProps: getReferenceRootProps,
        getInputProps: getReferenceInputProps,
        isDragActive: isReferenceDragActive,
    } = useDropzone({
        onDrop: onReferenceDatasetDrop,
        accept: { 'text/csv': ['.csv'] },
        multiple: false,
    });

    const {
        getRootProps: getCurrentRootProps,
        getInputProps: getCurrentInputProps,
        isDragActive: isCurrentDragActive,
    } = useDropzone({
        onDrop: onCurrentDatasetDrop,
        accept: { 'text/csv': ['.csv'] },
        multiple: false,
    });

    const {
        getRootProps: getModelRootProps,
        getInputProps: getModelInputProps,
        isDragActive: isModelDragActive,
    } = useDropzone({
        onDrop: onModelDrop,
        accept: {
            'application/octet-stream': ['.pkl'],
            'application/x-pickle': ['.pkl'],
            'application/onnx': ['.onnx']
        },
        multiple: false,
    });

    // Handler functions for dynamic configuration
    const handleModelTypeChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const newModelType = event.target.value as 'classification' | 'regression';
        const defaultMetrics = newModelType === 'classification'
            ? classificationMetrics.filter(m => m.default).map(m => m.id)
            : regressionMetrics.filter(m => m.default).map(m => m.id);

        // Get recommended statistical test for the model type
        const recommendedTest = statisticalTests.find(test =>
            test.applicableTo.includes(newModelType) && test.recommended
        );

        setUploadState(prev => ({
            ...prev,
            modelType: newModelType,
            selectedMetrics: defaultMetrics,
            statisticalTest: recommendedTest?.id || undefined,
            validationErrors: validateConfiguration({
                ...prev,
                modelType: newModelType,
                selectedMetrics: defaultMetrics
            })
        }));
    };

    const handleMetricToggle = (metricId: string) => {
        setUploadState(prev => {
            const newSelectedMetrics = prev.selectedMetrics.includes(metricId)
                ? prev.selectedMetrics.filter(id => id !== metricId)
                : [...prev.selectedMetrics, metricId];

            return {
                ...prev,
                selectedMetrics: newSelectedMetrics,
                validationErrors: validateConfiguration({
                    ...prev,
                    selectedMetrics: newSelectedMetrics
                })
            };
        });
    };

    const handleStatisticalTestChange = (event: React.ChangeEvent<{ value: unknown }>) => {
        setUploadState(prev => ({
            ...prev,
            statisticalTest: event.target.value as string,
            validationErrors: validateConfiguration({
                ...prev,
                statisticalTest: event.target.value as string
            })
        }));
    };

    const handleAnalysisNameChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setUploadState(prev => ({
            ...prev,
            analysisName: event.target.value,
            validationErrors: validateConfiguration({
                ...prev,
                analysisName: event.target.value
            })
        }));
    };

    // Validation function
    const validateConfiguration = (state: Partial<UploadState>): string[] => {
        const errors: string[] = [];

        if (!state.analysisName?.trim()) {
            errors.push('Analysis name is required');
        }

        if (state.model && !state.modelType) {
            errors.push('Please select a model type');
        }

        if (state.model && state.selectedMetrics && state.selectedMetrics.length === 0) {
            errors.push('Please select at least one performance metric');
        }

        if (state.model && state.modelType && state.selectedMetrics && state.selectedMetrics.length > 0 && !state.statisticalTest) {
            errors.push('Please select a statistical significance test');
        }

        return errors;
    };

    const isReadyForAnalysis = uploadState.referenceDataset &&
        uploadState.currentDataset &&
        uploadState.model &&
        !uploadState.isModelProcessing &&
        uploadState.modelType &&
        uploadState.selectedMetrics.length > 0 &&
        uploadState.analysisName.trim() &&
        uploadState.validationErrors.length === 0;

    const UploadCard: React.FC<{
        title: string;
        description: string;
        getRootProps: any;
        getInputProps: any;
        isDragActive: boolean;
        uploadProgress?: number;
        uploadedFile?: Dataset | ModelInfo;
        acceptedTypes: string;
        icon: React.ReactNode;
        isRequired?: boolean;
        isProcessing?: boolean;
        processingStage?: string;
    }> = ({
        title,
        description,
        getRootProps,
        getInputProps,
        isDragActive,
        uploadProgress,
        uploadedFile,
        acceptedTypes,
        icon,
        isRequired = true,
        isProcessing = false,
        processingStage
    }) => {
            const getProcessingMessage = () => {
                switch (processingStage) {
                    case 'uploading':
                        return 'Uploading model file...';
                    case 'analyzing':
                        return 'Analyzing model structure...';
                    case 'extracting-metadata':
                        return 'Extracting model metadata...';
                    case 'complete':
                        return 'Processing complete!';
                    default:
                        return 'Processing model...';
                }
            };

            return (
                <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                    <CardContent sx={{ flexGrow: 1 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                            {icon}
                            <Box sx={{ ml: 1 }}>
                                <Typography variant="h6" gutterBottom>
                                    {title} {isRequired && <Chip label="Required" size="small" color="primary" sx={{ ml: 1 }} />}
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                    {description}
                                </Typography>
                            </Box>
                        </Box>

                        {!uploadedFile ? (
                            <Box
                                {...getRootProps()}
                                sx={{
                                    border: 2,
                                    borderStyle: 'dashed',
                                    borderColor: isDragActive ? 'primary.main' : 'grey.300',
                                    borderRadius: 2,
                                    p: 3,
                                    textAlign: 'center',
                                    cursor: 'pointer',
                                    bgcolor: isDragActive ? 'primary.50' : 'background.default',
                                    transition: 'all 0.2s ease-in-out',
                                    '&:hover': {
                                        borderColor: 'primary.main',
                                        bgcolor: 'primary.50',
                                    },
                                }}
                            >
                                <input {...getInputProps()} />
                                <CloudUpload sx={{ fontSize: 48, color: 'grey.400', mb: 2 }} />
                                <Typography variant="body1" gutterBottom>
                                    {isDragActive ? 'Drop file here' : 'Drag & drop file here, or click to select'}
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                    Accepted: {acceptedTypes}
                                </Typography>
                            </Box>
                        ) : (
                            <Alert severity="success">
                                <Box display="flex" alignItems="center" justifyContent="space-between">
                                    <Box>
                                        <Typography variant="body2" fontWeight="bold">
                                            {uploadedFile.name}
                                        </Typography>
                                        {'rows' in uploadedFile && (
                                            <Typography variant="body2">
                                                {uploadedFile.rows.toLocaleString()} rows, {uploadedFile.columns.length} columns
                                            </Typography>
                                        )}
                                        {'type' in uploadedFile && (
                                            <Typography variant="body2">
                                                {uploadedFile.type} - {(uploadedFile.size / 1024 / 1024).toFixed(1)} MB
                                            </Typography>
                                        )}
                                    </Box>
                                    <CheckCircle color="success" />
                                </Box>
                            </Alert>
                        )}

                        {uploadProgress !== undefined && uploadProgress < 100 && (
                            <Box sx={{ mt: 2 }}>
                                <Typography variant="body2" gutterBottom>
                                    Uploading... {uploadProgress}%
                                </Typography>
                                <LinearProgress variant="determinate" value={uploadProgress} />
                            </Box>
                        )}

                        {isProcessing && (
                            <Box sx={{ mt: 2 }}>
                                <Typography variant="body2" gutterBottom color="primary">
                                    {getProcessingMessage()}
                                </Typography>
                                <LinearProgress color="secondary" />
                                {processingStage && (
                                    <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                                        Stage: {processingStage}
                                    </Typography>
                                )}
                            </Box>
                        )}
                    </CardContent>
                </Card>
            );
        }; return (
            <Box>
                <Typography variant="h4" gutterBottom>
                    Upload & Configuration
                </Typography>
                <Typography variant="body1" color="text.secondary" paragraph>
                    Upload your datasets and model to begin comprehensive drift analysis. The interface will adapt based on your uploads.
                </Typography>

                {/* Unified Upload Section */}
                <Paper sx={{ p: 3, mb: 3 }}>
                    <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                        <DatasetLinked sx={{ mr: 1 }} />
                        File Upload
                    </Typography>

                    <Grid container spacing={3}>
                        <Grid item xs={12} md={4}>
                            <UploadCard
                                title="Reference Dataset"
                                description="Historical data used as baseline for comparison"
                                getRootProps={getReferenceRootProps}
                                getInputProps={getReferenceInputProps}
                                isDragActive={isReferenceDragActive}
                                uploadProgress={uploadState.uploadProgress.reference}
                                uploadedFile={uploadState.referenceDataset}
                                acceptedTypes="CSV files"
                                icon={<Analytics color="primary" />}
                                isRequired={true}
                            />
                        </Grid>
                        <Grid item xs={12} md={4}>
                            <UploadCard
                                title="Current Dataset"
                                description="Recent data to analyze for potential drift"
                                getRootProps={getCurrentRootProps}
                                getInputProps={getCurrentInputProps}
                                isDragActive={isCurrentDragActive}
                                uploadProgress={uploadState.uploadProgress.current}
                                uploadedFile={uploadState.currentDataset}
                                acceptedTypes="CSV files"
                                icon={<Analytics color="primary" />}
                                isRequired={true}
                            />
                        </Grid>
                        <Grid item xs={12} md={4}>
                            <UploadCard
                                title="Trained Model"
                                description="Machine learning model for performance analysis"
                                getRootProps={getModelRootProps}
                                getInputProps={getModelInputProps}
                                isDragActive={isModelDragActive}
                                uploadProgress={uploadState.uploadProgress.model}
                                uploadedFile={uploadState.model}
                                acceptedTypes="PKL, ONNX files"
                                icon={<ModelTraining color="secondary" />}
                                isRequired={false}
                                isProcessing={uploadState.isModelProcessing}
                                processingStage={uploadState.modelProcessingStage}
                            />
                        </Grid>
                    </Grid>
                </Paper>

                {/* Basic Configuration - Always Visible */}
                <Paper sx={{ p: 3, mb: 3 }}>
                    <Typography variant="h5" gutterBottom>
                        Analysis Configuration
                    </Typography>

                    <Grid container spacing={3}>
                        <Grid item xs={12} md={6}>
                            <TextField
                                fullWidth
                                label="Analysis Name"
                                value={uploadState.analysisName}
                                onChange={handleAnalysisNameChange}
                                required
                                error={uploadState.validationErrors.includes('Analysis name is required')}
                                helperText={uploadState.validationErrors.includes('Analysis name is required') ? 'Analysis name is required' : ''}
                            />
                        </Grid>
                        <Grid item xs={12} md={6}>
                            <TextField
                                fullWidth
                                label="Description (Optional)"
                                value={config.description}
                                onChange={(e) => setConfig(prev => ({ ...prev, description: e.target.value }))}
                                multiline
                                rows={1}
                            />
                        </Grid>

                        <Grid item xs={12}>
                            <Divider sx={{ my: 2 }} />
                            <Typography variant="h6" gutterBottom>
                                Drift Detection Thresholds
                            </Typography>
                        </Grid>

                        <Grid item xs={12} md={4}>
                            <TextField
                                fullWidth
                                label="Low Threshold"
                                type="number"
                                value={config.drift_thresholds.low_threshold}
                                onChange={(e) => setConfig(prev => ({
                                    ...prev,
                                    drift_thresholds: {
                                        ...prev.drift_thresholds,
                                        low_threshold: parseFloat(e.target.value)
                                    }
                                }))}
                                inputProps={{ step: 0.1, min: 0 }}
                                helperText="Values below this threshold indicate low drift"
                            />
                        </Grid>
                        <Grid item xs={12} md={4}>
                            <TextField
                                fullWidth
                                label="Medium Threshold"
                                type="number"
                                value={config.drift_thresholds.medium_threshold}
                                onChange={(e) => setConfig(prev => ({
                                    ...prev,
                                    drift_thresholds: {
                                        ...prev.drift_thresholds,
                                        medium_threshold: parseFloat(e.target.value)
                                    }
                                }))}
                                inputProps={{ step: 0.1, min: 0 }}
                                helperText="Values between low and medium thresholds"
                            />
                        </Grid>
                        <Grid item xs={12} md={4}>
                            <TextField
                                fullWidth
                                label="High Threshold"
                                type="number"
                                value={config.drift_thresholds.high_threshold}
                                onChange={(e) => setConfig(prev => ({
                                    ...prev,
                                    drift_thresholds: {
                                        ...prev.drift_thresholds,
                                        high_threshold: parseFloat(e.target.value)
                                    }
                                }))}
                                inputProps={{ step: 0.1, min: 0 }}
                                helperText="Values above this threshold indicate high drift"
                            />
                        </Grid>
                    </Grid>
                </Paper>

                {/* Model-Dependent Configuration - Only show when model uploaded */}
                <Collapse in={!!uploadState.model && !uploadState.isModelProcessing}>
                    <Paper sx={{ p: 3, mb: 3 }}>
                        <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                            <ModelTraining sx={{ mr: 1 }} />
                            Model Analysis Configuration
                        </Typography>

                        <Alert severity="info" sx={{ mb: 3 }}>
                            <Typography variant="body2">
                                Configure model-specific analysis options. Select your model type and performance metrics for comprehensive drift analysis.
                            </Typography>
                        </Alert>

                        <Grid container spacing={3}>
                            {/* Model Type Selection */}
                            <Grid item xs={12}>
                                <FormControl component="fieldset">
                                    <FormLabel component="legend">
                                        <Typography variant="h6">Model Type</Typography>
                                    </FormLabel>
                                    <RadioGroup
                                        row
                                        value={uploadState.modelType || ''}
                                        onChange={handleModelTypeChange}
                                        sx={{ mt: 1 }}
                                    >
                                        <FormControlLabel
                                            value="classification"
                                            control={<Radio />}
                                            label="Classification Model"
                                        />
                                        <FormControlLabel
                                            value="regression"
                                            control={<Radio />}
                                            label="Regression Model"
                                        />
                                    </RadioGroup>
                                </FormControl>
                            </Grid>

                            {/* Performance Metrics Selection */}
                            {uploadState.modelType && (
                                <Grid item xs={12}>
                                    <Typography variant="h6" gutterBottom>
                                        Performance Metrics Selection
                                    </Typography>
                                    <Alert severity="info" sx={{ mb: 2 }}>
                                        <Typography variant="body2">
                                            Select metrics to evaluate model performance drift.
                                            Default metrics are pre-selected based on your model type.
                                            {uploadState.selectedMetrics.length > 0 && (
                                                ` Currently selected: ${uploadState.selectedMetrics.length} metrics.`
                                            )}
                                        </Typography>
                                    </Alert>

                                    <Box sx={{ mb: 2 }}>
                                        <Typography variant="subtitle1" gutterBottom>
                                            Core Metrics (Recommended)
                                        </Typography>
                                        <FormGroup>
                                            <Grid container spacing={1}>
                                                {(uploadState.modelType === 'classification' ? classificationMetrics : regressionMetrics)
                                                    .filter(metric => metric.default)
                                                    .map((metric) => (
                                                        <Grid item xs={12} sm={6} md={4} key={metric.id}>
                                                            <Tooltip title={metric.description} arrow placement="top">
                                                                <FormControlLabel
                                                                    control={
                                                                        <Checkbox
                                                                            checked={uploadState.selectedMetrics.includes(metric.id)}
                                                                            onChange={() => handleMetricToggle(metric.id)}
                                                                            color="primary"
                                                                        />
                                                                    }
                                                                    label={
                                                                        <Box>
                                                                            <Typography variant="body2" fontWeight="medium">
                                                                                {metric.label}
                                                                            </Typography>
                                                                        </Box>
                                                                    }
                                                                />
                                                            </Tooltip>
                                                        </Grid>
                                                    ))}
                                            </Grid>
                                        </FormGroup>
                                    </Box>

                                    <Box>
                                        <Typography variant="subtitle1" gutterBottom>
                                            Additional Metrics (Optional)
                                        </Typography>
                                        <FormGroup>
                                            <Grid container spacing={1}>
                                                {(uploadState.modelType === 'classification' ? classificationMetrics : regressionMetrics)
                                                    .filter(metric => !metric.default)
                                                    .map((metric) => (
                                                        <Grid item xs={12} sm={6} md={4} key={metric.id}>
                                                            <Tooltip title={metric.description} arrow placement="top">
                                                                <FormControlLabel
                                                                    control={
                                                                        <Checkbox
                                                                            checked={uploadState.selectedMetrics.includes(metric.id)}
                                                                            onChange={() => handleMetricToggle(metric.id)}
                                                                            color="primary"
                                                                        />
                                                                    }
                                                                    label={
                                                                        <Box>
                                                                            <Typography variant="body2">
                                                                                {metric.label}
                                                                            </Typography>
                                                                        </Box>
                                                                    }
                                                                />
                                                            </Tooltip>
                                                        </Grid>
                                                    ))}
                                            </Grid>
                                        </FormGroup>
                                    </Box>

                                    {uploadState.validationErrors.includes('Please select at least one performance metric') && (
                                        <Alert severity="error" sx={{ mt: 2 }}>
                                            Please select at least one performance metric.
                                        </Alert>
                                    )}
                                </Grid>
                            )}

                            {/* Statistical Significance Testing */}
                            {uploadState.modelType && uploadState.selectedMetrics.length > 0 && (
                                <Grid item xs={12}>
                                    <Typography variant="h6" gutterBottom>
                                        Statistical Significance Testing
                                    </Typography>
                                    <Alert severity="info" sx={{ mb: 2 }}>
                                        <Typography variant="body2">
                                            Choose a statistical test to determine if performance differences are statistically significant.
                                            Tests are filtered based on your selected model type.
                                        </Typography>
                                    </Alert>

                                    <FormControl fullWidth>
                                        <InputLabel>Statistical Test</InputLabel>
                                        <Select
                                            value={uploadState.statisticalTest || ''}
                                            onChange={handleStatisticalTestChange as any}
                                            label="Statistical Test"
                                        >
                                            {statisticalTests
                                                .filter(test => test.applicableTo.includes(uploadState.modelType!))
                                                .sort((a, b) => (b.recommended ? 1 : 0) - (a.recommended ? 1 : 0)) // Recommended first
                                                .map((test) => (
                                                    <MenuItem key={test.id} value={test.id}>
                                                        <Box sx={{ width: '100%' }}>
                                                            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                                                <Typography variant="body1" fontWeight={test.recommended ? 'bold' : 'normal'}>
                                                                    {test.label}
                                                                    {test.recommended && (
                                                                        <Chip label="Recommended" size="small" color="primary" sx={{ ml: 1 }} />
                                                                    )}
                                                                </Typography>
                                                                <Chip
                                                                    label={test.complexity}
                                                                    size="small"
                                                                    variant="outlined"
                                                                    color={test.complexity === 'Simple' ? 'success' : test.complexity === 'Moderate' ? 'warning' : 'error'}
                                                                />
                                                            </Box>
                                                            <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                                                                {test.description}
                                                            </Typography>
                                                            <Typography variant="caption" color="text.secondary">
                                                                Category: {test.category} | Assumptions: {test.assumptions.join(', ')}
                                                            </Typography>
                                                        </Box>
                                                    </MenuItem>
                                                ))}
                                        </Select>
                                    </FormControl>

                                    {uploadState.statisticalTest && (
                                        <Box sx={{ mt: 2 }}>
                                            {(() => {
                                                const selectedTest = statisticalTests.find(test => test.id === uploadState.statisticalTest);
                                                return selectedTest && (
                                                    <Alert severity="success">
                                                        <Typography variant="body2" fontWeight="bold">
                                                            Selected: {selectedTest.label}
                                                        </Typography>
                                                        <Typography variant="body2">
                                                            {selectedTest.tooltip}
                                                        </Typography>
                                                    </Alert>
                                                );
                                            })()}
                                        </Box>
                                    )}

                                    {uploadState.validationErrors.includes('Please select a statistical significance test') && (
                                        <Alert severity="error" sx={{ mt: 2 }}>
                                            Please select a statistical significance test.
                                        </Alert>
                                    )}
                                </Grid>
                            )}
                        </Grid>
                    </Paper>
                </Collapse>

                {/* Data Preview Section */}
                {(uploadState.referenceDataset || uploadState.currentDataset) && (
                    <Paper sx={{ p: 3, mb: 3 }}>
                        <Typography variant="h5" gutterBottom>
                            Data Preview
                        </Typography>

                        <Box sx={{ mb: 2 }}>
                            {uploadState.referenceDataset && (
                                <FormControlLabel
                                    control={
                                        <Switch
                                            checked={showPreview.reference}
                                            onChange={(e) => setShowPreview(prev => ({ ...prev, reference: e.target.checked }))}
                                        />
                                    }
                                    label="Show Reference Dataset Preview"
                                />
                            )}
                            {uploadState.currentDataset && (
                                <FormControlLabel
                                    control={
                                        <Switch
                                            checked={showPreview.current}
                                            onChange={(e) => setShowPreview(prev => ({ ...prev, current: e.target.checked }))}
                                        />
                                    }
                                    label="Show Current Dataset Preview"
                                    sx={{ ml: 2 }}
                                />
                            )}
                        </Box>

                        {showPreview.reference && uploadState.referenceDataset?.preview && (
                            <Box sx={{ mb: 3 }}>
                                <Typography variant="h6" gutterBottom>
                                    Reference Dataset Sample
                                </Typography>
                                <TableContainer component={Paper} variant="outlined">
                                    <Table size="small">
                                        <TableHead>
                                            <TableRow>
                                                {uploadState.referenceDataset.columns.map((col) => (
                                                    <TableCell key={col} sx={{ fontWeight: 'bold' }}>
                                                        {col}
                                                    </TableCell>
                                                ))}
                                            </TableRow>
                                        </TableHead>
                                        <TableBody>
                                            {uploadState.referenceDataset.preview.map((row, index) => (
                                                <TableRow key={index}>
                                                    {uploadState.referenceDataset!.columns.map((col) => (
                                                        <TableCell key={col}>
                                                            {row[col]}
                                                        </TableCell>
                                                    ))}
                                                </TableRow>
                                            ))}
                                        </TableBody>
                                    </Table>
                                </TableContainer>
                            </Box>
                        )}

                        {showPreview.current && uploadState.currentDataset?.preview && (
                            <Box>
                                <Typography variant="h6" gutterBottom>
                                    Current Dataset Sample
                                </Typography>
                                <TableContainer component={Paper} variant="outlined">
                                    <Table size="small">
                                        <TableHead>
                                            <TableRow>
                                                {uploadState.currentDataset.columns.map((col) => (
                                                    <TableCell key={col} sx={{ fontWeight: 'bold' }}>
                                                        {col}
                                                    </TableCell>
                                                ))}
                                            </TableRow>
                                        </TableHead>
                                        <TableBody>
                                            {uploadState.currentDataset.preview.map((row, index) => (
                                                <TableRow key={index}>
                                                    {uploadState.currentDataset!.columns.map((col) => (
                                                        <TableCell key={col}>
                                                            {row[col]}
                                                        </TableCell>
                                                    ))}
                                                </TableRow>
                                            ))}
                                        </TableBody>
                                    </Table>
                                </TableContainer>
                            </Box>
                        )}
                    </Paper>
                )}

                {/* Start Analysis Section */}
                <Paper sx={{ p: 3, textAlign: 'center' }}>
                    <Button
                        variant="contained"
                        size="large"
                        startIcon={<PlayArrow />}
                        disabled={!isReadyForAnalysis}
                        sx={{
                            px: 4,
                            py: 1.5,
                            fontSize: '1.1rem',
                            fontWeight: 600
                        }}
                    >
                        Start Drift Analysis
                    </Button>

                    <Box sx={{ mt: 2 }}>
                        {!isReadyForAnalysis && (
                            <Alert severity="info">
                                <Typography variant="body2">
                                    To start analysis, please ensure:
                                    <ul style={{ margin: '8px 0', paddingLeft: '20px' }}>
                                        {!uploadState.referenceDataset && <li>Upload reference dataset</li>}
                                        {!uploadState.currentDataset && <li>Upload current dataset</li>}
                                        {!uploadState.analysisName.trim() && <li>Provide analysis name</li>}
                                        {uploadState.model && !uploadState.modelType && <li>Select model type</li>}
                                        {uploadState.model && uploadState.selectedMetrics.length === 0 && <li>Select performance metrics</li>}
                                        {uploadState.model && uploadState.selectedMetrics.length > 0 && !uploadState.statisticalTest && <li>Select statistical significance test</li>}
                                        {uploadState.isModelProcessing && <li>Wait for model processing to complete</li>}
                                        {uploadState.validationErrors.length > 0 && <li>Fix validation errors above</li>}
                                    </ul>
                                </Typography>
                            </Alert>
                        )}

                        {isReadyForAnalysis && (
                            <Alert severity="success">
                                <Typography variant="body2" fontWeight="bold">
                                    ✓ Ready to analyze! All requirements satisfied.
                                </Typography>
                                <Typography variant="body2" sx={{ mt: 1 }}>
                                    Analysis will include:
                                    <ul style={{ margin: '4px 0', paddingLeft: '20px' }}>
                                        <li>Data drift analysis between reference and current datasets</li>
                                        {uploadState.model && (
                                            <>
                                                <li>Model performance comparison using {uploadState.selectedMetrics.length} metrics</li>
                                                {uploadState.statisticalTest && (
                                                    <li>Statistical significance testing using {statisticalTests.find(t => t.id === uploadState.statisticalTest)?.label}</li>
                                                )}
                                            </>
                                        )}
                                    </ul>
                                </Typography>
                            </Alert>
                        )}
                    </Box>
                </Paper>
            </Box>
        );
};

export default UploadTab;
