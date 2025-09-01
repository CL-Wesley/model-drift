import React, { useState, useCallback } from 'react';
import {
    Box,
    Paper,
    Typography,
    Grid,
    Button,
    Card,
    CardContent,
    CardActions,
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
} from '@mui/material';
import {
    CloudUpload,
    CheckCircle,
    Error,
    Visibility,
    PlayArrow,
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';

import { Dataset, ModelInfo } from '../../types';

interface UploadState {
    referenceModel?: ModelInfo;
    currentModel?: ModelInfo;
    evaluationDataset?: Dataset;
    uploadProgress: {
        referenceModel?: number;
        currentModel?: number;
        evaluationDataset?: number;
    };
}

interface MetricConfig {
    name: string;
    enabled: boolean;
    threshold?: number;
}

const ModelUpload: React.FC = () => {
    const [uploadState, setUploadState] = useState<UploadState>({
        uploadProgress: {},
    });

    const [metrics, setMetrics] = useState<MetricConfig[]>([
        { name: 'Accuracy', enabled: true },
        { name: 'Precision', enabled: true },
        { name: 'Recall', enabled: true },
        { name: 'F1 Score', enabled: true },
        { name: 'AUC-ROC', enabled: true },
        { name: 'AUC-PR', enabled: false },
        { name: 'MAE', enabled: false },
        { name: 'RMSE', enabled: false },
        { name: 'R²', enabled: false },
    ]);

    const [statisticalTests, setStatisticalTests] = useState<string[]>(['McNemar', 'Bootstrap']);
    const [significanceLevel, setSignificanceLevel] = useState<number>(0.05);
    const [confidenceInterval, setConfidenceInterval] = useState<number>(95);

    // Mock data for demonstration
    const mockReferenceModel: ModelInfo = {
        id: 'model-001',
        name: 'Credit Scoring Model v1.0',
        type: 'classification',
        features: ['credit_score', 'income', 'age', 'employment_length', 'debt_to_income'],
        accuracy: 0.89,
        created_date: '2023-06-15',
        feature_importance: {
            credit_score: 0.35,
            income: 0.25,
            age: 0.15,
            employment_length: 0.15,
            debt_to_income: 0.10,
        },
        format: 'pkl',
        size: 5.2,
    };

    const mockCurrentModel: ModelInfo = {
        id: 'model-002',
        name: 'Credit Scoring Model v1.1',
        type: 'classification',
        features: ['credit_score', 'income', 'age', 'employment_length', 'debt_to_income'],
        accuracy: 0.87,
        created_date: '2023-12-10',
        feature_importance: {
            credit_score: 0.32,
            income: 0.28,
            age: 0.12,
            employment_length: 0.18,
            debt_to_income: 0.10,
        },
        format: 'pkl',
        size: 5.4,
    };

    const mockEvaluationDataset: Dataset = {
        id: 'dataset-003',
        name: 'Evaluation Dataset Q4 2023',
        rows: 5000,
        columns: ['customer_id', 'credit_score', 'income', 'age', 'employment_length', 'debt_to_income', 'target'],
        dataTypes: {
            customer_id: 'string',
            credit_score: 'numeric',
            income: 'numeric',
            age: 'numeric',
            employment_length: 'numeric',
            debt_to_income: 'numeric',
            target: 'binary',
        },
        uploadDate: new Date('2023-12-15'),
        size: 2.3,
    };

    // Dropzone for reference model
    const onDropReferenceModel = useCallback((acceptedFiles: File[]) => {
        // Simulate upload progress
        let progress = 0;
        const interval = setInterval(() => {
            progress += 10;
            setUploadState(prev => ({
                ...prev,
                uploadProgress: {
                    ...prev.uploadProgress,
                    referenceModel: progress,
                },
            }));

            if (progress >= 100) {
                clearInterval(interval);
                // Set mock data after upload completes
                setTimeout(() => {
                    setUploadState(prev => ({
                        ...prev,
                        referenceModel: mockReferenceModel,
                    }));
                }, 500);
            }
        }, 300);
    }, []);

    // Dropzone for current model
    const onDropCurrentModel = useCallback((acceptedFiles: File[]) => {
        // Simulate upload progress
        let progress = 0;
        const interval = setInterval(() => {
            progress += 10;
            setUploadState(prev => ({
                ...prev,
                uploadProgress: {
                    ...prev.uploadProgress,
                    currentModel: progress,
                },
            }));

            if (progress >= 100) {
                clearInterval(interval);
                // Set mock data after upload completes
                setTimeout(() => {
                    setUploadState(prev => ({
                        ...prev,
                        currentModel: mockCurrentModel,
                    }));
                }, 500);
            }
        }, 300);
    }, []);

    // Dropzone for evaluation dataset
    const onDropEvaluationDataset = useCallback((acceptedFiles: File[]) => {
        // Simulate upload progress
        let progress = 0;
        const interval = setInterval(() => {
            progress += 10;
            setUploadState(prev => ({
                ...prev,
                uploadProgress: {
                    ...prev.uploadProgress,
                    evaluationDataset: progress,
                },
            }));

            if (progress >= 100) {
                clearInterval(interval);
                // Set mock data after upload completes
                setTimeout(() => {
                    setUploadState(prev => ({
                        ...prev,
                        evaluationDataset: mockEvaluationDataset,
                    }));
                }, 500);
            }
        }, 300);
    }, []);

    const { getRootProps: getReferenceModelRootProps, getInputProps: getReferenceModelInputProps } = useDropzone({
        onDrop: onDropReferenceModel,
        accept: {
            'application/octet-stream': ['.pkl', '.onnx'],
        },
        maxFiles: 1,
    });

    const { getRootProps: getCurrentModelRootProps, getInputProps: getCurrentModelInputProps } = useDropzone({
        onDrop: onDropCurrentModel,
        accept: {
            'application/octet-stream': ['.pkl', '.onnx'],
        },
        maxFiles: 1,
    });

    const { getRootProps: getEvaluationDatasetRootProps, getInputProps: getEvaluationDatasetInputProps } = useDropzone({
        onDrop: onDropEvaluationDataset,
        accept: {
            'text/csv': ['.csv'],
            'application/vnd.ms-excel': ['.xls'],
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
        },
        maxFiles: 1,
    });

    const handleMetricToggle = (index: number) => {
        const updatedMetrics = [...metrics];
        updatedMetrics[index].enabled = !updatedMetrics[index].enabled;
        setMetrics(updatedMetrics);
    };

    const handleStatisticalTestChange = (event: React.ChangeEvent<{ value: unknown }>) => {
        setStatisticalTests(event.target.value as string[]);
    };

    const handleSignificanceLevelChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setSignificanceLevel(Number(event.target.value));
    };

    const handleConfidenceIntervalChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setConfidenceInterval(Number(event.target.value));
    };

    const handleStartAnalysis = () => {
        // In a real application, this would trigger the analysis process
        console.log('Starting model drift analysis with:', {
            referenceModel: uploadState.referenceModel,
            currentModel: uploadState.currentModel,
            evaluationDataset: uploadState.evaluationDataset,
            metrics: metrics.filter(m => m.enabled).map(m => m.name),
            statisticalTests,
            significanceLevel,
            confidenceInterval,
        });
    };

    const isReadyForAnalysis = !!uploadState.referenceModel && !!uploadState.currentModel && !!uploadState.evaluationDataset;

    return (
        <Box>
            <Typography variant="h5" gutterBottom>
                Model Drift Analysis Configuration
            </Typography>
            <Typography variant="body1" paragraph>
                Upload your reference and current models along with an evaluation dataset to analyze model drift.
            </Typography>

            <Grid container spacing={3}>
                {/* Model Upload Section */}
                <Grid item xs={12} md={6}>
                    <Paper sx={{ p: 3, height: '100%' }}>
                        <Typography variant="h6" gutterBottom>
                            Model Upload
                        </Typography>

                        {/* Reference Model Upload */}
                        <Box sx={{ mb: 4 }}>
                            <Typography variant="subtitle1" gutterBottom>
                                Reference Model (Baseline)
                            </Typography>
                            {!uploadState.referenceModel ? (
                                <Box
                                    {...getReferenceModelRootProps()}
                                    sx={{
                                        border: '2px dashed #cccccc',
                                        borderRadius: 2,
                                        p: 3,
                                        textAlign: 'center',
                                        cursor: 'pointer',
                                        mb: 2,
                                        '&:hover': {
                                            borderColor: 'primary.main',
                                            bgcolor: 'rgba(0, 0, 0, 0.01)',
                                        },
                                    }}
                                >
                                    <input {...getReferenceModelInputProps()} />
                                    <CloudUpload sx={{ fontSize: 40, color: 'text.secondary', mb: 1 }} />
                                    <Typography>Drag and drop your reference model file here, or click to select</Typography>
                                    <Typography variant="caption" color="text.secondary">
                                        Supported formats: .pkl, .onnx
                                    </Typography>
                                </Box>
                            ) : (
                                <Card variant="outlined" sx={{ mb: 2 }}>
                                    <CardContent>
                                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                                            <CheckCircle color="success" sx={{ mr: 1 }} />
                                            <Typography variant="subtitle2">{uploadState.referenceModel.name}</Typography>
                                        </Box>
                                        <Typography variant="body2" color="text.secondary">
                                            Type: {uploadState.referenceModel.type} | Size: {uploadState.referenceModel.size} MB
                                        </Typography>
                                        <Typography variant="body2" color="text.secondary">
                                            Created: {uploadState.referenceModel.created_date}
                                        </Typography>
                                    </CardContent>
                                    <CardActions>
                                        <Button size="small" startIcon={<Visibility />}>
                                            View Details
                                        </Button>
                                    </CardActions>
                                </Card>
                            )}

                            {uploadState.uploadProgress.referenceModel !== undefined && uploadState.uploadProgress.referenceModel < 100 && (
                                <Box sx={{ width: '100%', mt: 1 }}>
                                    <LinearProgress variant="determinate" value={uploadState.uploadProgress.referenceModel} />
                                    <Typography variant="caption" sx={{ mt: 0.5, display: 'block' }}>
                                        Uploading... {uploadState.uploadProgress.referenceModel}%
                                    </Typography>
                                </Box>
                            )}
                        </Box>

                        {/* Current Model Upload */}
                        <Box sx={{ mb: 4 }}>
                            <Typography variant="subtitle1" gutterBottom>
                                Current Model
                            </Typography>
                            {!uploadState.currentModel ? (
                                <Box
                                    {...getCurrentModelRootProps()}
                                    sx={{
                                        border: '2px dashed #cccccc',
                                        borderRadius: 2,
                                        p: 3,
                                        textAlign: 'center',
                                        cursor: 'pointer',
                                        mb: 2,
                                        '&:hover': {
                                            borderColor: 'primary.main',
                                            bgcolor: 'rgba(0, 0, 0, 0.01)',
                                        },
                                    }}
                                >
                                    <input {...getCurrentModelInputProps()} />
                                    <CloudUpload sx={{ fontSize: 40, color: 'text.secondary', mb: 1 }} />
                                    <Typography>Drag and drop your current model file here, or click to select</Typography>
                                    <Typography variant="caption" color="text.secondary">
                                        Supported formats: .pkl, .onnx
                                    </Typography>
                                </Box>
                            ) : (
                                <Card variant="outlined" sx={{ mb: 2 }}>
                                    <CardContent>
                                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                                            <CheckCircle color="success" sx={{ mr: 1 }} />
                                            <Typography variant="subtitle2">{uploadState.currentModel.name}</Typography>
                                        </Box>
                                        <Typography variant="body2" color="text.secondary">
                                            Type: {uploadState.currentModel.type} | Size: {uploadState.currentModel.size} MB
                                        </Typography>
                                        <Typography variant="body2" color="text.secondary">
                                            Created: {uploadState.currentModel.created_date}
                                        </Typography>
                                    </CardContent>
                                    <CardActions>
                                        <Button size="small" startIcon={<Visibility />}>
                                            View Details
                                        </Button>
                                    </CardActions>
                                </Card>
                            )}

                            {uploadState.uploadProgress.currentModel !== undefined && uploadState.uploadProgress.currentModel < 100 && (
                                <Box sx={{ width: '100%', mt: 1 }}>
                                    <LinearProgress variant="determinate" value={uploadState.uploadProgress.currentModel} />
                                    <Typography variant="caption" sx={{ mt: 0.5, display: 'block' }}>
                                        Uploading... {uploadState.uploadProgress.currentModel}%
                                    </Typography>
                                </Box>
                            )}
                        </Box>

                        {/* Evaluation Dataset Upload */}
                        <Box>
                            <Typography variant="subtitle1" gutterBottom>
                                Evaluation Dataset
                            </Typography>
                            {!uploadState.evaluationDataset ? (
                                <Box
                                    {...getEvaluationDatasetRootProps()}
                                    sx={{
                                        border: '2px dashed #cccccc',
                                        borderRadius: 2,
                                        p: 3,
                                        textAlign: 'center',
                                        cursor: 'pointer',
                                        mb: 2,
                                        '&:hover': {
                                            borderColor: 'primary.main',
                                            bgcolor: 'rgba(0, 0, 0, 0.01)',
                                        },
                                    }}
                                >
                                    <input {...getEvaluationDatasetInputProps()} />
                                    <CloudUpload sx={{ fontSize: 40, color: 'text.secondary', mb: 1 }} />
                                    <Typography>Drag and drop your evaluation dataset here, or click to select</Typography>
                                    <Typography variant="caption" color="text.secondary">
                                        Supported formats: .csv, .xls, .xlsx
                                    </Typography>
                                </Box>
                            ) : (
                                <Card variant="outlined" sx={{ mb: 2 }}>
                                    <CardContent>
                                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                                            <CheckCircle color="success" sx={{ mr: 1 }} />
                                            <Typography variant="subtitle2">{uploadState.evaluationDataset.name}</Typography>
                                        </Box>
                                        <Typography variant="body2" color="text.secondary">
                                            Rows: {uploadState.evaluationDataset.rows.toLocaleString()} | Size: {uploadState.evaluationDataset.size} MB
                                        </Typography>
                                        <Typography variant="body2" color="text.secondary">
                                            Uploaded: {uploadState.evaluationDataset.uploadDate.toLocaleDateString()}
                                        </Typography>
                                    </CardContent>
                                    <CardActions>
                                        <Button size="small" startIcon={<Visibility />}>
                                            Preview Data
                                        </Button>
                                    </CardActions>
                                </Card>
                            )}

                            {uploadState.uploadProgress.evaluationDataset !== undefined && uploadState.uploadProgress.evaluationDataset < 100 && (
                                <Box sx={{ width: '100%', mt: 1 }}>
                                    <LinearProgress variant="determinate" value={uploadState.uploadProgress.evaluationDataset} />
                                    <Typography variant="caption" sx={{ mt: 0.5, display: 'block' }}>
                                        Uploading... {uploadState.uploadProgress.evaluationDataset}%
                                    </Typography>
                                </Box>
                            )}
                        </Box>
                    </Paper>
                </Grid>

                {/* Configuration Section */}
                <Grid item xs={12} md={6}>
                    <Paper sx={{ p: 3, height: '100%' }}>
                        <Typography variant="h6" gutterBottom>
                            Analysis Configuration
                        </Typography>

                        {/* Performance Metrics */}
                        <Box sx={{ mb: 4 }}>
                            <Typography variant="subtitle1" gutterBottom>
                                Performance Metrics
                            </Typography>
                            <Typography variant="body2" color="text.secondary" paragraph>
                                Select the metrics to include in your model drift analysis.
                            </Typography>

                            <Grid container spacing={2}>
                                {metrics.map((metric, index) => (
                                    <Grid item xs={6} sm={4} key={metric.name}>
                                        <FormControlLabel
                                            control={
                                                <Switch
                                                    checked={metric.enabled}
                                                    onChange={() => handleMetricToggle(index)}
                                                    color="primary"
                                                />
                                            }
                                            label={metric.name}
                                        />
                                    </Grid>
                                ))}
                            </Grid>
                        </Box>

                        {/* Statistical Testing */}
                        <Box sx={{ mb: 4 }}>
                            <Typography variant="subtitle1" gutterBottom>
                                Statistical Testing
                            </Typography>
                            <Typography variant="body2" color="text.secondary" paragraph>
                                Configure statistical tests for significance analysis.
                            </Typography>

                            <Grid container spacing={3}>
                                <Grid item xs={12}>
                                    <FormControl fullWidth size="small">
                                        <InputLabel>Statistical Tests</InputLabel>
                                        <Select
                                            multiple
                                            value={statisticalTests}
                                            onChange={handleStatisticalTestChange as any}
                                            renderValue={(selected) => (
                                                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                                                    {(selected as string[]).map((value) => (
                                                        <Chip key={value} label={value} size="small" />
                                                    ))}
                                                </Box>
                                            )}
                                        >
                                            <MenuItem value="McNemar">McNemar's Test</MenuItem>
                                            <MenuItem value="DeLong">DeLong's Test for AUC</MenuItem>
                                            <MenuItem value="5x2CV">5x2 Cross-validation paired t-test</MenuItem>
                                            <MenuItem value="Bootstrap">Bootstrap Confidence Intervals</MenuItem>
                                            <MenuItem value="DieboldMariano">Diebold-Mariano Test</MenuItem>
                                        </Select>
                                    </FormControl>
                                </Grid>

                                <Grid item xs={12} sm={6}>
                                    <TextField
                                        fullWidth
                                        label="Significance Level (α)"
                                        type="number"
                                        value={significanceLevel}
                                        onChange={handleSignificanceLevelChange}
                                        inputProps={{ min: 0.01, max: 0.1, step: 0.01 }}
                                        size="small"
                                        helperText="Common values: 0.05 (5%), 0.01 (1%)"
                                    />
                                </Grid>

                                <Grid item xs={12} sm={6}>
                                    <TextField
                                        fullWidth
                                        label="Confidence Interval (%)"
                                        type="number"
                                        value={confidenceInterval}
                                        onChange={handleConfidenceIntervalChange}
                                        inputProps={{ min: 80, max: 99, step: 1 }}
                                        size="small"
                                        helperText="Common values: 95%, 99%"
                                    />
                                </Grid>
                            </Grid>
                        </Box>

                        {/* Analysis Controls */}
                        <Box>
                            <Divider sx={{ my: 3 }} />
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <Box>
                                    <Typography variant="subtitle1">Ready for Analysis</Typography>
                                    <Typography variant="body2" color="text.secondary">
                                        {isReadyForAnalysis
                                            ? 'All required files are uploaded. You can start the analysis.'
                                            : 'Please upload all required files to proceed.'}
                                    </Typography>
                                </Box>
                                <Button
                                    variant="contained"
                                    color="primary"
                                    size="large"
                                    startIcon={<PlayArrow />}
                                    disabled={!isReadyForAnalysis}
                                    onClick={handleStartAnalysis}
                                >
                                    Start Analysis
                                </Button>
                            </Box>

                            {!isReadyForAnalysis && (
                                <Alert severity="info" sx={{ mt: 2 }}>
                                    <Typography variant="body2">
                                        To perform model drift analysis, please upload:
                                        <ul style={{ margin: '8px 0', paddingLeft: '20px' }}>
                                            {!uploadState.referenceModel && <li>Reference Model (Baseline)</li>}
                                            {!uploadState.currentModel && <li>Current Model</li>}
                                            {!uploadState.evaluationDataset && <li>Evaluation Dataset</li>}
                                        </ul>
                                    </Typography>
                                </Alert>
                            )}
                        </Box>
                    </Paper>
                </Grid>
            </Grid>
        </Box>
    );
};

export default ModelUpload;