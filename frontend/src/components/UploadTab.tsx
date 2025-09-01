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
}

const UploadTab: React.FC = () => {
    const [uploadState, setUploadState] = useState<UploadState>({
        uploadProgress: {}
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
            setUploadState(prev => ({
                ...prev,
                uploadProgress: { ...prev.uploadProgress, model: 0 }
            }));

            const progressInterval = setInterval(() => {
                setUploadState(prev => {
                    const currentProgress = prev.uploadProgress.model || 0;
                    if (currentProgress >= 100) {
                        clearInterval(progressInterval);
                        return {
                            ...prev,
                            model: { ...mockModelInfo, name: file.name },
                            uploadProgress: { ...prev.uploadProgress, model: 100 }
                        };
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

    const isReadyForAnalysis = uploadState.referenceDataset &&
        uploadState.currentDataset &&
        uploadState.model &&
        config.analysis_name.trim();

    const UploadCard: React.FC<{
        title: string;
        description: string;
        getRootProps: any;
        getInputProps: any;
        isDragActive: boolean;
        uploadProgress?: number;
        uploadedFile?: Dataset | ModelInfo;
        acceptedTypes: string;
    }> = ({
        title,
        description,
        getRootProps,
        getInputProps,
        isDragActive,
        uploadProgress,
        uploadedFile,
        acceptedTypes
    }) => (
            <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                <CardContent sx={{ flexGrow: 1 }}>
                    <Typography variant="h6" gutterBottom>
                        {title}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                        {description}
                    </Typography>

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

                    {uploadProgress !== undefined && uploadProgress < 100 && (
                        <Box sx={{ mt: 2 }}>
                            <Typography variant="body2" gutterBottom>
                                Uploading... {uploadProgress}%
                            </Typography>
                            <LinearProgress variant="determinate" value={uploadProgress} />
                        </Box>
                    )}

                    {uploadedFile && (
                        <Alert severity="success" sx={{ mt: 2 }}>
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
                </CardContent>
            </Card>
        );

    return (
        <Box>
            <Typography variant="h4" gutterBottom>
                Upload & Configuration
            </Typography>
            <Typography variant="body1" color="text.secondary" paragraph>
                Upload your reference dataset, current dataset, and trained model to begin drift analysis.
            </Typography>

            {/* Upload Section */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
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
                    />
                </Grid>
                <Grid item xs={12} md={4}>
                    <UploadCard
                        title="Trained Model"
                        description="Machine learning model for compatibility analysis"
                        getRootProps={getModelRootProps}
                        getInputProps={getModelInputProps}
                        isDragActive={isModelDragActive}
                        uploadProgress={uploadState.uploadProgress.model}
                        uploadedFile={uploadState.model}
                        acceptedTypes="PKL, ONNX files"
                    />
                </Grid>
            </Grid>

            {/* Configuration Panel */}
            <Paper sx={{ p: 3, mb: 3 }}>
                <Typography variant="h5" gutterBottom>
                    Analysis Configuration
                </Typography>

                <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                        <TextField
                            fullWidth
                            label="Analysis Name"
                            value={config.analysis_name}
                            onChange={(e) => setConfig(prev => ({ ...prev, analysis_name: e.target.value }))}
                            required
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
                            Drift Thresholds
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
                        />
                    </Grid>
                </Grid>
            </Paper>

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

            {/* Start Analysis Button */}
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

                {!isReadyForAnalysis && (
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                        Please upload all required files and provide an analysis name to continue
                    </Typography>
                )}

                {isReadyForAnalysis && (
                    <Typography variant="body2" color="success.main" sx={{ mt: 2 }}>
                        ✓ Ready to analyze! All files uploaded and configuration complete.
                    </Typography>
                )}
            </Paper>
        </Box>
    );
};

export default UploadTab;
