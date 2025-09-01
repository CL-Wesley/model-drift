import React, { useState } from 'react';
import {
    Box,
    Typography,
    Grid,
    Card,
    CardContent,
    Button,
    Chip,
    Alert,
    Paper,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    SelectChangeEvent
} from '@mui/material';
import {
    TrendingUp,
    Warning,
    CheckCircle,
    Speed,
    Assessment,
    Refresh
} from '@mui/icons-material';
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    LineChart,
    Line,
    PieChart,
    Pie,
    Cell,
    Legend
} from 'recharts';

import { mockDriftResults } from '../data/mockData';

const DriftDashboard: React.FC = () => {
    const [selectedFeature, setSelectedFeature] = useState<string>('credit_score');

    const handleFeatureChange = (event: SelectChangeEvent) => {
        setSelectedFeature(event.target.value);
    };

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'high': return '#dc3545';
            case 'medium': return '#ffc107';
            case 'low': return '#28a745';
            default: return '#6c757d';
        }
    };

    const getStatusIcon = (status: string) => {
        switch (status) {
            case 'high': return <Warning />;
            case 'medium': return <TrendingUp />;
            case 'low': return <CheckCircle />;
            default: return <Assessment />;
        }
    };

    // Prepare data for charts
    const featureDriftData = mockDriftResults.feature_analysis.map(feature => ({
        name: feature.feature,
        drift_score: feature.drift_score,
        status: feature.status,
        fill: getStatusColor(feature.status)
    }));

    const statusDistribution = [
        { name: 'High Drift', value: mockDriftResults.high_drift_features, fill: '#dc3545' },
        { name: 'Medium Drift', value: mockDriftResults.medium_drift_features, fill: '#ffc107' },
        { name: 'Low Drift', value: mockDriftResults.low_drift_features, fill: '#28a745' }
    ];

    const selectedFeatureData = mockDriftResults.feature_analysis.find(f => f.feature === selectedFeature);
    const histogramData = selectedFeatureData?.distribution_ref?.histogram?.map((bin, index) => {
        const currentBin = selectedFeatureData?.distribution_current?.histogram?.[index];
        return {
            bin: `${bin.bin_start}-${bin.bin_end}`,
            reference: bin.frequency * 100,
            current: currentBin ? currentBin.frequency * 100 : 0
        };
    }) || [];

    return (
        <Box>
            <Typography variant="h4" gutterBottom>
                Drift Analysis Dashboard
            </Typography>
            <Typography variant="body1" color="text.secondary" paragraph>
                Comprehensive overview of model drift detection results and key metrics.
            </Typography>

            {/* Executive Summary */}
            <Paper sx={{ p: 3, mb: 3, background: 'linear-gradient(135deg, #1f4e79 0%, #2e6da4 100%)', color: 'white' }}>
                <Grid container spacing={3} alignItems="center">
                    <Grid item xs={12} md={8}>
                        <Typography variant="h5" gutterBottom sx={{ fontWeight: 600 }}>
                            Executive Summary
                        </Typography>
                        <Typography variant="h2" sx={{ fontWeight: 700, mb: 1 }}>
                            {mockDriftResults.overall_drift_score.toFixed(1)}
                        </Typography>
                        <Typography variant="h6" sx={{ mb: 2 }}>
                            Overall Drift Score
                        </Typography>
                        <Typography variant="body1" sx={{ opacity: 0.9 }}>
                            {mockDriftResults.executive_summary}
                        </Typography>
                    </Grid>
                    <Grid item xs={12} md={4} sx={{ textAlign: 'center' }}>
                        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                            <Chip
                                icon={getStatusIcon(mockDriftResults.overall_status)}
                                label={`${mockDriftResults.overall_status.toUpperCase()} DRIFT`}
                                sx={{
                                    backgroundColor: getStatusColor(mockDriftResults.overall_status),
                                    color: 'white',
                                    fontSize: '1rem',
                                    fontWeight: 600,
                                    px: 2,
                                    py: 1
                                }}
                            />
                            <Typography variant="body2" sx={{ opacity: 0.8 }}>
                                Analysis completed: {mockDriftResults.analysis_timestamp.toLocaleDateString()}
                            </Typography>
                            <Button
                                variant="outlined"
                                startIcon={<Refresh />}
                                sx={{ color: 'white', borderColor: 'white', '&:hover': { borderColor: 'white', backgroundColor: 'rgba(255,255,255,0.1)' } }}
                            >
                                Re-run Analysis
                            </Button>
                        </Box>
                    </Grid>
                </Grid>
            </Paper>

            {/* Key Metrics */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
                <Grid item xs={12} sm={6} md={3}>
                    <Card>
                        <CardContent sx={{ textAlign: 'center' }}>
                            <Speed sx={{ fontSize: 40, color: '#dc3545', mb: 1 }} />
                            <Typography variant="h4" sx={{ fontWeight: 600, color: '#dc3545' }}>
                                {mockDriftResults.high_drift_features}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                                High Drift Features
                            </Typography>
                        </CardContent>
                    </Card>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                    <Card>
                        <CardContent sx={{ textAlign: 'center' }}>
                            <TrendingUp sx={{ fontSize: 40, color: '#ffc107', mb: 1 }} />
                            <Typography variant="h4" sx={{ fontWeight: 600, color: '#ffc107' }}>
                                {mockDriftResults.medium_drift_features}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                                Medium Drift Features
                            </Typography>
                        </CardContent>
                    </Card>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                    <Card>
                        <CardContent sx={{ textAlign: 'center' }}>
                            <CheckCircle sx={{ fontSize: 40, color: '#28a745', mb: 1 }} />
                            <Typography variant="h4" sx={{ fontWeight: 600, color: '#28a745' }}>
                                {(mockDriftResults.data_quality_score * 100).toFixed(0)}%
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                                Data Quality Score
                            </Typography>
                        </CardContent>
                    </Card>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                    <Card>
                        <CardContent sx={{ textAlign: 'center' }}>
                            <Assessment sx={{ fontSize: 40, color: '#6f42c1', mb: 1 }} />
                            <Typography variant="h4" sx={{ fontWeight: 600, color: '#6f42c1' }}>
                                {mockDriftResults.total_features}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                                Total Features Analyzed
                            </Typography>
                        </CardContent>
                    </Card>
                </Grid>
            </Grid>

            {/* Charts Section */}
            <Grid container spacing={3}>
                {/* Feature Drift Scores */}
                <Grid item xs={12} lg={8}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                Feature Drift Scores
                            </Typography>
                            <ResponsiveContainer width="100%" height={300}>
                                <BarChart data={featureDriftData}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="name" />
                                    <YAxis />
                                    <Tooltip
                                        formatter={(value: any) => [value.toFixed(2), 'Drift Score']}
                                        labelFormatter={(label) => `Feature: ${label}`}
                                    />
                                    <Bar dataKey="drift_score" fill="#1f4e79" />
                                </BarChart>
                            </ResponsiveContainer>
                        </CardContent>
                    </Card>
                </Grid>

                {/* Status Distribution */}
                <Grid item xs={12} lg={4}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                Drift Status Distribution
                            </Typography>
                            <ResponsiveContainer width="100%" height={300}>
                                <PieChart>
                                    <Pie
                                        data={statusDistribution}
                                        cx="50%"
                                        cy="50%"
                                        outerRadius={80}
                                        dataKey="value"
                                        label={({ name, value }) => `${name}: ${value}`}
                                    >
                                        {statusDistribution.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={entry.fill} />
                                        ))}
                                    </Pie>
                                    <Tooltip />
                                </PieChart>
                            </ResponsiveContainer>
                        </CardContent>
                    </Card>
                </Grid>

                {/* Distribution Comparison */}
                <Grid item xs={12}>
                    <Card>
                        <CardContent>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                                <Typography variant="h6">
                                    Distribution Comparison
                                </Typography>
                                <FormControl sx={{ minWidth: 200 }}>
                                    <InputLabel>Select Feature</InputLabel>
                                    <Select
                                        value={selectedFeature}
                                        label="Select Feature"
                                        onChange={handleFeatureChange}
                                    >
                                        {mockDriftResults.feature_analysis.map((feature) => (
                                            <MenuItem key={feature.feature} value={feature.feature}>
                                                {feature.feature}
                                            </MenuItem>
                                        ))}
                                    </Select>
                                </FormControl>
                            </Box>
                            {selectedFeatureData && (
                                <Box>
                                    <Alert severity={selectedFeatureData.status === 'high' ? 'error' : selectedFeatureData.status === 'medium' ? 'warning' : 'success'} sx={{ mb: 2 }}>
                                        <Typography variant="body2">
                                            <strong>{selectedFeatureData.feature}</strong> - Drift Score: {selectedFeatureData.drift_score.toFixed(2)} |
                                            KL Divergence: {selectedFeatureData.kl_divergence.toFixed(3)} |
                                            P-Value: {selectedFeatureData.p_value.toFixed(3)}
                                        </Typography>
                                    </Alert>
                                    <ResponsiveContainer width="100%" height={300}>
                                        <BarChart data={histogramData}>
                                            <CartesianGrid strokeDasharray="3 3" />
                                            <XAxis dataKey="bin" />
                                            <YAxis label={{ value: 'Frequency (%)', angle: -90, position: 'insideLeft' }} />
                                            <Tooltip formatter={(value: any) => [`${value.toFixed(1)}%`, '']} />
                                            <Legend />
                                            <Bar dataKey="reference" fill="#1f4e79" name="Reference Data" />
                                            <Bar dataKey="current" fill="#dc3545" name="Current Data" />
                                        </BarChart>
                                    </ResponsiveContainer>
                                </Box>
                            )}
                        </CardContent>
                    </Card>
                </Grid>
            </Grid>

            {/* Recommendations */}
            <Card sx={{ mt: 3 }}>
                <CardContent>
                    <Typography variant="h6" gutterBottom>
                        Key Recommendations
                    </Typography>
                    <Grid container spacing={2}>
                        {mockDriftResults.recommendations?.map((recommendation, index) => (
                            <Grid item xs={12} md={6} key={index}>
                                <Alert severity="info">
                                    {recommendation}
                                </Alert>
                            </Grid>
                        ))}
                    </Grid>
                </CardContent>
            </Card>
        </Box>
    );
};

export default DriftDashboard;
