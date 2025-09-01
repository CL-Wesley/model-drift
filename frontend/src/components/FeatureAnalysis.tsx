import React, { useState } from 'react';
import {
    Box,
    Typography,
    Grid,
    Card,
    CardContent,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    SelectChangeEvent,
    Chip,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Paper,
    Alert
} from '@mui/material';
import {
    TrendingUp,
    TrendingDown,
    Remove,
    DataUsage
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
    ScatterChart,
    Scatter
} from 'recharts';

import { mockDriftResults } from '../data/mockData';

const FeatureAnalysis: React.FC = () => {
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
            case 'high': return <TrendingUp />;
            case 'medium': return <Remove />;
            case 'low': return <TrendingDown />;
            default: return <DataUsage />;
        }
    };

    const selectedFeatureData = mockDriftResults.feature_analysis.find(f => f.feature === selectedFeature);

    // Prepare histogram data
    const histogramData = selectedFeatureData?.distribution_ref?.histogram?.map((bin, index) => {
        const currentBin = selectedFeatureData?.distribution_current?.histogram?.[index];
        return {
            bin: `${bin.bin_start}-${bin.bin_end}`,
            reference: bin.frequency * 100,
            current: currentBin ? currentBin.frequency * 100 : 0,
            binMidpoint: (bin.bin_start + bin.bin_end) / 2
        };
    }) || [];

    // Prepare summary statistics
    const summaryStats = selectedFeatureData ? [
        { metric: 'Mean', reference: selectedFeatureData.distribution_ref?.mean?.toFixed(2) || 'N/A', current: selectedFeatureData.distribution_current?.mean?.toFixed(2) || 'N/A' },
        { metric: 'Std Dev', reference: selectedFeatureData.distribution_ref?.std?.toFixed(2) || 'N/A', current: selectedFeatureData.distribution_current?.std?.toFixed(2) || 'N/A' },
        { metric: 'Min', reference: selectedFeatureData.distribution_ref?.min?.toFixed(2) || 'N/A', current: selectedFeatureData.distribution_current?.min?.toFixed(2) || 'N/A' },
        { metric: 'Max', reference: selectedFeatureData.distribution_ref?.max?.toFixed(2) || 'N/A', current: selectedFeatureData.distribution_current?.max?.toFixed(2) || 'N/A' },
        { metric: 'Q1 (25%)', reference: selectedFeatureData.distribution_ref?.q25?.toFixed(2) || 'N/A', current: selectedFeatureData.distribution_current?.q25?.toFixed(2) || 'N/A' },
        { metric: 'Median (50%)', reference: selectedFeatureData.distribution_ref?.q50?.toFixed(2) || 'N/A', current: selectedFeatureData.distribution_current?.q50?.toFixed(2) || 'N/A' },
        { metric: 'Q3 (75%)', reference: selectedFeatureData.distribution_ref?.q75?.toFixed(2) || 'N/A', current: selectedFeatureData.distribution_current?.q75?.toFixed(2) || 'N/A' },
    ] : [];

    // Prepare drift metrics
    const driftMetrics = selectedFeatureData ? [
        { metric: 'Drift Score', value: selectedFeatureData.drift_score.toFixed(3), interpretation: 'Overall drift magnitude' },
        { metric: 'KL Divergence', value: selectedFeatureData.kl_divergence.toFixed(4), interpretation: 'Relative entropy between distributions' },
        { metric: 'PSI (Population Stability Index)', value: selectedFeatureData.psi.toFixed(4), interpretation: 'Population stability measure' },
        { metric: 'KS Statistic', value: selectedFeatureData.ks_statistic.toFixed(4), interpretation: 'Kolmogorov-Smirnov test statistic' },
        { metric: 'P-Value', value: selectedFeatureData.p_value.toFixed(4), interpretation: 'Statistical significance of difference' },
    ] : [];

    return (
        <Box>
            <Typography variant="h4" gutterBottom>
                Feature Analysis
            </Typography>
            <Typography variant="body1" color="text.secondary" paragraph>
                Deep dive analysis of individual feature drift patterns and statistical metrics.
            </Typography>

            {/* Feature Selection */}
            <Card sx={{ mb: 3 }}>
                <CardContent>
                    <Grid container spacing={3} alignItems="center">
                        <Grid item xs={12} md={4}>
                            <FormControl fullWidth>
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
                        </Grid>
                        {selectedFeatureData && (
                            <Grid item xs={12} md={8}>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                                    <Chip
                                        icon={getStatusIcon(selectedFeatureData.status)}
                                        label={`${selectedFeatureData.status.toUpperCase()} DRIFT`}
                                        sx={{
                                            backgroundColor: getStatusColor(selectedFeatureData.status),
                                            color: 'white',
                                            fontWeight: 600
                                        }}
                                    />
                                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                                        Drift Score: {selectedFeatureData.drift_score.toFixed(2)}
                                    </Typography>
                                    <Chip
                                        label={selectedFeatureData.data_type}
                                        variant="outlined"
                                        sx={{ textTransform: 'capitalize' }}
                                    />
                                </Box>
                            </Grid>
                        )}
                    </Grid>
                </CardContent>
            </Card>

            {selectedFeatureData && (
                <>
                    {/* Feature Overview Alert */}
                    <Alert
                        severity={selectedFeatureData.status === 'high' ? 'error' : selectedFeatureData.status === 'medium' ? 'warning' : 'success'}
                        sx={{ mb: 3 }}
                    >
                        <Typography variant="body1">
                            <strong>{selectedFeatureData.feature}</strong> shows <strong>{selectedFeatureData.status}</strong> drift with a score of {selectedFeatureData.drift_score.toFixed(2)}.
                            {selectedFeatureData.status === 'high' && ' This requires immediate attention.'}
                            {selectedFeatureData.status === 'medium' && ' Monitor closely for further changes.'}
                            {selectedFeatureData.status === 'low' && ' No immediate action required.'}
                        </Typography>
                    </Alert>

                    <Grid container spacing={3}>
                        {/* Distribution Comparison Chart */}
                        <Grid item xs={12}>
                            <Card>
                                <CardContent>
                                    <Typography variant="h6" gutterBottom>
                                        Distribution Comparison: Reference vs Current
                                    </Typography>
                                    <ResponsiveContainer width="100%" height={400}>
                                        <BarChart data={histogramData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                                            <CartesianGrid strokeDasharray="3 3" />
                                            <XAxis dataKey="bin" />
                                            <YAxis label={{ value: 'Frequency (%)', angle: -90, position: 'insideLeft' }} />
                                            <Tooltip
                                                formatter={(value: any, name: string) => [`${value.toFixed(1)}%`, name === 'reference' ? 'Reference Data' : 'Current Data']}
                                                labelFormatter={(label) => `Bin: ${label}`}
                                            />
                                            <Bar dataKey="reference" fill="#1f4e79" name="Reference Data" opacity={0.8} />
                                            <Bar dataKey="current" fill="#dc3545" name="Current Data" opacity={0.8} />
                                        </BarChart>
                                    </ResponsiveContainer>
                                </CardContent>
                            </Card>
                        </Grid>

                        {/* Summary Statistics */}
                        <Grid item xs={12} lg={6}>
                            <Card>
                                <CardContent>
                                    <Typography variant="h6" gutterBottom>
                                        Summary Statistics
                                    </Typography>
                                    <TableContainer>
                                        <Table size="small">
                                            <TableHead>
                                                <TableRow>
                                                    <TableCell><strong>Metric</strong></TableCell>
                                                    <TableCell align="right"><strong>Reference</strong></TableCell>
                                                    <TableCell align="right"><strong>Current</strong></TableCell>
                                                    <TableCell align="right"><strong>Change</strong></TableCell>
                                                </TableRow>
                                            </TableHead>
                                            <TableBody>
                                                {summaryStats.map((stat) => {
                                                    const refVal = parseFloat(stat.reference);
                                                    const currVal = parseFloat(stat.current);
                                                    const change = !isNaN(refVal) && !isNaN(currVal) ?
                                                        ((currVal - refVal) / refVal * 100).toFixed(1) + '%' : 'N/A';
                                                    const changeColor = !isNaN(refVal) && !isNaN(currVal) ?
                                                        (currVal > refVal ? '#dc3545' : currVal < refVal ? '#28a745' : '#6c757d') : '#6c757d';

                                                    return (
                                                        <TableRow key={stat.metric}>
                                                            <TableCell>{stat.metric}</TableCell>
                                                            <TableCell align="right">{stat.reference}</TableCell>
                                                            <TableCell align="right">{stat.current}</TableCell>
                                                            <TableCell align="right" sx={{ color: changeColor, fontWeight: 500 }}>
                                                                {change}
                                                            </TableCell>
                                                        </TableRow>
                                                    );
                                                })}
                                            </TableBody>
                                        </Table>
                                    </TableContainer>
                                </CardContent>
                            </Card>
                        </Grid>

                        {/* Drift Metrics */}
                        <Grid item xs={12} lg={6}>
                            <Card>
                                <CardContent>
                                    <Typography variant="h6" gutterBottom>
                                        Drift Detection Metrics
                                    </Typography>
                                    <TableContainer>
                                        <Table size="small">
                                            <TableHead>
                                                <TableRow>
                                                    <TableCell><strong>Metric</strong></TableCell>
                                                    <TableCell align="right"><strong>Value</strong></TableCell>
                                                </TableRow>
                                            </TableHead>
                                            <TableBody>
                                                {driftMetrics.map((metric) => (
                                                    <TableRow key={metric.metric}>
                                                        <TableCell>
                                                            <Typography variant="body2">
                                                                <strong>{metric.metric}</strong>
                                                            </Typography>
                                                            <Typography variant="caption" color="text.secondary">
                                                                {metric.interpretation}
                                                            </Typography>
                                                        </TableCell>
                                                        <TableCell align="right">
                                                            <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                                                {metric.value}
                                                            </Typography>
                                                        </TableCell>
                                                    </TableRow>
                                                ))}
                                            </TableBody>
                                        </Table>
                                    </TableContainer>
                                </CardContent>
                            </Card>
                        </Grid>

                        {/* Data Quality Assessment */}
                        <Grid item xs={12}>
                            <Card>
                                <CardContent>
                                    <Typography variant="h6" gutterBottom>
                                        Data Quality Assessment
                                    </Typography>
                                    <Grid container spacing={3}>
                                        <Grid item xs={12} sm={6} md={3}>
                                            <Box sx={{ textAlign: 'center', p: 2, backgroundColor: '#f8f9fa', borderRadius: 2 }}>
                                                <Typography variant="h4" sx={{ fontWeight: 600, color: selectedFeatureData.missing_values_ref > 0 ? '#dc3545' : '#28a745' }}>
                                                    {selectedFeatureData.missing_values_ref}
                                                </Typography>
                                                <Typography variant="body2" color="text.secondary">
                                                    Missing Values (Ref)
                                                </Typography>
                                            </Box>
                                        </Grid>
                                        <Grid item xs={12} sm={6} md={3}>
                                            <Box sx={{ textAlign: 'center', p: 2, backgroundColor: '#f8f9fa', borderRadius: 2 }}>
                                                <Typography variant="h4" sx={{ fontWeight: 600, color: selectedFeatureData.missing_values_current > 0 ? '#dc3545' : '#28a745' }}>
                                                    {selectedFeatureData.missing_values_current}
                                                </Typography>
                                                <Typography variant="body2" color="text.secondary">
                                                    Missing Values (Current)
                                                </Typography>
                                            </Box>
                                        </Grid>
                                        <Grid item xs={12} sm={6} md={3}>
                                            <Box sx={{ textAlign: 'center', p: 2, backgroundColor: '#f8f9fa', borderRadius: 2 }}>
                                                <Typography variant="h4" sx={{ fontWeight: 600, color: '#6f42c1' }}>
                                                    {selectedFeatureData.data_type === 'numerical' ? 'Continuous' : 'Categorical'}
                                                </Typography>
                                                <Typography variant="body2" color="text.secondary">
                                                    Data Type
                                                </Typography>
                                            </Box>
                                        </Grid>
                                        <Grid item xs={12} sm={6} md={3}>
                                            <Box sx={{ textAlign: 'center', p: 2, backgroundColor: '#f8f9fa', borderRadius: 2 }}>
                                                <Typography variant="h4" sx={{ fontWeight: 600, color: '#1f4e79' }}>
                                                    {selectedFeatureData.p_value < 0.05 ? 'Significant' : 'Not Significant'}
                                                </Typography>
                                                <Typography variant="body2" color="text.secondary">
                                                    Statistical Test
                                                </Typography>
                                            </Box>
                                        </Grid>
                                    </Grid>
                                </CardContent>
                            </Card>
                        </Grid>
                    </Grid>
                </>
            )}
        </Box>
    );
};

export default FeatureAnalysis;
