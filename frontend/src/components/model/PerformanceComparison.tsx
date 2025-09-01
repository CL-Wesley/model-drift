import React from 'react';
import {
    Box,
    Paper,
    Typography,
    Grid,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Chip,
    Card,
    CardContent,
    Tooltip,
    IconButton,
    Divider,
} from '@mui/material';
import { Info, TrendingDown, TrendingUp, TrendingFlat } from '@mui/icons-material';
import { 
    Radar, 
    RadarChart, 
    PolarGrid, 
    PolarAngleAxis, 
    PolarRadiusAxis, 
    ResponsiveContainer,
    Legend
} from 'recharts';

// Mock data for performance comparison
const performanceData = {
    referenceModel: {
        name: 'Credit Scoring Model v1.0',
        metrics: {
            accuracy: 0.89,
            precision: 0.86,
            recall: 0.83,
            f1Score: 0.84,
            aucRoc: 0.92,
            aucPr: 0.88,
        },
    },
    currentModel: {
        name: 'Credit Scoring Model v1.1',
        metrics: {
            accuracy: 0.87,
            precision: 0.84,
            recall: 0.81,
            f1Score: 0.82,
            aucRoc: 0.90,
            aucPr: 0.85,
        },
    },
    deltas: {
        accuracy: -0.02,
        precision: -0.02,
        recall: -0.02,
        f1Score: -0.02,
        aucRoc: -0.02,
        aucPr: -0.03,
    },
    significance: {
        accuracy: { pValue: 0.032, significant: true },
        precision: { pValue: 0.041, significant: true },
        recall: { pValue: 0.039, significant: true },
        f1Score: { pValue: 0.028, significant: true },
        aucRoc: { pValue: 0.045, significant: true },
        aucPr: { pValue: 0.062, significant: false },
    },
    confidenceIntervals: {
        accuracy: { lower: -0.035, upper: -0.005 },
        precision: { lower: -0.038, upper: -0.002 },
        recall: { lower: -0.042, upper: -0.001 },
        f1Score: { lower: -0.039, upper: -0.003 },
        aucRoc: { lower: -0.037, upper: -0.004 },
        aucPr: { lower: -0.048, upper: -0.012 },
    },
    riskLevel: 'Medium',
};

// Format data for radar chart
const radarData = [
    {
        metric: 'Accuracy',
        reference: performanceData.referenceModel.metrics.accuracy * 100,
        current: performanceData.currentModel.metrics.accuracy * 100,
    },
    {
        metric: 'Precision',
        reference: performanceData.referenceModel.metrics.precision * 100,
        current: performanceData.currentModel.metrics.precision * 100,
    },
    {
        metric: 'Recall',
        reference: performanceData.referenceModel.metrics.recall * 100,
        current: performanceData.currentModel.metrics.recall * 100,
    },
    {
        metric: 'F1 Score',
        reference: performanceData.referenceModel.metrics.f1Score * 100,
        current: performanceData.currentModel.metrics.f1Score * 100,
    },
    {
        metric: 'AUC-ROC',
        reference: performanceData.referenceModel.metrics.aucRoc * 100,
        current: performanceData.currentModel.metrics.aucRoc * 100,
    },
    {
        metric: 'AUC-PR',
        reference: performanceData.referenceModel.metrics.aucPr * 100,
        current: performanceData.currentModel.metrics.aucPr * 100,
    },
];

const PerformanceComparison: React.FC = () => {
    // Helper function to determine trend icon and color
    const getTrendIndicator = (delta: number) => {
        if (delta > 0.005) {
            return { icon: <TrendingUp color="success" />, color: 'success' };
        } else if (delta < -0.005) {
            return { icon: <TrendingDown color="error" />, color: 'error' };
        } else {
            return { icon: <TrendingFlat color="info" />, color: 'info' };
        }
    };

    // Helper function to format percentage
    const formatPercent = (value: number) => {
        return `${(value * 100).toFixed(1)}%`;
    };

    // Helper function to format confidence interval
    const formatCI = (lower: number, upper: number) => {
        return `[${(lower * 100).toFixed(1)}%, ${(upper * 100).toFixed(1)}%]`;
    };

    return (
        <Box>
            <Typography variant="h5" gutterBottom>
                Performance Comparison
            </Typography>
            <Typography variant="body1" paragraph>
                Compare the performance metrics between your reference and current models.
            </Typography>

            {/* Executive Overview */}
            <Paper sx={{ p: 3, mb: 3 }}>
                <Typography variant="h6" gutterBottom>
                    Executive Overview
                </Typography>
                <Grid container spacing={3}>
                    <Grid item xs={12} md={4}>
                        <Card sx={{ height: '100%', bgcolor: 'background.default' }}>
                            <CardContent>
                                <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                                    Overall Performance Change
                                </Typography>
                                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                                    <Typography variant="h4" component="div" sx={{ mr: 1 }}>
                                        {formatPercent(performanceData.deltas.accuracy)}
                                    </Typography>
                                    {getTrendIndicator(performanceData.deltas.accuracy).icon}
                                </Box>
                                <Typography variant="body2" color="text.secondary">
                                    Average change across all metrics: {formatPercent(
                                        Object.values(performanceData.deltas).reduce((a, b) => a + b, 0) / Object.values(performanceData.deltas).length
                                    )}
                                </Typography>
                            </CardContent>
                        </Card>
                    </Grid>
                    <Grid item xs={12} md={4}>
                        <Card sx={{ height: '100%', bgcolor: 'background.default' }}>
                            <CardContent>
                                <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                                    Risk Assessment
                                </Typography>
                                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                                    <Chip 
                                        label={performanceData.riskLevel} 
                                        color={performanceData.riskLevel === 'High' ? 'error' : 
                                               performanceData.riskLevel === 'Medium' ? 'warning' : 'success'}
                                        sx={{ fontSize: '1.1rem', py: 2, px: 1 }}
                                    />
                                </Box>
                                <Typography variant="body2" color="text.secondary">
                                    {performanceData.significance.accuracy.significant ? 
                                        'Statistically significant performance degradation detected.' : 
                                        'No statistically significant performance degradation detected.'}
                                </Typography>
                            </CardContent>
                        </Card>
                    </Grid>
                    <Grid item xs={12} md={4}>
                        <Card sx={{ height: '100%', bgcolor: 'background.default' }}>
                            <CardContent>
                                <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                                    Confidence Assessment
                                </Typography>
                                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                                    <Typography variant="h6" component="div">
                                        95% Confidence Interval
                                    </Typography>
                                </Box>
                                <Typography variant="body2" color="text.secondary">
                                    Accuracy: {formatCI(
                                        performanceData.confidenceIntervals.accuracy.lower,
                                        performanceData.confidenceIntervals.accuracy.upper
                                    )}
                                </Typography>
                            </CardContent>
                        </Card>
                    </Grid>
                </Grid>
            </Paper>

            {/* Radar Chart Comparison */}
            <Paper sx={{ p: 3, mb: 3 }}>
                <Typography variant="h6" gutterBottom>
                    Metric Comparison
                </Typography>
                <Box sx={{ height: 400, width: '100%' }}>
                    <ResponsiveContainer width="100%" height="100%">
                        <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
                            <PolarGrid />
                            <PolarAngleAxis dataKey="metric" />
                            <PolarRadiusAxis angle={30} domain={[0, 100]} />
                            <Radar
                                name="Reference Model"
                                dataKey="reference"
                                stroke="#8884d8"
                                fill="#8884d8"
                                fillOpacity={0.6}
                            />
                            <Radar
                                name="Current Model"
                                dataKey="current"
                                stroke="#82ca9d"
                                fill="#82ca9d"
                                fillOpacity={0.6}
                            />
                            <Legend />
                        </RadarChart>
                    </ResponsiveContainer>
                </Box>
            </Paper>

            {/* Detailed Metrics Table */}
            <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                    Detailed Metrics Comparison
                </Typography>
                <TableContainer>
                    <Table>
                        <TableHead>
                            <TableRow>
                                <TableCell>Metric</TableCell>
                                <TableCell align="right">Reference Model</TableCell>
                                <TableCell align="right">Current Model</TableCell>
                                <TableCell align="right">Delta</TableCell>
                                <TableCell align="right">Significance (p-value)</TableCell>
                                <TableCell align="right">95% CI</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            <TableRow>
                                <TableCell component="th" scope="row">Accuracy</TableCell>
                                <TableCell align="right">{formatPercent(performanceData.referenceModel.metrics.accuracy)}</TableCell>
                                <TableCell align="right">{formatPercent(performanceData.currentModel.metrics.accuracy)}</TableCell>
                                <TableCell align="right">
                                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                                        {formatPercent(performanceData.deltas.accuracy)}
                                        {getTrendIndicator(performanceData.deltas.accuracy).icon}
                                    </Box>
                                </TableCell>
                                <TableCell align="right">
                                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                                        {performanceData.significance.accuracy.pValue.toFixed(3)}
                                        {performanceData.significance.accuracy.significant && (
                                            <Tooltip title="Statistically significant at α=0.05">
                                                <IconButton size="small">
                                                    <Info fontSize="small" color="warning" />
                                                </IconButton>
                                            </Tooltip>
                                        )}
                                    </Box>
                                </TableCell>
                                <TableCell align="right">
                                    {formatCI(
                                        performanceData.confidenceIntervals.accuracy.lower,
                                        performanceData.confidenceIntervals.accuracy.upper
                                    )}
                                </TableCell>
                            </TableRow>

                            <TableRow>
                                <TableCell component="th" scope="row">Precision</TableCell>
                                <TableCell align="right">{formatPercent(performanceData.referenceModel.metrics.precision)}</TableCell>
                                <TableCell align="right">{formatPercent(performanceData.currentModel.metrics.precision)}</TableCell>
                                <TableCell align="right">
                                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                                        {formatPercent(performanceData.deltas.precision)}
                                        {getTrendIndicator(performanceData.deltas.precision).icon}
                                    </Box>
                                </TableCell>
                                <TableCell align="right">
                                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                                        {performanceData.significance.precision.pValue.toFixed(3)}
                                        {performanceData.significance.precision.significant && (
                                            <Tooltip title="Statistically significant at α=0.05">
                                                <IconButton size="small">
                                                    <Info fontSize="small" color="warning" />
                                                </IconButton>
                                            </Tooltip>
                                        )}
                                    </Box>
                                </TableCell>
                                <TableCell align="right">
                                    {formatCI(
                                        performanceData.confidenceIntervals.precision.lower,
                                        performanceData.confidenceIntervals.precision.upper
                                    )}
                                </TableCell>
                            </TableRow>

                            <TableRow>
                                <TableCell component="th" scope="row">Recall</TableCell>
                                <TableCell align="right">{formatPercent(performanceData.referenceModel.metrics.recall)}</TableCell>
                                <TableCell align="right">{formatPercent(performanceData.currentModel.metrics.recall)}</TableCell>
                                <TableCell align="right">
                                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                                        {formatPercent(performanceData.deltas.recall)}
                                        {getTrendIndicator(performanceData.deltas.recall).icon}
                                    </Box>
                                </TableCell>
                                <TableCell align="right">
                                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                                        {performanceData.significance.recall.pValue.toFixed(3)}
                                        {performanceData.significance.recall.significant && (
                                            <Tooltip title="Statistically significant at α=0.05">
                                                <IconButton size="small">
                                                    <Info fontSize="small" color="warning" />
                                                </IconButton>
                                            </Tooltip>
                                        )}
                                    </Box>
                                </TableCell>
                                <TableCell align="right">
                                    {formatCI(
                                        performanceData.confidenceIntervals.recall.lower,
                                        performanceData.confidenceIntervals.recall.upper
                                    )}
                                </TableCell>
                            </TableRow>

                            <TableRow>
                                <TableCell component="th" scope="row">F1 Score</TableCell>
                                <TableCell align="right">{formatPercent(performanceData.referenceModel.metrics.f1Score)}</TableCell>
                                <TableCell align="right">{formatPercent(performanceData.currentModel.metrics.f1Score)}</TableCell>
                                <TableCell align="right">
                                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                                        {formatPercent(performanceData.deltas.f1Score)}
                                        {getTrendIndicator(performanceData.deltas.f1Score).icon}
                                    </Box>
                                </TableCell>
                                <TableCell align="right">
                                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                                        {performanceData.significance.f1Score.pValue.toFixed(3)}
                                        {performanceData.significance.f1Score.significant && (
                                            <Tooltip title="Statistically significant at α=0.05">
                                                <IconButton size="small">
                                                    <Info fontSize="small" color="warning" />
                                                </IconButton>
                                            </Tooltip>
                                        )}
                                    </Box>
                                </TableCell>
                                <TableCell align="right">
                                    {formatCI(
                                        performanceData.confidenceIntervals.f1Score.lower,
                                        performanceData.confidenceIntervals.f1Score.upper
                                    )}
                                </TableCell>
                            </TableRow>

                            <TableRow>
                                <TableCell component="th" scope="row">AUC-ROC</TableCell>
                                <TableCell align="right">{formatPercent(performanceData.referenceModel.metrics.aucRoc)}</TableCell>
                                <TableCell align="right">{formatPercent(performanceData.currentModel.metrics.aucRoc)}</TableCell>
                                <TableCell align="right">
                                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                                        {formatPercent(performanceData.deltas.aucRoc)}
                                        {getTrendIndicator(performanceData.deltas.aucRoc).icon}
                                    </Box>
                                </TableCell>
                                <TableCell align="right">
                                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                                        {performanceData.significance.aucRoc.pValue.toFixed(3)}
                                        {performanceData.significance.aucRoc.significant && (
                                            <Tooltip title="Statistically significant at α=0.05">
                                                <IconButton size="small">
                                                    <Info fontSize="small" color="warning" />
                                                </IconButton>
                                            </Tooltip>
                                        )}
                                    </Box>
                                </TableCell>
                                <TableCell align="right">
                                    {formatCI(
                                        performanceData.confidenceIntervals.aucRoc.lower,
                                        performanceData.confidenceIntervals.aucRoc.upper
                                    )}
                                </TableCell>
                            </TableRow>

                            <TableRow>
                                <TableCell component="th" scope="row">AUC-PR</TableCell>
                                <TableCell align="right">{formatPercent(performanceData.referenceModel.metrics.aucPr)}</TableCell>
                                <TableCell align="right">{formatPercent(performanceData.currentModel.metrics.aucPr)}</TableCell>
                                <TableCell align="right">
                                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                                        {formatPercent(performanceData.deltas.aucPr)}
                                        {getTrendIndicator(performanceData.deltas.aucPr).icon}
                                    </Box>
                                </TableCell>
                                <TableCell align="right">
                                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                                        {performanceData.significance.aucPr.pValue.toFixed(3)}
                                        {performanceData.significance.aucPr.significant && (
                                            <Tooltip title="Statistically significant at α=0.05">
                                                <IconButton size="small">
                                                    <Info fontSize="small" color="warning" />
                                                </IconButton>
                                            </Tooltip>
                                        )}
                                    </Box>
                                </TableCell>
                                <TableCell align="right">
                                    {formatCI(
                                        performanceData.confidenceIntervals.aucPr.lower,
                                        performanceData.confidenceIntervals.aucPr.upper
                                    )}
                                </TableCell>
                            </TableRow>
                        </TableBody>
                    </Table>
                </TableContainer>
            </Paper>
        </Box>
    );
};

export default PerformanceComparison;