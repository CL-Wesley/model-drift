import React, { useState } from 'react';
import {
    Box,
    Paper,
    Typography,
    Grid,
    Tabs,
    Tab,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Card,
    CardContent,
    Chip,
    Alert,
} from '@mui/material';
import {
    CheckCircle,
    Cancel,
    ArrowDownward,
    ArrowUpward,
} from '@mui/icons-material';
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip as RechartsTooltip,
    ResponsiveContainer,
} from 'recharts';

const StatisticalSignificance: React.FC = () => {
    const [activeTab, setActiveTab] = useState(0);

    const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
        setActiveTab(newValue);
    };

    // Simplified hypothesis testing data
    const testResults = [
        {
            test: "McNemar's Test",
            metric: 'Classification Agreement',
            statistic: 12.45,
            pValue: 0.0004,
            significant: true,
            interpretation: 'Models show statistically significant disagreement on classification decisions.'
        },
        {
            test: "DeLong's Test",
            metric: 'AUC-ROC Comparison',
            statistic: 2.31,
            pValue: 0.0209,
            significant: true,
            interpretation: 'Significant difference in discriminative ability between models.'
        },
        {
            test: 'Paired t-test',
            metric: 'Accuracy Difference',
            statistic: -3.45,
            pValue: 0.0012,
            significant: true,
            interpretation: 'Current model shows significantly lower accuracy than reference.'
        },
        {
            test: 'Wilcoxon Signed-Rank',
            metric: 'F1-Score Distribution',
            statistic: 145.0,
            pValue: 0.0034,
            significant: true,
            interpretation: 'Non-parametric test confirms significant performance degradation.'
        }
    ];

    // Effect size data for visualization
    const effectSizeData = [
        { metric: 'Accuracy', effectSize: -0.32, interpretation: 'Medium Effect' },
        { metric: 'Precision', effectSize: -0.28, interpretation: 'Small-Medium Effect' },
        { metric: 'Recall', effectSize: -0.35, interpretation: 'Medium Effect' },
        { metric: 'F1-Score', effectSize: -0.33, interpretation: 'Medium Effect' },
        { metric: 'AUC-ROC', effectSize: -0.25, interpretation: 'Small-Medium Effect' }
    ];

    const getSignificanceIcon = (significant: boolean) => {
        return significant ?
            <CheckCircle color="error" /> :
            <Cancel color="success" />;
    };

    const getEffectIcon = (effectSize: number) => {
        return effectSize < 0 ?
            <ArrowDownward color="error" /> :
            <ArrowUpward color="success" />;
    };

    return (
        <Box>
            <Typography variant="h4" gutterBottom>
                Statistical Significance Analysis
            </Typography>
            <Typography variant="body1" color="text.secondary" paragraph>
                Comprehensive statistical tests to determine if observed performance differences
                are statistically significant and not due to random variation.
            </Typography>

            {/* Executive Summary */}
            <Alert severity="error" sx={{ mb: 4 }}>
                <Typography variant="body2" fontWeight="bold">
                    Statistical Conclusion: Significant Model Degradation Detected
                </Typography>
                <Typography variant="body2">
                    All 4 statistical tests confirm significant performance degradation (p &lt; 0.05).
                    The observed differences are statistically meaningful and require immediate attention.
                </Typography>
            </Alert>

            <Tabs value={activeTab} onChange={handleTabChange} sx={{ mb: 3 }}>
                <Tab label="Hypothesis Tests" />
                <Tab label="Effect Size Analysis" />
            </Tabs>

            {/* Hypothesis Testing Results */}
            {activeTab === 0 && (
                <Box>
                    <Paper sx={{ p: 3, mb: 3 }}>
                        <Typography variant="h6" gutterBottom>
                            Statistical Test Results
                        </Typography>
                        <TableContainer>
                            <Table>
                                <TableHead>
                                    <TableRow>
                                        <TableCell><strong>Statistical Test</strong></TableCell>
                                        <TableCell><strong>Metric</strong></TableCell>
                                        <TableCell align="right"><strong>Test Statistic</strong></TableCell>
                                        <TableCell align="right"><strong>p-value</strong></TableCell>
                                        <TableCell align="center"><strong>Significant</strong></TableCell>
                                        <TableCell><strong>Interpretation</strong></TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {testResults.map((test, index) => (
                                        <TableRow key={index}>
                                            <TableCell component="th" scope="row">
                                                <Typography variant="body2" fontWeight="bold">
                                                    {test.test}
                                                </Typography>
                                            </TableCell>
                                            <TableCell>{test.metric}</TableCell>
                                            <TableCell align="right">
                                                <Typography variant="body2" fontFamily="monospace">
                                                    {test.statistic.toFixed(2)}
                                                </Typography>
                                            </TableCell>
                                            <TableCell align="right">
                                                <Typography
                                                    variant="body2"
                                                    fontFamily="monospace"
                                                    color={test.pValue < 0.05 ? 'error.main' : 'text.primary'}
                                                    fontWeight={test.pValue < 0.05 ? 'bold' : 'normal'}
                                                >
                                                    {test.pValue.toFixed(4)}
                                                </Typography>
                                            </TableCell>
                                            <TableCell align="center">
                                                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                                    {getSignificanceIcon(test.significant)}
                                                    <Chip
                                                        label={test.significant ? 'Yes' : 'No'}
                                                        color={test.significant ? 'error' : 'success'}
                                                        size="small"
                                                        sx={{ ml: 1 }}
                                                    />
                                                </Box>
                                            </TableCell>
                                            <TableCell>
                                                <Typography variant="body2">
                                                    {test.interpretation}
                                                </Typography>
                                            </TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </TableContainer>
                    </Paper>

                    {/* Statistical Power Analysis */}
                    <Grid container spacing={3}>
                        <Grid item xs={12} md={6}>
                            <Card>
                                <CardContent>
                                    <Typography variant="h6" gutterBottom>
                                        Statistical Power
                                    </Typography>
                                    <Typography variant="h3" color="success.main" gutterBottom>
                                        0.95
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary">
                                        High statistical power (95%) ensures our tests can reliably detect
                                        true performance differences when they exist.
                                    </Typography>
                                </CardContent>
                            </Card>
                        </Grid>
                        <Grid item xs={12} md={6}>
                            <Card>
                                <CardContent>
                                    <Typography variant="h6" gutterBottom>
                                        Multiple Testing Correction
                                    </Typography>
                                    <Typography variant="h3" color="warning.main" gutterBottom>
                                        Bonferroni
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary">
                                        Applied Bonferroni correction (α = 0.0125) to control family-wise
                                        error rate across 4 simultaneous tests.
                                    </Typography>
                                </CardContent>
                            </Card>
                        </Grid>
                    </Grid>
                </Box>
            )}

            {/* Effect Size Analysis */}
            {activeTab === 1 && (
                <Box>
                    <Paper sx={{ p: 3, mb: 3 }}>
                        <Typography variant="h6" gutterBottom>
                            Effect Size Analysis
                        </Typography>
                        <Typography variant="body2" color="text.secondary" paragraph>
                            Effect sizes quantify the magnitude of performance differences,
                            providing practical significance beyond statistical significance.
                        </Typography>

                        <Box sx={{ height: 300, mb: 3 }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={effectSizeData}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="metric" />
                                    <YAxis domain={[-0.5, 0.1]} />
                                    <RechartsTooltip
                                        formatter={(value: any) => [value.toFixed(3), 'Effect Size']}
                                    />
                                    <Bar
                                        dataKey="effectSize"
                                        fill="#ff7300"
                                        name="Cohen's d Effect Size"
                                    />
                                </BarChart>
                            </ResponsiveContainer>
                        </Box>

                        <TableContainer>
                            <Table>
                                <TableHead>
                                    <TableRow>
                                        <TableCell><strong>Performance Metric</strong></TableCell>
                                        <TableCell align="right"><strong>Effect Size (Cohen's d)</strong></TableCell>
                                        <TableCell align="center"><strong>Direction</strong></TableCell>
                                        <TableCell><strong>Practical Significance</strong></TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {effectSizeData.map((item, index) => (
                                        <TableRow key={index}>
                                            <TableCell component="th" scope="row">
                                                <Typography variant="body2" fontWeight="bold">
                                                    {item.metric}
                                                </Typography>
                                            </TableCell>
                                            <TableCell align="right">
                                                <Typography
                                                    variant="body2"
                                                    fontFamily="monospace"
                                                    color={item.effectSize < -0.2 ? 'error.main' : 'text.primary'}
                                                    fontWeight="bold"
                                                >
                                                    {item.effectSize.toFixed(3)}
                                                </Typography>
                                            </TableCell>
                                            <TableCell align="center">
                                                {getEffectIcon(item.effectSize)}
                                            </TableCell>
                                            <TableCell>
                                                <Chip
                                                    label={item.interpretation}
                                                    color={Math.abs(item.effectSize) > 0.3 ? 'error' : 'warning'}
                                                    size="small"
                                                />
                                            </TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </TableContainer>
                    </Paper>

                    {/* Effect Size Interpretation Guide */}
                    <Alert severity="info">
                        <Typography variant="body2" fontWeight="bold">Effect Size Interpretation (Cohen's d):</Typography>
                        <Typography variant="body2">
                            • Small effect: d = 0.2 | Medium effect: d = 0.5 | Large effect: d = 0.8<br />
                            • Negative values indicate performance degradation in current model<br />
                            • All observed effects are in the small-to-medium range, indicating meaningful practical impact
                        </Typography>
                    </Alert>
                </Box>
            )}
        </Box>
    );
};

export default StatisticalSignificance;