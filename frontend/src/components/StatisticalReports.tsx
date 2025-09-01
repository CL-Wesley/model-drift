import React, { useState } from 'react';
import {
    Box,
    Typography,
    Grid,
    Card,
    CardContent,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Paper,
    Accordion,
    AccordionSummary,
    AccordionDetails,
    Chip,
    Button,
    Alert,
    Tabs,
    Tab
} from '@mui/material';
import {
    ExpandMore,
    Assessment,
    TrendingUp,
    FileDownload,
    Summarize
} from '@mui/icons-material';

import { mockDriftResults } from '../data/mockData';

interface TabPanelProps {
    children?: React.ReactNode;
    index: number;
    value: number;
}

function TabPanel(props: TabPanelProps) {
    const { children, value, index, ...other } = props;

    return (
        <div
            role="tabpanel"
            hidden={value !== index}
            id={`report-tabpanel-${index}`}
            aria-labelledby={`report-tab-${index}`}
            {...other}
        >
            {value === index && (
                <Box sx={{ p: 3 }}>
                    {children}
                </Box>
            )}
        </div>
    );
}

const StatisticalReports: React.FC = () => {
    const [tabValue, setTabValue] = useState(0);

    const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
        setTabValue(newValue);
    };

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'high': return '#dc3545';
            case 'medium': return '#ffc107';
            case 'low': return '#28a745';
            default: return '#6c757d';
        }
    };

    const getStatusSeverity = (status: string): "error" | "warning" | "success" => {
        switch (status) {
            case 'high': return 'error';
            case 'medium': return 'warning';
            case 'low': return 'success';
            default: return 'success';
        }
    };

    // Generate correlation analysis (mock data)
    const correlationData = [
        { feature1: 'age', feature2: 'income', correlation: 0.32, drift_correlation: 0.28 },
        { feature1: 'credit_score', feature2: 'loan_amount', correlation: -0.45, drift_correlation: -0.52 },
        { feature1: 'income', feature2: 'loan_amount', correlation: 0.67, drift_correlation: 0.71 },
        { feature1: 'age', feature2: 'credit_score', correlation: -0.23, drift_correlation: -0.31 },
    ];

    return (
        <Box>
            <Typography variant="h4" gutterBottom>
                Statistical Reports
            </Typography>
            <Typography variant="body1" color="text.secondary" paragraph>
                Comprehensive statistical analysis and detailed drift detection reports.
            </Typography>

            {/* Executive Summary Card */}
            <Card sx={{ mb: 3, background: 'linear-gradient(135deg, #6f42c1 0%, #8e44ad 100%)', color: 'white' }}>
                <CardContent>
                    <Grid container spacing={3} alignItems="center">
                        <Grid item xs={12} md={8}>
                            <Typography variant="h5" gutterBottom sx={{ fontWeight: 600 }}>
                                Executive Summary
                            </Typography>
                            <Typography variant="body1" sx={{ mb: 2, opacity: 0.9 }}>
                                {mockDriftResults.executive_summary}
                            </Typography>
                            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                                <Chip
                                    label={`Overall Status: ${mockDriftResults.overall_status.toUpperCase()}`}
                                    sx={{ backgroundColor: 'rgba(255,255,255,0.2)', color: 'white' }}
                                />
                                <Chip
                                    label={`${mockDriftResults.total_features} Features Analyzed`}
                                    sx={{ backgroundColor: 'rgba(255,255,255,0.2)', color: 'white' }}
                                />
                                <Chip
                                    label={`${(mockDriftResults.data_quality_score * 100).toFixed(0)}% Data Quality`}
                                    sx={{ backgroundColor: 'rgba(255,255,255,0.2)', color: 'white' }}
                                />
                            </Box>
                        </Grid>
                        <Grid item xs={12} md={4} sx={{ textAlign: 'center' }}>
                            <Typography variant="h2" sx={{ fontWeight: 700, mb: 1 }}>
                                {mockDriftResults.overall_drift_score.toFixed(1)}
                            </Typography>
                            <Typography variant="h6">
                                Overall Drift Score
                            </Typography>
                            <Button
                                variant="outlined"
                                startIcon={<FileDownload />}
                                sx={{
                                    mt: 2,
                                    color: 'white',
                                    borderColor: 'white',
                                    '&:hover': { borderColor: 'white', backgroundColor: 'rgba(255,255,255,0.1)' }
                                }}
                            >
                                Download Report
                            </Button>
                        </Grid>
                    </Grid>
                </CardContent>
            </Card>

            {/* Report Tabs */}
            <Card>
                <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                    <Tabs value={tabValue} onChange={handleTabChange} aria-label="statistical reports tabs">
                        <Tab icon={<Assessment />} label="Feature Analysis" />
                        <Tab icon={<TrendingUp />} label="Correlation Analysis" />
                        <Tab icon={<Summarize />} label="Statistical Tests" />
                    </Tabs>
                </Box>

                {/* Feature Analysis Tab */}
                <TabPanel value={tabValue} index={0}>
                    <Typography variant="h6" gutterBottom>
                        Detailed Feature Drift Analysis
                    </Typography>
                    <TableContainer component={Paper} variant="outlined">
                        <Table>
                            <TableHead>
                                <TableRow sx={{ backgroundColor: '#f8f9fa' }}>
                                    <TableCell><strong>Feature</strong></TableCell>
                                    <TableCell align="center"><strong>Status</strong></TableCell>
                                    <TableCell align="right"><strong>Drift Score</strong></TableCell>
                                    <TableCell align="right"><strong>KL Divergence</strong></TableCell>
                                    <TableCell align="right"><strong>PSI</strong></TableCell>
                                    <TableCell align="right"><strong>KS Statistic</strong></TableCell>
                                    <TableCell align="right"><strong>P-Value</strong></TableCell>
                                    <TableCell align="center"><strong>Data Type</strong></TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {mockDriftResults.feature_analysis.map((feature) => (
                                    <TableRow key={feature.feature} sx={{ '&:nth-of-type(odd)': { backgroundColor: '#fafafa' } }}>
                                        <TableCell sx={{ fontWeight: 600 }}>{feature.feature}</TableCell>
                                        <TableCell align="center">
                                            <Chip
                                                label={feature.status.toUpperCase()}
                                                size="small"
                                                sx={{
                                                    backgroundColor: getStatusColor(feature.status),
                                                    color: 'white',
                                                    fontWeight: 600
                                                }}
                                            />
                                        </TableCell>
                                        <TableCell align="right">
                                            <Typography sx={{ fontWeight: 600, color: getStatusColor(feature.status) }}>
                                                {feature.drift_score.toFixed(3)}
                                            </Typography>
                                        </TableCell>
                                        <TableCell align="right">{feature.kl_divergence.toFixed(4)}</TableCell>
                                        <TableCell align="right">{feature.psi.toFixed(4)}</TableCell>
                                        <TableCell align="right">{feature.ks_statistic.toFixed(4)}</TableCell>
                                        <TableCell align="right">
                                            <Typography sx={{ color: feature.p_value < 0.05 ? '#dc3545' : '#28a745' }}>
                                                {feature.p_value.toFixed(4)}
                                            </Typography>
                                        </TableCell>
                                        <TableCell align="center">
                                            <Chip
                                                label={feature.data_type}
                                                variant="outlined"
                                                size="small"
                                                sx={{ textTransform: 'capitalize' }}
                                            />
                                        </TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </TableContainer>

                    {/* Feature Analysis Accordions */}
                    <Box sx={{ mt: 3 }}>
                        <Typography variant="h6" gutterBottom>
                            Detailed Analysis by Feature
                        </Typography>
                        {mockDriftResults.feature_analysis.map((feature) => (
                            <Accordion key={feature.feature} sx={{ mb: 1 }}>
                                <AccordionSummary expandIcon={<ExpandMore />}>
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, width: '100%' }}>
                                        <Typography sx={{ fontWeight: 600 }}>{feature.feature}</Typography>
                                        <Chip
                                            label={feature.status.toUpperCase()}
                                            size="small"
                                            sx={{
                                                backgroundColor: getStatusColor(feature.status),
                                                color: 'white'
                                            }}
                                        />
                                        <Typography sx={{ marginLeft: 'auto', color: 'text.secondary' }}>
                                            Drift Score: {feature.drift_score.toFixed(3)}
                                        </Typography>
                                    </Box>
                                </AccordionSummary>
                                <AccordionDetails>
                                    <Alert severity={getStatusSeverity(feature.status)} sx={{ mb: 2 }}>
                                        <Typography variant="body2">
                                            <strong>{feature.feature}</strong> shows <strong>{feature.status}</strong> drift.
                                            {feature.status === 'high' && ' Immediate attention required - consider model retraining.'}
                                            {feature.status === 'medium' && ' Monitor closely and consider data validation.'}
                                            {feature.status === 'low' && ' No immediate action required.'}
                                        </Typography>
                                    </Alert>

                                    <Grid container spacing={2}>
                                        <Grid item xs={12} md={6}>
                                            <Typography variant="subtitle2" gutterBottom>Statistical Metrics</Typography>
                                            <Table size="small">
                                                <TableBody>
                                                    <TableRow>
                                                        <TableCell>KL Divergence</TableCell>
                                                        <TableCell align="right">{feature.kl_divergence.toFixed(4)}</TableCell>
                                                    </TableRow>
                                                    <TableRow>
                                                        <TableCell>Population Stability Index</TableCell>
                                                        <TableCell align="right">{feature.psi.toFixed(4)}</TableCell>
                                                    </TableRow>
                                                    <TableRow>
                                                        <TableCell>KS Statistic</TableCell>
                                                        <TableCell align="right">{feature.ks_statistic.toFixed(4)}</TableCell>
                                                    </TableRow>
                                                    <TableRow>
                                                        <TableCell>P-Value</TableCell>
                                                        <TableCell align="right">{feature.p_value.toFixed(4)}</TableCell>
                                                    </TableRow>
                                                </TableBody>
                                            </Table>
                                        </Grid>
                                        <Grid item xs={12} md={6}>
                                            <Typography variant="subtitle2" gutterBottom>Data Quality</Typography>
                                            <Table size="small">
                                                <TableBody>
                                                    <TableRow>
                                                        <TableCell>Missing Values (Ref)</TableCell>
                                                        <TableCell align="right">{feature.missing_values_ref}</TableCell>
                                                    </TableRow>
                                                    <TableRow>
                                                        <TableCell>Missing Values (Current)</TableCell>
                                                        <TableCell align="right">{feature.missing_values_current}</TableCell>
                                                    </TableRow>
                                                    <TableRow>
                                                        <TableCell>Data Type</TableCell>
                                                        <TableCell align="right" sx={{ textTransform: 'capitalize' }}>
                                                            {feature.data_type}
                                                        </TableCell>
                                                    </TableRow>
                                                </TableBody>
                                            </Table>
                                        </Grid>
                                    </Grid>
                                </AccordionDetails>
                            </Accordion>
                        ))}
                    </Box>
                </TabPanel>

                {/* Correlation Analysis Tab */}
                <TabPanel value={tabValue} index={1}>
                    <Typography variant="h6" gutterBottom>
                        Feature Correlation Analysis
                    </Typography>
                    <Typography variant="body2" color="text.secondary" paragraph>
                        Analysis of feature relationships and how correlations have changed between reference and current datasets.
                    </Typography>

                    <TableContainer component={Paper} variant="outlined">
                        <Table>
                            <TableHead>
                                <TableRow sx={{ backgroundColor: '#f8f9fa' }}>
                                    <TableCell><strong>Feature Pair</strong></TableCell>
                                    <TableCell align="right"><strong>Reference Correlation</strong></TableCell>
                                    <TableCell align="right"><strong>Current Correlation</strong></TableCell>
                                    <TableCell align="right"><strong>Change</strong></TableCell>
                                    <TableCell align="center"><strong>Impact</strong></TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {correlationData.map((corr, index) => {
                                    const change = corr.drift_correlation - corr.correlation;
                                    const absChange = Math.abs(change);
                                    const impact = absChange > 0.1 ? 'High' : absChange > 0.05 ? 'Medium' : 'Low';
                                    const impactColor = absChange > 0.1 ? '#dc3545' : absChange > 0.05 ? '#ffc107' : '#28a745';

                                    return (
                                        <TableRow key={index} sx={{ '&:nth-of-type(odd)': { backgroundColor: '#fafafa' } }}>
                                            <TableCell sx={{ fontWeight: 600 }}>
                                                {corr.feature1} ↔ {corr.feature2}
                                            </TableCell>
                                            <TableCell align="right">{corr.correlation.toFixed(3)}</TableCell>
                                            <TableCell align="right">{corr.drift_correlation.toFixed(3)}</TableCell>
                                            <TableCell align="right">
                                                <Typography sx={{
                                                    color: change > 0 ? '#dc3545' : change < 0 ? '#28a745' : '#6c757d',
                                                    fontWeight: 600
                                                }}>
                                                    {change > 0 ? '+' : ''}{change.toFixed(3)}
                                                </Typography>
                                            </TableCell>
                                            <TableCell align="center">
                                                <Chip
                                                    label={impact}
                                                    size="small"
                                                    sx={{
                                                        backgroundColor: impactColor,
                                                        color: 'white'
                                                    }}
                                                />
                                            </TableCell>
                                        </TableRow>
                                    );
                                })}
                            </TableBody>
                        </Table>
                    </TableContainer>
                </TabPanel>

                {/* Statistical Tests Tab */}
                <TabPanel value={tabValue} index={2}>
                    <Typography variant="h6" gutterBottom>
                        Statistical Test Results
                    </Typography>
                    <Typography variant="body2" color="text.secondary" paragraph>
                        Detailed results from various statistical tests used to detect distribution differences.
                    </Typography>

                    <Grid container spacing={3}>
                        <Grid item xs={12} md={6}>
                            <Card variant="outlined">
                                <CardContent>
                                    <Typography variant="h6" gutterBottom>
                                        Kolmogorov-Smirnov Test
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary" paragraph>
                                        Tests for differences in continuous distributions
                                    </Typography>
                                    <Table size="small">
                                        <TableHead>
                                            <TableRow>
                                                <TableCell>Feature</TableCell>
                                                <TableCell align="right">KS Statistic</TableCell>
                                                <TableCell align="right">P-Value</TableCell>
                                                <TableCell align="center">Result</TableCell>
                                            </TableRow>
                                        </TableHead>
                                        <TableBody>
                                            {mockDriftResults.feature_analysis
                                                .filter(f => f.data_type === 'numerical')
                                                .map((feature) => (
                                                    <TableRow key={feature.feature}>
                                                        <TableCell>{feature.feature}</TableCell>
                                                        <TableCell align="right">{feature.ks_statistic.toFixed(4)}</TableCell>
                                                        <TableCell align="right">{feature.p_value.toFixed(4)}</TableCell>
                                                        <TableCell align="center">
                                                            <Chip
                                                                label={feature.p_value < 0.05 ? 'Significant' : 'Not Significant'}
                                                                size="small"
                                                                color={feature.p_value < 0.05 ? 'error' : 'success'}
                                                            />
                                                        </TableCell>
                                                    </TableRow>
                                                ))}
                                        </TableBody>
                                    </Table>
                                </CardContent>
                            </Card>
                        </Grid>

                        <Grid item xs={12} md={6}>
                            <Card variant="outlined">
                                <CardContent>
                                    <Typography variant="h6" gutterBottom>
                                        Chi-Square Test
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary" paragraph>
                                        Tests for differences in categorical distributions
                                    </Typography>
                                    <Table size="small">
                                        <TableHead>
                                            <TableRow>
                                                <TableCell>Feature</TableCell>
                                                <TableCell align="right">Chi-Square</TableCell>
                                                <TableCell align="right">P-Value</TableCell>
                                                <TableCell align="center">Result</TableCell>
                                            </TableRow>
                                        </TableHead>
                                        <TableBody>
                                            {mockDriftResults.feature_analysis
                                                .filter(f => f.data_type === 'categorical')
                                                .map((feature) => (
                                                    <TableRow key={feature.feature}>
                                                        <TableCell>{feature.feature}</TableCell>
                                                        <TableCell align="right">{(feature.ks_statistic * 10).toFixed(3)}</TableCell>
                                                        <TableCell align="right">{feature.p_value.toFixed(4)}</TableCell>
                                                        <TableCell align="center">
                                                            <Chip
                                                                label={feature.p_value < 0.05 ? 'Significant' : 'Not Significant'}
                                                                size="small"
                                                                color={feature.p_value < 0.05 ? 'error' : 'success'}
                                                            />
                                                        </TableCell>
                                                    </TableRow>
                                                ))}
                                        </TableBody>
                                    </Table>
                                </CardContent>
                            </Card>
                        </Grid>
                    </Grid>
                </TabPanel>
            </Card>
        </Box>
    );
};

export default StatisticalReports;
