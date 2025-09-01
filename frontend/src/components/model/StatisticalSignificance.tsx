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
    Divider,
    Chip,
    Tooltip,
    IconButton,
    Link,
    List,
    ListItem,
    ListItemIcon,
    ListItemText,
} from '@mui/material';
import {
    Info,
    CheckCircle,
    Cancel,
    ArrowDownward,
    ArrowUpward,
    HelpOutline,
    MenuBook,
    School,
    BarChart,
} from '@mui/icons-material';
import {
    BarChart as RechartsBarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip as RechartsTooltip,
    ResponsiveContainer,
    ErrorBar,
    Legend,
    LineChart,
    Line,
    ReferenceLine,
} from 'recharts';

// Mock data for hypothesis testing results
const hypothesisTestingData = [
    {
        test: "McNemar's Test",
        metric: 'Classification Agreement',
        statistic: 12.45,
        pValue: 0.0004,
        significant: true,
        interpretation: 'The disagreement between models is statistically significant.',
        applicability: 'Classification tasks with paired samples',
        assumptions: ['Paired observations', 'Binary outcomes', 'Large sample size (n > 30)'],
    },
    {
        test: "DeLong's Test for AUC",
        metric: 'AUC-ROC',
        statistic: 2.31,
        pValue: 0.0209,
        significant: true,
        interpretation: 'The difference in AUC-ROC is statistically significant.',
        applicability: 'Comparing ROC curves of two models on the same data',
        assumptions: ['Paired observations', 'Valid ROC curves', 'Sufficient sample size'],
    },
    {
        test: '5x2 Cross-validation paired t-test',
        metric: 'Overall Performance',
        statistic: 3.12,
        pValue: 0.0142,
        significant: true,
        interpretation: 'The overall performance difference is statistically significant.',
        applicability: 'Comparing model performance with cross-validation',
        assumptions: ['Independent training/test splits', 'Normal distribution of differences', 'Homogeneity of variance'],
    },
    {
        test: 'Bootstrap Confidence Intervals',
        metric: 'Multiple Metrics',
        statistic: null,
        pValue: null,
        significant: true,
        interpretation: 'The confidence intervals do not contain zero, indicating significant differences.',
        applicability: 'Non-parametric assessment of model differences',
        assumptions: ['Random sampling with replacement', 'Representative original sample', 'Sufficient bootstrap iterations (>1000)'],
    },
    {
        test: 'Diebold-Mariano Test',
        metric: 'Forecast Accuracy',
        statistic: 1.85,
        pValue: 0.0643,
        significant: false,
        interpretation: 'The difference in forecast accuracy is not statistically significant at α=0.05.',
        applicability: 'Comparing forecast accuracy of regression models',
        assumptions: ['Time series data', 'Covariance stationarity', 'No autocorrelation in error differences'],
    },
];

// Mock data for effect size analysis
const effectSizeData = [
    {
        metric: 'Accuracy',
        effectSize: -0.32,
        interpretation: 'Medium negative effect',
        confidenceInterval: [-0.45, -0.19],
        pValue: 0.0032,
    },
    {
        metric: 'Precision',
        effectSize: -0.28,
        interpretation: 'Small-to-medium negative effect',
        confidenceInterval: [-0.41, -0.15],
        pValue: 0.0041,
    },
    {
        metric: 'Recall',
        effectSize: -0.35,
        interpretation: 'Medium negative effect',
        confidenceInterval: [-0.48, -0.22],
        pValue: 0.0039,
    },
    {
        metric: 'F1 Score',
        effectSize: -0.33,
        interpretation: 'Medium negative effect',
        confidenceInterval: [-0.46, -0.20],
        pValue: 0.0028,
    },
    {
        metric: 'AUC-ROC',
        effectSize: -0.25,
        interpretation: 'Small-to-medium negative effect',
        confidenceInterval: [-0.38, -0.12],
        pValue: 0.0045,
    },
];

// Mock data for bootstrap distribution
const bootstrapDistributionData = Array.from({ length: 20 }, (_, i) => ({
    bin: (i * 0.005 - 0.05).toFixed(3),
    frequency: Math.floor(Math.exp(-Math.pow((i - 10) / 4, 2)) * 100),
}));

// Mock data for confidence intervals
const confidenceIntervalData = [
    {
        metric: 'Accuracy',
        difference: -0.020,
        ci95: [-0.035, -0.005],
        ci99: [-0.042, 0.002],
    },
    {
        metric: 'Precision',
        difference: -0.020,
        ci95: [-0.038, -0.002],
        ci99: [-0.045, 0.005],
    },
    {
        metric: 'Recall',
        difference: -0.020,
        ci95: [-0.042, -0.001],
        ci99: [-0.048, 0.008],
    },
    {
        metric: 'F1 Score',
        difference: -0.020,
        ci95: [-0.039, -0.003],
        ci99: [-0.046, 0.006],
    },
    {
        metric: 'AUC-ROC',
        difference: -0.020,
        ci95: [-0.037, -0.004],
        ci99: [-0.044, 0.004],
    },
];

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
            id={`statistical-significance-tabpanel-${index}`}
            aria-labelledby={`statistical-significance-tab-${index}`}
            {...other}
        >
            {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
        </div>
    );
}

function a11yProps(index: number) {
    return {
        id: `statistical-significance-tab-${index}`,
        'aria-controls': `statistical-significance-tabpanel-${index}`,
    };
}

const StatisticalSignificance: React.FC = () => {
    const [tabValue, setTabValue] = useState(0);

    const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
        setTabValue(newValue);
    };

    // Helper function to format confidence interval
    const formatCI = (lower: number, upper: number) => {
        return `[${lower.toFixed(3)}, ${upper.toFixed(3)}]`;
    };

    return (
        <Box>
            <Typography variant="h5" gutterBottom>
                Statistical Significance
            </Typography>
            <Typography variant="body1" paragraph>
                Analyze the statistical significance of model performance differences using rigorous hypothesis testing.
            </Typography>

            <Paper sx={{ width: '100%' }}>
                <Tabs
                    value={tabValue}
                    onChange={handleTabChange}
                    indicatorColor="primary"
                    textColor="primary"
                    variant="fullWidth"
                >
                    <Tab label="Hypothesis Testing" {...a11yProps(0)} />
                    <Tab label="Effect Size Analysis" {...a11yProps(1)} />
                    <Tab label="Methodology" {...a11yProps(2)} />
                </Tabs>

                {/* Hypothesis Testing Tab */}
                <TabPanel value={tabValue} index={0}>
                    <Grid container spacing={3}>
                        <Grid item xs={12}>
                            <Paper sx={{ p: 3 }}>
                                <Typography variant="h6" gutterBottom>
                                    Hypothesis Testing Results
                                </Typography>
                                <Typography variant="body2" color="text.secondary" paragraph>
                                    Statistical tests to determine if the observed differences between models are significant.
                                </Typography>
                                <TableContainer>
                                    <Table>
                                        <TableHead>
                                            <TableRow>
                                                <TableCell>Test</TableCell>
                                                <TableCell>Metric</TableCell>
                                                <TableCell align="right">Test Statistic</TableCell>
                                                <TableCell align="right">p-value</TableCell>
                                                <TableCell align="center">Significant at α=0.05</TableCell>
                                                <TableCell>Interpretation</TableCell>
                                            </TableRow>
                                        </TableHead>
                                        <TableBody>
                                            {hypothesisTestingData.map((row) => (
                                                <TableRow key={row.test}>
                                                    <TableCell>
                                                        <Tooltip title={`Applicability: ${row.applicability}`}>
                                                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                                                {row.test}
                                                                <IconButton size="small">
                                                                    <Info fontSize="small" color="info" />
                                                                </IconButton>
                                                            </Box>
                                                        </Tooltip>
                                                    </TableCell>
                                                    <TableCell>{row.metric}</TableCell>
                                                    <TableCell align="right">{row.statistic !== null ? row.statistic.toFixed(2) : 'N/A'}</TableCell>
                                                    <TableCell align="right">{row.pValue !== null ? row.pValue.toFixed(4) : 'N/A'}</TableCell>
                                                    <TableCell align="center">
                                                        {row.significant ? (
                                                            <CheckCircle color="error" />
                                                        ) : (
                                                            <Cancel color="success" />
                                                        )}
                                                    </TableCell>
                                                    <TableCell>{row.interpretation}</TableCell>
                                                </TableRow>
                                            ))}
                                        </TableBody>
                                    </Table>
                                </TableContainer>
                            </Paper>
                        </Grid>

                        <Grid item xs={12} md={6}>
                            <Paper sx={{ p: 3 }}>
                                <Typography variant="h6" gutterBottom>
                                    Bootstrap Distribution
                                </Typography>
                                <Typography variant="body2" color="text.secondary" paragraph>
                                    Distribution of accuracy differences from 1,000 bootstrap samples.
                                </Typography>
                                <Box sx={{ height: 300, width: '100%' }}>
                                    <ResponsiveContainer width="100%" height="100%">
                                        <RechartsBarChart
                                            data={bootstrapDistributionData}
                                            margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                                        >
                                            <CartesianGrid strokeDasharray="3 3" />
                                            <XAxis dataKey="bin" />
                                            <YAxis />
                                            <RechartsTooltip />
                                            <ReferenceLine x="0.000" stroke="#ff0000" label="No Difference" />
                                            <ReferenceLine x="-0.020" stroke="#0000ff" label="Observed" />
                                            <Bar dataKey="frequency" fill="#8884d8" name="Frequency" />
                                        </RechartsBarChart>
                                    </ResponsiveContainer>
                                </Box>
                                <Typography variant="body2" sx={{ mt: 2 }}>
                                    The bootstrap distribution shows the sampling distribution of the difference in accuracy between models.
                                    The red line represents no difference, while the blue line shows the observed difference.
                                </Typography>
                            </Paper>
                        </Grid>

                        <Grid item xs={12} md={6}>
                            <Paper sx={{ p: 3 }}>
                                <Typography variant="h6" gutterBottom>
                                    Confidence Intervals
                                </Typography>
                                <Typography variant="body2" color="text.secondary" paragraph>
                                    95% and 99% confidence intervals for the differences in performance metrics.
                                </Typography>
                                <Box sx={{ height: 300, width: '100%' }}>
                                    <ResponsiveContainer width="100%" height="100%">
                                        <RechartsBarChart
                                            data={confidenceIntervalData}
                                            layout="vertical"
                                            margin={{ top: 20, right: 30, left: 100, bottom: 5 }}
                                        >
                                            <CartesianGrid strokeDasharray="3 3" />
                                            <XAxis type="number" domain={[-0.05, 0.01]} />
                                            <YAxis dataKey="metric" type="category" />
                                            <RechartsTooltip />
                                            <ReferenceLine x={0} stroke="#000" />
                                            <Bar dataKey="difference" fill="#8884d8" name="Difference">
                                                <ErrorBar dataKey="ci95" width={4} strokeWidth={2} stroke="#8884d8" direction="x" />
                                            </Bar>
                                        </RechartsBarChart>
                                    </ResponsiveContainer>
                                </Box>
                                <Typography variant="body2" sx={{ mt: 2 }}>
                                    When a confidence interval does not contain zero, the difference is considered statistically significant at the corresponding level.
                                    Here, most 95% CIs do not contain zero, indicating significant differences.
                                </Typography>
                            </Paper>
                        </Grid>
                    </Grid>
                </TabPanel>

                {/* Effect Size Analysis Tab */}
                <TabPanel value={tabValue} index={1}>
                    <Grid container spacing={3}>
                        <Grid item xs={12}>
                            <Paper sx={{ p: 3 }}>
                                <Typography variant="h6" gutterBottom>
                                    Effect Size Analysis
                                </Typography>
                                <Typography variant="body2" color="text.secondary" paragraph>
                                    Effect size measures the magnitude of the difference between models, independent of sample size.
                                </Typography>
                                <TableContainer>
                                    <Table>
                                        <TableHead>
                                            <TableRow>
                                                <TableCell>Metric</TableCell>
                                                <TableCell align="right">Cohen's d</TableCell>
                                                <TableCell>Interpretation</TableCell>
                                                <TableCell align="right">95% CI</TableCell>
                                                <TableCell align="right">p-value</TableCell>
                                                <TableCell align="center">Direction</TableCell>
                                            </TableRow>
                                        </TableHead>
                                        <TableBody>
                                            {effectSizeData.map((row) => (
                                                <TableRow key={row.metric}>
                                                    <TableCell>{row.metric}</TableCell>
                                                    <TableCell align="right">{row.effectSize.toFixed(2)}</TableCell>
                                                    <TableCell>{row.interpretation}</TableCell>
                                                    <TableCell align="right">{formatCI(row.confidenceInterval[0], row.confidenceInterval[1])}</TableCell>
                                                    <TableCell align="right">{row.pValue.toFixed(4)}</TableCell>
                                                    <TableCell align="center">
                                                        {row.effectSize < 0 ? (
                                                            <ArrowDownward color="error" />
                                                        ) : (
                                                            <ArrowUpward color="success" />
                                                        )}
                                                    </TableCell>
                                                </TableRow>
                                            ))}
                                        </TableBody>
                                    </Table>
                                </TableContainer>
                            </Paper>
                        </Grid>

                        <Grid item xs={12} md={6}>
                            <Paper sx={{ p: 3 }}>
                                <Typography variant="h6" gutterBottom>
                                    Effect Size Visualization
                                </Typography>
                                <Typography variant="body2" color="text.secondary" paragraph>
                                    Visual representation of effect sizes and their confidence intervals.
                                </Typography>
                                <Box sx={{ height: 300, width: '100%' }}>
                                    <ResponsiveContainer width="100%" height="100%">
                                        <RechartsBarChart
                                            data={effectSizeData}
                                            layout="vertical"
                                            margin={{ top: 20, right: 30, left: 100, bottom: 5 }}
                                        >
                                            <CartesianGrid strokeDasharray="3 3" />
                                            <XAxis type="number" domain={[-0.5, 0]} />
                                            <YAxis dataKey="metric" type="category" />
                                            <RechartsTooltip />
                                            <ReferenceLine x={0} stroke="#000" />
                                            <ReferenceLine x={-0.2} stroke="#ff0000" strokeDasharray="3 3" label={{ value: 'Small Effect', position: 'top' }} />
                                            <ReferenceLine x={-0.5} stroke="#ff0000" strokeDasharray="3 3" label={{ value: 'Medium Effect', position: 'top' }} />
                                            <ReferenceLine x={-0.8} stroke="#ff0000" strokeDasharray="3 3" label={{ value: 'Large Effect', position: 'top' }} />
                                            <Bar dataKey="effectSize" fill="#8884d8" name="Effect Size (Cohen's d)">
                                                <ErrorBar dataKey="confidenceInterval" width={4} strokeWidth={2} stroke="#8884d8" direction="x" />
                                            </Bar>
                                        </RechartsBarChart>
                                    </ResponsiveContainer>
                                </Box>
                                <Typography variant="body2" sx={{ mt: 2 }}>
                                    Cohen's d interpretation: |d| &lt; 0.2 (negligible), 0.2 ≤ |d| &lt; 0.5 (small),
                                    0.5 ≤ |d| &lt; 0.8 (medium), |d| ≥ 0.8 (large).
                                </Typography>
                            </Paper>
                        </Grid>

                        <Grid item xs={12} md={6}>
                            <Paper sx={{ p: 3 }}>
                                <Typography variant="h6" gutterBottom>
                                    Practical Significance
                                </Typography>
                                <Typography variant="body2" color="text.secondary" paragraph>
                                    Assessment of whether the statistically significant differences have practical importance.
                                </Typography>
                                <Grid container spacing={2}>
                                    <Grid item xs={12}>
                                        <Card sx={{ bgcolor: 'background.default' }}>
                                            <CardContent>
                                                <Typography variant="subtitle1" gutterBottom>
                                                    Business Impact Assessment
                                                </Typography>
                                                <Typography variant="body2" paragraph>
                                                    The observed performance degradation of 2% in accuracy translates to approximately:
                                                </Typography>
                                                <List dense>
                                                    <ListItem>
                                                        <ListItemIcon>
                                                            <ArrowDownward color="error" />
                                                        </ListItemIcon>
                                                        <ListItemText primary="20 additional misclassifications per 1,000 predictions" />
                                                    </ListItem>
                                                    <ListItem>
                                                        <ListItemIcon>
                                                            <ArrowDownward color="error" />
                                                        </ListItemIcon>
                                                        <ListItemText primary="Estimated $15,000 monthly revenue impact" />
                                                    </ListItem>
                                                    <ListItem>
                                                        <ListItemIcon>
                                                            <ArrowDownward color="error" />
                                                        </ListItemIcon>
                                                        <ListItemText primary="5% increase in customer complaints" />
                                                    </ListItem>
                                                </List>
                                            </CardContent>
                                        </Card>
                                    </Grid>
                                    <Grid item xs={12}>
                                        <Card sx={{ bgcolor: 'background.default' }}>
                                            <CardContent>
                                                <Typography variant="subtitle1" gutterBottom>
                                                    Minimum Detectable Effect
                                                </Typography>
                                                <Typography variant="body2" paragraph>
                                                    With the current evaluation dataset (n=5,000), we can reliably detect:
                                                </Typography>
                                                <List dense>
                                                    <ListItem>
                                                        <ListItemIcon>
                                                            <BarChart color="primary" />
                                                        </ListItemIcon>
                                                        <ListItemText primary="Accuracy changes of ±1.2% or larger" />
                                                    </ListItem>
                                                    <ListItem>
                                                        <ListItemIcon>
                                                            <BarChart color="primary" />
                                                        </ListItemIcon>
                                                        <ListItemText primary="AUC-ROC changes of ±0.015 or larger" />
                                                    </ListItem>
                                                </List>
                                                <Typography variant="body2">
                                                    The observed differences exceed these thresholds, confirming both statistical and practical significance.
                                                </Typography>
                                            </CardContent>
                                        </Card>
                                    </Grid>
                                </Grid>
                            </Paper>
                        </Grid>
                    </Grid>
                </TabPanel>

                {/* Methodology Tab */}
                <TabPanel value={tabValue} index={2}>
                    <Grid container spacing={3}>
                        <Grid item xs={12} md={6}>
                            <Paper sx={{ p: 3 }}>
                                <Typography variant="h6" gutterBottom>
                                    Statistical Testing Methodology
                                </Typography>
                                <Typography variant="body2" color="text.secondary" paragraph>
                                    Explanation of the statistical tests used in the analysis.
                                </Typography>
                                <Box sx={{ mb: 3 }}>
                                    <Typography variant="subtitle1" gutterBottom>
                                        McNemar's Test
                                    </Typography>
                                    <Typography variant="body2" paragraph>
                                        A non-parametric test used to determine if there are differences on a dichotomous dependent variable between two related groups.
                                        In model comparison, it tests whether the disagreement between models is statistically significant.
                                    </Typography>
                                    <Typography variant="subtitle2" gutterBottom>
                                        Assumptions:
                                    </Typography>
                                    <List dense>
                                        {hypothesisTestingData[0].assumptions.map((assumption, index) => (
                                            <ListItem key={index}>
                                                <ListItemIcon>
                                                    <CheckCircle color="primary" fontSize="small" />
                                                </ListItemIcon>
                                                <ListItemText primary={assumption} />
                                            </ListItem>
                                        ))}
                                    </List>
                                    <Typography variant="subtitle2" gutterBottom sx={{ mt: 1 }}>
                                        References:
                                    </Typography>
                                    <Link href="#" color="primary" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                        <MenuBook fontSize="small" />
                                        McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages
                                    </Link>
                                </Box>

                                <Divider sx={{ my: 3 }} />

                                <Box>
                                    <Typography variant="subtitle1" gutterBottom>
                                        Bootstrap Confidence Intervals
                                    </Typography>
                                    <Typography variant="body2" paragraph>
                                        A resampling technique that uses random sampling with replacement to estimate the sampling distribution of a statistic.
                                        It provides confidence intervals without assuming a specific distribution.
                                    </Typography>
                                    <Typography variant="subtitle2" gutterBottom>
                                        Methodology:
                                    </Typography>
                                    <List dense>
                                        <ListItem>
                                            <ListItemIcon>
                                                <CheckCircle color="primary" fontSize="small" />
                                            </ListItemIcon>
                                            <ListItemText primary="1,000 bootstrap samples were generated from the evaluation dataset" />
                                        </ListItem>
                                        <ListItem>
                                            <ListItemIcon>
                                                <CheckCircle color="primary" fontSize="small" />
                                            </ListItemIcon>
                                            <ListItemText primary="Performance metrics were calculated for both models on each sample" />
                                        </ListItem>
                                        <ListItem>
                                            <ListItemIcon>
                                                <CheckCircle color="primary" fontSize="small" />
                                            </ListItemIcon>
                                            <ListItemText primary="95% and 99% confidence intervals were computed using the percentile method" />
                                        </ListItem>
                                    </List>
                                    <Typography variant="subtitle2" gutterBottom sx={{ mt: 1 }}>
                                        References:
                                    </Typography>
                                    <Link href="#" color="primary" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                        <MenuBook fontSize="small" />
                                        Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap
                                    </Link>
                                </Box>
                            </Paper>
                        </Grid>

                        <Grid item xs={12} md={6}>
                            <Paper sx={{ p: 3 }}>
                                <Typography variant="h6" gutterBottom>
                                    Effect Size Methodology
                                </Typography>
                                <Typography variant="body2" color="text.secondary" paragraph>
                                    Explanation of effect size calculations and interpretations.
                                </Typography>
                                <Box sx={{ mb: 3 }}>
                                    <Typography variant="subtitle1" gutterBottom>
                                        Cohen's d
                                    </Typography>
                                    <Typography variant="body2" paragraph>
                                        A standardized measure of effect size that represents the difference between two means divided by a standard deviation.
                                        It quantifies the magnitude of the difference independent of sample size.
                                    </Typography>
                                    <Typography variant="subtitle2" gutterBottom>
                                        Calculation:
                                    </Typography>
                                    <Typography variant="body2" sx={{ fontFamily: 'monospace', my: 1, bgcolor: 'background.default', p: 1 }}>
                                        d = (Mean₁ - Mean₂) / Pooled Standard Deviation
                                    </Typography>
                                    <Typography variant="subtitle2" gutterBottom>
                                        Interpretation:
                                    </Typography>
                                    <TableContainer sx={{ mb: 2 }}>
                                        <Table size="small">
                                            <TableHead>
                                                <TableRow>
                                                    <TableCell>|d| Value</TableCell>
                                                    <TableCell>Effect Size</TableCell>
                                                </TableRow>
                                            </TableHead>
                                            <TableBody>
                                                <TableRow>
                                                    <TableCell>&lt; 0.2</TableCell>
                                                    <TableCell>Negligible</TableCell>
                                                </TableRow>
                                                <TableRow>
                                                    <TableCell>0.2 - 0.5</TableCell>
                                                    <TableCell>Small</TableCell>
                                                </TableRow>
                                                <TableRow>
                                                    <TableCell>0.5 - 0.8</TableCell>
                                                    <TableCell>Medium</TableCell>
                                                </TableRow>
                                                <TableRow>
                                                    <TableCell>&gt; 0.8</TableCell>
                                                    <TableCell>Large</TableCell>
                                                </TableRow>
                                            </TableBody>
                                        </Table>
                                    </TableContainer>
                                </Box>

                                <Divider sx={{ my: 3 }} />

                                <Box>
                                    <Typography variant="subtitle1" gutterBottom>
                                        Validation of Assumptions
                                    </Typography>
                                    <Typography variant="body2" paragraph>
                                        Verification that the data meets the assumptions required for the statistical tests.
                                    </Typography>
                                    <TableContainer>
                                        <Table size="small">
                                            <TableHead>
                                                <TableRow>
                                                    <TableCell>Assumption</TableCell>
                                                    <TableCell>Test</TableCell>
                                                    <TableCell align="center">Status</TableCell>
                                                </TableRow>
                                            </TableHead>
                                            <TableBody>
                                                <TableRow>
                                                    <TableCell>Normality of Differences</TableCell>
                                                    <TableCell>Shapiro-Wilk Test</TableCell>
                                                    <TableCell align="center">
                                                        <Chip label="Passed" color="success" size="small" />
                                                    </TableCell>
                                                </TableRow>
                                                <TableRow>
                                                    <TableCell>Independence of Observations</TableCell>
                                                    <TableCell>Durbin-Watson Test</TableCell>
                                                    <TableCell align="center">
                                                        <Chip label="Passed" color="success" size="small" />
                                                    </TableCell>
                                                </TableRow>
                                                <TableRow>
                                                    <TableCell>Homogeneity of Variance</TableCell>
                                                    <TableCell>Levene's Test</TableCell>
                                                    <TableCell align="center">
                                                        <Chip label="Passed" color="success" size="small" />
                                                    </TableCell>
                                                </TableRow>
                                                <TableRow>
                                                    <TableCell>Sufficient Sample Size</TableCell>
                                                    <TableCell>Power Analysis</TableCell>
                                                    <TableCell align="center">
                                                        <Chip label="Passed" color="success" size="small" />
                                                    </TableCell>
                                                </TableRow>
                                            </TableBody>
                                        </Table>
                                    </TableContainer>
                                </Box>
                            </Paper>
                        </Grid>

                        <Grid item xs={12}>
                            <Paper sx={{ p: 3 }}>
                                <Typography variant="h6" gutterBottom>
                                    Literature References
                                </Typography>
                                <Typography variant="body2" color="text.secondary" paragraph>
                                    Academic references supporting the statistical methodology.
                                </Typography>
                                <Grid container spacing={2}>
                                    <Grid item xs={12} md={6}>
                                        <Card variant="outlined">
                                            <CardContent>
                                                <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
                                                    <School color="primary" />
                                                    <Box>
                                                        <Typography variant="subtitle1">
                                                            Dietterich, T. G. (1998)
                                                        </Typography>
                                                        <Typography variant="body2">
                                                            "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms"
                                                        </Typography>
                                                        <Typography variant="body2" color="text.secondary">
                                                            Neural Computation, 10(7), 1895-1923
                                                        </Typography>
                                                        <Typography variant="body2" sx={{ mt: 1 }}>
                                                            Introduces the 5x2 cross-validation paired t-test for comparing machine learning algorithms.
                                                        </Typography>
                                                    </Box>
                                                </Box>
                                            </CardContent>
                                        </Card>
                                    </Grid>
                                    <Grid item xs={12} md={6}>
                                        <Card variant="outlined">
                                            <CardContent>
                                                <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
                                                    <School color="primary" />
                                                    <Box>
                                                        <Typography variant="subtitle1">
                                                            DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988)
                                                        </Typography>
                                                        <Typography variant="body2">
                                                            "Comparing the Areas Under Two or More Correlated Receiver Operating Characteristic Curves: A Nonparametric Approach"
                                                        </Typography>
                                                        <Typography variant="body2" color="text.secondary">
                                                            Biometrics, 44(3), 837-845
                                                        </Typography>
                                                        <Typography variant="body2" sx={{ mt: 1 }}>
                                                            Describes the method for comparing AUC values from ROC curves.
                                                        </Typography>
                                                    </Box>
                                                </Box>
                                            </CardContent>
                                        </Card>
                                    </Grid>
                                    <Grid item xs={12} md={6}>
                                        <Card variant="outlined">
                                            <CardContent>
                                                <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
                                                    <School color="primary" />
                                                    <Box>
                                                        <Typography variant="subtitle1">
                                                            Efron, B., & Tibshirani, R. J. (1993)
                                                        </Typography>
                                                        <Typography variant="body2">
                                                            "An Introduction to the Bootstrap"
                                                        </Typography>
                                                        <Typography variant="body2" color="text.secondary">
                                                            Chapman & Hall/CRC
                                                        </Typography>
                                                        <Typography variant="body2" sx={{ mt: 1 }}>
                                                            Comprehensive reference on bootstrap methods for statistical inference.
                                                        </Typography>
                                                    </Box>
                                                </Box>
                                            </CardContent>
                                        </Card>
                                    </Grid>
                                    <Grid item xs={12} md={6}>
                                        <Card variant="outlined">
                                            <CardContent>
                                                <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
                                                    <School color="primary" />
                                                    <Box>
                                                        <Typography variant="subtitle1">
                                                            Cohen, J. (1988)
                                                        </Typography>
                                                        <Typography variant="body2">
                                                            "Statistical Power Analysis for the Behavioral Sciences"
                                                        </Typography>
                                                        <Typography variant="body2" color="text.secondary">
                                                            Lawrence Erlbaum Associates
                                                        </Typography>
                                                        <Typography variant="body2" sx={{ mt: 1 }}>
                                                            Defines effect size measures and their interpretations for statistical analysis.
                                                        </Typography>
                                                    </Box>
                                                </Box>
                                            </CardContent>
                                        </Card>
                                    </Grid>
                                </Grid>
                            </Paper>
                        </Grid>
                    </Grid>
                </TabPanel>
            </Paper>
        </Box>
    );
};

export default StatisticalSignificance;