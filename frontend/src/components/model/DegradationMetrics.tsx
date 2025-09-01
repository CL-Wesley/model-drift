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
} from '@mui/material';
import {
    ScatterChart,
    Scatter,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip as RechartsTooltip,
    ResponsiveContainer,
    BarChart,
    Bar,
    Legend,
    LineChart,
    Line,
    AreaChart,
    Area,
} from 'recharts';

// Mock data for model disagreement analysis
const predictionScatterData = Array.from({ length: 100 }, (_, i) => ({
    id: i,
    referenceModel: Math.random() * 0.8 + 0.1, // Random values between 0.1 and 0.9
    currentModel: Math.random() * 0.8 + 0.1 + (Math.random() - 0.5) * 0.2, // Add some noise
}));

// Mock data for confidence score distribution
const confidenceDistributionData = [
    { confidence: '0.0-0.1', reference: 5, current: 8 },
    { confidence: '0.1-0.2', reference: 12, current: 15 },
    { confidence: '0.2-0.3', reference: 18, current: 22 },
    { confidence: '0.3-0.4', reference: 25, current: 28 },
    { confidence: '0.4-0.5', reference: 32, current: 35 },
    { confidence: '0.5-0.6', reference: 45, current: 40 },
    { confidence: '0.6-0.7', reference: 58, current: 48 },
    { confidence: '0.7-0.8', reference: 72, current: 62 },
    { confidence: '0.8-0.9', reference: 85, current: 75 },
    { confidence: '0.9-1.0', reference: 95, current: 82 },
];

// Mock data for calibration curve
const calibrationCurveData = [
    { bin: '0.0-0.1', referenceFreq: 0.05, currentFreq: 0.08, perfect: 0.05 },
    { bin: '0.1-0.2', referenceFreq: 0.15, currentFreq: 0.18, perfect: 0.15 },
    { bin: '0.2-0.3', referenceFreq: 0.25, currentFreq: 0.22, perfect: 0.25 },
    { bin: '0.3-0.4', referenceFreq: 0.35, currentFreq: 0.32, perfect: 0.35 },
    { bin: '0.4-0.5', referenceFreq: 0.45, currentFreq: 0.42, perfect: 0.45 },
    { bin: '0.5-0.6', referenceFreq: 0.55, currentFreq: 0.58, perfect: 0.55 },
    { bin: '0.6-0.7', referenceFreq: 0.65, currentFreq: 0.68, perfect: 0.65 },
    { bin: '0.7-0.8', referenceFreq: 0.75, currentFreq: 0.72, perfect: 0.75 },
    { bin: '0.8-0.9', referenceFreq: 0.85, currentFreq: 0.82, perfect: 0.85 },
    { bin: '0.9-1.0', referenceFreq: 0.95, currentFreq: 0.92, perfect: 0.95 },
];

// Mock data for feature importance drift
const featureImportanceData = [
    { feature: 'credit_score', reference: 0.35, current: 0.32, delta: -0.03 },
    { feature: 'income', reference: 0.25, current: 0.28, delta: 0.03 },
    { feature: 'age', reference: 0.15, current: 0.12, delta: -0.03 },
    { feature: 'employment_length', reference: 0.15, current: 0.18, delta: 0.03 },
    { feature: 'debt_to_income', reference: 0.10, current: 0.10, delta: 0.00 },
];

// KL Divergence value
const klDivergence = 0.082;

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
            id={`degradation-metrics-tabpanel-${index}`}
            aria-labelledby={`degradation-metrics-tab-${index}`}
            {...other}
        >
            {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
        </div>
    );
}

function a11yProps(index: number) {
    return {
        id: `degradation-metrics-tab-${index}`,
        'aria-controls': `degradation-metrics-tabpanel-${index}`,
    };
}

const DegradationMetrics: React.FC = () => {
    const [tabValue, setTabValue] = useState(0);

    const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
        setTabValue(newValue);
    };

    return (
        <Box>
            <Typography variant="h5" gutterBottom>
                Degradation Metrics
            </Typography>
            <Typography variant="body1" paragraph>
                Analyze model degradation through prediction disagreement, confidence calibration, and feature importance drift.
            </Typography>

            <Paper sx={{ width: '100%' }}>
                <Tabs
                    value={tabValue}
                    onChange={handleTabChange}
                    indicatorColor="primary"
                    textColor="primary"
                    variant="fullWidth"
                >
                    <Tab label="Model Disagreement" {...a11yProps(0)} />
                    <Tab label="Confidence Analysis" {...a11yProps(1)} />
                    <Tab label="Feature Importance Drift" {...a11yProps(2)} />
                </Tabs>

                {/* Model Disagreement Tab */}
                <TabPanel value={tabValue} index={0}>
                    <Grid container spacing={3}>
                        <Grid item xs={12} lg={8}>
                            <Paper sx={{ p: 3, height: '100%' }}>
                                <Typography variant="h6" gutterBottom>
                                    Prediction Scatter Plot
                                </Typography>
                                <Typography variant="body2" color="text.secondary" paragraph>
                                    This scatter plot shows the prediction probabilities from both models. Points along the diagonal indicate agreement, while points far from the diagonal show disagreement.
                                </Typography>
                                <Box sx={{ height: 400, width: '100%' }}>
                                    <ResponsiveContainer width="100%" height="100%">
                                        <ScatterChart
                                            margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                                        >
                                            <CartesianGrid />
                                            <XAxis 
                                                type="number" 
                                                dataKey="referenceModel" 
                                                name="Reference Model" 
                                                domain={[0, 1]} 
                                                label={{ value: 'Reference Model Prediction', position: 'bottom' }}
                                            />
                                            <YAxis 
                                                type="number" 
                                                dataKey="currentModel" 
                                                name="Current Model" 
                                                domain={[0, 1]} 
                                                label={{ value: 'Current Model Prediction', angle: -90, position: 'left' }}
                                            />
                                            <RechartsTooltip 
                                                cursor={{ strokeDasharray: '3 3' }} 
                                                formatter={(value: any, name: string) => [
                                                    `${parseFloat(value).toFixed(3)}`, 
                                                    name === 'referenceModel' ? 'Reference Model' : 'Current Model'
                                                ]}
                                            />
                                            <Scatter 
                                                name="Predictions" 
                                                data={predictionScatterData} 
                                                fill="#8884d8" 
                                            />
                                            {/* Add a diagonal reference line */}
                                            <Line 
                                                type="monotone" 
                                                dataKey="referenceModel" 
                                                stroke="#ff7300" 
                                                dot={false} 
                                                activeDot={false}
                                                legendType="none"
                                            />
                                        </ScatterChart>
                                    </ResponsiveContainer>
                                </Box>
                            </Paper>
                        </Grid>
                        <Grid item xs={12} lg={4}>
                            <Paper sx={{ p: 3, height: '100%' }}>
                                <Typography variant="h6" gutterBottom>
                                    Disagreement Analysis
                                </Typography>
                                <Box sx={{ mb: 3 }}>
                                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                                        Prediction Difference Statistics
                                    </Typography>
                                    <TableContainer>
                                        <Table size="small">
                                            <TableBody>
                                                <TableRow>
                                                    <TableCell>Mean Absolute Difference</TableCell>
                                                    <TableCell align="right">0.078</TableCell>
                                                </TableRow>
                                                <TableRow>
                                                    <TableCell>Max Difference</TableCell>
                                                    <TableCell align="right">0.215</TableCell>
                                                </TableRow>
                                                <TableRow>
                                                    <TableCell>Standard Deviation</TableCell>
                                                    <TableCell align="right">0.092</TableCell>
                                                </TableRow>
                                                <TableRow>
                                                    <TableCell>Pearson Correlation</TableCell>
                                                    <TableCell align="right">0.876</TableCell>
                                                </TableRow>
                                            </TableBody>
                                        </Table>
                                    </TableContainer>
                                </Box>

                                <Divider sx={{ my: 3 }} />

                                <Box>
                                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                                        Decision Threshold Analysis
                                    </Typography>
                                    <TableContainer>
                                        <Table size="small">
                                            <TableHead>
                                                <TableRow>
                                                    <TableCell>Threshold</TableCell>
                                                    <TableCell align="right">Agreement %</TableCell>
                                                    <TableCell align="right">Disagreement %</TableCell>
                                                </TableRow>
                                            </TableHead>
                                            <TableBody>
                                                <TableRow>
                                                    <TableCell>0.3</TableCell>
                                                    <TableCell align="right">92.5%</TableCell>
                                                    <TableCell align="right">7.5%</TableCell>
                                                </TableRow>
                                                <TableRow>
                                                    <TableCell>0.5</TableCell>
                                                    <TableCell align="right">89.2%</TableCell>
                                                    <TableCell align="right">10.8%</TableCell>
                                                </TableRow>
                                                <TableRow>
                                                    <TableCell>0.7</TableCell>
                                                    <TableCell align="right">85.7%</TableCell>
                                                    <TableCell align="right">14.3%</TableCell>
                                                </TableRow>
                                            </TableBody>
                                        </Table>
                                    </TableContainer>
                                </Box>
                            </Paper>
                        </Grid>
                    </Grid>
                </TabPanel>

                {/* Confidence Analysis Tab */}
                <TabPanel value={tabValue} index={1}>
                    <Grid container spacing={3}>
                        <Grid item xs={12} md={6}>
                            <Paper sx={{ p: 3 }}>
                                <Typography variant="h6" gutterBottom>
                                    Confidence Score Distribution
                                </Typography>
                                <Typography variant="body2" color="text.secondary" paragraph>
                                    Comparison of confidence score distributions between reference and current models.
                                </Typography>
                                <Box sx={{ height: 400, width: '100%' }}>
                                    <ResponsiveContainer width="100%" height="100%">
                                        <BarChart
                                            data={confidenceDistributionData}
                                            margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                                        >
                                            <CartesianGrid strokeDasharray="3 3" />
                                            <XAxis dataKey="confidence" />
                                            <YAxis />
                                            <RechartsTooltip />
                                            <Legend />
                                            <Bar dataKey="reference" name="Reference Model" fill="#8884d8" />
                                            <Bar dataKey="current" name="Current Model" fill="#82ca9d" />
                                        </BarChart>
                                    </ResponsiveContainer>
                                </Box>
                                <Box sx={{ mt: 2 }}>
                                    <Typography variant="subtitle2" gutterBottom>
                                        KL Divergence: {klDivergence.toFixed(3)}
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary">
                                        KL Divergence measures the difference between probability distributions. Lower values indicate more similar distributions.
                                    </Typography>
                                </Box>
                            </Paper>
                        </Grid>
                        <Grid item xs={12} md={6}>
                            <Paper sx={{ p: 3 }}>
                                <Typography variant="h6" gutterBottom>
                                    Calibration Curve
                                </Typography>
                                <Typography variant="body2" color="text.secondary" paragraph>
                                    Reliability diagram showing predicted probabilities vs. observed frequencies. A well-calibrated model follows the diagonal line.
                                </Typography>
                                <Box sx={{ height: 400, width: '100%' }}>
                                    <ResponsiveContainer width="100%" height="100%">
                                        <LineChart
                                            data={calibrationCurveData}
                                            margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                                        >
                                            <CartesianGrid strokeDasharray="3 3" />
                                            <XAxis dataKey="bin" />
                                            <YAxis />
                                            <RechartsTooltip />
                                            <Legend />
                                            <Line 
                                                type="monotone" 
                                                dataKey="perfect" 
                                                name="Perfect Calibration" 
                                                stroke="#ff7300" 
                                                strokeDasharray="5 5" 
                                            />
                                            <Line 
                                                type="monotone" 
                                                dataKey="referenceFreq" 
                                                name="Reference Model" 
                                                stroke="#8884d8" 
                                                activeDot={{ r: 8 }} 
                                            />
                                            <Line 
                                                type="monotone" 
                                                dataKey="currentFreq" 
                                                name="Current Model" 
                                                stroke="#82ca9d" 
                                                activeDot={{ r: 8 }} 
                                            />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </Box>
                            </Paper>
                        </Grid>
                        <Grid item xs={12}>
                            <Paper sx={{ p: 3 }}>
                                <Typography variant="h6" gutterBottom>
                                    Confidence Metrics
                                </Typography>
                                <Grid container spacing={3}>
                                    <Grid item xs={12} sm={6} md={3}>
                                        <Card sx={{ bgcolor: 'background.default' }}>
                                            <CardContent>
                                                <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                                                    Brier Score
                                                </Typography>
                                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                                                    <Typography variant="h5">0.124</Typography>
                                                    <Typography variant="body2" color="error">
                                                        +0.018
                                                    </Typography>
                                                </Box>
                                                <Typography variant="caption" color="text.secondary">
                                                    Lower is better. Reference: 0.106
                                                </Typography>
                                            </CardContent>
                                        </Card>
                                    </Grid>
                                    <Grid item xs={12} sm={6} md={3}>
                                        <Card sx={{ bgcolor: 'background.default' }}>
                                            <CardContent>
                                                <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                                                    ECE (Expected Calibration Error)
                                                </Typography>
                                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                                                    <Typography variant="h5">0.052</Typography>
                                                    <Typography variant="body2" color="error">
                                                        +0.012
                                                    </Typography>
                                                </Box>
                                                <Typography variant="caption" color="text.secondary">
                                                    Lower is better. Reference: 0.040
                                                </Typography>
                                            </CardContent>
                                        </Card>
                                    </Grid>
                                    <Grid item xs={12} sm={6} md={3}>
                                        <Card sx={{ bgcolor: 'background.default' }}>
                                            <CardContent>
                                                <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                                                    MCE (Maximum Calibration Error)
                                                </Typography>
                                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                                                    <Typography variant="h5">0.078</Typography>
                                                    <Typography variant="body2" color="error">
                                                        +0.015
                                                    </Typography>
                                                </Box>
                                                <Typography variant="caption" color="text.secondary">
                                                    Lower is better. Reference: 0.063
                                                </Typography>
                                            </CardContent>
                                        </Card>
                                    </Grid>
                                    <Grid item xs={12} sm={6} md={3}>
                                        <Card sx={{ bgcolor: 'background.default' }}>
                                            <CardContent>
                                                <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                                                    Confidence Entropy
                                                </Typography>
                                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                                                    <Typography variant="h5">0.682</Typography>
                                                    <Typography variant="body2" color="success">
                                                        -0.024
                                                    </Typography>
                                                </Box>
                                                <Typography variant="caption" color="text.secondary">
                                                    Lower indicates more certainty. Reference: 0.706
                                                </Typography>
                                            </CardContent>
                                        </Card>
                                    </Grid>
                                </Grid>
                            </Paper>
                        </Grid>
                    </Grid>
                </TabPanel>

                {/* Feature Importance Drift Tab */}
                <TabPanel value={tabValue} index={2}>
                    <Grid container spacing={3}>
                        <Grid item xs={12} md={8}>
                            <Paper sx={{ p: 3 }}>
                                <Typography variant="h6" gutterBottom>
                                    Feature Importance Comparison
                                </Typography>
                                <Typography variant="body2" color="text.secondary" paragraph>
                                    Comparison of feature importance values between reference and current models.
                                </Typography>
                                <Box sx={{ height: 400, width: '100%' }}>
                                    <ResponsiveContainer width="100%" height="100%">
                                        <BarChart
                                            data={featureImportanceData}
                                            layout="vertical"
                                            margin={{ top: 20, right: 30, left: 100, bottom: 5 }}
                                        >
                                            <CartesianGrid strokeDasharray="3 3" />
                                            <XAxis type="number" domain={[0, 0.4]} />
                                            <YAxis dataKey="feature" type="category" />
                                            <RechartsTooltip />
                                            <Legend />
                                            <Bar dataKey="reference" name="Reference Model" fill="#8884d8" />
                                            <Bar dataKey="current" name="Current Model" fill="#82ca9d" />
                                        </BarChart>
                                    </ResponsiveContainer>
                                </Box>
                            </Paper>
                        </Grid>
                        <Grid item xs={12} md={4}>
                            <Paper sx={{ p: 3, height: '100%' }}>
                                <Typography variant="h6" gutterBottom>
                                    Feature Importance Drift
                                </Typography>
                                <Typography variant="body2" color="text.secondary" paragraph>
                                    Analysis of changes in feature importance between models.
                                </Typography>
                                <TableContainer>
                                    <Table>
                                        <TableHead>
                                            <TableRow>
                                                <TableCell>Feature</TableCell>
                                                <TableCell align="right">Delta</TableCell>
                                                <TableCell align="right">% Change</TableCell>
                                            </TableRow>
                                        </TableHead>
                                        <TableBody>
                                            {featureImportanceData.map((row) => (
                                                <TableRow key={row.feature}>
                                                    <TableCell component="th" scope="row">
                                                        {row.feature}
                                                    </TableCell>
                                                    <TableCell align="right" sx={{
                                                        color: row.delta > 0 ? 'success.main' : 
                                                               row.delta < 0 ? 'error.main' : 'text.primary'
                                                    }}>
                                                        {row.delta > 0 ? '+' : ''}{row.delta.toFixed(3)}
                                                    </TableCell>
                                                    <TableCell align="right" sx={{
                                                        color: row.delta > 0 ? 'success.main' : 
                                                               row.delta < 0 ? 'error.main' : 'text.primary'
                                                    }}>
                                                        {row.delta > 0 ? '+' : ''}{((row.delta / row.reference) * 100).toFixed(1)}%
                                                    </TableCell>
                                                </TableRow>
                                            ))}
                                        </TableBody>
                                    </Table>
                                </TableContainer>

                                <Box sx={{ mt: 4 }}>
                                    <Typography variant="subtitle2" gutterBottom>
                                        Impact Assessment
                                    </Typography>
                                    <Typography variant="body2" paragraph>
                                        The model shows moderate feature importance drift, with employment_length and income gaining importance while credit_score and age decreasing in importance.
                                    </Typography>
                                    <Typography variant="body2">
                                        This shift may indicate changing economic conditions where employment stability and current income are becoming more predictive than historical credit scores.
                                    </Typography>
                                </Box>
                            </Paper>
                        </Grid>
                        <Grid item xs={12}>
                            <Paper sx={{ p: 3 }}>
                                <Typography variant="h6" gutterBottom>
                                    Feature Importance Drift Over Time
                                </Typography>
                                <Typography variant="body2" color="text.secondary" paragraph>
                                    Trend analysis of feature importance changes across model versions.
                                </Typography>
                                <Box sx={{ height: 400, width: '100%' }}>
                                    <ResponsiveContainer width="100%" height="100%">
                                        <AreaChart
                                            data={[
                                                { version: 'v1.0', credit_score: 0.38, income: 0.22, age: 0.18, employment_length: 0.12, debt_to_income: 0.10 },
                                                { version: 'v1.1', credit_score: 0.35, income: 0.25, age: 0.15, employment_length: 0.15, debt_to_income: 0.10 },
                                                { version: 'v1.2', credit_score: 0.32, income: 0.28, age: 0.12, employment_length: 0.18, debt_to_income: 0.10 },
                                            ]}
                                            margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                                        >
                                            <CartesianGrid strokeDasharray="3 3" />
                                            <XAxis dataKey="version" />
                                            <YAxis />
                                            <RechartsTooltip />
                                            <Legend />
                                            <Area type="monotone" dataKey="credit_score" stackId="1" stroke="#8884d8" fill="#8884d8" />
                                            <Area type="monotone" dataKey="income" stackId="1" stroke="#82ca9d" fill="#82ca9d" />
                                            <Area type="monotone" dataKey="age" stackId="1" stroke="#ffc658" fill="#ffc658" />
                                            <Area type="monotone" dataKey="employment_length" stackId="1" stroke="#ff8042" fill="#ff8042" />
                                            <Area type="monotone" dataKey="debt_to_income" stackId="1" stroke="#0088fe" fill="#0088fe" />
                                        </AreaChart>
                                    </ResponsiveContainer>
                                </Box>
                            </Paper>
                        </Grid>
                    </Grid>
                </TabPanel>
            </Paper>
        </Box>
    );
};

export default DegradationMetrics;