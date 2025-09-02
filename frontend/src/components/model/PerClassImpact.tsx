import React, { useState } from 'react';
import {
    Box,
    Typography,
    Grid,
    Paper,
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
    Tab,
    Tabs,
    LinearProgress,
    Tooltip,
    IconButton,
} from '@mui/material';
import {
    Info as InfoIcon,
    TrendingDown,
    Warning,
    Error,
} from '@mui/icons-material';
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip as ChartTooltip,
    ResponsiveContainer,
    LineChart,
    Line,
    Legend,
    RadarChart,
    PolarGrid,
    PolarAngleAxis,
    PolarRadiusAxis,
    Radar,
} from 'recharts';

interface PerClassImpactProps {
    classData: any;
}

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
            id={`impact-tabpanel-${index}`}
            aria-labelledby={`impact-tab-${index}`}
            {...other}
        >
            {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
        </div>
    );
}

const PerClassImpact: React.FC<PerClassImpactProps> = ({ classData }) => {
    const [tabValue, setTabValue] = useState(0);

    const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
        setTabValue(newValue);
    };

    // Mock performance impact data
    const performanceImpact = {
        'Class_A': {
            precision: { reference: 0.92, current: 0.88, change: -0.04 },
            recall: { reference: 0.89, current: 0.84, change: -0.05 },
            f1Score: { reference: 0.90, current: 0.86, change: -0.04 },
            specificity: { reference: 0.95, current: 0.93, change: -0.02 },
            npv: { reference: 0.94, current: 0.91, change: -0.03 },
            impactSeverity: 'moderate',
            primaryCause: 'Reduced sample representation',
            riskLevel: 'medium'
        },
        'Class_B': {
            precision: { reference: 0.85, current: 0.82, change: -0.03 },
            recall: { reference: 0.87, current: 0.89, change: 0.02 },
            f1Score: { reference: 0.86, current: 0.85, change: -0.01 },
            specificity: { reference: 0.91, current: 0.88, change: -0.03 },
            npv: { reference: 0.92, current: 0.93, change: 0.01 },
            impactSeverity: 'low',
            primaryCause: 'Stable performance despite changes',
            riskLevel: 'low'
        },
        'Class_C': {
            precision: { reference: 0.78, current: 0.65, change: -0.13 },
            recall: { reference: 0.82, current: 0.58, change: -0.24 },
            f1Score: { reference: 0.80, current: 0.61, change: -0.19 },
            specificity: { reference: 0.89, current: 0.85, change: -0.04 },
            npv: { reference: 0.91, current: 0.86, change: -0.05 },
            impactSeverity: 'high',
            primaryCause: 'Severe underrepresentation (60% reduction)',
            riskLevel: 'high'
        },
        'Class_D': {
            precision: { reference: 0.83, current: 0.79, change: -0.04 },
            recall: { reference: 0.80, current: 0.76, change: -0.04 },
            f1Score: { reference: 0.81, current: 0.77, change: -0.04 },
            specificity: { reference: 0.93, current: 0.91, change: -0.02 },
            npv: { reference: 0.92, current: 0.89, change: -0.03 },
            impactSeverity: 'moderate',
            primaryCause: 'Moderate sample reduction',
            riskLevel: 'medium'
        }
    };

    const getSeverityColor = (severity: string) => {
        switch (severity) {
            case 'low': return 'success';
            case 'moderate': return 'warning';
            case 'high': return 'error';
            default: return 'default';
        }
    };

    const getImpactIcon = (severity: string) => {
        switch (severity) {
            case 'low': return <InfoIcon color="success" />;
            case 'moderate': return <Warning color="warning" />;
            case 'high': return <Error color="error" />;
            default: return <InfoIcon />;
        }
    };

    // Prepare data for performance comparison chart
    const performanceComparisonData = Object.entries(performanceImpact).map(([className, metrics]) => ({
        className,
        referencePrecision: metrics.precision.reference,
        currentPrecision: metrics.precision.current,
        referenceRecall: metrics.recall.reference,
        currentRecall: metrics.recall.current,
        referenceF1: metrics.f1Score.reference,
        currentF1: metrics.f1Score.current,
    }));

    // Prepare radar chart data for comprehensive view
    const radarData = Object.entries(performanceImpact).map(([className, metrics]) => ({
        className,
        precision: metrics.precision.current * 100,
        recall: metrics.recall.current * 100,
        f1Score: metrics.f1Score.current * 100,
        specificity: metrics.specificity.current * 100,
        npv: metrics.npv.current * 100,
        fullMark: 100
    }));

    // Confusion matrix heatmap data (simplified representation)
    const confusionMatrixData = [
        { actual: 'Class_A', predicted: 'Class_A', value: 0.88, color: '#4CAF50' },
        { actual: 'Class_A', predicted: 'Class_B', value: 0.07, color: '#FFC107' },
        { actual: 'Class_A', predicted: 'Class_C', value: 0.03, color: '#FF9800' },
        { actual: 'Class_A', predicted: 'Class_D', value: 0.02, color: '#FF5722' },
        { actual: 'Class_B', predicted: 'Class_A', value: 0.05, color: '#FF9800' },
        { actual: 'Class_B', predicted: 'Class_B', value: 0.82, color: '#4CAF50' },
        { actual: 'Class_B', predicted: 'Class_C', value: 0.08, color: '#FFC107' },
        { actual: 'Class_B', predicted: 'Class_D', value: 0.05, color: '#FF9800' },
        { actual: 'Class_C', predicted: 'Class_A', value: 0.15, color: '#FF5722' },
        { actual: 'Class_C', predicted: 'Class_B', value: 0.20, color: '#FF5722' },
        { actual: 'Class_C', predicted: 'Class_C', value: 0.58, color: '#FF9800' },
        { actual: 'Class_C', predicted: 'Class_D', value: 0.07, color: '#FFC107' },
        { actual: 'Class_D', predicted: 'Class_A', value: 0.08, color: '#FFC107' },
        { actual: 'Class_D', predicted: 'Class_B', value: 0.10, color: '#FFC107' },
        { actual: 'Class_D', predicted: 'Class_C', value: 0.06, color: '#FF9800' },
        { actual: 'Class_D', predicted: 'Class_D', value: 0.76, color: '#4CAF50' },
    ];

    return (
        <Box>
            <Typography variant="h5" gutterBottom>
                Per-Class Performance Impact
            </Typography>
            <Typography variant="body1" color="text.secondary" paragraph>
                Analysis of how class imbalance affects individual class performance metrics and prediction accuracy.
            </Typography>

            {/* Impact Overview Cards */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
                {Object.entries(performanceImpact).map(([className, metrics]) => (
                    <Grid item xs={12} sm={6} md={3} key={className}>
                        <Card sx={{ height: '100%', border: metrics.riskLevel === 'high' ? 2 : 1, borderColor: metrics.riskLevel === 'high' ? 'error.main' : 'divider' }}>
                            <CardContent>
                                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                                    <Typography variant="h6">{className}</Typography>
                                    {getImpactIcon(metrics.impactSeverity)}
                                </Box>

                                <Box sx={{ mb: 2 }}>
                                    <Typography variant="body2" color="text.secondary" gutterBottom>
                                        F1-Score Impact
                                    </Typography>
                                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                                        <Typography variant="h5" sx={{ mr: 1 }}>
                                            {(metrics.f1Score.change * 100).toFixed(1)}%
                                        </Typography>
                                        <TrendingDown color={metrics.f1Score.change < -0.05 ? 'error' : 'warning'} />
                                    </Box>
                                    <LinearProgress
                                        variant="determinate"
                                        value={Math.abs(metrics.f1Score.change) * 500}
                                        color={getSeverityColor(metrics.impactSeverity) as any}
                                        sx={{ mb: 1 }}
                                    />
                                </Box>

                                <Box sx={{ mb: 2 }}>
                                    <Typography variant="body2" color="text.secondary">
                                        Current F1: {metrics.f1Score.current.toFixed(3)}
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary">
                                        Reference F1: {metrics.f1Score.reference.toFixed(3)}
                                    </Typography>
                                </Box>

                                <Chip
                                    label={`${metrics.impactSeverity} impact`}
                                    color={getSeverityColor(metrics.impactSeverity) as any}
                                    size="small"
                                    sx={{ mb: 1 }}
                                />

                                <Typography variant="caption" display="block" color="text.secondary">
                                    {metrics.primaryCause}
                                </Typography>
                            </CardContent>
                        </Card>
                    </Grid>
                ))}
            </Grid>

            {/* Detailed Analysis Tabs */}
            <Paper sx={{ mb: 4 }}>
                <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                    <Tabs value={tabValue} onChange={handleTabChange}>
                        <Tab label="Performance Metrics" />
                        <Tab label="Confusion Analysis" />
                        <Tab label="Risk Assessment" />
                    </Tabs>
                </Box>

                <TabPanel value={tabValue} index={0}>
                    {/* Performance Metrics Comparison */}
                    <Grid container spacing={3}>
                        <Grid item xs={12} md={8}>
                            <Typography variant="h6" gutterBottom>
                                Performance Metrics Comparison
                            </Typography>
                            <Box sx={{ height: 300 }}>
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={performanceComparisonData}>
                                        <CartesianGrid strokeDasharray="3 3" />
                                        <XAxis dataKey="className" />
                                        <YAxis domain={[0, 1]} />
                                        <ChartTooltip formatter={(value: any) => [value.toFixed(3), '']} />
                                        <Legend />
                                        <Bar dataKey="referencePrecision" fill="#8884d8" name="Reference Precision" />
                                        <Bar dataKey="currentPrecision" fill="#82ca9d" name="Current Precision" />
                                        <Bar dataKey="referenceRecall" fill="#ffc658" name="Reference Recall" />
                                        <Bar dataKey="currentRecall" fill="#ff7300" name="Current Recall" />
                                    </BarChart>
                                </ResponsiveContainer>
                            </Box>
                        </Grid>

                        <Grid item xs={12} md={4}>
                            <Typography variant="h6" gutterBottom>
                                Overall Performance Radar
                            </Typography>
                            <Box sx={{ height: 300 }}>
                                <ResponsiveContainer width="100%" height="100%">
                                    <RadarChart data={[radarData[2]]}> {/* Class_C as most impacted */}
                                        <PolarGrid />
                                        <PolarAngleAxis dataKey="className" />
                                        <PolarRadiusAxis angle={90} domain={[0, 100]} />
                                        <Radar
                                            name="Class_C Performance"
                                            dataKey="precision"
                                            stroke="#ff7300"
                                            fill="#ff7300"
                                            fillOpacity={0.6}
                                        />
                                    </RadarChart>
                                </ResponsiveContainer>
                            </Box>
                        </Grid>
                    </Grid>
                </TabPanel>

                <TabPanel value={tabValue} index={1}>
                    {/* Confusion Matrix Analysis */}
                    <Typography variant="h6" gutterBottom>
                        Confusion Matrix Impact Analysis
                    </Typography>
                    <Grid container spacing={3}>
                        <Grid item xs={12} md={8}>
                            <TableContainer>
                                <Table>
                                    <TableHead>
                                        <TableRow>
                                            <TableCell><strong>Actual \ Predicted</strong></TableCell>
                                            <TableCell align="center"><strong>Class_A</strong></TableCell>
                                            <TableCell align="center"><strong>Class_B</strong></TableCell>
                                            <TableCell align="center"><strong>Class_C</strong></TableCell>
                                            <TableCell align="center"><strong>Class_D</strong></TableCell>
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        {['Class_A', 'Class_B', 'Class_C', 'Class_D'].map((actualClass) => (
                                            <TableRow key={actualClass}>
                                                <TableCell component="th" scope="row">
                                                    <strong>{actualClass}</strong>
                                                </TableCell>
                                                {['Class_A', 'Class_B', 'Class_C', 'Class_D'].map((predictedClass) => {
                                                    const cellData = confusionMatrixData.find(
                                                        item => item.actual === actualClass && item.predicted === predictedClass
                                                    );
                                                    return (
                                                        <TableCell key={predictedClass} align="center">
                                                            <Box
                                                                sx={{
                                                                    p: 1,
                                                                    borderRadius: 1,
                                                                    backgroundColor: cellData?.color,
                                                                    color: 'white',
                                                                    fontWeight: 'bold'
                                                                }}
                                                            >
                                                                {cellData?.value.toFixed(2)}
                                                            </Box>
                                                        </TableCell>
                                                    );
                                                })}
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </TableContainer>
                        </Grid>
                        <Grid item xs={12} md={4}>
                            <Alert severity="error" sx={{ mb: 2 }}>
                                <Typography variant="body2" fontWeight="bold">
                                    Critical Misclassification Pattern
                                </Typography>
                                <Typography variant="body2">
                                    Class_C shows 42% misclassification rate, primarily confused with Class_A (15%) and Class_B (20%).
                                </Typography>
                            </Alert>
                            <Alert severity="warning">
                                <Typography variant="body2" fontWeight="bold">
                                    Impact on Business Metrics
                                </Typography>
                                <ul style={{ margin: '8px 0', paddingLeft: '20px' }}>
                                    <li>False negatives for Class_C increased by 180%</li>
                                    <li>Cross-class confusion up 25% overall</li>
                                    <li>Decision boundary uncertainty increased</li>
                                </ul>
                            </Alert>
                        </Grid>
                    </Grid>
                </TabPanel>

                <TabPanel value={tabValue} index={2}>
                    {/* Risk Assessment */}
                    <Typography variant="h6" gutterBottom>
                        Risk Assessment & Mitigation Strategies
                    </Typography>
                    <Grid container spacing={3}>
                        {Object.entries(performanceImpact).map(([className, metrics]) => (
                            <Grid item xs={12} md={6} key={className}>
                                <Paper sx={{ p: 3, border: 1, borderColor: getSeverityColor(metrics.riskLevel) + '.main' }}>
                                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                                        <Typography variant="h6" sx={{ mr: 1 }}>{className}</Typography>
                                        <Chip
                                            label={`${metrics.riskLevel} risk`}
                                            color={getSeverityColor(metrics.riskLevel) as any}
                                            size="small"
                                        />
                                    </Box>

                                    <Typography variant="body2" color="text.secondary" paragraph>
                                        <strong>Primary Issue:</strong> {metrics.primaryCause}
                                    </Typography>

                                    <Typography variant="body2" fontWeight="bold" gutterBottom>
                                        Impact Metrics:
                                    </Typography>
                                    <Box sx={{ pl: 2, mb: 2 }}>
                                        <Typography variant="body2">
                                            • Precision: {(metrics.precision.change * 100).toFixed(1)}% change
                                        </Typography>
                                        <Typography variant="body2">
                                            • Recall: {(metrics.recall.change * 100).toFixed(1)}% change
                                        </Typography>
                                        <Typography variant="body2">
                                            • F1-Score: {(metrics.f1Score.change * 100).toFixed(1)}% change
                                        </Typography>
                                    </Box>

                                    <Typography variant="body2" fontWeight="bold" gutterBottom>
                                        Recommended Actions:
                                    </Typography>
                                    <Box sx={{ pl: 2 }}>
                                        {metrics.riskLevel === 'high' && (
                                            <>
                                                <Typography variant="body2">• Immediate data collection for underrepresented class</Typography>
                                                <Typography variant="body2">• Apply SMOTE or similar oversampling techniques</Typography>
                                                <Typography variant="body2">• Consider ensemble methods with class-specific models</Typography>
                                                <Typography variant="body2">• Implement custom loss functions with class weights</Typography>
                                            </>
                                        )}
                                        {metrics.riskLevel === 'medium' && (
                                            <>
                                                <Typography variant="body2">• Monitor performance trends closely</Typography>
                                                <Typography variant="body2">• Consider stratified sampling for new data</Typography>
                                                <Typography variant="body2">• Adjust classification thresholds if needed</Typography>
                                            </>
                                        )}
                                        {metrics.riskLevel === 'low' && (
                                            <>
                                                <Typography variant="body2">• Continue regular monitoring</Typography>
                                                <Typography variant="body2">• Maintain current data collection strategy</Typography>
                                            </>
                                        )}
                                    </Box>
                                </Paper>
                            </Grid>
                        ))}
                    </Grid>
                </TabPanel>
            </Paper>

            {/* Summary Alert */}
            <Alert severity="warning">
                <Typography variant="body2" fontWeight="bold">Performance Impact Summary:</Typography>
                <Typography variant="body2">
                    Class imbalance has resulted in significant performance degradation, with Class_C showing the most severe impact (-19% F1-score).
                    Immediate intervention through targeted data collection and rebalancing techniques is recommended to restore model performance
                    and prevent further degradation in minority class predictions.
                </Typography>
            </Alert>
        </Box>
    );
};

export default PerClassImpact;
