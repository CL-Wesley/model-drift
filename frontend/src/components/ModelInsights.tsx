import React from 'react';
import {
    Box,
    Typography,
    Grid,
    Card,
    CardContent,
    Chip,
    LinearProgress,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Paper,
    Alert,
    List,
    ListItem,
    ListItemIcon,
    ListItemText,
    Divider
} from '@mui/material';
import {
    ModelTraining,
    TrendingUp,
    Warning,
    CheckCircle,
    Lightbulb,
    Schedule,
    Assessment,
    Speed
} from '@mui/icons-material';
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    RadarChart,
    PolarGrid,
    PolarAngleAxis,
    PolarRadiusAxis,
    Radar,
    Legend
} from 'recharts';

import { mockModelInfo, mockDriftResults } from '../data/mockData';

const ModelInsights: React.FC = () => {
    const getStatusColor = (status: string) => {
        switch (status) {
            case 'high': return '#dc3545';
            case 'medium': return '#ffc107';
            case 'low': return '#28a745';
            default: return '#6c757d';
        }
    };

    // Calculate model performance impact prediction
    const calculateModelImpact = () => {
        const highDriftWeight = 0.7;
        const mediumDriftWeight = 0.3;
        const lowDriftWeight = 0.1;

        const impact = (
            mockDriftResults.high_drift_features * highDriftWeight +
            mockDriftResults.medium_drift_features * mediumDriftWeight +
            mockDriftResults.low_drift_features * lowDriftWeight
        ) / mockDriftResults.total_features;

        return Math.min(impact * 100, 100);
    };

    const modelImpact = calculateModelImpact();
    const getImpactSeverity = (): "error" | "warning" | "success" => {
        if (modelImpact > 60) return 'error';
        if (modelImpact > 30) return 'warning';
        return 'success';
    };

    // Prepare feature importance vs drift data
    const featureImportanceVsDrift = mockDriftResults.feature_analysis
        .filter(f => mockModelInfo.feature_importance?.[f.feature])
        .map(feature => ({
            feature: feature.feature,
            importance: (mockModelInfo.feature_importance?.[feature.feature] || 0) * 100,
            drift_score: feature.drift_score,
            status: feature.status
        }))
        .sort((a, b) => b.importance - a.importance);

    // Prepare radar chart data for model risk assessment
    const riskAssessmentData = [
        {
            subject: 'Data Quality',
            score: mockDriftResults.data_quality_score * 100,
            fullMark: 100
        },
        {
            subject: 'Feature Stability',
            score: 100 - (mockDriftResults.overall_drift_score / 3 * 100),
            fullMark: 100
        },
        {
            subject: 'Model Compatibility',
            score: mockDriftResults.model_compatibility_status === 'compatible' ? 90 :
                mockDriftResults.model_compatibility_status === 'warning' ? 60 : 30,
            fullMark: 100
        },
        {
            subject: 'Performance Risk',
            score: 100 - modelImpact,
            fullMark: 100
        },
        {
            subject: 'Retraining Urgency',
            score: mockDriftResults.overall_status === 'low' ? 90 :
                mockDriftResults.overall_status === 'medium' ? 50 : 20,
            fullMark: 100
        }
    ];

    return (
        <Box>
            <Typography variant="h4" gutterBottom>
                Model Insights
            </Typography>
            <Typography variant="body1" color="text.secondary" paragraph>
                Analysis of model performance impact and recommendations for maintaining model effectiveness.
            </Typography>

            {/* Model Information Overview */}
            <Card sx={{ mb: 3, background: 'linear-gradient(135deg, #28a745 0%, #20c997 100%)', color: 'white' }}>
                <CardContent>
                    <Grid container spacing={3} alignItems="center">
                        <Grid item xs={12} md={8}>
                            <Typography variant="h5" gutterBottom sx={{ fontWeight: 600 }}>
                                {mockModelInfo.name}
                            </Typography>
                            <Typography variant="h6" sx={{ mb: 2, opacity: 0.9 }}>
                                {mockModelInfo.type} • Accuracy: {(mockModelInfo.accuracy! * 100).toFixed(1)}%
                            </Typography>
                            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                                <Chip
                                    label={`${mockModelInfo.features.length} Features`}
                                    sx={{ backgroundColor: 'rgba(255,255,255,0.2)', color: 'white' }}
                                />
                                <Chip
                                    label={`Created: ${new Date(mockModelInfo.created_date).toLocaleDateString()}`}
                                    sx={{ backgroundColor: 'rgba(255,255,255,0.2)', color: 'white' }}
                                />
                                <Chip
                                    label={`Format: ${mockModelInfo.format.toUpperCase()}`}
                                    sx={{ backgroundColor: 'rgba(255,255,255,0.2)', color: 'white' }}
                                />
                            </Box>
                        </Grid>
                        <Grid item xs={12} md={4} sx={{ textAlign: 'center' }}>
                            <ModelTraining sx={{ fontSize: 60, mb: 1, opacity: 0.8 }} />
                            <Typography variant="h6">
                                Model Status: {mockDriftResults.model_compatibility_status.toUpperCase()}
                            </Typography>
                        </Grid>
                    </Grid>
                </CardContent>
            </Card>

            <Grid container spacing={3}>
                {/* Model Performance Impact */}
                <Grid item xs={12} lg={6}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                Predicted Performance Impact
                            </Typography>
                            <Alert severity={getImpactSeverity()} sx={{ mb: 3 }}>
                                <Typography variant="body2">
                                    <strong>{modelImpact.toFixed(1)}% performance degradation risk</strong> detected based on current drift patterns.
                                    {modelImpact > 60 && ' Immediate model retraining recommended.'}
                                    {modelImpact > 30 && modelImpact <= 60 && ' Consider scheduling model retraining soon.'}
                                    {modelImpact <= 30 && ' Model performance appears stable.'}
                                </Typography>
                            </Alert>

                            <Box sx={{ mb: 2 }}>
                                <Typography variant="body2" gutterBottom>
                                    Performance Risk Level: {modelImpact.toFixed(1)}%
                                </Typography>
                                <LinearProgress
                                    variant="determinate"
                                    value={modelImpact}
                                    sx={{
                                        height: 10,
                                        borderRadius: 5,
                                        backgroundColor: '#e0e0e0',
                                        '& .MuiLinearProgress-bar': {
                                            backgroundColor: getStatusColor(modelImpact > 60 ? 'high' : modelImpact > 30 ? 'medium' : 'low')
                                        }
                                    }}
                                />
                            </Box>

                            <Typography variant="subtitle2" gutterBottom sx={{ mt: 3 }}>
                                Impact Assessment Breakdown:
                            </Typography>
                            <Table size="small">
                                <TableBody>
                                    <TableRow>
                                        <TableCell>High Drift Features</TableCell>
                                        <TableCell align="right">{mockDriftResults.high_drift_features} features</TableCell>
                                        <TableCell align="right">70% weight</TableCell>
                                    </TableRow>
                                    <TableRow>
                                        <TableCell>Medium Drift Features</TableCell>
                                        <TableCell align="right">{mockDriftResults.medium_drift_features} features</TableCell>
                                        <TableCell align="right">30% weight</TableCell>
                                    </TableRow>
                                    <TableRow>
                                        <TableCell>Low Drift Features</TableCell>
                                        <TableCell align="right">{mockDriftResults.low_drift_features} features</TableCell>
                                        <TableCell align="right">10% weight</TableCell>
                                    </TableRow>
                                </TableBody>
                            </Table>
                        </CardContent>
                    </Card>
                </Grid>

                {/* Model Risk Assessment Radar */}
                <Grid item xs={12} lg={6}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                Model Risk Assessment
                            </Typography>
                            <ResponsiveContainer width="100%" height={300}>
                                <RadarChart data={riskAssessmentData}>
                                    <PolarGrid />
                                    <PolarAngleAxis dataKey="subject" />
                                    <PolarRadiusAxis domain={[0, 100]} />
                                    <Radar
                                        name="Risk Score"
                                        dataKey="score"
                                        stroke="#1f4e79"
                                        fill="#1f4e79"
                                        fillOpacity={0.3}
                                    />
                                </RadarChart>
                            </ResponsiveContainer>
                            <Typography variant="caption" color="text.secondary">
                                Higher scores indicate lower risk. Scores below 50 require attention.
                            </Typography>
                        </CardContent>
                    </Card>
                </Grid>

                {/* Feature Importance vs Drift */}
                <Grid item xs={12}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                Feature Importance vs Drift Score Analysis
                            </Typography>
                            <Typography variant="body2" color="text.secondary" paragraph>
                                Critical analysis showing which important features are experiencing drift.
                            </Typography>

                            <ResponsiveContainer width="100%" height={400}>
                                <BarChart data={featureImportanceVsDrift} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="feature" />
                                    <YAxis yAxisId="left" orientation="left" label={{ value: 'Importance (%)', angle: -90, position: 'insideLeft' }} />
                                    <YAxis yAxisId="right" orientation="right" label={{ value: 'Drift Score', angle: 90, position: 'insideRight' }} />
                                    <Tooltip
                                        formatter={(value: any, name: string) => [
                                            name === 'importance' ? `${value.toFixed(1)}%` : value.toFixed(2),
                                            name === 'importance' ? 'Feature Importance' : 'Drift Score'
                                        ]}
                                    />
                                    <Legend />
                                    <Bar yAxisId="left" dataKey="importance" fill="#1f4e79" name="Feature Importance (%)" />
                                    <Bar yAxisId="right" dataKey="drift_score" fill="#dc3545" name="Drift Score" />
                                </BarChart>
                            </ResponsiveContainer>

                            <Alert severity="info" sx={{ mt: 2 }}>
                                <Typography variant="body2">
                                    <strong>Key Insight:</strong> Features with both high importance and high drift (like credit_score)
                                    pose the greatest risk to model performance and should be prioritized for investigation.
                                </Typography>
                            </Alert>
                        </CardContent>
                    </Card>
                </Grid>

                {/* Action Recommendations */}
                <Grid item xs={12} lg={8}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                Recommended Actions
                            </Typography>

                            <List>
                                <ListItem>
                                    <ListItemIcon>
                                        <Warning sx={{ color: '#dc3545' }} />
                                    </ListItemIcon>
                                    <ListItemText
                                        primary="Immediate Actions"
                                        secondary="Investigate credit_score drift - this is your most important feature with high drift"
                                    />
                                </ListItem>
                                <Divider variant="inset" component="li" />

                                <ListItem>
                                    <ListItemIcon>
                                        <Schedule sx={{ color: '#ffc107' }} />
                                    </ListItemIcon>
                                    <ListItemText
                                        primary="Short-term (1-2 weeks)"
                                        secondary="Validate data collection processes and check for systematic changes in income data"
                                    />
                                </ListItem>
                                <Divider variant="inset" component="li" />

                                <ListItem>
                                    <ListItemIcon>
                                        <Assessment sx={{ color: '#1f4e79' }} />
                                    </ListItemIcon>
                                    <ListItemText
                                        primary="Medium-term (1 month)"
                                        secondary="Plan model retraining with recent data to address distribution shifts"
                                    />
                                </ListItem>
                                <Divider variant="inset" component="li" />

                                <ListItem>
                                    <ListItemIcon>
                                        <Lightbulb sx={{ color: '#28a745' }} />
                                    </ListItemIcon>
                                    <ListItemText
                                        primary="Long-term Strategy"
                                        secondary="Implement automated drift monitoring with alerts at feature importance thresholds"
                                    />
                                </ListItem>
                            </List>
                        </CardContent>
                    </Card>
                </Grid>

                {/* Model Metadata */}
                <Grid item xs={12} lg={4}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                Model Metadata
                            </Typography>

                            <Table size="small">
                                <TableBody>
                                    <TableRow>
                                        <TableCell>Model Type</TableCell>
                                        <TableCell align="right">{mockModelInfo.type}</TableCell>
                                    </TableRow>
                                    <TableRow>
                                        <TableCell>Accuracy</TableCell>
                                        <TableCell align="right">{(mockModelInfo.accuracy! * 100).toFixed(1)}%</TableCell>
                                    </TableRow>
                                    <TableRow>
                                        <TableCell>Features</TableCell>
                                        <TableCell align="right">{mockModelInfo.features.length}</TableCell>
                                    </TableRow>
                                    <TableRow>
                                        <TableCell>Created</TableCell>
                                        <TableCell align="right">{new Date(mockModelInfo.created_date).toLocaleDateString()}</TableCell>
                                    </TableRow>
                                    <TableRow>
                                        <TableCell>File Size</TableCell>
                                        <TableCell align="right">{(mockModelInfo.size / 1024 / 1024).toFixed(1)} MB</TableCell>
                                    </TableRow>
                                    <TableRow>
                                        <TableCell>Format</TableCell>
                                        <TableCell align="right">{mockModelInfo.format.toUpperCase()}</TableCell>
                                    </TableRow>
                                </TableBody>
                            </Table>

                            <Divider sx={{ my: 2 }} />

                            <Typography variant="subtitle2" gutterBottom>
                                Feature Importance Rankings
                            </Typography>
                            {Object.entries(mockModelInfo.feature_importance || {})
                                .sort(([, a], [, b]) => b - a)
                                .map(([feature, importance]) => (
                                    <Box key={feature} sx={{ mb: 1 }}>
                                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                                            <Typography variant="body2">{feature}</Typography>
                                            <Typography variant="body2">{(importance * 100).toFixed(1)}%</Typography>
                                        </Box>
                                        <LinearProgress
                                            variant="determinate"
                                            value={importance * 100}
                                            sx={{ height: 6, borderRadius: 3 }}
                                        />
                                    </Box>
                                ))}
                        </CardContent>
                    </Card>
                </Grid>

                {/* Retraining Recommendations */}
                <Grid item xs={12}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                Model Retraining Schedule Recommendations
                            </Typography>

                            <Grid container spacing={3}>
                                <Grid item xs={12} md={4}>
                                    <Box sx={{ textAlign: 'center', p: 3, backgroundColor: '#f8f9fa', borderRadius: 2 }}>
                                        <Speed sx={{ fontSize: 48, color: '#dc3545', mb: 2 }} />
                                        <Typography variant="h6" sx={{ fontWeight: 600, color: '#dc3545' }}>
                                            Immediate
                                        </Typography>
                                        <Typography variant="body2" color="text.secondary">
                                            High drift detected in critical features
                                        </Typography>
                                        <Chip
                                            label="Within 1 week"
                                            sx={{ mt: 1, backgroundColor: '#dc3545', color: 'white' }}
                                        />
                                    </Box>
                                </Grid>

                                <Grid item xs={12} md={4}>
                                    <Box sx={{ textAlign: 'center', p: 3, backgroundColor: '#f8f9fa', borderRadius: 2 }}>
                                        <TrendingUp sx={{ fontSize: 48, color: '#ffc107', mb: 2 }} />
                                        <Typography variant="h6" sx={{ fontWeight: 600, color: '#ffc107' }}>
                                            Scheduled
                                        </Typography>
                                        <Typography variant="body2" color="text.secondary">
                                            Regular maintenance retraining
                                        </Typography>
                                        <Chip
                                            label="Every 3 months"
                                            sx={{ mt: 1, backgroundColor: '#ffc107', color: 'white' }}
                                        />
                                    </Box>
                                </Grid>

                                <Grid item xs={12} md={4}>
                                    <Box sx={{ textAlign: 'center', p: 3, backgroundColor: '#f8f9fa', borderRadius: 2 }}>
                                        <CheckCircle sx={{ fontSize: 48, color: '#28a745', mb: 2 }} />
                                        <Typography variant="h6" sx={{ fontWeight: 600, color: '#28a745' }}>
                                            Monitoring
                                        </Typography>
                                        <Typography variant="body2" color="text.secondary">
                                            Continuous drift monitoring
                                        </Typography>
                                        <Chip
                                            label="Daily checks"
                                            sx={{ mt: 1, backgroundColor: '#28a745', color: 'white' }}
                                        />
                                    </Box>
                                </Grid>
                            </Grid>
                        </CardContent>
                    </Card>
                </Grid>
            </Grid>
        </Box>
    );
};

export default ModelInsights;
