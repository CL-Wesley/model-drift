import React, { useState } from 'react';
import {
    Box,
    Paper,
    Typography,
    Grid,
    Card,
    CardContent,
    Divider,
    Chip,
    List,
    ListItem,
    ListItemIcon,
    ListItemText,
    Button,
    Tabs,
    Tab,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    LinearProgress,
    Tooltip,
    IconButton,
} from '@mui/material';
import {
    Warning,
    Error,
    Info,
    CheckCircle,
    Timeline,
    TrendingDown,
    TrendingUp,
    AttachMoney,
    Schedule,
    Build,
    School,
    BarChart,
    ShowChart,
    HelpOutline,
    ArrowForward,
} from '@mui/icons-material';
import {
    BarChart as RechartsBarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip as RechartsTooltip,
    ResponsiveContainer,
    Legend,
    LineChart,
    Line,
    PieChart,
    Pie,
    Cell,
} from 'recharts';

// Mock data for immediate actions
const immediateActionsData = [
    {
        id: 1,
        action: 'Retrain model with recent data',
        urgency: 'high',
        impact: 'high',
        effort: 'medium',
        description: 'The model shows significant drift in key features. Retraining with recent data will likely restore performance.',
        estimatedTime: '3-5 days',
        resources: ['Data Science Team', 'Computing Resources', 'Data Engineering Support'],
    },
    {
        id: 2,
        action: 'Implement feature monitoring for customer behavior metrics',
        urgency: 'medium',
        impact: 'high',
        effort: 'medium',
        description: 'Customer behavior features show the most significant drift. Implementing real-time monitoring will provide early warnings for future drift.',
        estimatedTime: '1-2 weeks',
        resources: ['ML Engineers', 'Data Engineers', 'DevOps'],
    },
    {
        id: 3,
        action: 'Adjust decision thresholds temporarily',
        urgency: 'high',
        impact: 'medium',
        effort: 'low',
        description: 'As an immediate mitigation, adjusting decision thresholds can partially compensate for the performance degradation.',
        estimatedTime: '1-2 days',
        resources: ['Data Scientist', 'Product Manager'],
    },
    {
        id: 4,
        action: 'Investigate data quality issues in source systems',
        urgency: 'medium',
        impact: 'medium',
        effort: 'medium',
        description: 'Some drift patterns suggest potential data quality issues in upstream systems that should be investigated.',
        estimatedTime: '1 week',
        resources: ['Data Engineers', 'Data Quality Team'],
    },
];

// Mock data for strategic planning
const strategicPlanningData = [
    {
        id: 1,
        strategy: 'Implement automated retraining pipeline',
        timeframe: 'Q3 2023',
        impact: 'high',
        description: 'Develop an automated pipeline to retrain models when drift exceeds thresholds, reducing response time and manual effort.',
        keyMilestones: [
            'Design pipeline architecture (2 weeks)',
            'Develop monitoring components (3 weeks)',
            'Implement automated retraining (4 weeks)',
            'Testing and validation (2 weeks)',
        ],
        dependencies: ['CI/CD infrastructure', 'Model versioning system'],
    },
    {
        id: 2,
        strategy: 'Enhance feature engineering for stability',
        timeframe: 'Q4 2023',
        impact: 'high',
        description: 'Redesign feature engineering to create more drift-resistant features, focusing on ratio-based and normalized features.',
        keyMilestones: [
            'Research drift-resistant feature techniques (3 weeks)',
            'Prototype and evaluate new features (4 weeks)',
            'Implement in production pipeline (3 weeks)',
        ],
        dependencies: ['Research time', 'Feature store implementation'],
    },
    {
        id: 3,
        strategy: 'Develop ensemble approach with multiple model types',
        timeframe: 'Q1-Q2 2024',
        impact: 'medium',
        description: 'Create an ensemble of different model types to increase robustness to different types of drift.',
        keyMilestones: [
            'Research appropriate ensemble techniques (4 weeks)',
            'Develop prototype ensemble (6 weeks)',
            'Evaluate performance and drift resistance (4 weeks)',
            'Production implementation (6 weeks)',
        ],
        dependencies: ['Model serving infrastructure updates', 'Research resources'],
    },
];

// Mock data for business impact assessment
const businessImpactData = {
    currentImpact: {
        revenueImpact: -120000,
        customerSatisfaction: -8,
        operationalEfficiency: -12,
        riskExposure: 15,
    },
    mitigationImpact: {
        revenueRecovery: 105000,
        customerSatisfactionImprovement: 7,
        operationalEfficiencyImprovement: 10,
        riskReduction: 12,
    },
    roi: {
        implementationCost: 85000,
        annualizedBenefit: 420000,
        paybackPeriod: 2.5,
        roi: 394,
    },
    timelineData: [
        { month: 'Jan', performance: 0.92 },
        { month: 'Feb', performance: 0.91 },
        { month: 'Mar', performance: 0.90 },
        { month: 'Apr', performance: 0.89 },
        { month: 'May', performance: 0.87 },
        { month: 'Jun', performance: 0.85 },
        { month: 'Jul', performance: 0.84, projected: 0.89 },
        { month: 'Aug', projected: 0.91 },
        { month: 'Sep', projected: 0.92 },
    ],
    costBreakdown: [
        { name: 'Data Collection', value: 15000 },
        { name: 'Model Retraining', value: 35000 },
        { name: 'Testing & Validation', value: 20000 },
        { name: 'Implementation', value: 10000 },
        { name: 'Monitoring Setup', value: 5000 },
    ],
};

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
            id={`recommendations-tabpanel-${index}`}
            aria-labelledby={`recommendations-tab-${index}`}
            {...other}
        >
            {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
        </div>
    );
}

function a11yProps(index: number) {
    return {
        id: `recommendations-tab-${index}`,
        'aria-controls': `recommendations-tabpanel-${index}`,
    };
}

const Recommendations: React.FC = () => {
    const [tabValue, setTabValue] = useState(0);

    const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
        setTabValue(newValue);
    };

    // Helper function to render urgency chip
    const renderUrgencyChip = (urgency: string) => {
        switch (urgency) {
            case 'high':
                return <Chip label="High Urgency" color="error" size="small" icon={<Error />} />;
            case 'medium':
                return <Chip label="Medium Urgency" color="warning" size="small" icon={<Warning />} />;
            case 'low':
                return <Chip label="Low Urgency" color="info" size="small" icon={<Info />} />;
            default:
                return <Chip label="Unknown" size="small" />;
        }
    };

    // Helper function to render impact chip
    const renderImpactChip = (impact: string) => {
        switch (impact) {
            case 'high':
                return <Chip label="High Impact" color="primary" size="small" />;
            case 'medium':
                return <Chip label="Medium Impact" color="secondary" size="small" />;
            case 'low':
                return <Chip label="Low Impact" size="small" />;
            default:
                return <Chip label="Unknown" size="small" />;
        }
    };

    // Helper function to render effort chip
    const renderEffortChip = (effort: string) => {
        switch (effort) {
            case 'high':
                return <Chip label="High Effort" color="default" size="small" />;
            case 'medium':
                return <Chip label="Medium Effort" color="default" size="small" />;
            case 'low':
                return <Chip label="Low Effort" color="success" size="small" />;
            default:
                return <Chip label="Unknown" size="small" />;
        }
    };

    // Colors for pie chart
    const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];

    return (
        <Box>
            <Typography variant="h5" gutterBottom>
                Recommendations
            </Typography>
            <Typography variant="body1" paragraph>
                Based on the model drift analysis, here are recommended actions, strategic planning, and business impact assessment.
            </Typography>

            <Paper sx={{ width: '100%' }}>
                <Tabs
                    value={tabValue}
                    onChange={handleTabChange}
                    indicatorColor="primary"
                    textColor="primary"
                    variant="fullWidth"
                >
                    <Tab label="Immediate Actions" {...a11yProps(0)} />
                    <Tab label="Strategic Planning" {...a11yProps(1)} />
                    <Tab label="Business Impact" {...a11yProps(2)} />
                </Tabs>

                {/* Immediate Actions Tab */}
                <TabPanel value={tabValue} index={0}>
                    <Grid container spacing={3}>
                        <Grid item xs={12}>
                            <Paper sx={{ p: 3 }}>
                                <Typography variant="h6" gutterBottom>
                                    Recommended Immediate Actions
                                </Typography>
                                <Typography variant="body2" color="text.secondary" paragraph>
                                    Prioritized actions to address the detected model drift, based on urgency, impact, and effort required.
                                </Typography>

                                {immediateActionsData.map((action) => (
                                    <Card key={action.id} sx={{ mb: 2, border: action.urgency === 'high' ? '1px solid #f44336' : 'none' }}>
                                        <CardContent>
                                            <Grid container spacing={2}>
                                                <Grid item xs={12} sm={8}>
                                                    <Typography variant="h6" gutterBottom>
                                                        {action.action}
                                                    </Typography>
                                                    <Typography variant="body2" paragraph>
                                                        {action.description}
                                                    </Typography>
                                                    <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                                                        {renderUrgencyChip(action.urgency)}
                                                        {renderImpactChip(action.impact)}
                                                        {renderEffortChip(action.effort)}
                                                    </Box>
                                                    <Grid container spacing={2}>
                                                        <Grid item xs={12} sm={6}>
                                                            <Typography variant="subtitle2" gutterBottom>
                                                                Estimated Time:
                                                            </Typography>
                                                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                                <Schedule fontSize="small" color="action" />
                                                                <Typography variant="body2">{action.estimatedTime}</Typography>
                                                            </Box>
                                                        </Grid>
                                                        <Grid item xs={12} sm={6}>
                                                            <Typography variant="subtitle2" gutterBottom>
                                                                Required Resources:
                                                            </Typography>
                                                            <List dense disablePadding>
                                                                {action.resources.map((resource, index) => (
                                                                    <ListItem key={index} disablePadding>
                                                                        <ListItemIcon sx={{ minWidth: 24 }}>
                                                                            <Build fontSize="small" color="action" />
                                                                        </ListItemIcon>
                                                                        <ListItemText primary={resource} primaryTypographyProps={{ variant: 'body2' }} />
                                                                    </ListItem>
                                                                ))}
                                                            </List>
                                                        </Grid>
                                                    </Grid>
                                                </Grid>
                                                <Grid item xs={12} sm={4}>
                                                    <Card variant="outlined" sx={{ height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                                                        <CardContent>
                                                            <Typography variant="subtitle2" gutterBottom>
                                                                Priority Assessment
                                                            </Typography>
                                                            <Box sx={{ mb: 2 }}>
                                                                <Typography variant="body2" gutterBottom>
                                                                    Urgency:
                                                                </Typography>
                                                                <LinearProgress
                                                                    variant="determinate"
                                                                    value={action.urgency === 'high' ? 100 : action.urgency === 'medium' ? 60 : 30}
                                                                    color={action.urgency === 'high' ? 'error' : action.urgency === 'medium' ? 'warning' : 'info'}
                                                                    sx={{ height: 8, borderRadius: 4 }}
                                                                />
                                                            </Box>
                                                            <Box sx={{ mb: 2 }}>
                                                                <Typography variant="body2" gutterBottom>
                                                                    Impact:
                                                                </Typography>
                                                                <LinearProgress
                                                                    variant="determinate"
                                                                    value={action.impact === 'high' ? 100 : action.impact === 'medium' ? 60 : 30}
                                                                    color={action.impact === 'high' ? 'primary' : action.impact === 'medium' ? 'secondary' : 'info'}
                                                                    sx={{ height: 8, borderRadius: 4 }}
                                                                />
                                                            </Box>
                                                            <Box>
                                                                <Typography variant="body2" gutterBottom>
                                                                    Effort:
                                                                </Typography>
                                                                <LinearProgress
                                                                    variant="determinate"
                                                                    value={action.effort === 'high' ? 100 : action.effort === 'medium' ? 60 : 30}
                                                                    color={action.effort === 'low' ? 'success' : 'secondary'}
                                                                    sx={{ height: 8, borderRadius: 4 }}
                                                                />
                                                            </Box>
                                                        </CardContent>
                                                        <Box sx={{ p: 2, mt: 'auto' }}>
                                                            <Button variant="contained" color="primary" fullWidth endIcon={<ArrowForward />}>
                                                                Implement
                                                            </Button>
                                                        </Box>
                                                    </Card>
                                                </Grid>
                                            </Grid>
                                        </CardContent>
                                    </Card>
                                ))}
                            </Paper>
                        </Grid>
                    </Grid>
                </TabPanel>

                {/* Strategic Planning Tab */}
                <TabPanel value={tabValue} index={1}>
                    <Grid container spacing={3}>
                        <Grid item xs={12}>
                            <Paper sx={{ p: 3 }}>
                                <Typography variant="h6" gutterBottom>
                                    Strategic Planning
                                </Typography>
                                <Typography variant="body2" color="text.secondary" paragraph>
                                    Long-term strategies to improve model robustness, monitoring, and maintenance processes.
                                </Typography>

                                {strategicPlanningData.map((strategy) => (
                                    <Card key={strategy.id} sx={{ mb: 3 }}>
                                        <CardContent>
                                            <Grid container spacing={2}>
                                                <Grid item xs={12} sm={8}>
                                                    <Typography variant="h6" gutterBottom>
                                                        {strategy.strategy}
                                                    </Typography>
                                                    <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                                                        <Chip
                                                            label={strategy.timeframe}
                                                            color="primary"
                                                            size="small"
                                                            icon={<Timeline />}
                                                        />
                                                        {renderImpactChip(strategy.impact)}
                                                    </Box>
                                                    <Typography variant="body2" paragraph>
                                                        {strategy.description}
                                                    </Typography>

                                                    <Divider sx={{ my: 2 }} />

                                                    <Grid container spacing={2}>
                                                        <Grid item xs={12} sm={6}>
                                                            <Typography variant="subtitle2" gutterBottom>
                                                                Key Milestones:
                                                            </Typography>
                                                            <List dense disablePadding>
                                                                {strategy.keyMilestones.map((milestone, index) => (
                                                                    <ListItem key={index} disablePadding>
                                                                        <ListItemIcon sx={{ minWidth: 24 }}>
                                                                            <CheckCircle fontSize="small" color="primary" />
                                                                        </ListItemIcon>
                                                                        <ListItemText primary={milestone} primaryTypographyProps={{ variant: 'body2' }} />
                                                                    </ListItem>
                                                                ))}
                                                            </List>
                                                        </Grid>
                                                        <Grid item xs={12} sm={6}>
                                                            <Typography variant="subtitle2" gutterBottom>
                                                                Dependencies:
                                                            </Typography>
                                                            <List dense disablePadding>
                                                                {strategy.dependencies.map((dependency, index) => (
                                                                    <ListItem key={index} disablePadding>
                                                                        <ListItemIcon sx={{ minWidth: 24 }}>
                                                                            <ArrowForward fontSize="small" color="action" />
                                                                        </ListItemIcon>
                                                                        <ListItemText primary={dependency} primaryTypographyProps={{ variant: 'body2' }} />
                                                                    </ListItem>
                                                                ))}
                                                            </List>
                                                        </Grid>
                                                    </Grid>
                                                </Grid>
                                                <Grid item xs={12} sm={4}>
                                                    <Card variant="outlined" sx={{ height: '100%' }}>
                                                        <CardContent>
                                                            <Typography variant="subtitle2" gutterBottom>
                                                                Implementation Timeline
                                                            </Typography>
                                                            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, mt: 2 }}>
                                                                {strategy.keyMilestones.map((milestone, index) => {
                                                                    const duration = milestone.match(/\((.*?)\)/);
                                                                    const milestoneName = milestone.replace(/\(.*?\)/, '').trim();
                                                                    const progress = 100 - (index * 100) / strategy.keyMilestones.length;

                                                                    return (
                                                                        <Box key={index} sx={{ mb: 2 }}>
                                                                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                                                                                <Typography variant="body2">{milestoneName}</Typography>
                                                                                <Typography variant="body2" color="text.secondary">
                                                                                    {duration ? duration[1] : ''}
                                                                                </Typography>
                                                                            </Box>
                                                                            <LinearProgress
                                                                                variant="determinate"
                                                                                value={index === 0 ? progress : 0}
                                                                                color="primary"
                                                                                sx={{ height: 8, borderRadius: 4 }}
                                                                            />
                                                                        </Box>
                                                                    );
                                                                })}
                                                            </Box>
                                                        </CardContent>
                                                    </Card>
                                                </Grid>
                                            </Grid>
                                        </CardContent>
                                    </Card>
                                ))}
                            </Paper>
                        </Grid>
                    </Grid>
                </TabPanel>

                {/* Business Impact Tab */}
                <TabPanel value={tabValue} index={2}>
                    <Grid container spacing={3}>
                        <Grid item xs={12} md={6}>
                            <Paper sx={{ p: 3 }}>
                                <Typography variant="h6" gutterBottom>
                                    Current Business Impact
                                </Typography>
                                <Typography variant="body2" color="text.secondary" paragraph>
                                    Quantified business impact of the detected model drift.
                                </Typography>

                                <Grid container spacing={2}>
                                    <Grid item xs={12} sm={6}>
                                        <Card sx={{ bgcolor: 'error.light', color: 'error.contrastText' }}>
                                            <CardContent>
                                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                    <AttachMoney />
                                                    <Typography variant="h6">
                                                        ${Math.abs(businessImpactData.currentImpact.revenueImpact).toLocaleString()}
                                                    </Typography>
                                                </Box>
                                                <Typography variant="body2">Estimated Revenue Impact</Typography>
                                            </CardContent>
                                        </Card>
                                    </Grid>
                                    <Grid item xs={12} sm={6}>
                                        <Card sx={{ bgcolor: 'warning.light', color: 'warning.contrastText' }}>
                                            <CardContent>
                                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                    <TrendingDown />
                                                    <Typography variant="h6">
                                                        {businessImpactData.currentImpact.customerSatisfaction}%
                                                    </Typography>
                                                </Box>
                                                <Typography variant="body2">Customer Satisfaction Decrease</Typography>
                                            </CardContent>
                                        </Card>
                                    </Grid>
                                    <Grid item xs={12} sm={6}>
                                        <Card sx={{ bgcolor: 'warning.light', color: 'warning.contrastText' }}>
                                            <CardContent>
                                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                    <TrendingDown />
                                                    <Typography variant="h6">
                                                        {businessImpactData.currentImpact.operationalEfficiency}%
                                                    </Typography>
                                                </Box>
                                                <Typography variant="body2">Operational Efficiency Decrease</Typography>
                                            </CardContent>
                                        </Card>
                                    </Grid>
                                    <Grid item xs={12} sm={6}>
                                        <Card sx={{ bgcolor: 'error.light', color: 'error.contrastText' }}>
                                            <CardContent>
                                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                    <TrendingUp />
                                                    <Typography variant="h6">
                                                        {businessImpactData.currentImpact.riskExposure}%
                                                    </Typography>
                                                </Box>
                                                <Typography variant="body2">Risk Exposure Increase</Typography>
                                            </CardContent>
                                        </Card>
                                    </Grid>
                                </Grid>

                                <Box sx={{ mt: 3 }}>
                                    <Typography variant="subtitle1" gutterBottom>
                                        Performance Projection
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary" paragraph>
                                        Historical and projected model performance after implementing recommendations.
                                    </Typography>
                                    <Box sx={{ height: 300, width: '100%' }}>
                                        <ResponsiveContainer width="100%" height="100%">
                                            <LineChart
                                                data={businessImpactData.timelineData}
                                                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                                            >
                                                <CartesianGrid strokeDasharray="3 3" />
                                                <XAxis dataKey="month" />
                                                <YAxis domain={[0.8, 0.95]} />
                                                <RechartsTooltip />
                                                <Legend />
                                                <Line
                                                    type="monotone"
                                                    dataKey="performance"
                                                    stroke="#8884d8"
                                                    name="Actual Performance"
                                                    strokeWidth={2}
                                                />
                                                <Line
                                                    type="monotone"
                                                    dataKey="projected"
                                                    stroke="#82ca9d"
                                                    strokeDasharray="5 5"
                                                    name="Projected After Fix"
                                                    strokeWidth={2}
                                                />
                                            </LineChart>
                                        </ResponsiveContainer>
                                    </Box>
                                </Box>
                            </Paper>
                        </Grid>

                        <Grid item xs={12} md={6}>
                            <Paper sx={{ p: 3 }}>
                                <Typography variant="h6" gutterBottom>
                                    ROI Analysis
                                </Typography>
                                <Typography variant="body2" color="text.secondary" paragraph>
                                    Return on investment analysis for implementing the recommended actions.
                                </Typography>

                                <Grid container spacing={2}>
                                    <Grid item xs={12} sm={6}>
                                        <Card sx={{ bgcolor: 'primary.light', color: 'primary.contrastText' }}>
                                            <CardContent>
                                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                    <AttachMoney />
                                                    <Typography variant="h6">
                                                        ${businessImpactData.roi.implementationCost.toLocaleString()}
                                                    </Typography>
                                                </Box>
                                                <Typography variant="body2">Implementation Cost</Typography>
                                            </CardContent>
                                        </Card>
                                    </Grid>
                                    <Grid item xs={12} sm={6}>
                                        <Card sx={{ bgcolor: 'success.light', color: 'success.contrastText' }}>
                                            <CardContent>
                                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                    <AttachMoney />
                                                    <Typography variant="h6">
                                                        ${businessImpactData.roi.annualizedBenefit.toLocaleString()}
                                                    </Typography>
                                                </Box>
                                                <Typography variant="body2">Annualized Benefit</Typography>
                                            </CardContent>
                                        </Card>
                                    </Grid>
                                    <Grid item xs={12} sm={6}>
                                        <Card sx={{ bgcolor: 'info.light', color: 'info.contrastText' }}>
                                            <CardContent>
                                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                    <Schedule />
                                                    <Typography variant="h6">
                                                        {businessImpactData.roi.paybackPeriod} months
                                                    </Typography>
                                                </Box>
                                                <Typography variant="body2">Payback Period</Typography>
                                            </CardContent>
                                        </Card>
                                    </Grid>
                                    <Grid item xs={12} sm={6}>
                                        <Card sx={{ bgcolor: 'success.light', color: 'success.contrastText' }}>
                                            <CardContent>
                                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                    <TrendingUp />
                                                    <Typography variant="h6">
                                                        {businessImpactData.roi.roi}%
                                                    </Typography>
                                                </Box>
                                                <Typography variant="body2">ROI (1 Year)</Typography>
                                            </CardContent>
                                        </Card>
                                    </Grid>
                                </Grid>

                                <Box sx={{ mt: 3 }}>
                                    <Typography variant="subtitle1" gutterBottom>
                                        Implementation Cost Breakdown
                                    </Typography>
                                    <Box sx={{ height: 300, width: '100%' }}>
                                        <ResponsiveContainer width="100%" height="100%">
                                            <PieChart>
                                                <Pie
                                                    data={businessImpactData.costBreakdown}
                                                    cx="50%"
                                                    cy="50%"
                                                    labelLine={false}
                                                    outerRadius={80}
                                                    fill="#8884d8"
                                                    dataKey="value"
                                                    nameKey="name"
                                                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                                                >
                                                    {businessImpactData.costBreakdown.map((entry, index) => (
                                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                                    ))}
                                                </Pie>
                                                <RechartsTooltip formatter={(value) => `$${value.toLocaleString()}`} />
                                            </PieChart>
                                        </ResponsiveContainer>
                                    </Box>
                                </Box>

                                <Box sx={{ mt: 3 }}>
                                    <Typography variant="subtitle1" gutterBottom>
                                        Expected Benefits After Implementation
                                    </Typography>
                                    <TableContainer>
                                        <Table size="small">
                                            <TableHead>
                                                <TableRow>
                                                    <TableCell>Metric</TableCell>
                                                    <TableCell align="right">Current Impact</TableCell>
                                                    <TableCell align="right">After Mitigation</TableCell>
                                                    <TableCell align="right">Recovery</TableCell>
                                                </TableRow>
                                            </TableHead>
                                            <TableBody>
                                                <TableRow>
                                                    <TableCell>Revenue</TableCell>
                                                    <TableCell align="right" sx={{ color: 'error.main' }}>
                                                        -${Math.abs(businessImpactData.currentImpact.revenueImpact).toLocaleString()}
                                                    </TableCell>
                                                    <TableCell align="right" sx={{ color: 'success.main' }}>
                                                        +${businessImpactData.mitigationImpact.revenueRecovery.toLocaleString()}
                                                    </TableCell>
                                                    <TableCell align="right">
                                                        {Math.round((businessImpactData.mitigationImpact.revenueRecovery / Math.abs(businessImpactData.currentImpact.revenueImpact)) * 100)}%
                                                    </TableCell>
                                                </TableRow>
                                                <TableRow>
                                                    <TableCell>Customer Satisfaction</TableCell>
                                                    <TableCell align="right" sx={{ color: 'error.main' }}>
                                                        -{businessImpactData.currentImpact.customerSatisfaction}%
                                                    </TableCell>
                                                    <TableCell align="right" sx={{ color: 'success.main' }}>
                                                        +{businessImpactData.mitigationImpact.customerSatisfactionImprovement}%
                                                    </TableCell>
                                                    <TableCell align="right">
                                                        {Math.round((businessImpactData.mitigationImpact.customerSatisfactionImprovement / businessImpactData.currentImpact.customerSatisfaction) * 100)}%
                                                    </TableCell>
                                                </TableRow>
                                                <TableRow>
                                                    <TableCell>Operational Efficiency</TableCell>
                                                    <TableCell align="right" sx={{ color: 'error.main' }}>
                                                        -{businessImpactData.currentImpact.operationalEfficiency}%
                                                    </TableCell>
                                                    <TableCell align="right" sx={{ color: 'success.main' }}>
                                                        +{businessImpactData.mitigationImpact.operationalEfficiencyImprovement}%
                                                    </TableCell>
                                                    <TableCell align="right">
                                                        {Math.round((businessImpactData.mitigationImpact.operationalEfficiencyImprovement / businessImpactData.currentImpact.operationalEfficiency) * 100)}%
                                                    </TableCell>
                                                </TableRow>
                                                <TableRow>
                                                    <TableCell>Risk Exposure</TableCell>
                                                    <TableCell align="right" sx={{ color: 'error.main' }}>
                                                        +{businessImpactData.currentImpact.riskExposure}%
                                                    </TableCell>
                                                    <TableCell align="right" sx={{ color: 'success.main' }}>
                                                        -{businessImpactData.mitigationImpact.riskReduction}%
                                                    </TableCell>
                                                    <TableCell align="right">
                                                        {Math.round((businessImpactData.mitigationImpact.riskReduction / businessImpactData.currentImpact.riskExposure) * 100)}%
                                                    </TableCell>
                                                </TableRow>
                                            </TableBody>
                                        </Table>
                                    </TableContainer>
                                </Box>
                            </Paper>
                        </Grid>
                    </Grid>
                </TabPanel>
            </Paper>
        </Box>
    );
};

export default Recommendations;