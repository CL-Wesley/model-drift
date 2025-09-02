import React, { useState } from 'react';
import {
    Box,
    Paper,
    Typography,
    Grid,
    Tabs,
    Tab,
    Alert,
    Chip,
    Card,
    CardContent,
    LinearProgress,
} from '@mui/material';
import {
    BarChart,
    PieChart,
    Assessment,
    Balance,
    TrendingDown,
    Warning,
} from '@mui/icons-material';

import ClassDistributionOverview from './model/ClassDistributionOverview';
import ImbalanceMetrics from './model/ImbalanceMetrics';
import PerClassImpact from './model/PerClassImpact';

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
            id={`class-imbalance-tabpanel-${index}`}
            aria-labelledby={`class-imbalance-tab-${index}`}
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

function a11yProps(index: number) {
    return {
        id: `class-imbalance-tab-${index}`,
        'aria-controls': `class-imbalance-tabpanel-${index}`,
    };
}

const ClassImbalanceAnalysis: React.FC = () => {
    const [activeTab, setActiveTab] = useState(0);

    const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
        setActiveTab(newValue);
    };

    // Mock data for class imbalance analysis
    const mockClassData = {
        overall_imbalance_score: 2.8,
        severity_level: 'Medium' as 'Low' | 'Medium' | 'High',
        total_samples: { reference: 10000, current: 8500 },
        class_counts: {
            reference: { 'Class_A': 7000, 'Class_B': 2500, 'Class_C': 500 },
            current: { 'Class_A': 5500, 'Class_B': 2800, 'Class_C': 200 }
        },
        class_percentages: {
            reference: { 'Class_A': 70.0, 'Class_B': 25.0, 'Class_C': 5.0 },
            current: { 'Class_A': 64.7, 'Class_B': 32.9, 'Class_C': 2.4 }
        },
        imbalance_metrics: {
            imbalance_ratio: 27.5,
            gini_coefficient: { reference: 0.42, current: 0.51 },
            shannon_entropy: { reference: 1.18, current: 1.02 },
            chi_square_test: { statistic: 145.67, p_value: 0.001 }
        }
    };

    const getSeverityColor = (severity: string) => {
        switch (severity) {
            case 'Low': return 'success';
            case 'Medium': return 'warning';
            case 'High': return 'error';
            default: return 'info';
        }
    };

    return (
        <Box>
            {/* Header Section */}
            <Box sx={{ mb: 3 }}>
                <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                    <Balance sx={{ mr: 2, fontSize: 40 }} />
                    Class Distribution & Imbalance Analysis
                </Typography>
                <Typography variant="body1" color="text.secondary" paragraph>
                    Comprehensive analysis of class distribution changes and their impact on model performance.
                    Detect imbalances that could affect prediction accuracy and reliability.
                </Typography>
            </Box>

            {/* Overview Cards */}
            <Grid container spacing={3} sx={{ mb: 3 }}>
                <Grid item xs={12} md={3}>
                    <Card>
                        <CardContent>
                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                                <Assessment color="primary" sx={{ mr: 1 }} />
                                <Typography variant="h6">Imbalance Score</Typography>
                            </Box>
                            <Typography variant="h4" color="primary">
                                {mockClassData.overall_imbalance_score}
                            </Typography>
                            <Chip
                                label={mockClassData.severity_level}
                                color={getSeverityColor(mockClassData.severity_level) as any}
                                size="small"
                                sx={{ mt: 1 }}
                            />
                        </CardContent>
                    </Card>
                </Grid>
                <Grid item xs={12} md={3}>
                    <Card>
                        <CardContent>
                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                                <BarChart color="secondary" sx={{ mr: 1 }} />
                                <Typography variant="h6">Sample Change</Typography>
                            </Box>
                            <Typography variant="h4" color="secondary">
                                {Math.round((mockClassData.total_samples.current / mockClassData.total_samples.reference - 1) * 100)}%
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                                {mockClassData.total_samples.current.toLocaleString()} current samples
                            </Typography>
                        </CardContent>
                    </Card>
                </Grid>
                <Grid item xs={12} md={3}>
                    <Card>
                        <CardContent>
                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                                <TrendingDown color="error" sx={{ mr: 1 }} />
                                <Typography variant="h6">Most Affected</Typography>
                            </Box>
                            <Typography variant="h4" color="error">
                                Class_C
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                                -60% reduction
                            </Typography>
                        </CardContent>
                    </Card>
                </Grid>
                <Grid item xs={12} md={3}>
                    <Card>
                        <CardContent>
                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                                <Warning color="warning" sx={{ mr: 1 }} />
                                <Typography variant="h6">Chi-Square</Typography>
                            </Box>
                            <Typography variant="h4" color="warning">
                                145.67
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                                p-value: 0.001
                            </Typography>
                        </CardContent>
                    </Card>
                </Grid>
            </Grid>

            {/* Alert for significant imbalance */}
            {mockClassData.severity_level !== 'Low' && (
                <Alert severity={getSeverityColor(mockClassData.severity_level) as any} sx={{ mb: 3 }}>
                    <Typography variant="body1" fontWeight="bold">
                        {mockClassData.severity_level} Class Imbalance Detected
                    </Typography>
                    <Typography variant="body2">
                        The current dataset shows significant class distribution changes compared to the reference.
                        This may impact model performance and requires attention.
                    </Typography>
                </Alert>
            )}

            {/* Detailed Analysis Tabs */}
            <Paper sx={{ width: '100%' }}>
                <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                    <Tabs
                        value={activeTab}
                        onChange={handleTabChange}
                        aria-label="class imbalance analysis tabs"
                        variant="scrollable"
                        scrollButtons="auto"
                    >
                        <Tab
                            label="Distribution Overview"
                            icon={<PieChart />}
                            iconPosition="start"
                            {...a11yProps(0)}
                        />
                        <Tab
                            label="Imbalance Metrics"
                            icon={<Assessment />}
                            iconPosition="start"
                            {...a11yProps(1)}
                        />
                        <Tab
                            label="Per-Class Impact"
                            icon={<BarChart />}
                            iconPosition="start"
                            {...a11yProps(2)}
                        />
                    </Tabs>
                </Box>

                <TabPanel value={activeTab} index={0}>
                    <ClassDistributionOverview classData={mockClassData} />
                </TabPanel>
                <TabPanel value={activeTab} index={1}>
                    <ImbalanceMetrics classData={mockClassData} />
                </TabPanel>
                <TabPanel value={activeTab} index={2}>
                    <PerClassImpact classData={mockClassData} />
                </TabPanel>
            </Paper>
        </Box>
    );
};

export default ClassImbalanceAnalysis;
