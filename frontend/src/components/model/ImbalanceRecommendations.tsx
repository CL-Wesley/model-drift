import React, { useState } from 'react';
import {
    Box,
    Typography,
    Grid,
    Paper,
    Card,
    CardContent,
    CardActions,
    Button,
    Chip,
    Alert,
    Divider,
    List,
    ListItem,
    ListItemIcon,
    ListItemText,
    Accordion,
    AccordionSummary,
    AccordionDetails,
    Stepper,
    Step,
    StepLabel,
    StepContent,
    LinearProgress,
    Tooltip,
    IconButton,
} from '@mui/material';
import {
    ExpandMore as ExpandMoreIcon,
    CheckCircle,
    Warning,
    Error,
    Info,
    Timeline,
    Speed,
    Security,
    TrendingUp,
    Psychology,
    DataUsage,
    ModelTraining,
    Assignment,
} from '@mui/icons-material';

interface ImbalanceRecommendationsProps {
    classData: any;
}

const ImbalanceRecommendations: React.FC<ImbalanceRecommendationsProps> = ({ classData }) => {
    const [activeStep, setActiveStep] = useState(0);
    const [expandedAccordion, setExpandedAccordion] = useState<string | false>(false);

    const handleAccordionChange = (panel: string) => (event: React.SyntheticEvent, isExpanded: boolean) => {
        setExpandedAccordion(isExpanded ? panel : false);
    };

    // Comprehensive recommendation engine
    const recommendations = {
        immediate: [
            {
                id: 'data_collection',
                title: 'Targeted Data Collection',
                priority: 'critical',
                effort: 'high',
                impact: 'high',
                timeframe: '1-2 weeks',
                description: 'Collect additional samples for Class_C to address 60% reduction in representation',
                details: [
                    'Target 300% increase in Class_C samples (from 120 to 480 samples)',
                    'Use stratified sampling to maintain representative distribution',
                    'Implement quality checks to ensure new samples meet data standards',
                    'Consider synthetic data generation if real samples are unavailable'
                ],
                estimatedCost: '$5,000 - $15,000',
                riskLevel: 'low',
                prerequisites: ['Data collection pipeline', 'Quality assurance process']
            },
            {
                id: 'smote_oversampling',
                title: 'SMOTE Oversampling Implementation',
                priority: 'high',
                effort: 'medium',
                impact: 'high',
                timeframe: '3-5 days',
                description: 'Apply Synthetic Minority Oversampling Technique to balance training data',
                details: [
                    'Implement SMOTE with k=5 nearest neighbors for Class_C',
                    'Use BorderlineSMOTE for improved boundary decision learning',
                    'Apply ADASYN for adaptive synthetic sample generation',
                    'Validate synthetic samples quality and distribution'
                ],
                estimatedCost: '$2,000 - $5,000',
                riskLevel: 'medium',
                prerequisites: ['Python environment', 'imbalanced-learn library']
            },
            {
                id: 'threshold_optimization',
                title: 'Classification Threshold Optimization',
                priority: 'medium',
                effort: 'low',
                impact: 'medium',
                timeframe: '1-2 days',
                description: 'Optimize decision thresholds for each class to improve minority class recall',
                details: [
                    'Use precision-recall curves to find optimal thresholds',
                    'Apply class-specific thresholds based on business requirements',
                    'Implement F1-score optimization for balanced performance',
                    'Consider cost-sensitive threshold selection'
                ],
                estimatedCost: '$500 - $1,500',
                riskLevel: 'low',
                prerequisites: ['Model evaluation framework', 'Business requirement analysis']
            }
        ],
        mediumTerm: [
            {
                id: 'ensemble_methods',
                title: 'Ensemble Methods with Class Balancing',
                priority: 'high',
                effort: 'high',
                impact: 'high',
                timeframe: '2-3 weeks',
                description: 'Implement ensemble approaches specifically designed for imbalanced datasets',
                details: [
                    'Deploy BalancedRandomForest with class-specific sampling',
                    'Implement EasyEnsemble for balanced bootstrap sampling',
                    'Use BalancedBagging with different resampling strategies',
                    'Apply RUSBoost for boosting with random undersampling'
                ],
                estimatedCost: '$8,000 - $20,000',
                riskLevel: 'medium',
                prerequisites: ['ML infrastructure', 'Ensemble framework setup']
            },
            {
                id: 'cost_sensitive_learning',
                title: 'Cost-Sensitive Learning Framework',
                priority: 'medium',
                effort: 'medium',
                impact: 'high',
                timeframe: '1-2 weeks',
                description: 'Implement cost-sensitive algorithms that penalize minority class misclassification',
                details: [
                    'Apply class weights inversely proportional to class frequency',
                    'Implement focal loss for hard example mining',
                    'Use MetaCost wrapper for cost-sensitive classification',
                    'Deploy CostSensitiveRandomForest with custom cost matrices'
                ],
                estimatedCost: '$5,000 - $12,000',
                riskLevel: 'low',
                prerequisites: ['Cost matrix definition', 'Business impact assessment']
            },
            {
                id: 'feature_engineering',
                title: 'Class-Specific Feature Engineering',
                priority: 'medium',
                effort: 'high',
                impact: 'medium',
                timeframe: '2-4 weeks',
                description: 'Develop features that better discriminate minority classes',
                details: [
                    'Analyze feature importance for each class separately',
                    'Create class-specific feature interactions',
                    'Implement domain-specific feature engineering for Class_C',
                    'Use feature selection optimized for imbalanced datasets'
                ],
                estimatedCost: '$10,000 - $25,000',
                riskLevel: 'medium',
                prerequisites: ['Domain expertise', 'Feature engineering pipeline']
            }
        ],
        longTerm: [
            {
                id: 'data_pipeline_redesign',
                title: 'Comprehensive Data Pipeline Redesign',
                priority: 'high',
                effort: 'very_high',
                impact: 'very_high',
                timeframe: '2-3 months',
                description: 'Redesign data collection and processing pipeline to prevent future imbalances',
                details: [
                    'Implement real-time class distribution monitoring',
                    'Set up automated alerts for distribution drift',
                    'Create adaptive sampling strategies based on current distribution',
                    'Develop feedback loops for continuous balance maintenance'
                ],
                estimatedCost: '$30,000 - $75,000',
                riskLevel: 'high',
                prerequisites: ['Infrastructure overhaul', 'Team training', 'Business process changes']
            },
            {
                id: 'advanced_ml_techniques',
                title: 'Advanced ML Techniques for Imbalanced Data',
                priority: 'medium',
                effort: 'very_high',
                impact: 'high',
                timeframe: '3-4 months',
                description: 'Implement state-of-the-art techniques for handling class imbalance',
                details: [
                    'Deploy deep learning with focal loss and class-balanced sampling',
                    'Implement GANs for high-quality synthetic minority sample generation',
                    'Use meta-learning approaches for few-shot learning of minority classes',
                    'Apply self-supervised learning for better feature representations'
                ],
                estimatedCost: '$50,000 - $100,000',
                riskLevel: 'high',
                prerequisites: ['Deep learning infrastructure', 'Research team', 'Extended timeline']
            }
        ]
    };

    const implementationSteps = [
        {
            label: 'Assessment & Planning',
            description: 'Analyze current state and plan intervention strategy',
            tasks: ['Measure current imbalance severity', 'Define success metrics', 'Prioritize recommendations'],
            duration: '2-3 days'
        },
        {
            label: 'Quick Wins Implementation',
            description: 'Deploy immediate solutions with high impact and low effort',
            tasks: ['Optimize classification thresholds', 'Apply SMOTE oversampling', 'Implement class weights'],
            duration: '1 week'
        },
        {
            label: 'Data Collection & Enhancement',
            description: 'Execute targeted data collection for minority classes',
            tasks: ['Collect additional Class_C samples', 'Validate data quality', 'Update training dataset'],
            duration: '2-3 weeks'
        },
        {
            label: 'Model Improvement',
            description: 'Implement ensemble methods and cost-sensitive learning',
            tasks: ['Deploy ensemble algorithms', 'Tune hyperparameters', 'Validate performance improvements'],
            duration: '3-4 weeks'
        },
        {
            label: 'Long-term Infrastructure',
            description: 'Build sustainable solutions for ongoing balance maintenance',
            tasks: ['Redesign data pipeline', 'Implement monitoring systems', 'Create feedback loops'],
            duration: '2-3 months'
        }
    ];

    const getPriorityColor = (priority: string) => {
        switch (priority) {
            case 'critical': return 'error';
            case 'high': return 'warning';
            case 'medium': return 'info';
            case 'low': return 'success';
            default: return 'default';
        }
    };

    const getEffortLevel = (effort: string) => {
        const levels = { 'low': 25, 'medium': 50, 'high': 75, 'very_high': 100 };
        return levels[effort as keyof typeof levels] || 0;
    };

    const RecommendationCard = ({ rec, category }: { rec: any, category: string }) => (
        <Card sx={{ mb: 2, border: rec.priority === 'critical' ? 2 : 1, borderColor: rec.priority === 'critical' ? 'error.main' : 'divider' }}>
            <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                    <Typography variant="h6">{rec.title}</Typography>
                    <Box>
                        <Chip
                            label={rec.priority}
                            color={getPriorityColor(rec.priority) as any}
                            size="small"
                            sx={{ mr: 1 }}
                        />
                        <Chip
                            label={rec.timeframe}
                            variant="outlined"
                            size="small"
                        />
                    </Box>
                </Box>

                <Typography variant="body2" color="text.secondary" paragraph>
                    {rec.description}
                </Typography>

                <Grid container spacing={2} sx={{ mb: 2 }}>
                    <Grid item xs={12} sm={4}>
                        <Typography variant="body2" fontWeight="bold">Effort Level</Typography>
                        <LinearProgress
                            variant="determinate"
                            value={getEffortLevel(rec.effort)}
                            color="primary"
                            sx={{ mt: 1, mb: 1 }}
                        />
                        <Typography variant="caption">{rec.effort}</Typography>
                    </Grid>
                    <Grid item xs={12} sm={4}>
                        <Typography variant="body2" fontWeight="bold">Expected Impact</Typography>
                        <LinearProgress
                            variant="determinate"
                            value={getEffortLevel(rec.impact)}
                            color="success"
                            sx={{ mt: 1, mb: 1 }}
                        />
                        <Typography variant="caption">{rec.impact}</Typography>
                    </Grid>
                    <Grid item xs={12} sm={4}>
                        <Typography variant="body2" fontWeight="bold">Risk Level</Typography>
                        <Chip
                            label={rec.riskLevel}
                            color={getPriorityColor(rec.riskLevel) as any}
                            size="small"
                            sx={{ mt: 1 }}
                        />
                    </Grid>
                </Grid>

                <Typography variant="body2" fontWeight="bold" gutterBottom>
                    Estimated Cost: {rec.estimatedCost}
                </Typography>

                <Accordion
                    expanded={expandedAccordion === rec.id}
                    onChange={handleAccordionChange(rec.id)}
                    sx={{ boxShadow: 'none', border: 1, borderColor: 'divider' }}
                >
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                        <Typography variant="body2" fontWeight="bold">Implementation Details</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                        <List dense>
                            {rec.details.map((detail: string, index: number) => (
                                <ListItem key={index} sx={{ py: 0.5 }}>
                                    <ListItemIcon sx={{ minWidth: 32 }}>
                                        <CheckCircle color="primary" fontSize="small" />
                                    </ListItemIcon>
                                    <ListItemText primary={detail} />
                                </ListItem>
                            ))}
                        </List>

                        <Divider sx={{ my: 2 }} />

                        <Typography variant="body2" fontWeight="bold" gutterBottom>
                            Prerequisites:
                        </Typography>
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                            {rec.prerequisites.map((prereq: string, index: number) => (
                                <Chip key={index} label={prereq} size="small" variant="outlined" />
                            ))}
                        </Box>
                    </AccordionDetails>
                </Accordion>
            </CardContent>
            <CardActions>
                <Button size="small" variant="contained" color="primary">
                    Start Implementation
                </Button>
                <Button size="small" variant="outlined">
                    View Resources
                </Button>
                <Button size="small">
                    Export Plan
                </Button>
            </CardActions>
        </Card>
    );

    return (
        <Box>
            <Typography variant="h5" gutterBottom>
                Intelligent Recommendation Engine
            </Typography>
            <Typography variant="body1" color="text.secondary" paragraph>
                AI-powered recommendations for addressing class imbalance with prioritized action plans,
                cost estimates, and implementation roadmaps.
            </Typography>

            {/* Executive Summary */}
            <Alert severity="warning" sx={{ mb: 4 }}>
                <Typography variant="body2" fontWeight="bold">Executive Summary:</Typography>
                <Typography variant="body2">
                    Critical class imbalance detected with Class_C showing 60% sample reduction. Immediate action required
                    to prevent further model performance degradation. Estimated total cost for comprehensive solution:
                    $15,000 - $40,000 over 6-8 weeks. Priority focus on data collection and SMOTE implementation.
                </Typography>
            </Alert>

            {/* Implementation Roadmap */}
            <Paper sx={{ p: 3, mb: 4 }}>
                <Typography variant="h6" gutterBottom>
                    Implementation Roadmap
                </Typography>
                <Stepper activeStep={activeStep} orientation="vertical">
                    {implementationSteps.map((step, index) => (
                        <Step key={index}>
                            <StepLabel>
                                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                    <Typography variant="subtitle1">{step.label}</Typography>
                                    <Chip label={step.duration} size="small" sx={{ ml: 2 }} />
                                </Box>
                            </StepLabel>
                            <StepContent>
                                <Typography variant="body2" color="text.secondary" paragraph>
                                    {step.description}
                                </Typography>
                                <List dense>
                                    {step.tasks.map((task, taskIndex) => (
                                        <ListItem key={taskIndex} sx={{ py: 0.5 }}>
                                            <ListItemIcon sx={{ minWidth: 32 }}>
                                                <Assignment fontSize="small" color="primary" />
                                            </ListItemIcon>
                                            <ListItemText primary={task} />
                                        </ListItem>
                                    ))}
                                </List>
                                <Box sx={{ mt: 2 }}>
                                    <Button
                                        variant="contained"
                                        onClick={() => setActiveStep(index + 1)}
                                        sx={{ mr: 1 }}
                                        disabled={index === implementationSteps.length - 1}
                                    >
                                        {index === implementationSteps.length - 1 ? 'Complete' : 'Next Step'}
                                    </Button>
                                    <Button
                                        disabled={index === 0}
                                        onClick={() => setActiveStep(index - 1)}
                                    >
                                        Back
                                    </Button>
                                </Box>
                            </StepContent>
                        </Step>
                    ))}
                </Stepper>
            </Paper>

            {/* Immediate Actions */}
            <Box sx={{ mb: 4 }}>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                    <Error color="error" sx={{ mr: 1 }} />
                    Immediate Actions (1-2 weeks)
                </Typography>
                {recommendations.immediate.map((rec) => (
                    <RecommendationCard key={rec.id} rec={rec} category="immediate" />
                ))}
            </Box>

            {/* Medium-term Solutions */}
            <Box sx={{ mb: 4 }}>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                    <Timeline color="warning" sx={{ mr: 1 }} />
                    Medium-term Solutions (2-8 weeks)
                </Typography>
                {recommendations.mediumTerm.map((rec) => (
                    <RecommendationCard key={rec.id} rec={rec} category="medium" />
                ))}
            </Box>

            {/* Long-term Strategic Initiatives */}
            <Box sx={{ mb: 4 }}>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                    <TrendingUp color="info" sx={{ mr: 1 }} />
                    Long-term Strategic Initiatives (2-4 months)
                </Typography>
                {recommendations.longTerm.map((rec) => (
                    <RecommendationCard key={rec.id} rec={rec} category="longterm" />
                ))}
            </Box>

            {/* Resource Requirements Summary */}
            <Paper sx={{ p: 3, mb: 4 }}>
                <Typography variant="h6" gutterBottom>
                    Resource Requirements Summary
                </Typography>
                <Grid container spacing={3}>
                    <Grid item xs={12} md={3}>
                        <Box sx={{ textAlign: 'center', p: 2, border: 1, borderColor: 'divider', borderRadius: 1 }}>
                            <DataUsage color="primary" sx={{ fontSize: 40, mb: 1 }} />
                            <Typography variant="h6">$20K - $50K</Typography>
                            <Typography variant="body2" color="text.secondary">Total Budget</Typography>
                        </Box>
                    </Grid>
                    <Grid item xs={12} md={3}>
                        <Box sx={{ textAlign: 'center', p: 2, border: 1, borderColor: 'divider', borderRadius: 1 }}>
                            <Speed color="warning" sx={{ fontSize: 40, mb: 1 }} />
                            <Typography variant="h6">6-8 weeks</Typography>
                            <Typography variant="body2" color="text.secondary">Implementation Time</Typography>
                        </Box>
                    </Grid>
                    <Grid item xs={12} md={3}>
                        <Box sx={{ textAlign: 'center', p: 2, border: 1, borderColor: 'divider', borderRadius: 1 }}>
                            <Psychology color="success" sx={{ fontSize: 40, mb: 1 }} />
                            <Typography variant="h6">2-3 FTE</Typography>
                            <Typography variant="body2" color="text.secondary">Team Size</Typography>
                        </Box>
                    </Grid>
                    <Grid item xs={12} md={3}>
                        <Box sx={{ textAlign: 'center', p: 2, border: 1, borderColor: 'divider', borderRadius: 1 }}>
                            <Security color="error" sx={{ fontSize: 40, mb: 1 }} />
                            <Typography variant="h6">Medium</Typography>
                            <Typography variant="body2" color="text.secondary">Risk Level</Typography>
                        </Box>
                    </Grid>
                </Grid>
            </Paper>

            {/* Success Metrics */}
            <Alert severity="info">
                <Typography variant="body2" fontWeight="bold">Success Metrics to Track:</Typography>
                <ul style={{ margin: '8px 0', paddingLeft: '20px' }}>
                    <li>Class_C F1-score improvement from 0.61 to target 0.75+ (23% increase)</li>
                    <li>Overall model accuracy maintained above 85%</li>
                    <li>Class imbalance ratio reduced from 3.5:1 to target 2:1</li>
                    <li>False negative rate for minority classes below 15%</li>
                    <li>Chi-square test p-value for distribution change above 0.05</li>
                </ul>
            </Alert>
        </Box>
    );
};

export default ImbalanceRecommendations;
