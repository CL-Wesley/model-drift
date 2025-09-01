import React, { useState } from 'react';
import {
    AppBar,
    Toolbar,
    Typography,
    Box,
    Container,
    ThemeProvider,
    createTheme,
    CssBaseline,
    Paper,
    Breadcrumbs,
    Link
} from '@mui/material';
import {
    Home,
    NavigateNext
} from '@mui/icons-material';

import { TabType } from './types';
import Sidebar from './components/Sidebar';

// Data Drift components
import UploadTab from './components/UploadTab';
import DriftDashboard from './components/DriftDashboard';
import FeatureAnalysis from './components/FeatureAnalysis';
import StatisticalReports from './components/StatisticalReports';
import ModelInsights from './components/ModelInsights';
import ExportAlerts from './components/ExportAlerts';

// Model Drift components
import ModelUpload from './components/model/ModelUpload';
import PerformanceComparison from './components/model/PerformanceComparison';
import DegradationMetrics from './components/model/DegradationMetrics';
import StatisticalSignificance from './components/model/StatisticalSignificance';
import Recommendations from './components/model/Recommendations';

// Professional enterprise theme
const theme = createTheme({
    palette: {
        primary: {
            main: '#1f4e79',
            light: '#2e6da4',
            dark: '#0d3a5f',
        },
        secondary: {
            main: '#6c757d',
            light: '#f8f9fa',
            dark: '#495057',
        },
        success: {
            main: '#28a745',
        },
        warning: {
            main: '#ffc107',
        },
        error: {
            main: '#dc3545',
        },
        background: {
            default: '#f8f9fa',
            paper: '#ffffff',
        },
    },
    typography: {
        fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
        h1: {
            fontWeight: 600,
            fontSize: '2.5rem',
        },
        h2: {
            fontWeight: 600,
            fontSize: '2rem',
        },
        h3: {
            fontWeight: 600,
            fontSize: '1.75rem',
        },
        h4: {
            fontWeight: 600,
            fontSize: '1.5rem',
        },
        h5: {
            fontWeight: 600,
            fontSize: '1.25rem',
        },
        h6: {
            fontWeight: 600,
            fontSize: '1rem',
        },
        body1: {
            fontSize: '1rem',
            lineHeight: 1.6,
        },
        button: {
            textTransform: 'none',
            fontWeight: 500,
        },
    },
    shape: {
        borderRadius: 8,
    },
    spacing: 8,
    components: {
        MuiPaper: {
            styleOverrides: {
                root: {
                    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                },
            },
        },
        MuiButton: {
            styleOverrides: {
                root: {
                    borderRadius: 8,
                    padding: '10px 24px',
                    boxShadow: 'none',
                    '&:hover': {
                        boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
                    },
                },
            },
        },
        MuiCard: {
            styleOverrides: {
                root: {
                    borderRadius: 12,
                    boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                },
            },
        },
    },
});

// Navigation structure is now in Sidebar component

function App() {
    const [activeTab, setActiveTab] = useState<TabType>('home');
    const [sidebarOpen, setSidebarOpen] = useState<boolean>(true);

    const handleTabChange = (id: string) => {
        setActiveTab(id as TabType);
    };

    const toggleSidebar = () => {
        setSidebarOpen(!sidebarOpen);
    };

    // Get breadcrumb path based on active tab
    const getBreadcrumbs = () => {
        if (activeTab === 'home') {
            return [{ label: 'Home', path: '/' }];
        }
        
        if (activeTab.startsWith('model-') || activeTab === 'performance-comparison' || 
            activeTab === 'degradation-metrics' || activeTab === 'statistical-significance' || 
            activeTab === 'recommendations') {
            return [
                { label: 'Home', path: '/' },
                { label: 'Model Drift Analysis', path: '/model-drift' },
                { 
                    label: getTabLabel(activeTab), 
                    path: `/model-drift/${activeTab}` 
                }
            ];
        }
        
        return [
            { label: 'Home', path: '/' },
            { label: 'Data Drift Analysis', path: '/data-drift' },
            { 
                label: getTabLabel(activeTab), 
                path: `/data-drift/${activeTab}` 
            }
        ];
    };

    // Get tab label based on id
    const getTabLabel = (tabId: string): string => {
        switch (tabId) {
            case 'home': return 'Platform Home';
            case 'upload': return 'Upload & Configuration';
            case 'dashboard': return 'Drift Analysis Dashboard';
            case 'feature-analysis': return 'Feature Deep Dive';
            case 'statistical-reports': return 'Statistical Reports';
            case 'model-insights': return 'Model Insights';
            case 'export-alerts': return 'Export & Alerts';
            case 'model-upload': return 'Model Upload & Config';
            case 'performance-comparison': return 'Performance Comparison';
            case 'degradation-metrics': return 'Degradation Metrics';
            case 'statistical-significance': return 'Statistical Significance';
            case 'recommendations': return 'Recommendations';
            case 'settings': return 'Platform Settings';
            default: return 'Unknown';
        }
    };

    const renderTabContent = () => {
        switch (activeTab) {
            case 'home':
                return (
                    <Box sx={{ p: 3 }}>
                        <Typography variant="h4" gutterBottom>Welcome to Drift Detection Platform</Typography>
                        <Typography variant="body1" paragraph>
                            This platform helps you detect and analyze both data drift and model drift in your machine learning systems.
                            Use the sidebar navigation to explore different features.
                        </Typography>
                        <Typography variant="h6" gutterBottom>Quick Start:</Typography>
                        <Typography variant="body1">
                            1. For Data Drift Analysis: Upload reference and current datasets in the Upload & Configuration section.
                        </Typography>
                        <Typography variant="body1">
                            2. For Model Drift Analysis: Upload reference and current models along with evaluation datasets.
                        </Typography>
                    </Box>
                );
            // Data Drift Analysis tabs
            case 'upload':
                return <UploadTab />;
            case 'dashboard':
                return <DriftDashboard />;
            case 'feature-analysis':
                return <FeatureAnalysis />;
            case 'statistical-reports':
                return <StatisticalReports />;
            case 'model-insights':
                return <ModelInsights />;
            case 'export-alerts':
                return <ExportAlerts />;
                
            // Model Drift Analysis tabs
            case 'model-upload':
                return <ModelUpload />;
            case 'performance-comparison':
                return <PerformanceComparison />;
            case 'degradation-metrics':
                return <DegradationMetrics />;
            case 'statistical-significance':
                return <StatisticalSignificance />;
            case 'recommendations':
                return <Recommendations />;
                
            // Settings tab
            case 'settings':
                return (
                    <Box sx={{ p: 3 }}>
                        <Typography variant="h5">Coming Soon: Platform Settings</Typography>
                        <Typography variant="body1" sx={{ mt: 2 }}>
                            The settings section is under development and will be implemented soon.
                        </Typography>
                    </Box>
                );
            default:
                return <UploadTab />;
        }
    };

    return (
        <ThemeProvider theme={theme}>
            <CssBaseline />
            <Box sx={{ display: 'flex', minHeight: '100vh' }}>
                {/* Sidebar */}
                <Sidebar 
                    open={sidebarOpen} 
                    onToggle={toggleSidebar} 
                    activeItem={activeTab} 
                    onItemSelect={handleTabChange} 
                />

                {/* Main Content Area */}
                <Box sx={{ 
                    flexGrow: 1, 
                    display: 'flex', 
                    flexDirection: 'column',
                    transition: theme.transitions.create('margin', {
                        easing: theme.transitions.easing.sharp,
                        duration: theme.transitions.duration.leavingScreen,
                    }),
                    marginLeft: 0,
                }}>
                    {/* Header */}
                    <AppBar 
                        position="static" 
                        elevation={1} 
                        color="default" 
                        sx={{ 
                            borderBottom: `1px solid ${theme.palette.divider}`,
                            bgcolor: 'background.paper'
                        }}
                    >
                        <Toolbar>
                            <Typography variant="h6" component="div" sx={{ flexGrow: 1, fontWeight: 600, color: 'text.primary' }}>
                                {getTabLabel(activeTab)}
                            </Typography>
                            <Typography variant="body2" sx={{ opacity: 0.8 }}>
                                Enterprise Analytics Suite
                            </Typography>
                        </Toolbar>
                    </AppBar>

                    {/* Breadcrumbs */}
                    <Paper 
                        square 
                        elevation={0} 
                        sx={{ 
                            py: 1.5, 
                            px: 3, 
                            borderBottom: `1px solid ${theme.palette.divider}`,
                            bgcolor: 'background.default'
                        }}
                    >
                        <Breadcrumbs 
                            separator={<NavigateNext fontSize="small" />} 
                            aria-label="breadcrumb"
                        >
                            {getBreadcrumbs().map((crumb, index) => {
                                const isLast = index === getBreadcrumbs().length - 1;
                                return isLast ? (
                                    <Typography key={crumb.path} color="text.primary" fontWeight={500}>
                                        {crumb.label}
                                    </Typography>
                                ) : (
                                    <Link 
                                        key={crumb.path} 
                                        color="inherit" 
                                        sx={{ 
                                            cursor: 'pointer',
                                            textDecoration: 'none',
                                            '&:hover': { textDecoration: 'underline' }
                                        }}
                                        onClick={() => {
                                            if (crumb.label === 'Home') {
                                                handleTabChange('home');
                                            } else if (crumb.label === 'Data Drift Analysis') {
                                                handleTabChange('upload');
                                            } else if (crumb.label === 'Model Drift Analysis') {
                                                handleTabChange('model-upload');
                                            }
                                        }}
                                    >
                                        {index === 0 && <Home fontSize="small" sx={{ mr: 0.5, verticalAlign: 'text-bottom' }} />}
                                        {crumb.label}
                                    </Link>
                                );
                            })}
                        </Breadcrumbs>
                    </Paper>

                    {/* Main Content */}
                    <Box sx={{ flexGrow: 1, bgcolor: 'background.default', overflow: 'auto' }}>
                        <Container maxWidth={false} sx={{ py: 3 }}>
                            {renderTabContent()}
                        </Container>
                    </Box>
                </Box>

                {/* Footer */}
                <Paper
                    component="footer"
                    square
                    elevation={1}
                    sx={{
                        py: 2,
                        px: 3,
                        bgcolor: 'background.paper',
                        borderTop: 1,
                        borderColor: 'divider'
                    }}
                >
                    <Container maxWidth={false}>
                        <Typography variant="body2" color="text.secondary" align="center">
                            © 2024 Model Drift Detection Platform - Enterprise Edition
                        </Typography>
                    </Container>
                </Paper>
            </Box>
        </ThemeProvider>
    );
}

export default App;
