import React, { useState } from 'react';
import {
    AppBar,
    Toolbar,
    Typography,
    Tabs,
    Tab,
    Box,
    Container,
    ThemeProvider,
    createTheme,
    CssBaseline,
    Paper
} from '@mui/material';
import {
    CloudUpload,
    Dashboard,
    Analytics,
    Assessment,
    Insights,
    GetApp
} from '@mui/icons-material';

import { TabType } from './types';
import UploadTab from './components/UploadTab';
import DriftDashboard from './components/DriftDashboard';
import FeatureAnalysis from './components/FeatureAnalysis';
import StatisticalReports from './components/StatisticalReports';
import ModelInsights from './components/ModelInsights';
import ExportAlerts from './components/ExportAlerts';

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

const tabConfig = [
    {
        id: 'upload' as TabType,
        label: 'Upload & Configuration',
        icon: <CloudUpload />,
    },
    {
        id: 'dashboard' as TabType,
        label: 'Drift Analysis Dashboard',
        icon: <Dashboard />,
    },
    {
        id: 'feature-analysis' as TabType,
        label: 'Feature Analysis',
        icon: <Analytics />,
    },
    {
        id: 'statistical-reports' as TabType,
        label: 'Statistical Reports',
        icon: <Assessment />,
    },
    {
        id: 'model-insights' as TabType,
        label: 'Model Insights',
        icon: <Insights />,
    },
    {
        id: 'export-alerts' as TabType,
        label: 'Export & Alerts',
        icon: <GetApp />,
    },
];

function App() {
    const [activeTab, setActiveTab] = useState<TabType>('upload');

    const handleTabChange = (event: React.SyntheticEvent, newValue: TabType) => {
        setActiveTab(newValue);
    };

    const renderTabContent = () => {
        switch (activeTab) {
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
            default:
                return <UploadTab />;
        }
    };

    return (
        <ThemeProvider theme={theme}>
            <CssBaseline />
            <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
                {/* Header */}
                <AppBar position="static" elevation={1}>
                    <Toolbar>
                        <Typography variant="h6" component="div" sx={{ flexGrow: 1, fontWeight: 600 }}>
                            Model Drift Detection Platform
                        </Typography>
                        <Typography variant="body2" sx={{ opacity: 0.8 }}>
                            Enterprise Analytics Suite
                        </Typography>
                    </Toolbar>
                </AppBar>

                {/* Navigation Tabs */}
                <Paper square elevation={1}>
                    <Container maxWidth={false}>
                        <Tabs
                            value={activeTab}
                            onChange={handleTabChange}
                            variant="scrollable"
                            scrollButtons="auto"
                            sx={{
                                borderBottom: 1,
                                borderColor: 'divider',
                                '& .MuiTab-root': {
                                    minHeight: 64,
                                    textTransform: 'none',
                                    fontSize: '0.95rem',
                                    fontWeight: 500,
                                    py: 2,
                                },
                            }}
                        >
                            {tabConfig.map((tab) => (
                                <Tab
                                    key={tab.id}
                                    value={tab.id}
                                    label={tab.label}
                                    icon={tab.icon}
                                    iconPosition="start"
                                />
                            ))}
                        </Tabs>
                    </Container>
                </Paper>

                {/* Main Content */}
                <Box sx={{ flexGrow: 1, bgcolor: 'background.default', py: 3 }}>
                    <Container maxWidth={false}>
                        {renderTabContent()}
                    </Container>
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
