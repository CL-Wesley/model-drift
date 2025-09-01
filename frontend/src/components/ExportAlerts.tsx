import React, { useState } from 'react';
import {
    Box,
    Typography,
    Grid,
    Card,
    CardContent,
    Button,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    TextField,
    Switch,
    FormControlLabel,
    Chip,
    Alert,
    List,
    ListItem,
    ListItemIcon,
    ListItemText,
    Divider,
    Paper,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    SelectChangeEvent
} from '@mui/material';
import {
    FileDownload,
    Email,
    NotificationsActive,
    Schedule,
    Settings,
    PictureAsPdf,
    TableView,
    Code,
    InsertChart,
    CheckCircle,
    Warning
} from '@mui/icons-material';

import { mockDriftResults } from '../data/mockData';

interface AlertConfig {
    enabled: boolean;
    email: string;
    webhook: string;
    threshold: number;
    frequency: 'daily' | 'weekly' | 'monthly';
}

const ExportAlerts: React.FC = () => {
    const [exportFormat, setExportFormat] = useState<'pdf' | 'csv' | 'html' | 'json'>('pdf');
    const [includeCharts, setIncludeCharts] = useState(true);
    const [includeDetails, setIncludeDetails] = useState(true);
    const [includeRecommendations, setIncludeRecommendations] = useState(true);

    const [alertConfig, setAlertConfig] = useState<AlertConfig>({
        enabled: false,
        email: '',
        webhook: '',
        threshold: 2.0,
        frequency: 'weekly'
    });

    const [exportHistory] = useState([
        { date: new Date('2024-09-01'), format: 'PDF', size: '2.3 MB', status: 'Completed' },
        { date: new Date('2024-08-28'), format: 'CSV', size: '156 KB', status: 'Completed' },
        { date: new Date('2024-08-25'), format: 'HTML', size: '1.8 MB', status: 'Completed' },
    ]);

    const handleExportFormatChange = (event: SelectChangeEvent) => {
        setExportFormat(event.target.value as 'pdf' | 'csv' | 'html' | 'json');
    };

    const handleFrequencyChange = (event: SelectChangeEvent) => {
        setAlertConfig(prev => ({ ...prev, frequency: event.target.value as 'daily' | 'weekly' | 'monthly' }));
    };

    const handleExport = () => {
        // Mock export functionality
        alert(`Exporting ${exportFormat.toUpperCase()} report...`);
    };

    const getFormatIcon = (format: string) => {
        switch (format.toLowerCase()) {
            case 'pdf': return <PictureAsPdf />;
            case 'csv': return <TableView />;
            case 'html': return <InsertChart />;
            case 'json': return <Code />;
            default: return <FileDownload />;
        }
    };

    const getFormatDescription = (format: string) => {
        switch (format) {
            case 'pdf': return 'Professional report with charts and analysis - perfect for presentations';
            case 'csv': return 'Raw data export for further analysis in Excel or other tools';
            case 'html': return 'Interactive web report with embedded charts and navigation';
            case 'json': return 'Structured data format for API integration and custom applications';
            default: return '';
        }
    };

    return (
        <Box>
            <Typography variant="h4" gutterBottom>
                Export & Alerts
            </Typography>
            <Typography variant="body1" color="text.secondary" paragraph>
                Generate reports, configure automated alerts, and set up monitoring for continuous drift detection.
            </Typography>

            <Grid container spacing={3}>
                {/* Export Reports Section */}
                <Grid item xs={12} lg={8}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                Export Analysis Reports
                            </Typography>

                            <Grid container spacing={3}>
                                <Grid item xs={12} md={6}>
                                    <FormControl fullWidth>
                                        <InputLabel>Report Format</InputLabel>
                                        <Select
                                            value={exportFormat}
                                            label="Report Format"
                                            onChange={handleExportFormatChange}
                                        >
                                            <MenuItem value="pdf">PDF Report</MenuItem>
                                            <MenuItem value="csv">CSV Data</MenuItem>
                                            <MenuItem value="html">HTML Report</MenuItem>
                                            <MenuItem value="json">JSON Data</MenuItem>
                                        </Select>
                                    </FormControl>

                                    <Alert severity="info" sx={{ mt: 2 }}>
                                        <Typography variant="body2">
                                            {getFormatDescription(exportFormat)}
                                        </Typography>
                                    </Alert>
                                </Grid>

                                <Grid item xs={12} md={6}>
                                    <Typography variant="subtitle2" gutterBottom>
                                        Export Options
                                    </Typography>

                                    <FormControlLabel
                                        control={
                                            <Switch
                                                checked={includeCharts}
                                                onChange={(e) => setIncludeCharts(e.target.checked)}
                                            />
                                        }
                                        label="Include Charts & Visualizations"
                                    />

                                    <FormControlLabel
                                        control={
                                            <Switch
                                                checked={includeDetails}
                                                onChange={(e) => setIncludeDetails(e.target.checked)}
                                            />
                                        }
                                        label="Include Detailed Statistics"
                                        sx={{ display: 'block' }}
                                    />

                                    <FormControlLabel
                                        control={
                                            <Switch
                                                checked={includeRecommendations}
                                                onChange={(e) => setIncludeRecommendations(e.target.checked)}
                                            />
                                        }
                                        label="Include Recommendations"
                                        sx={{ display: 'block' }}
                                    />
                                </Grid>
                            </Grid>

                            <Box sx={{ mt: 3, textAlign: 'center' }}>
                                <Button
                                    variant="contained"
                                    size="large"
                                    startIcon={getFormatIcon(exportFormat)}
                                    onClick={handleExport}
                                    sx={{ px: 4 }}
                                >
                                    Export {exportFormat.toUpperCase()} Report
                                </Button>
                            </Box>

                            {/* Preview of what will be included */}
                            <Paper sx={{ mt: 3, p: 2, backgroundColor: '#f8f9fa' }}>
                                <Typography variant="subtitle2" gutterBottom>
                                    Report Contents Preview:
                                </Typography>
                                <List dense>
                                    <ListItem>
                                        <ListItemIcon><CheckCircle sx={{ color: '#28a745', fontSize: 20 }} /></ListItemIcon>
                                        <ListItemText primary="Executive Summary" secondary="Overall drift analysis and key findings" />
                                    </ListItem>
                                    <ListItem>
                                        <ListItemIcon><CheckCircle sx={{ color: '#28a745', fontSize: 20 }} /></ListItemIcon>
                                        <ListItemText primary="Feature Analysis" secondary="Individual feature drift scores and statistics" />
                                    </ListItem>
                                    {includeCharts && (
                                        <ListItem>
                                            <ListItemIcon><CheckCircle sx={{ color: '#28a745', fontSize: 20 }} /></ListItemIcon>
                                            <ListItemText primary="Charts & Visualizations" secondary="Drift heatmaps, distribution comparisons, trends" />
                                        </ListItem>
                                    )}
                                    {includeDetails && (
                                        <ListItem>
                                            <ListItemIcon><CheckCircle sx={{ color: '#28a745', fontSize: 20 }} /></ListItemIcon>
                                            <ListItemText primary="Statistical Details" secondary="KL divergence, PSI, KS tests, p-values" />
                                        </ListItem>
                                    )}
                                    {includeRecommendations && (
                                        <ListItem>
                                            <ListItemIcon><CheckCircle sx={{ color: '#28a745', fontSize: 20 }} /></ListItemIcon>
                                            <ListItemText primary="Action Recommendations" secondary="Specific steps for addressing drift issues" />
                                        </ListItem>
                                    )}
                                </List>
                            </Paper>
                        </CardContent>
                    </Card>
                </Grid>

                {/* Export History */}
                <Grid item xs={12} lg={4}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                Export History
                            </Typography>

                            <TableContainer>
                                <Table size="small">
                                    <TableHead>
                                        <TableRow>
                                            <TableCell>Date</TableCell>
                                            <TableCell>Format</TableCell>
                                            <TableCell>Size</TableCell>
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        {exportHistory.map((export_, index) => (
                                            <TableRow key={index}>
                                                <TableCell>{export_.date.toLocaleDateString()}</TableCell>
                                                <TableCell>
                                                    <Chip label={export_.format} size="small" />
                                                </TableCell>
                                                <TableCell>{export_.size}</TableCell>
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </TableContainer>

                            <Alert severity="info" sx={{ mt: 2 }}>
                                <Typography variant="body2">
                                    Reports are automatically archived for 90 days
                                </Typography>
                            </Alert>
                        </CardContent>
                    </Card>
                </Grid>

                {/* Alert Configuration Section */}
                <Grid item xs={12}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                Alert Configuration
                            </Typography>
                            <Typography variant="body2" color="text.secondary" paragraph>
                                Set up automated alerts to be notified when drift thresholds are exceeded.
                            </Typography>

                            <Grid container spacing={3}>
                                <Grid item xs={12} md={6}>
                                    <FormControlLabel
                                        control={
                                            <Switch
                                                checked={alertConfig.enabled}
                                                onChange={(e) => setAlertConfig(prev => ({ ...prev, enabled: e.target.checked }))}
                                            />
                                        }
                                        label="Enable Automated Alerts"
                                        sx={{ mb: 2 }}
                                    />

                                    <TextField
                                        fullWidth
                                        label="Email Notifications"
                                        value={alertConfig.email}
                                        onChange={(e) => setAlertConfig(prev => ({ ...prev, email: e.target.value }))}
                                        disabled={!alertConfig.enabled}
                                        placeholder="admin@company.com, team@company.com"
                                        sx={{ mb: 2 }}
                                        helperText="Comma-separated email addresses"
                                    />

                                    <TextField
                                        fullWidth
                                        label="Webhook URL (Optional)"
                                        value={alertConfig.webhook}
                                        onChange={(e) => setAlertConfig(prev => ({ ...prev, webhook: e.target.value }))}
                                        disabled={!alertConfig.enabled}
                                        placeholder="https://hooks.slack.com/..."
                                        sx={{ mb: 2 }}
                                        helperText="Slack, Teams, or custom webhook"
                                    />
                                </Grid>

                                <Grid item xs={12} md={6}>
                                    <TextField
                                        fullWidth
                                        label="Alert Threshold"
                                        type="number"
                                        value={alertConfig.threshold}
                                        onChange={(e) => setAlertConfig(prev => ({ ...prev, threshold: parseFloat(e.target.value) }))}
                                        disabled={!alertConfig.enabled}
                                        inputProps={{ step: 0.1, min: 0 }}
                                        sx={{ mb: 2 }}
                                        helperText="Trigger alerts when drift score exceeds this value"
                                    />

                                    <FormControl fullWidth disabled={!alertConfig.enabled}>
                                        <InputLabel>Monitoring Frequency</InputLabel>
                                        <Select
                                            value={alertConfig.frequency}
                                            label="Monitoring Frequency"
                                            onChange={handleFrequencyChange}
                                        >
                                            <MenuItem value="daily">Daily</MenuItem>
                                            <MenuItem value="weekly">Weekly</MenuItem>
                                            <MenuItem value="monthly">Monthly</MenuItem>
                                        </Select>
                                    </FormControl>
                                </Grid>
                            </Grid>

                            {alertConfig.enabled && (
                                <Alert severity="success" sx={{ mt: 3 }}>
                                    <Typography variant="body2">
                                        <strong>Alerts Configured!</strong> You will receive notifications when drift scores exceed {alertConfig.threshold}
                                        on a {alertConfig.frequency} basis.
                                    </Typography>
                                </Alert>
                            )}
                        </CardContent>
                    </Card>
                </Grid>

                {/* Alert Status & History */}
                <Grid item xs={12} md={6}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                Alert Status
                            </Typography>

                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                                <NotificationsActive sx={{ mr: 1, color: alertConfig.enabled ? '#28a745' : '#6c757d' }} />
                                <Typography variant="body2">
                                    Alerts are currently <strong>{alertConfig.enabled ? 'ENABLED' : 'DISABLED'}</strong>
                                </Typography>
                            </Box>

                            <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                                Current Alert Triggers:
                            </Typography>

                            {mockDriftResults.feature_analysis
                                .filter(f => f.drift_score > (alertConfig.threshold || 2.0))
                                .map((feature) => (
                                    <Alert key={feature.feature} severity="warning" sx={{ mb: 1 }}>
                                        <Typography variant="body2">
                                            <strong>{feature.feature}</strong>: {feature.drift_score.toFixed(2)}
                                            (threshold: {alertConfig.threshold})
                                        </Typography>
                                    </Alert>
                                ))}

                            {mockDriftResults.feature_analysis.filter(f => f.drift_score > (alertConfig.threshold || 2.0)).length === 0 && (
                                <Alert severity="success">
                                    <Typography variant="body2">
                                        No features currently exceed the alert threshold
                                    </Typography>
                                </Alert>
                            )}
                        </CardContent>
                    </Card>
                </Grid>

                {/* Integration Options */}
                <Grid item xs={12} md={6}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                Integration Options
                            </Typography>

                            <List>
                                <ListItem>
                                    <ListItemIcon>
                                        <Email sx={{ color: '#1f4e79' }} />
                                    </ListItemIcon>
                                    <ListItemText
                                        primary="Email Notifications"
                                        secondary="Send detailed alerts to team members"
                                    />
                                </ListItem>
                                <Divider variant="inset" component="li" />

                                <ListItem>
                                    <ListItemIcon>
                                        <Settings sx={{ color: '#1f4e79' }} />
                                    </ListItemIcon>
                                    <ListItemText
                                        primary="Webhook Integration"
                                        secondary="Connect to Slack, Teams, or custom systems"
                                    />
                                </ListItem>
                                <Divider variant="inset" component="li" />

                                <ListItem>
                                    <ListItemIcon>
                                        <Code sx={{ color: '#1f4e79' }} />
                                    </ListItemIcon>
                                    <ListItemText
                                        primary="API Access"
                                        secondary="Programmatic access to drift analysis results"
                                    />
                                </ListItem>
                                <Divider variant="inset" component="li" />

                                <ListItem>
                                    <ListItemIcon>
                                        <Schedule sx={{ color: '#1f4e79' }} />
                                    </ListItemIcon>
                                    <ListItemText
                                        primary="Scheduled Reports"
                                        secondary="Automated report generation and delivery"
                                    />
                                </ListItem>
                            </List>
                        </CardContent>
                    </Card>
                </Grid>

                {/* API Endpoints for Integration */}
                <Grid item xs={12}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                API Endpoints for Integration
                            </Typography>
                            <Typography variant="body2" color="text.secondary" paragraph>
                                Use these endpoints to integrate drift detection into your existing workflows.
                            </Typography>

                            <TableContainer component={Paper} variant="outlined">
                                <Table size="small">
                                    <TableHead>
                                        <TableRow sx={{ backgroundColor: '#f8f9fa' }}>
                                            <TableCell><strong>Endpoint</strong></TableCell>
                                            <TableCell><strong>Method</strong></TableCell>
                                            <TableCell><strong>Description</strong></TableCell>
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        <TableRow>
                                            <TableCell><code>/api/v1/analysis/{'{id}'}/results</code></TableCell>
                                            <TableCell><Chip label="GET" size="small" color="success" /></TableCell>
                                            <TableCell>Get complete drift analysis results</TableCell>
                                        </TableRow>
                                        <TableRow>
                                            <TableCell><code>/api/v1/analysis/{'{id}'}/dashboard</code></TableCell>
                                            <TableCell><Chip label="GET" size="small" color="success" /></TableCell>
                                            <TableCell>Get dashboard summary data</TableCell>
                                        </TableRow>
                                        <TableRow>
                                            <TableCell><code>/api/v1/export/{'{id}'}/report</code></TableCell>
                                            <TableCell><Chip label="POST" size="small" color="primary" /></TableCell>
                                            <TableCell>Generate and download reports</TableCell>
                                        </TableRow>
                                        <TableRow>
                                            <TableCell><code>/api/v1/alerts/configure</code></TableCell>
                                            <TableCell><Chip label="POST" size="small" color="primary" /></TableCell>
                                            <TableCell>Configure alert settings</TableCell>
                                        </TableRow>
                                    </TableBody>
                                </Table>
                            </TableContainer>
                        </CardContent>
                    </Card>
                </Grid>
            </Grid>
        </Box>
    );
};

export default ExportAlerts;
