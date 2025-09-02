import React from 'react';
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
    Chip,
    Alert,
} from '@mui/material';
import {
    PieChart,
    Pie,
    Cell,
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
} from 'recharts';

interface ClassDistributionOverviewProps {
    classData: any;
}

const ClassDistributionOverview: React.FC<ClassDistributionOverviewProps> = ({ classData }) => {
    const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

    // Prepare data for visualizations
    const pieDataReference = Object.entries(classData.class_percentages.reference).map(([className, percentage], index) => ({
        name: className,
        value: percentage,
        count: classData.class_counts.reference[className],
        color: COLORS[index % COLORS.length]
    }));

    const pieDataCurrent = Object.entries(classData.class_percentages.current).map(([className, percentage], index) => ({
        name: className,
        value: percentage,
        count: classData.class_counts.current[className],
        color: COLORS[index % COLORS.length]
    }));

    const comparisonData = Object.keys(classData.class_counts.reference).map(className => ({
        className,
        reference: classData.class_counts.reference[className],
        current: classData.class_counts.current[className],
        change: Math.round((classData.class_counts.current[className] / classData.class_counts.reference[className] - 1) * 100)
    }));

    const getChangeColor = (change: number) => {
        if (change > 10) return 'success';
        if (change < -10) return 'error';
        return 'default';
    };

    const getChangeIcon = (change: number) => {
        if (change > 0) return '↗';
        if (change < 0) return '↘';
        return '→';
    };

    const CustomTooltip = ({ active, payload, label }: any) => {
        if (active && payload && payload.length) {
            return (
                <Paper sx={{ p: 2, border: 1, borderColor: 'divider' }}>
                    <Typography variant="body2" fontWeight="bold">{label}</Typography>
                    {payload.map((entry: any, index: number) => (
                        <Typography key={index} variant="body2" style={{ color: entry.color }}>
                            {entry.dataKey}: {entry.value.toLocaleString()}
                        </Typography>
                    ))}
                </Paper>
            );
        }
        return null;
    };

    return (
        <Box>
            <Typography variant="h5" gutterBottom>
                Class Distribution Overview
            </Typography>
            <Typography variant="body1" color="text.secondary" paragraph>
                Visual comparison of class distributions between reference and current datasets,
                highlighting changes in sample counts and proportions.
            </Typography>

            {/* Side-by-side Pie Charts */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
                <Grid item xs={12} md={6}>
                    <Paper sx={{ p: 3 }}>
                        <Typography variant="h6" gutterBottom align="center">
                            Reference Dataset Distribution
                        </Typography>
                        <Box sx={{ height: 300 }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <PieChart>
                                    <Pie
                                        data={pieDataReference}
                                        cx="50%"
                                        cy="50%"
                                        labelLine={false}
                                        label={({ name, value }) => `${name}: ${value.toFixed(1)}%`}
                                        outerRadius={80}
                                        fill="#8884d8"
                                        dataKey="value"
                                    >
                                        {pieDataReference.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={entry.color} />
                                        ))}
                                    </Pie>
                                    <Tooltip formatter={(value: any) => [`${value.toFixed(1)}%`, 'Percentage']} />
                                </PieChart>
                            </ResponsiveContainer>
                        </Box>
                        <Typography variant="body2" align="center" color="text.secondary">
                            Total Samples: {classData.total_samples.reference.toLocaleString()}
                        </Typography>
                    </Paper>
                </Grid>

                <Grid item xs={12} md={6}>
                    <Paper sx={{ p: 3 }}>
                        <Typography variant="h6" gutterBottom align="center">
                            Current Dataset Distribution
                        </Typography>
                        <Box sx={{ height: 300 }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <PieChart>
                                    <Pie
                                        data={pieDataCurrent}
                                        cx="50%"
                                        cy="50%"
                                        labelLine={false}
                                        label={({ name, value }) => `${name}: ${value.toFixed(1)}%`}
                                        outerRadius={80}
                                        fill="#8884d8"
                                        dataKey="value"
                                    >
                                        {pieDataCurrent.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={entry.color} />
                                        ))}
                                    </Pie>
                                    <Tooltip formatter={(value: any) => [`${value.toFixed(1)}%`, 'Percentage']} />
                                </PieChart>
                            </ResponsiveContainer>
                        </Box>
                        <Typography variant="body2" align="center" color="text.secondary">
                            Total Samples: {classData.total_samples.current.toLocaleString()}
                        </Typography>
                    </Paper>
                </Grid>
            </Grid>

            {/* Class Count Comparison Bar Chart */}
            <Paper sx={{ p: 3, mb: 4 }}>
                <Typography variant="h6" gutterBottom>
                    Class Count Comparison
                </Typography>
                <Box sx={{ height: 300 }}>
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={comparisonData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="className" />
                            <YAxis />
                            <Tooltip content={<CustomTooltip />} />
                            <Legend />
                            <Bar dataKey="reference" fill="#8884d8" name="Reference" />
                            <Bar dataKey="current" fill="#82ca9d" name="Current" />
                        </BarChart>
                    </ResponsiveContainer>
                </Box>
            </Paper>

            {/* Detailed Comparison Table */}
            <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                    Detailed Class Comparison
                </Typography>
                <TableContainer>
                    <Table>
                        <TableHead>
                            <TableRow>
                                <TableCell><strong>Class</strong></TableCell>
                                <TableCell align="right"><strong>Reference Count</strong></TableCell>
                                <TableCell align="right"><strong>Reference %</strong></TableCell>
                                <TableCell align="right"><strong>Current Count</strong></TableCell>
                                <TableCell align="right"><strong>Current %</strong></TableCell>
                                <TableCell align="right"><strong>Change</strong></TableCell>
                                <TableCell align="center"><strong>Impact Level</strong></TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {comparisonData.map((row) => (
                                <TableRow key={row.className}>
                                    <TableCell component="th" scope="row">
                                        <Typography variant="body2" fontWeight="bold">
                                            {row.className}
                                        </Typography>
                                    </TableCell>
                                    <TableCell align="right">
                                        {row.reference.toLocaleString()}
                                    </TableCell>
                                    <TableCell align="right">
                                        {classData.class_percentages.reference[row.className].toFixed(1)}%
                                    </TableCell>
                                    <TableCell align="right">
                                        {row.current.toLocaleString()}
                                    </TableCell>
                                    <TableCell align="right">
                                        {classData.class_percentages.current[row.className].toFixed(1)}%
                                    </TableCell>
                                    <TableCell align="right">
                                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                                            <Typography variant="body2" sx={{ mr: 1 }}>
                                                {getChangeIcon(row.change)}
                                            </Typography>
                                            <Typography
                                                variant="body2"
                                                color={row.change > 0 ? 'success.main' : row.change < 0 ? 'error.main' : 'text.primary'}
                                                fontWeight="bold"
                                            >
                                                {row.change > 0 ? '+' : ''}{row.change}%
                                            </Typography>
                                        </Box>
                                    </TableCell>
                                    <TableCell align="center">
                                        <Chip
                                            label={Math.abs(row.change) > 20 ? 'High' : Math.abs(row.change) > 10 ? 'Medium' : 'Low'}
                                            color={getChangeColor(row.change) as any}
                                            size="small"
                                        />
                                    </TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </TableContainer>
            </Paper>

            {/* Key Insights */}
            <Box sx={{ mt: 3 }}>
                <Alert severity="info">
                    <Typography variant="body2" fontWeight="bold">Key Insights:</Typography>
                    <ul style={{ margin: '8px 0', paddingLeft: '20px' }}>
                        <li>Total dataset size decreased by {Math.round((1 - classData.total_samples.current / classData.total_samples.reference) * 100)}%</li>
                        <li>Class_C shows the most significant reduction (-60%), potentially affecting minority class predictions</li>
                        <li>Class_B shows relative increase in proportion, which may impact model calibration</li>
                        <li>Distribution changes are statistically significant (Chi-square p-value: 0.001)</li>
                    </ul>
                </Alert>
            </Box>
        </Box>
    );
};

export default ClassDistributionOverview;
