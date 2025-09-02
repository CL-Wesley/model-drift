import React, { useState } from 'react';
import {
    Box,
    Drawer,
    List,
    ListItem,
    ListItemButton,
    ListItemIcon,
    ListItemText,
    Divider,
    IconButton,
    Typography,
    Collapse,
    Tooltip,
    useTheme
} from '@mui/material';
import {
    ChevronLeft,
    ChevronRight,
    Home,
    CloudUpload,
    Dashboard,
    Analytics,
    Assessment,
    Notifications,
    Settings,
    BarChart,
    ShowChart,
    CompareArrows,
    TrendingDown
} from '@mui/icons-material';

// Define the navigation item structure
export interface NavItem {
    id: string;
    label: string;
    icon: React.ReactNode;
    path?: string;
    children?: NavItem[];
}

// Define the sidebar props
interface SidebarProps {
    open: boolean;
    onToggle: () => void;
    activeItem: string;
    onItemSelect: (id: string) => void;
}

// Define the navigation structure
const navigationItems: NavItem[] = [
    {
        id: 'home',
        label: 'Platform Home',
        icon: <Home />,
        path: '/'
    },
    {
        id: 'data-drift',
        label: 'Data Drift Analysis',
        icon: <BarChart />,
        children: [
            {
                id: 'upload',
                label: 'Upload & Configuration',
                icon: <CloudUpload />,
                path: '/data-drift/upload'
            },
            {
                id: 'dashboard',
                label: 'Drift Analysis Dashboard',
                icon: <Dashboard />,
                path: '/data-drift/dashboard'
            },
            {
                id: 'feature-analysis',
                label: 'Feature Deep Dive',
                icon: <Analytics />,
                path: '/data-drift/feature-analysis'
            },
            {
                id: 'class-imbalance',
                label: 'Class Imbalance Analysis',
                icon: <Assessment />,
                path: '/data-drift/class-imbalance'
            },
            {
                id: 'statistical-reports',
                label: 'Statistical Reports',
                icon: <Assessment />,
                path: '/data-drift/statistical-reports'
            },
            {
                id: 'export-alerts',
                label: 'Export & Alerts',
                icon: <Notifications />,
                path: '/data-drift/export-alerts'
            }
        ]
    },
    {
        id: 'model-drift',
        label: 'Model Drift Analysis',
        icon: <ShowChart />,
        children: [
            {
                id: 'performance-comparison',
                label: 'Performance Comparison',
                icon: <CompareArrows />,
                path: '/model-drift/performance'
            },
            {
                id: 'degradation-metrics',
                label: 'Degradation Metrics',
                icon: <TrendingDown />,
                path: '/model-drift/degradation'
            },
            {
                id: 'statistical-significance',
                label: 'Statistical Significance',
                icon: <Assessment />,
                path: '/model-drift/significance'
            }
        ]
    },
    {
        id: 'settings',
        label: 'Platform Settings',
        icon: <Settings />,
        path: '/settings'
    }
];

const Sidebar: React.FC<SidebarProps> = ({ open, onToggle, activeItem, onItemSelect }) => {
    const theme = useTheme();
    const [expandedItems, setExpandedItems] = useState<string[]>(['data-drift']);

    // Handle expanding/collapsing sections
    const handleExpandClick = (itemId: string) => {
        setExpandedItems(prev => {
            if (prev.includes(itemId)) {
                return prev.filter(id => id !== itemId);
            } else {
                return [...prev, itemId];
            }
        });
    };

    // Check if an item is active (either the item itself or one of its children)
    const isItemActive = (item: NavItem): boolean => {
        if (item.id === activeItem) return true;
        if (item.children) {
            return item.children.some(child => child.id === activeItem);
        }
        return false;
    };

    // Render a navigation item
    const renderNavItem = (item: NavItem, level: number = 0) => {
        const isActive = isItemActive(item);
        const hasChildren = item.children && item.children.length > 0;
        const isExpanded = expandedItems.includes(item.id);

        return (
            <React.Fragment key={item.id}>
                <ListItem
                    disablePadding
                    sx={{
                        display: 'block',
                        pl: level * (open ? 2 : 0)
                    }}
                >
                    <ListItemButton
                        sx={{
                            minHeight: 48,
                            justifyContent: open ? 'initial' : 'center',
                            px: 2.5,
                            bgcolor: isActive ? `${theme.palette.primary.main}15` : 'transparent',
                            '&:hover': {
                                bgcolor: isActive
                                    ? `${theme.palette.primary.main}25`
                                    : `${theme.palette.primary.main}10`,
                            },
                            borderLeft: isActive ? `4px solid ${theme.palette.primary.main}` : '4px solid transparent',
                        }}
                        onClick={() => {
                            if (hasChildren) {
                                handleExpandClick(item.id);
                            } else {
                                onItemSelect(item.id);
                            }
                        }}
                    >
                        <Tooltip title={open ? '' : item.label} placement="right" arrow>
                            <ListItemIcon
                                sx={{
                                    minWidth: 0,
                                    mr: open ? 2 : 'auto',
                                    justifyContent: 'center',
                                    color: isActive ? theme.palette.primary.main : 'inherit'
                                }}
                            >
                                {item.icon}
                            </ListItemIcon>
                        </Tooltip>
                        {open && (
                            <>
                                <ListItemText
                                    primary={item.label}
                                    sx={{
                                        opacity: open ? 1 : 0,
                                        color: isActive ? theme.palette.primary.main : 'inherit',
                                        '& .MuiTypography-root': {
                                            fontWeight: isActive ? 600 : 400
                                        }
                                    }}
                                />
                                {hasChildren && (
                                    isExpanded ? <ChevronLeft /> : <ChevronRight />
                                )}
                            </>
                        )}
                    </ListItemButton>
                </ListItem>
                {hasChildren && open && (
                    <Collapse in={isExpanded} timeout="auto" unmountOnExit>
                        <List component="div" disablePadding>
                            {item.children!.map(child => renderNavItem(child, level + 1))}
                        </List>
                    </Collapse>
                )}
            </React.Fragment>
        );
    };

    return (
        <Drawer
            variant="permanent"
            open={open}
            sx={{
                width: open ? 250 : 60,
                flexShrink: 0,
                whiteSpace: 'nowrap',
                boxSizing: 'border-box',
                transition: theme.transitions.create('width', {
                    easing: theme.transitions.easing.sharp,
                    duration: theme.transitions.duration.enteringScreen,
                }),
                '& .MuiDrawer-paper': {
                    width: open ? 250 : 60,
                    overflowX: 'hidden',
                    transition: theme.transitions.create('width', {
                        easing: theme.transitions.easing.sharp,
                        duration: theme.transitions.duration.enteringScreen,
                    }),
                    boxSizing: 'border-box',
                    borderRight: `1px solid ${theme.palette.divider}`,
                    bgcolor: theme.palette.background.paper,
                },
            }}
        >
            <Box sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: open ? 'space-between' : 'center',
                padding: theme.spacing(2),
                borderBottom: `1px solid ${theme.palette.divider}`,
                minHeight: 64
            }}>
                {open && (
                    <Typography variant="h6" noWrap component="div" sx={{ fontWeight: 600 }}>
                        Drift Detection
                    </Typography>
                )}
                <IconButton onClick={onToggle}>
                    {open ? <ChevronLeft /> : <ChevronRight />}
                </IconButton>
            </Box>
            <List sx={{ pt: 1 }}>
                {navigationItems.map(item => renderNavItem(item))}
            </List>
        </Drawer>
    );
};

export default Sidebar;