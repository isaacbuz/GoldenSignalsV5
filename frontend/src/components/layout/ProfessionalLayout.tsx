/**
 * ProfessionalLayout.tsx
 * 
 * SINGLE SOURCE OF TRUTH FOR APP LAYOUT
 * DO NOT CREATE ALTERNATIVE LAYOUTS
 * ALL PAGES MUST USE THIS LAYOUT
 * 
 * Professional trading platform layout with:
 * - Fixed header with status indicators
 * - Collapsible sidebar navigation
 * - Main content area with consistent spacing
 * - Footer with system information
 */

import React, { useState, useEffect, createContext } from 'react';
import {
  Box,
  AppBar,
  Toolbar,
  Drawer,
  IconButton,
  Typography,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Badge,
  Chip,
  Container,
  useTheme,
  useMediaQuery,
  Avatar,
  Tooltip,
  Paper,
  TextField,
  Autocomplete,
  InputAdornment,
  ToggleButton,
  ToggleButtonGroup,
  Menu,
  MenuItem,
  Button,
  Stack,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  ShowChart as ChartIcon,
  Psychology as AIIcon,
  AccountBalance as PortfolioIcon,
  Settings as SettingsIcon,
  Notifications as NotificationsIcon,
  Circle as CircleIcon,
  TrendingUp as TrendingUpIcon,
  Speed as SpeedIcon,
  Security as SecurityIcon,
  Search as SearchIcon,
  Diamond as DiamondIcon,
  AutoAwesome as AutoIcon,
  QueryStats as AnalysisIcon,
  CandlestickChart as OptionsIcon,
  Insights as InsightsIcon,
  Analytics as AnalyticsIcon,
  FlashOn as SignalsIcon,
  ArrowDropDown as ArrowDownIcon,
  AccessTime as TimeIcon,
  Bolt as BoltIcon,
  Close as CloseIcon,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';

interface ProfessionalLayoutProps {
  children: React.ReactNode;
}

interface SystemStatus {
  backend: 'online' | 'offline' | 'connecting';
  websocket: 'connected' | 'disconnected' | 'reconnecting';
  latency: number;
}

interface LayoutContextType {
  selectedSymbol: any;
  setSelectedSymbol: (symbol: any) => void;
  timeframe: string;
  setTimeframe: (tf: string) => void;
  strategy: string;
  setStrategy: (strategy: string) => void;
}

export const LayoutContext = createContext<LayoutContextType | null>(null);

const DRAWER_WIDTH = 200;
const HEADER_HEIGHT = 48;

/**
 * Main navigation items - focused on signal generation & analysis
 */
const navigationItems = [
  { 
    id: 'signal-generation',
    title: 'Signal Generator', 
    icon: <SignalsIcon />, 
    path: '/signals',
    description: 'AI-powered signals',
    badge: 'LIVE'
  },
  { 
    id: 'options-analysis',
    title: 'Options Analysis', 
    icon: <OptionsIcon />, 
    path: '/options',
    description: 'Greeks & strategies'
  },
  { 
    id: 'market-scanner',
    title: 'Market Scanner', 
    icon: <AnalysisIcon />, 
    path: '/scanner',
    description: 'Find opportunities'
  },
  { 
    id: 'flow-analysis',
    title: 'Flow Analysis', 
    icon: <SpeedIcon />, 
    path: '/flow',
    description: 'Unusual activity'
  },
  { 
    id: 'insights',
    title: 'AI Insights', 
    icon: <InsightsIcon />, 
    path: '/insights',
    description: 'Deep analysis'
  },
  { 
    id: 'backtesting',
    title: 'Backtest Lab', 
    icon: <AnalyticsIcon />, 
    path: '/backtest',
    description: 'Strategy testing'
  },
];

// Stock symbols database for search
const STOCK_SYMBOLS = [
  { symbol: 'AAPL', name: 'Apple Inc.', sector: 'Technology' },
  { symbol: 'MSFT', name: 'Microsoft Corporation', sector: 'Technology' },
  { symbol: 'GOOGL', name: 'Alphabet Inc.', sector: 'Technology' },
  { symbol: 'AMZN', name: 'Amazon.com Inc.', sector: 'Consumer' },
  { symbol: 'NVDA', name: 'NVIDIA Corporation', sector: 'Technology' },
  { symbol: 'META', name: 'Meta Platforms', sector: 'Technology' },
  { symbol: 'TSLA', name: 'Tesla Inc.', sector: 'Automotive' },
  { symbol: 'JPM', name: 'JPMorgan Chase', sector: 'Finance' },
  { symbol: 'SPY', name: 'SPDR S&P 500 ETF', sector: 'ETF' },
  { symbol: 'QQQ', name: 'Invesco QQQ Trust', sector: 'ETF' },
];

// Timeframe options
const TIMEFRAMES = [
  { value: '1m', label: '1M' },
  { value: '5m', label: '5M' },
  { value: '15m', label: '15M' },
  { value: '1h', label: '1H' },
  { value: '4h', label: '4H' },
  { value: '1d', label: '1D' },
];

const ProfessionalLayout: React.FC<ProfessionalLayoutProps> = ({ children }) => {
  const theme = useTheme();
  const navigate = useNavigate();
  const location = useLocation();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState<any>({ symbol: 'SPY', name: 'SPDR S&P 500 ETF' });
  const [searchValue, setSearchValue] = useState('');
  const [searchOpen, setSearchOpen] = useState(false);
  const [timeframe, setTimeframe] = useState('15m');
  const [strategy, setStrategy] = useState('momentum');
  const [signalMode, setSignalMode] = useState('auto');
  
  // Market status logic
  const getMarketStatus = () => {
    const now = new Date();
    const day = now.getDay();
    const hour = now.getHours();
    const minutes = now.getMinutes();
    const time = hour + minutes / 60;
    
    // Weekend check
    if (day === 0 || day === 6) {
      return { status: 'CLOSED', color: '#FF4757', bgColor: 'rgba(255, 71, 87, 0.15)' };
    }
    
    // Weekday market hours (9:30 AM - 4:00 PM EST)
    // Adjust for your timezone as needed
    if (time >= 9.5 && time < 16) {
      return { status: 'LIVE', color: '#00D4AA', bgColor: 'rgba(0, 212, 170, 0.15)' };
    } else if ((time >= 4 && time < 9.5) || (time >= 16 && time < 20)) {
      return { status: 'AFTER HRS', color: '#FFA500', bgColor: 'rgba(255, 165, 0, 0.15)' };
    } else {
      return { status: 'CLOSED', color: '#FF4757', bgColor: 'rgba(255, 71, 87, 0.15)' };
    }
  };
  
  const [marketStatus, setMarketStatus] = useState(getMarketStatus());
  
  // Update market status every minute
  useEffect(() => {
    const interval = setInterval(() => {
      setMarketStatus(getMarketStatus());
    }, 60000); // Update every minute
    
    return () => clearInterval(interval);
  }, []);
  
  // Dropdown anchors
  const [timeframeAnchor, setTimeframeAnchor] = useState<null | HTMLElement>(null);
  const [strategyAnchor, setStrategyAnchor] = useState<null | HTMLElement>(null);
  const [modeAnchor, setModeAnchor] = useState<null | HTMLElement>(null);
  
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    backend: 'connecting',
    websocket: 'disconnected',
    latency: 0,
  });

  // Check system health
  useEffect(() => {
    const checkHealth = async () => {
      const startTime = Date.now();
      try {
        const response = await fetch('http://localhost:8000/api/v1/health');
        const latency = Date.now() - startTime;
        if (response.ok) {
          setSystemStatus(prev => ({
            ...prev,
            backend: 'online',
            latency,
          }));
        }
      } catch {
        setSystemStatus(prev => ({
          ...prev,
          backend: 'offline',
        }));
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, []);

  // WebSocket status monitoring
  useEffect(() => {
    // This would connect to your actual WebSocket service
    const checkWebSocket = () => {
      // Placeholder - integrate with actual WebSocket service
      setSystemStatus(prev => ({
        ...prev,
        websocket: 'connected',
      }));
    };
    checkWebSocket();
  }, []);

  const handleDrawerToggle = () => {
    setDrawerOpen(!drawerOpen);
  };

  // Dropdown Menus
  const timeframeOptions = [
    { value: '1m', label: '1 Min', description: 'Scalping' },
    { value: '5m', label: '5 Min', description: 'Day Trading' },
    { value: '15m', label: '15 Min', description: 'Intraday' },
    { value: '30m', label: '30 Min', description: 'Short Term' },
    { value: '1h', label: '1 Hour', description: 'Swing' },
    { value: '4h', label: '4 Hours', description: 'Position' },
    { value: '1d', label: '1 Day', description: 'Daily' },
    { value: '1w', label: '1 Week', description: 'Weekly' },
  ];

  const strategyOptions = [
    { value: 'momentum', label: 'Momentum', description: 'Follow the trend' },
    { value: 'reversal', label: 'Reversal', description: 'Catch the turn' },
    { value: 'breakout', label: 'Breakout', description: 'New highs/lows' },
    { value: 'volatility', label: 'Volatility', description: 'Range expansion' },
    { value: 'meanReversion', label: 'Mean Reversion', description: 'Back to average' },
    { value: 'arbitrage', label: 'Arbitrage', description: 'Price discrepancy' },
  ];

  const modeOptions = [
    { value: 'auto', label: 'Automatic', description: 'AI generates signals', icon: '🤖' },
    { value: 'manual', label: 'Manual', description: 'You analyze', icon: '👤' },
    { value: 'backtest', label: 'Backtest', description: 'Test strategies', icon: '📊' },
    { value: 'paper', label: 'Paper Trade', description: 'Practice mode', icon: '📝' },
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online':
      case 'connected':
        return '#00D4AA';
      case 'connecting':
      case 'reconnecting':
        return '#FFA500';
      default:
        return '#FF4757';
    }
  };

  return (
    <LayoutContext.Provider value={{ 
      selectedSymbol, 
      setSelectedSymbol, 
      timeframe, 
      setTimeframe,
      strategy,
      setStrategy 
    }}>
      <Box sx={{ 
        display: 'flex', 
        minHeight: '100vh',
        backgroundColor: theme.palette.background.default 
      }}>
      {/* Simplified Header AppBar */}
      <AppBar
        position="fixed"
        sx={{
          zIndex: theme.zIndex.drawer + 1,
          backgroundColor: '#0a0a0a',
          borderBottom: '1px solid rgba(255, 215, 0, 0.1)',
          boxShadow: 'none',
        }}
      >
        <Toolbar sx={{ height: HEADER_HEIGHT, px: 3, justifyContent: 'space-between' }}>
          {/* Left Section - Logo */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <IconButton
              color="inherit"
              edge="start"
              onClick={handleDrawerToggle}
              size="small"
            >
              <MenuIcon sx={{ fontSize: 20 }} />
            </IconButton>
            <DiamondIcon sx={{ color: '#FFD700', fontSize: 20 }} />
            <Typography 
              variant="h6" 
              sx={{ 
                fontWeight: 700,
                fontSize: '0.9rem',
                background: 'linear-gradient(135deg, #FFD700 0%, #FFA000 100%)',
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              GoldenSignals AI
            </Typography>
          </Box>

          {/* Right Section - Notifications and User */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Tooltip title="Notifications">
              <IconButton size="small" color="inherit">
                <Badge badgeContent={3} color="error">
                  <NotificationsIcon sx={{ fontSize: 18 }} />
                </Badge>
              </IconButton>
            </Tooltip>

            <Avatar 
              sx={{ 
                width: 28, 
                height: 28,
                backgroundColor: '#FFD700',
                color: '#000',
                fontWeight: 600,
                fontSize: '0.75rem',
              }}
            >
              U
            </Avatar>
          </Box>
        </Toolbar>
      </AppBar>


      {/* Sidebar Drawer */}
      <Drawer
        variant={isMobile ? 'temporary' : 'persistent'}
        open={drawerOpen}
        onClose={handleDrawerToggle}
        sx={{
          width: drawerOpen ? DRAWER_WIDTH : 0,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: DRAWER_WIDTH,
            boxSizing: 'border-box',
            backgroundColor: 'rgba(15, 15, 15, 0.98)',
            borderRight: '1px solid rgba(255, 215, 0, 0.1)',
            mt: `${HEADER_HEIGHT}px`,
          },
        }}
      >
        <Box sx={{ overflow: 'auto', py: 2 }}>
          {/* Signal Generation Stats */}
          <Box sx={{ px: 2, pb: 2 }}>
            <Paper 
              sx={{ 
                p: 2, 
                backgroundColor: 'rgba(255, 215, 0, 0.05)',
                border: '1px solid rgba(255, 215, 0, 0.1)',
              }}
            >
              <Typography variant="subtitle2" sx={{ color: '#FFD700', mb: 1, fontSize: '0.75rem' }}>
                Today's Signals
              </Typography>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.7rem' }}>
                  Generated:
                </Typography>
                <Typography variant="caption" sx={{ fontWeight: 600, fontSize: '0.7rem' }}>
                  47
                </Typography>
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.7rem' }}>
                  Win Rate:
                </Typography>
                <Typography variant="caption" sx={{ color: '#00D4AA', fontWeight: 600, fontSize: '0.7rem' }}>
                  87.2%
                </Typography>
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.7rem' }}>
                  Avg Confidence:
                </Typography>
                <Typography variant="caption" sx={{ color: '#FFD700', fontWeight: 600, fontSize: '0.7rem' }}>
                  91%
                </Typography>
              </Box>
            </Paper>
          </Box>

          <Divider sx={{ borderColor: 'rgba(255, 215, 0, 0.1)' }} />

          {/* Navigation Items */}
          <List sx={{ px: 1, py: 2 }}>
            {navigationItems.map((item) => {
              const isActive = location.pathname === item.path;
              return (
                <ListItem
                  key={item.id}
                  button
                  onClick={() => navigate(item.path)}
                  sx={{
                    borderRadius: 2,
                    mb: 1,
                    backgroundColor: isActive ? 'rgba(255, 215, 0, 0.1)' : 'transparent',
                    borderLeft: isActive ? '3px solid #FFD700' : '3px solid transparent',
                    '&:hover': {
                      backgroundColor: 'rgba(255, 215, 0, 0.05)',
                    },
                  }}
                >
                  <ListItemIcon sx={{ 
                    color: isActive ? '#FFD700' : '#94A3B8',
                    minWidth: 40,
                  }}>
                    {item.icon}
                  </ListItemIcon>
                  <ListItemText 
                    primary={item.title}
                    secondary={item.description}
                    primaryTypographyProps={{
                      fontWeight: isActive ? 600 : 400,
                      fontSize: '0.875rem',
                      color: isActive ? '#FFD700' : '#E2E8F0',
                    }}
                    secondaryTypographyProps={{
                      fontSize: '0.7rem',
                      color: '#64748B',
                    }}
                  />
                  {item.badge && (
                    <Chip
                      label={item.badge}
                      size="small"
                      sx={{
                        height: 16,
                        fontSize: '0.65rem',
                        backgroundColor: 'rgba(0, 212, 170, 0.2)',
                        color: '#00D4AA',
                        fontWeight: 700,
                      }}
                    />
                  )}
                </ListItem>
              );
            })}
          </List>

          <Divider sx={{ borderColor: 'rgba(255, 215, 0, 0.1)' }} />

          {/* AI Analysis Status */}
          <Box sx={{ px: 2, pt: 2 }}>
            <Box sx={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: 1,
              p: 1.5,
              backgroundColor: 'rgba(255, 215, 0, 0.05)',
              borderRadius: 2,
              border: '1px solid rgba(255, 215, 0, 0.2)',
            }}>
              <AIIcon sx={{ color: '#FFD700', fontSize: 20 }} />
              <Box>
                <Typography variant="caption" sx={{ color: '#FFD700', fontSize: '0.7rem' }}>
                  AI Engine Status
                </Typography>
                <Typography variant="caption" display="block" sx={{ color: '#00D4AA', fontSize: '0.65rem' }}>
                  4 Agents Active
                </Typography>
              </Box>
            </Box>
          </Box>
        </Box>
      </Drawer>

      {/* Main Content Area */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          px: 0, // Remove padding - let dashboard handle it
          py: 0,
          mt: `${HEADER_HEIGHT}px`,
          ml: !isMobile && drawerOpen ? `${DRAWER_WIDTH}px` : 0,
          transition: theme.transitions.create(['margin'], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
          minHeight: `calc(100vh - ${HEADER_HEIGHT}px)`,
          width: !isMobile && drawerOpen ? `calc(100% - ${DRAWER_WIDTH}px)` : '100%',
        }}
      >
        <Box sx={{ height: '100%', width: '100%' }}>
          {children}
        </Box>
      </Box>

      {/* Dropdown Menus */}
      <Menu
        anchorEl={timeframeAnchor}
        open={Boolean(timeframeAnchor)}
        onClose={() => setTimeframeAnchor(null)}
        PaperProps={{
          sx: {
            backgroundColor: '#0a0a0a',
            border: '1px solid rgba(255, 215, 0, 0.2)',
            minWidth: 180,
          }
        }}
      >
        {timeframeOptions.map((option) => (
          <MenuItem
            key={option.value}
            selected={timeframe === option.value}
            onClick={() => {
              setTimeframe(option.value);
              setTimeframeAnchor(null);
            }}
            sx={{
              fontSize: '0.85rem',
              '&.Mui-selected': {
                backgroundColor: 'rgba(255, 215, 0, 0.1)',
                color: '#FFD700',
              },
              '&:hover': {
                backgroundColor: 'rgba(255, 215, 0, 0.05)',
              },
            }}
          >
            <Box sx={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
              <Typography sx={{ fontWeight: timeframe === option.value ? 600 : 400 }}>
                {option.label}
              </Typography>
              <Typography variant="caption" sx={{ color: 'text.secondary', ml: 2 }}>
                {option.description}
              </Typography>
            </Box>
          </MenuItem>
        ))}
      </Menu>

      <Menu
        anchorEl={strategyAnchor}
        open={Boolean(strategyAnchor)}
        onClose={() => setStrategyAnchor(null)}
        PaperProps={{
          sx: {
            backgroundColor: '#0a0a0a',
            border: '1px solid rgba(255, 215, 0, 0.2)',
            minWidth: 200,
          }
        }}
      >
        {strategyOptions.map((option) => (
          <MenuItem
            key={option.value}
            selected={strategy === option.value}
            onClick={() => {
              setStrategy(option.value);
              setStrategyAnchor(null);
            }}
            sx={{
              fontSize: '0.85rem',
              '&.Mui-selected': {
                backgroundColor: 'rgba(255, 215, 0, 0.1)',
                color: '#FFD700',
              },
              '&:hover': {
                backgroundColor: 'rgba(255, 215, 0, 0.05)',
              },
            }}
          >
            <Box>
              <Typography sx={{ fontWeight: strategy === option.value ? 600 : 400 }}>
                {option.label}
              </Typography>
              <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                {option.description}
              </Typography>
            </Box>
          </MenuItem>
        ))}
      </Menu>

      <Menu
        anchorEl={modeAnchor}
        open={Boolean(modeAnchor)}
        onClose={() => setModeAnchor(null)}
        PaperProps={{
          sx: {
            backgroundColor: '#0a0a0a',
            border: '1px solid rgba(255, 215, 0, 0.2)',
            minWidth: 200,
          }
        }}
      >
        {modeOptions.map((option) => (
          <MenuItem
            key={option.value}
            selected={signalMode === option.value}
            onClick={() => {
              setSignalMode(option.value);
              setModeAnchor(null);
            }}
            sx={{
              fontSize: '0.85rem',
              '&.Mui-selected': {
                backgroundColor: 'rgba(0, 212, 170, 0.1)',
                color: '#00D4AA',
              },
              '&:hover': {
                backgroundColor: 'rgba(255, 215, 0, 0.05)',
              },
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Typography sx={{ fontSize: '1rem' }}>{option.icon}</Typography>
              <Box>
                <Typography sx={{ fontWeight: signalMode === option.value ? 600 : 400 }}>
                  {option.label}
                </Typography>
                <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                  {option.description}
                </Typography>
              </Box>
            </Box>
          </MenuItem>
        ))}
      </Menu>

      {/* Global styles for animations */}
      <style>{`
        @keyframes pulse {
          0% {
            opacity: 1;
          }
          50% {
            opacity: 0.5;
          }
          100% {
            opacity: 1;
          }
        }
      `}</style>
      
      {/* Removed footer for maximum chart space */}
      </Box>
    </LayoutContext.Provider>
  );
};

export default ProfessionalLayout;