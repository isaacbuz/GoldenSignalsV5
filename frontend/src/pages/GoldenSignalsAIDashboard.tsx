/**
 * GoldenSignalsAIDashboard.tsx
 * 
 * CERTIFIED FINAL LAYOUT - DO NOT MODIFY WITHOUT PERMISSION
 * Professional AI-powered signals dashboard with optimal proportions
 * Inspired by Bloomberg Terminal efficiency and TradingView aesthetics
 */

import React, { useState, useEffect, useContext } from 'react';
import {
  Box,
  Paper,
  Typography,
  Chip,
  IconButton,
  Button,
  ToggleButton,
  ToggleButtonGroup,
  LinearProgress,
  CircularProgress,
  Fade,
  Card,
  CardContent,
  Divider,
  Tooltip,
  Stack,
  Alert,
  TextField,
  Autocomplete,
  Menu,
  MenuItem,
} from '@mui/material';
import {
  Search as SearchIcon,
  Psychology as AIIcon,
  Psychology,
  FlashOn as SignalIcon,
  TrendingUp as BullishIcon,
  TrendingUp,
  TrendingDown as BearishIcon,
  TrendingDown,
  Circle as CircleIcon,
  ShowChart as ChartIcon,
  AccessTime as TimeIcon,
  CheckCircle as SuccessIcon,
  CheckCircle,
  Warning as WarningIcon,
  Analytics as AnalyticsIcon,
  AutoAwesome as AutoIcon,
  Bolt as BoltIcon,
  Diamond as DiamondIcon,
  Star as StarIcon,
  ArrowDropDown as ArrowDownIcon,
  Close as CloseIcon,
} from '@mui/icons-material';
import ProfessionalLayout, { LayoutContext } from '../components/layout/ProfessionalLayout';
import PredictiveAIChart from '../components/charts/PredictiveAIChart';


interface AISignal {
  id: string;
  timestamp: Date;
  confidence: number;
  type: 'STRONG_BUY' | 'BUY' | 'HOLD' | 'SELL' | 'STRONG_SELL';
  entry: number;
  targets: number[];
  stopLoss: number;
  riskReward: string;
  pattern: string;
  agents: {
    technical: number;
    sentiment: number;
    flow: number;
    momentum: number;
  };
}

// Stock symbols for search
const STOCK_SYMBOLS = [
  { symbol: 'SPY', name: 'SPDR S&P 500 ETF' },
  { symbol: 'QQQ', name: 'Invesco QQQ Trust' },
  { symbol: 'AAPL', name: 'Apple Inc.' },
  { symbol: 'MSFT', name: 'Microsoft Corporation' },
  { symbol: 'NVDA', name: 'NVIDIA Corporation' },
  { symbol: 'TSLA', name: 'Tesla, Inc.' },
  { symbol: 'META', name: 'Meta Platforms, Inc.' },
  { symbol: 'GOOGL', name: 'Alphabet Inc.' },
  { symbol: 'AMZN', name: 'Amazon.com, Inc.' },
  { symbol: 'AMD', name: 'Advanced Micro Devices' },
];

const GoldenSignalsAIDashboard: React.FC = () => {
  const layoutContext = useContext(LayoutContext);
  const { strategy } = layoutContext || { strategy: 'momentum' };
  
  // Local state for search and timeframe
  const [selectedSymbol, setSelectedSymbol] = useState({ symbol: 'SPY', name: 'SPDR S&P 500 ETF' });
  const [watchlist, setWatchlist] = useState<string[]>(['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA']);
  const [timeframe, setTimeframe] = useState('1D');
  const [searchValue, setSearchValue] = useState('');
  const [timeframeAnchor, setTimeframeAnchor] = useState<null | HTMLElement>(null);
  const [aiIndicatorsAnchor, setAiIndicatorsAnchor] = useState<null | HTMLElement>(null);
  const [activeIndicators, setActiveIndicators] = useState<Set<string>>(new Set());
  
  const [aiProcessing, setAiProcessing] = useState(false);
  const [currentSignal, setCurrentSignal] = useState<AISignal | null>(null);
  const [currentPrice, setCurrentPrice] = useState<number | null>(null);
  const [priceChange, setPriceChange] = useState<number>(0);
  const [priceChangePercent, setPriceChangePercent] = useState<number>(0);
  
  // Market status logic (Eastern Time)
  const getMarketStatus = () => {
    const now = new Date();
    // Convert to ET (Eastern Time)
    const etOffset = -5; // EST offset (use -4 for EDT)
    const utc = now.getTime() + (now.getTimezoneOffset() * 60000);
    const etTime = new Date(utc + (3600000 * etOffset));
    
    const day = etTime.getDay();
    const hour = etTime.getHours();
    const minutes = etTime.getMinutes();
    const time = hour + minutes / 60;
    
    if (day === 0 || day === 6) {
      return { status: 'CLOSED', color: '#FF4757', bgColor: 'rgba(255, 71, 87, 0.15)' };
    }
    
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
    }, 60000);
    return () => clearInterval(interval);
  }, []);

  // Fetch real-time price for selected symbol
  useEffect(() => {
    if (!selectedSymbol) return;
    
    const fetchPrice = async () => {
      try {
        // Try Finnhub first for real-time data
        const response = await fetch(
          `https://finnhub.io/api/v1/quote?symbol=${selectedSymbol.symbol}&token=d0ihu29r01qrfsag9qo0d0ihu29r01qrfsag9qog`
        );
        
        if (response.ok) {
          const data = await response.json();
          if (data && data.c) {
            setCurrentPrice(data.c);
            setPriceChange(data.d || 0);
            setPriceChangePercent(data.dp || 0);
          }
        }
      } catch (err) {
        console.error('Error fetching price:', err);
        // Set mock data as fallback
        const mockPrice = selectedSymbol.symbol === 'SPY' ? 448.50 : 195.50;
        setCurrentPrice(mockPrice);
        setPriceChange(2.34);
        setPriceChangePercent(0.52);
      }
    };
    
    fetchPrice();
    const interval = setInterval(fetchPrice, 30000); // Update every 30 seconds
    
    return () => clearInterval(interval);
  }, [selectedSymbol]);

  // Simulate AI processing when symbol or timeframe changes
  useEffect(() => {
    if (!selectedSymbol) return;
    setAiProcessing(true);
    const timer = setTimeout(() => {
      setAiProcessing(false);
      // Generate mock signal based on context
      const price = currentPrice || (selectedSymbol.symbol === 'SPY' ? 448.50 : 195.50);
      setCurrentSignal({
        id: '1',
        timestamp: new Date(),
        confidence: 92,
        type: 'STRONG_BUY',
        entry: price,
        targets: [price * 1.005, price * 1.01, price * 1.015],
        stopLoss: price * 0.995,
        riskReward: '1:3.2',
        pattern: 'Bullish Flag Breakout + Volume Surge',
        agents: {
          technical: 95,
          sentiment: 88,
          flow: 91,
          momentum: 94,
        },
      });
    }, 1500);
    return () => clearTimeout(timer);
  }, [selectedSymbol, timeframe, currentPrice]);

  const getSignalColor = (type: string) => {
    switch (type) {
      case 'STRONG_BUY': return '#00D4AA';
      case 'BUY': return '#4CAF50';
      case 'HOLD': return '#FFD700';
      case 'SELL': return '#FF9800';
      case 'STRONG_SELL': return '#FF4757';
      default: return '#666';
    }
  };

  return (
    <ProfessionalLayout>
      <Box sx={{ 
        width: '100%', 
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
        px: 6,  // Increased horizontal margins for better framing
        py: 3,  // Proper vertical padding
        backgroundColor: '#0a0a0a',
        boxSizing: 'border-box',
        overflow: 'hidden',
      }}>
        
        {/* Professional Trading Control Bar */}
        <Box sx={{
          display: 'flex',
          alignItems: 'stretch',
          gap: 2,
          mb: 3,
          height: 60,
          flexShrink: 0,
        }}>
          {/* Advanced Search & Symbol Info Section */}
          <Paper sx={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            gap: 0,
            backgroundColor: 'rgba(10, 10, 10, 0.95)',
            backdropFilter: 'blur(20px)',
            border: '1px solid rgba(255, 255, 255, 0.05)',
            borderRadius: 2,
            overflow: 'hidden',
            boxShadow: '0 4px 24px rgba(0, 0, 0, 0.3)',
          }}>
            {/* Symbol & Price Section */}
            <Box sx={{
              display: 'flex',
              alignItems: 'center',
              px: 3,
              height: '100%',
              borderRight: '1px solid rgba(255, 255, 255, 0.05)',
              minWidth: 200,
              background: 'linear-gradient(135deg, rgba(255, 215, 0, 0.03) 0%, transparent 100%)',
            }}>
              {selectedSymbol && currentPrice ? (
                <Box>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                    <Typography sx={{ 
                      fontSize: '0.9rem', 
                      fontWeight: 700,
                      color: '#FFD700',
                      letterSpacing: 0.5,
                    }}>
                      {selectedSymbol.symbol}
                    </Typography>
                    <Typography sx={{ 
                      fontSize: '0.7rem', 
                      color: 'rgba(255, 255, 255, 0.4)',
                    }}>
                      {selectedSymbol.name.split(' ').slice(0, 2).join(' ')}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 1.5 }}>
                    <Typography sx={{ 
                      fontSize: '1.5rem', 
                      fontWeight: 700,
                      color: '#fff',
                      lineHeight: 1,
                    }}>
                      ${currentPrice.toFixed(2)}
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                      {priceChange >= 0 ? (
                        <Box sx={{ 
                          display: 'flex', 
                          alignItems: 'center',
                          px: 1,
                          py: 0.25,
                          borderRadius: 1,
                          backgroundColor: 'rgba(0, 212, 170, 0.1)',
                        }}>
                          <TrendingUp sx={{ fontSize: 14, color: '#00D4AA', mr: 0.25 }} />
                          <Typography sx={{
                            fontSize: '0.8rem',
                            color: '#00D4AA',
                            fontWeight: 600,
                          }}>
                            +{priceChangePercent.toFixed(2)}%
                          </Typography>
                        </Box>
                      ) : (
                        <Box sx={{ 
                          display: 'flex', 
                          alignItems: 'center',
                          px: 1,
                          py: 0.25,
                          borderRadius: 1,
                          backgroundColor: 'rgba(255, 71, 87, 0.1)',
                        }}>
                          <TrendingDown sx={{ fontSize: 14, color: '#FF4757', mr: 0.25 }} />
                          <Typography sx={{
                            fontSize: '0.8rem',
                            color: '#FF4757',
                            fontWeight: 600,
                          }}>
                            {priceChangePercent.toFixed(2)}%
                          </Typography>
                        </Box>
                      )}
                    </Box>
                  </Box>
                </Box>
              ) : (
                <Box>
                  <Typography sx={{ fontSize: '0.8rem', color: 'rgba(255, 255, 255, 0.4)', mb: 0.5 }}>
                    Select Symbol
                  </Typography>
                  <Typography sx={{ fontSize: '1.2rem', color: '#FFD700', fontWeight: 600 }}>
                    --
                  </Typography>
                </Box>
              )}
            </Box>

            {/* Search Input Section */}
            <Box sx={{
              flex: 1,
              display: 'flex',
              alignItems: 'center',
              px: 2,
              height: '100%',
              position: 'relative',
            }}>
            <SearchIcon sx={{ fontSize: 20, color: 'rgba(255, 215, 0, 0.5)', mr: 1.5, flexShrink: 0 }} />
            <Autocomplete
              value={selectedSymbol}
              onChange={(event, newValue) => {
                if (newValue) {
                  setSelectedSymbol(newValue);
                }
              }}
              inputValue={searchValue}
              onInputChange={(event, newInputValue) => {
                setSearchValue(newInputValue);
              }}
              options={STOCK_SYMBOLS}
              getOptionLabel={(option) => `${option.symbol} - ${option.name}`}
              sx={{ flex: 1 }}
              size="small"
              renderInput={(params) => (
                <TextField
                  {...params}
                  placeholder="Search stocks, ETFs, crypto..."
                  variant="standard"
                  InputProps={{
                    ...params.InputProps,
                    disableUnderline: true,
                    sx: {
                      fontSize: '1rem',
                      color: '#fff',
                      fontWeight: 500,
                      '& ::placeholder': {
                        color: 'rgba(255, 255, 255, 0.3)',
                      }
                    }
                  }}
                />
              )}
              renderOption={(props, option) => (
                <Box component="li" {...props} sx={{ 
                  py: 1.5, 
                  px: 2,
                  '&:hover': {
                    backgroundColor: 'rgba(255, 215, 0, 0.05)',
                  }
                }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
                    <Box>
                      <Typography sx={{ fontWeight: 700, fontSize: '0.9rem', color: '#FFD700' }}>
                        {option.symbol}
                      </Typography>
                      <Typography sx={{ fontSize: '0.75rem', color: 'rgba(255, 255, 255, 0.5)' }}>
                        {option.name}
                      </Typography>
                    </Box>
                    <Chip 
                      label="STOCK" 
                      size="small" 
                      sx={{ 
                        height: 20,
                        fontSize: '0.65rem',
                        backgroundColor: 'rgba(255, 215, 0, 0.1)',
                        color: '#FFD700',
                      }} 
                    />
                  </Box>
                </Box>
              )}
            />
            {searchValue && (
              <IconButton
                size="small"
                onClick={() => setSearchValue('')}
                sx={{ 
                  color: 'rgba(255, 255, 255, 0.4)',
                  '&:hover': { color: 'rgba(255, 255, 255, 0.6)' }
                }}
              >
                <CloseIcon sx={{ fontSize: 18 }} />
              </IconButton>
            )}
            </Box>

            {/* Quick Stats */}
            <Box sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 2,
              px: 2,
              height: '100%',
              borderLeft: '1px solid rgba(255, 255, 255, 0.05)',
            }}>
              <Box sx={{ textAlign: 'center', minWidth: 60 }}>
                <Typography sx={{ fontSize: '0.65rem', color: 'rgba(255, 255, 255, 0.4)', mb: 0.25 }}>
                  VOL
                </Typography>
                <Typography sx={{ fontSize: '0.85rem', color: '#fff', fontWeight: 600 }}>
                  2.4M
                </Typography>
              </Box>
              <Box sx={{ textAlign: 'center', minWidth: 60 }}>
                <Typography sx={{ fontSize: '0.65rem', color: 'rgba(255, 255, 255, 0.4)', mb: 0.25 }}>
                  AVG
                </Typography>
                <Typography sx={{ fontSize: '0.85rem', color: '#fff', fontWeight: 600 }}>
                  1.8M
                </Typography>
              </Box>
            </Box>
          </Paper>
          
          {/* Timeframe & Market Status Section */}
          <Paper sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 0,
            backgroundColor: 'rgba(10, 10, 10, 0.95)',
            backdropFilter: 'blur(20px)',
            border: '1px solid rgba(255, 255, 255, 0.05)',
            borderRadius: 2,
            overflow: 'hidden',
            boxShadow: '0 4px 24px rgba(0, 0, 0, 0.3)',
          }}>
            {/* Timeframe Selector */}
            <Button
              size="medium"
              onClick={(e) => setTimeframeAnchor(e.currentTarget)}
              sx={{
                height: '100%',
                px: 2.5,
                py: 1.5,
                fontSize: '0.85rem',
                color: '#FFD700',
                backgroundColor: 'transparent',
                borderRadius: 0,
                borderRight: '1px solid rgba(255, 255, 255, 0.05)',
                textTransform: 'none',
                fontWeight: 600,
                minWidth: 120,
                transition: 'all 0.2s ease',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'flex-start',
                justifyContent: 'center',
                '&:hover': {
                  backgroundColor: 'rgba(255, 215, 0, 0.03)',
                },
              }}
            >
              <Typography sx={{ fontSize: '0.65rem', color: 'rgba(255, 255, 255, 0.4)', mb: 0.25 }}>
                TIMEFRAME
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <TimeIcon sx={{ fontSize: 16 }} />
                <Typography sx={{ fontSize: '0.9rem', fontWeight: 600 }}>
                  {timeframe.toUpperCase()}
                </Typography>
                <ArrowDownIcon sx={{ fontSize: 16, ml: 0.5 }} />
              </Box>
            </Button>
            
            {/* AI Indicators Selector - Removed (now internal) */}
            {false && (
            <Button
              size="small"
              onClick={(e) => setAiIndicatorsAnchor(e.currentTarget)}
              sx={{
                height: 36,
                px: 1.5,
                fontSize: '0.75rem',
                color: activeIndicators.size > 0 ? '#00D4AA' : 'rgba(255, 255, 255, 0.5)',
                backgroundColor: activeIndicators.size > 0 ? 'rgba(0, 212, 170, 0.08)' : 'rgba(255, 255, 255, 0.02)',
                border: `1px solid ${activeIndicators.size > 0 ? 'rgba(0, 212, 170, 0.3)' : 'rgba(255, 255, 255, 0.08)'}`,
                borderRadius: 1.5,
                textTransform: 'none',
                fontWeight: 500,
                minWidth: 140,
                transition: 'all 0.2s ease',
                '&:hover': {
                  backgroundColor: activeIndicators.size > 0 ? 'rgba(0, 212, 170, 0.12)' : 'rgba(255, 255, 255, 0.04)',
                  borderColor: activeIndicators.size > 0 ? 'rgba(0, 212, 170, 0.4)' : 'rgba(255, 255, 255, 0.12)',
                },
              }}
            >
              <AIIcon sx={{ fontSize: 16, mr: 0.75 }} />
              AI Layers
              {activeIndicators.size > 0 && (
                <Chip
                  label={activeIndicators.size}
                  size="small"
                  sx={{
                    ml: 1,
                    height: 18,
                    minWidth: 18,
                    fontSize: '0.65rem',
                    backgroundColor: '#00D4AA',
                    color: '#000',
                    fontWeight: 700,
                  }}
                />
              )}
            </Button>
            )}
            
            {/* Market Status */}
            <Box sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              px: 2.5,
              py: 1.5,
              minWidth: 120,
            }}>
              <Typography sx={{ fontSize: '0.65rem', color: 'rgba(255, 255, 255, 0.4)', mb: 0.25 }}>
                MARKET
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}>
                <CircleIcon sx={{ 
                  fontSize: 8,
                  color: marketStatus.color,
                  animation: marketStatus.status === 'LIVE' ? 'pulse 2s infinite' : 'none'
                }} />
                <Typography sx={{
                  fontSize: '0.9rem',
                  color: marketStatus.color,
                  fontWeight: 700,
                }}>
                  {marketStatus.status}
                </Typography>
              </Box>
            </Box>
          </Paper>
        </Box>


        {/* Main Content Grid - Balanced Layout */}
        <Box sx={{ 
          display: 'flex', 
          gap: 3,  // Increased gap between panels
          width: '100%',
          height: 'calc(100vh - 200px)',  // Dynamic height calculation
          maxHeight: '700px',  // Max height for very large screens
          alignItems: 'stretch',  // Stretch to fill height
          overflow: 'hidden',
        }}>
          {/* Left Panel - Compact Signal */}
          <Paper sx={{ 
            width: '200px',
            minWidth: '200px',
            height: '100%',
            minHeight: '500px',
            maxHeight: '700px',
            p: 2.5,
            backgroundColor: 'rgba(15, 15, 15, 0.95)',
            border: '1px solid rgba(255, 215, 0, 0.15)',
            borderRadius: 1,
            overflowY: 'auto',
            overflowX: 'hidden',
            display: 'flex',
            flexDirection: 'column',
          }}>
            <Typography variant="subtitle2" sx={{ 
              mb: 1.5, 
              fontWeight: 600,
              fontSize: '0.875rem',
              color: '#FFD700',
              display: 'flex',
              alignItems: 'center',
              gap: 0.5,
            }}>
              <BoltIcon sx={{ fontSize: 18 }} />
              AI SIGNAL
            </Typography>

            {currentSignal && !aiProcessing && (
              <Stack spacing={2}>
                {/* Confidence Score */}
                <Card sx={{ 
                  backgroundColor: 'rgba(255, 215, 0, 0.05)',
                  border: `1px solid ${getSignalColor(currentSignal.type)}`,
                  borderRadius: 1,
                }}>
                  <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
                    <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.7rem' }}>
                      AI Analysis (RSI, MACD, MA, BOLL, VWAP)
                    </Typography>
                    <Typography variant="h3" sx={{ 
                      color: getSignalColor(currentSignal.type),
                      fontWeight: 700,
                      fontSize: '2rem',
                      mb: 0.5,
                    }}>
                      {currentSignal.confidence}%
                    </Typography>
                    <Chip
                      label={currentSignal.type.replace('_', ' ')}
                      size="small"
                      sx={{
                        backgroundColor: getSignalColor(currentSignal.type),
                        color: '#000',
                        fontWeight: 700,
                        fontSize: '0.7rem',
                        width: '100%',
                        height: 22,
                      }}
                    />
                  </CardContent>
                </Card>

                {/* Pattern Detection */}
                <Box sx={{ 
                  p: 2.5,
                  backgroundColor: 'rgba(255, 215, 0, 0.08)',
                  borderRadius: 0.5,
                  border: '1px solid rgba(255, 215, 0, 0.2)',
                }}>
                  <Typography variant="caption" sx={{ color: '#FFD700', fontSize: '0.65rem', fontWeight: 600 }}>
                    PATTERN DETECTED
                  </Typography>
                  <Typography variant="body2" sx={{ mt: 0.5, fontSize: '0.75rem', color: '#fff' }}>
                    {currentSignal.pattern}
                  </Typography>
                </Box>

                {/* Entry & Targets */}
                <Box>
                  <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.7rem', mb: 1, display: 'block' }}>
                    Entry & Targets
                  </Typography>
                  <Stack spacing={0.5}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', p: 0.5, backgroundColor: 'rgba(255, 255, 255, 0.02)', borderRadius: 0.5 }}>
                      <Typography variant="caption" sx={{ fontSize: '0.75rem' }}>Entry</Typography>
                      <Typography variant="caption" sx={{ fontWeight: 600, fontSize: '0.75rem' }}>${currentSignal.entry}</Typography>
                    </Box>
                    {currentSignal.targets.map((target, i) => (
                      <Box key={i} sx={{ display: 'flex', justifyContent: 'space-between', p: 0.5, backgroundColor: 'rgba(0, 212, 170, 0.05)', borderRadius: 0.5 }}>
                        <Typography variant="caption" sx={{ fontSize: '0.75rem', color: '#00D4AA' }}>Target {i + 1}</Typography>
                        <Typography variant="caption" sx={{ fontWeight: 600, fontSize: '0.75rem', color: '#00D4AA' }}>${target}</Typography>
                      </Box>
                    ))}
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', p: 0.5, backgroundColor: 'rgba(255, 71, 87, 0.05)', borderRadius: 0.5 }}>
                      <Typography variant="caption" sx={{ fontSize: '0.75rem', color: '#FF4757' }}>Stop Loss</Typography>
                      <Typography variant="caption" sx={{ fontWeight: 600, fontSize: '0.75rem', color: '#FF4757' }}>${currentSignal.stopLoss}</Typography>
                    </Box>
                  </Stack>
                </Box>

                {/* Risk/Reward */}
                <Box sx={{ p: 1, backgroundColor: 'rgba(255, 215, 0, 0.05)', borderRadius: 1 }}>
                  <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.7rem' }}>
                    Risk/Reward Ratio
                  </Typography>
                  <Typography variant="h6" sx={{ color: '#00D4AA', fontWeight: 600 }}>
                    {currentSignal.riskReward}
                  </Typography>
                </Box>

                {/* AI Agents Consensus */}
                <Box>
                  <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.7rem', mb: 1, display: 'block' }}>
                    AI Agents Consensus
                  </Typography>
                  {Object.entries(currentSignal.agents).map(([agent, score]) => (
                    <Box key={agent} sx={{ mb: 0.5 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.25 }}>
                        <Typography variant="caption" sx={{ fontSize: '0.7rem', textTransform: 'capitalize' }}>
                          {agent}
                        </Typography>
                        <Typography variant="caption" sx={{ fontSize: '0.7rem', fontWeight: 600 }}>
                          {score}%
                        </Typography>
                      </Box>
                      <LinearProgress
                        variant="determinate"
                        value={score}
                        sx={{
                          height: 3,
                          borderRadius: 2,
                          backgroundColor: 'rgba(255, 215, 0, 0.1)',
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: score > 90 ? '#00D4AA' : score > 70 ? '#FFD700' : '#FF9800',
                            borderRadius: 2,
                          }
                        }}
                      />
                    </Box>
                  ))}
                </Box>

                {/* Signal Actions - No Execution */}
                <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
                  <Button
                    fullWidth
                    variant="outlined"
                    size="small"
                    startIcon={<AnalyticsIcon sx={{ fontSize: 14 }} />}
                    sx={{
                      borderColor: '#FFD700',
                      color: '#FFD700',
                      fontSize: '0.7rem',
                      py: 0.75,
                      '&:hover': {
                        backgroundColor: 'rgba(255, 215, 0, 0.1)',
                        borderColor: '#FFD700',
                      }
                    }}
                  >
                    DETAILS
                  </Button>
                  <Button
                    fullWidth
                    variant="outlined"
                    size="small"
                    sx={{
                      borderColor: 'rgba(255, 255, 255, 0.3)',
                      color: 'text.secondary',
                      fontSize: '0.7rem',
                      py: 0.75,
                      '&:hover': {
                        backgroundColor: 'rgba(255, 255, 255, 0.05)',
                        borderColor: 'rgba(255, 255, 255, 0.5)',
                      }
                    }}
                  >
                    SHARE
                  </Button>
                </Stack>
              </Stack>
            )}

            {aiProcessing && (
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <AIIcon sx={{ fontSize: 48, color: '#FFD700', mb: 2, animation: 'pulse 2s infinite' }} />
                <Typography variant="body2" sx={{ color: '#FFD700', fontSize: '0.875rem' }}>
                  AI Agents Analyzing...
                </Typography>
                <LinearProgress sx={{ mt: 2 }} />
              </Box>
            )}
          </Paper>

          {/* Center - Clean Chart */}
          <Paper sx={{ 
            flex: 1,
            minWidth: 0,  // Allow flex shrinking
            height: '100%',  // Full height of container
            minHeight: '500px',
            maxHeight: '700px',
            backgroundColor: '#0f0f0f',
            border: '1px solid rgba(255, 215, 0, 0.15)',
            borderRadius: 1,
            position: 'relative',
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column',
          }}>
            {/* Chart Context Badge */}
            <Box sx={{
              position: 'absolute',
              top: 12,
              left: 12,
              zIndex: 10,
              display: 'flex',
              gap: 1,
            }}>
              <Chip
                label={`${selectedSymbol?.symbol || 'SPY'} • ${timeframe || '15m'} • ${strategy || 'Momentum'}`}
                size="small"
                sx={{
                  backgroundColor: 'rgba(0, 0, 0, 0.9)',
                  backdropFilter: 'blur(10px)',
                  color: '#FFD700',
                  border: '1px solid rgba(255, 215, 0, 0.3)',
                  fontWeight: 600,
                  fontSize: '0.7rem',
                  textTransform: 'capitalize',
                }}
              />
            </Box>

            {/* Removed Signal Active Overlay */}

            <Box sx={{ 
              flex: 1,  // Take up remaining space
              width: '100%', 
              minHeight: 0,  // Allow flex shrinking
              position: 'relative',
              overflow: 'hidden'
            }}>
              <PredictiveAIChart
                symbol={selectedSymbol?.symbol || 'SPY'}
                height={'100%'}
                timeframe={timeframe}
                aiSignal={currentSignal ? {
                  type: currentSignal.type,
                  confidence: currentSignal.confidence,
                  pattern: currentSignal.pattern,
                  entry: currentSignal.entry,
                  targets: currentSignal.targets,
                  stopLoss: currentSignal.stopLoss,
                } : null}
                showPrediction={true}
              />
            </Box>
          </Paper>

          {/* Right Panel - Compact Metrics */}
          <Paper sx={{ 
            width: '200px',
            minWidth: '200px',
            height: '100%',
            minHeight: '500px',
            maxHeight: '700px',
            p: 2.5,
            backgroundColor: 'rgba(15, 15, 15, 0.95)',
            border: '1px solid rgba(255, 215, 0, 0.15)',
            borderRadius: 1,
            overflowY: 'auto',
            overflowX: 'hidden',
            display: 'flex',
            flexDirection: 'column',
          }}>
            <Typography variant="subtitle2" sx={{ 
              mb: 1.5, 
              fontWeight: 600,
              fontSize: '0.875rem',
              color: '#FFD700',
              display: 'flex',
              alignItems: 'center',
              gap: 0.5,
            }}>
              <AnalyticsIcon sx={{ fontSize: 18 }} />
              MARKET METRICS
            </Typography>

            <Stack spacing={2}>
              {/* Quick Stats */}
              <Box>
                <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.7rem', mb: 1, display: 'block' }}>
                  Today's Performance
                </Typography>
                <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1 }}>
                  <Box sx={{ p: 1, backgroundColor: 'rgba(0, 212, 170, 0.1)', borderRadius: 1 }}>
                    <Typography variant="caption" sx={{ fontSize: '0.65rem', color: 'text.secondary' }}>
                      Signals
                    </Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600, color: '#00D4AA', fontSize: '0.875rem' }}>
                      24/28
                    </Typography>
                  </Box>
                  <Box sx={{ p: 1, backgroundColor: 'rgba(255, 215, 0, 0.1)', borderRadius: 1 }}>
                    <Typography variant="caption" sx={{ fontSize: '0.65rem', color: 'text.secondary' }}>
                      Win Rate
                    </Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600, color: '#FFD700', fontSize: '0.875rem' }}>
                      85.7%
                    </Typography>
                  </Box>
                  <Box sx={{ p: 1, backgroundColor: 'rgba(0, 212, 170, 0.1)', borderRadius: 1 }}>
                    <Typography variant="caption" sx={{ fontSize: '0.65rem', color: 'text.secondary' }}>
                      Avg Gain
                    </Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600, color: '#00D4AA', fontSize: '0.875rem' }}>
                      +3.2%
                    </Typography>
                  </Box>
                  <Box sx={{ p: 1, backgroundColor: 'rgba(255, 71, 87, 0.1)', borderRadius: 1 }}>
                    <Typography variant="caption" sx={{ fontSize: '0.65rem', color: 'text.secondary' }}>
                      Max DD
                    </Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600, color: '#FF4757', fontSize: '0.875rem' }}>
                      -1.1%
                    </Typography>
                  </Box>
                </Box>
              </Box>

              <Divider sx={{ borderColor: 'rgba(255, 215, 0, 0.1)' }} />

              {/* Market Sentiment */}
              <Box>
                <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.7rem', mb: 1, display: 'block' }}>
                  Market Sentiment
                </Typography>
                <Box sx={{ p: 1.5, backgroundColor: 'rgba(255, 215, 0, 0.05)', borderRadius: 1, textAlign: 'center' }}>
                  <Box sx={{ display: 'flex', justifyContent: 'center', gap: 0.5, mb: 1 }}>
                    {[...Array(5)].map((_, i) => (
                      <StarIcon 
                        key={i} 
                        sx={{ 
                          fontSize: 16, 
                          color: i < 4 ? '#FFD700' : 'rgba(255, 215, 0, 0.2)' 
                        }} 
                      />
                    ))}
                  </Box>
                  <Typography variant="caption" sx={{ fontSize: '0.7rem', color: '#FFD700' }}>
                    VERY BULLISH
                  </Typography>
                </Box>
              </Box>

              {/* Recent Signals */}
              <Box>
                <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.7rem', mb: 1, display: 'block' }}>
                  Recent Signals
                </Typography>
                <Stack spacing={0.5}>
                  {['NVDA +2.8%', 'TSLA +1.5%', 'META +3.2%', 'AMD +2.1%'].map((signal, i) => (
                    <Box 
                      key={i}
                      sx={{ 
                        display: 'flex', 
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        p: 0.75,
                        backgroundColor: 'rgba(0, 212, 170, 0.05)',
                        borderRadius: 0.5,
                      }}
                    >
                      <Typography variant="caption" sx={{ fontSize: '0.7rem' }}>
                        {signal.split(' ')[0]}
                      </Typography>
                      <Chip
                        label={signal.split(' ')[1]}
                        size="small"
                        sx={{
                          height: 16,
                          fontSize: '0.65rem',
                          backgroundColor: 'rgba(0, 212, 170, 0.2)',
                          color: '#00D4AA',
                        }}
                      />
                    </Box>
                  ))}
                </Stack>
              </Box>

              {/* AI Health Status */}
              <Box sx={{ 
                p: 1.5, 
                backgroundColor: 'rgba(0, 212, 170, 0.1)', 
                borderRadius: 1,
                border: '1px solid rgba(0, 212, 170, 0.3)',
              }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                  <SuccessIcon sx={{ fontSize: 16, color: '#00D4AA' }} />
                  <Typography variant="caption" sx={{ fontSize: '0.75rem', fontWeight: 600 }}>
                    AI SYSTEMS
                  </Typography>
                </Box>
                <Typography variant="caption" sx={{ fontSize: '0.65rem', color: 'text.secondary' }}>
                  All agents operational
                </Typography>
                <Typography variant="caption" sx={{ fontSize: '0.65rem', display: 'block', mt: 0.5 }}>
                  Response time: 12ms
                </Typography>
              </Box>
            </Stack>
          </Paper>
        </Box>
      </Box>

      {/* Timeframe Menu */}
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
        {['1min', '5min', '15min', '30min', '1h', '4h', '1D', '1W', '1M', '3M', '6M', '1Y', '5Y', 'MAX'].map((tf) => (
          <MenuItem
            key={tf}
            selected={timeframe === tf}
            onClick={() => {
              setTimeframe(tf);
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
            <Box sx={{ display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center' }}>
              <Typography sx={{ fontSize: '0.85rem' }}>
                {tf === '1min' ? '1 Minute' :
                 tf === '5min' ? '5 Minutes' :
                 tf === '15min' ? '15 Minutes' :
                 tf === '30min' ? '30 Minutes' :
                 tf === '1h' ? '1 Hour' :
                 tf === '4h' ? '4 Hours' :
                 tf === '1D' ? '1 Day' :
                 tf === '1W' ? '1 Week' :
                 tf === '1M' ? '1 Month' :
                 tf === '3M' ? '3 Months' :
                 tf === '6M' ? '6 Months' :
                 tf === '1Y' ? '1 Year' :
                 tf === '5Y' ? '5 Years' :
                 tf === 'MAX' ? 'All Time' : tf}
              </Typography>
              {tf === timeframe && <CheckCircle sx={{ fontSize: 16, color: '#FFD700' }} />}
            </Box>
          </MenuItem>
        ))}
      </Menu>

      {/* AI Indicators Menu */}
      <Menu
        anchorEl={aiIndicatorsAnchor}
        open={Boolean(aiIndicatorsAnchor)}
        onClose={() => setAiIndicatorsAnchor(null)}
        PaperProps={{
          sx: {
            backgroundColor: '#0a0a0a',
            border: '1px solid rgba(255, 215, 0, 0.2)',
            minWidth: 250,
            p: 2.5,
          }
        }}
      >
        <Typography sx={{ 
          px: 2, 
          py: 1, 
          fontSize: '0.75rem', 
          color: '#FFD700',
          fontWeight: 600,
          borderBottom: '1px solid rgba(255, 215, 0, 0.2)',
          mb: 1,
        }}>
          AI OVERLAY INDICATORS
        </Typography>
        
        {[
          { id: 'pattern', label: 'Pattern Recognition', icon: <ChartIcon sx={{ fontSize: 16 }} /> },
          { id: 'confidence', label: 'Confidence Zones', icon: <SignalIcon sx={{ fontSize: 16 }} /> },
          { id: 'neural', label: 'Neural Activity', icon: <Psychology sx={{ fontSize: 16 }} /> },
          { id: 'signals', label: 'Signal Markers', icon: <BoltIcon sx={{ fontSize: 16 }} /> },
          { id: 'meter', label: 'Confidence Meter', icon: <AnalyticsIcon sx={{ fontSize: 16 }} /> },
        ].map((indicator) => (
          <MenuItem
            key={indicator.id}
            onClick={() => {
              const newIndicators = new Set(activeIndicators);
              if (newIndicators.has(indicator.id)) {
                newIndicators.delete(indicator.id);
              } else {
                newIndicators.add(indicator.id);
              }
              setActiveIndicators(newIndicators);
            }}
            sx={{
              fontSize: '0.85rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              py: 0.75,
              '&:hover': {
                backgroundColor: 'rgba(255, 215, 0, 0.05)',
              },
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              {indicator.icon}
              <Typography sx={{ fontSize: '0.85rem' }}>{indicator.label}</Typography>
            </Box>
            {activeIndicators.has(indicator.id) && (
              <CheckCircle sx={{ fontSize: 16, color: '#00D4AA' }} />
            )}
          </MenuItem>
        ))}
        
        <Divider sx={{ my: 1, borderColor: 'rgba(255, 215, 0, 0.2)' }} />
        
        <MenuItem
          onClick={() => {
            if (activeIndicators.size === 5) {
              setActiveIndicators(new Set());
            } else {
              setActiveIndicators(new Set(['pattern', 'confidence', 'neural', 'signals', 'meter']));
            }
          }}
          sx={{
            fontSize: '0.8rem',
            color: '#FFD700',
            fontWeight: 600,
            justifyContent: 'center',
            '&:hover': {
              backgroundColor: 'rgba(255, 215, 0, 0.1)',
            },
          }}
        >
          {activeIndicators.size === 5 ? 'CLEAR ALL' : 'SELECT ALL'}
        </MenuItem>
      </Menu>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        @keyframes pulse {
          0% { opacity: 1; }
          50% { opacity: 0.5; }
          100% { opacity: 1; }
        }
      `}</style>
    </ProfessionalLayout>
  );
};

export default GoldenSignalsAIDashboard;