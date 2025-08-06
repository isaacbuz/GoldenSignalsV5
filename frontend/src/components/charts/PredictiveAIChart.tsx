/**
 * PredictiveAIChart.tsx
 * 
 * AI-powered chart with predictive line visualization
 * Shows historical data and AI predictions with confidence bands
 * Inspired by Palantir's approach to operational intelligence
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  createChart,
  IChartApi,
  ISeriesApi,
  ColorType,
  CrosshairMode,
  LineStyle,
  Time,
  SeriesMarkerPosition,
  SeriesMarkerShape,
} from 'lightweight-charts';
import { Box, Typography, Chip, CircularProgress, Alert, LinearProgress, Tooltip } from '@mui/material';
import { TrendingUp, TrendingDown, AutoAwesome, Psychology, Timeline } from '@mui/icons-material';

interface PredictiveAIChartProps {
  symbol: string;
  height?: number | string;
  timeframe?: string;
  aiSignal?: AISignalData | null;
  showPrediction?: boolean;
}

interface AISignalData {
  type: 'STRONG_BUY' | 'BUY' | 'HOLD' | 'SELL' | 'STRONG_SELL';
  confidence: number;
  pattern?: string;
  patternCandles?: number[];
  entry?: number;
  targets?: number[];
  stopLoss?: number;
}

interface MarketData {
  time: Time;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface PredictionData {
  time: Time;
  value: number;
  upperBound: number;
  lowerBound: number;
  confidence: number;
}

interface LiveQuote {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  timestamp: number;
}

// API Configuration
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const WS_BASE = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';

const PredictiveAIChart: React.FC<PredictiveAIChartProps> = ({
  symbol = 'SPY',
  height = '100%',
  timeframe = '15min',
  aiSignal,
  showPrediction = true,
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const predictionLineRef = useRef<ISeriesApi<'Line'> | null>(null);
  const upperBoundLineRef = useRef<ISeriesApi<'Line'> | null>(null);
  const lowerBoundLineRef = useRef<ISeriesApi<'Line'> | null>(null);
  const confidenceAreaRef = useRef<ISeriesApi<'Area'> | null>(null);

  const [marketData, setMarketData] = useState<MarketData[]>([]);
  const [predictionData, setPredictionData] = useState<PredictionData[]>([]);
  const [liveQuote, setLiveQuote] = useState<LiveQuote | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isPredicting, setIsPredicting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [predictionConfidence, setPredictionConfidence] = useState<number>(0);
  
  // Debug logging
  useEffect(() => {
    console.log('PredictiveAIChart mounted', { symbol, timeframe, height });
  }, [symbol, timeframe, height]);

  // Fetch historical data with extended range
  const fetchHistoricalData = useCallback(async () => {
    console.log('fetchHistoricalData called for', symbol);
    setIsLoading(true);
    setError(null);
    
    try {
      // Always use mock data first to ensure chart renders
      const mockData = generateMockData(timeframe);
      setMarketData(mockData);
      console.log('Mock data set:', mockData.length, 'points');
      
      // Then try to fetch real data in background
      // Determine period and interval based on timeframe
      // Note: intervals must match yfinance format exactly
      let period = '1d';
      let interval = '15m'; // Default
      
      // Map timeframe to appropriate period for realistic data
      switch (timeframe) {
        case '1D':
        case '1d':
          period = '5d';
          interval = '15m';
          break;
        case '1W':
        case '1w':
          period = '1mo';
          interval = '1h';
          break;
        case '1M':
        case '1mo':
          period = '3mo';
          interval = '1d';
          break;
        case '3M':
        case '3mo':
          period = '1y';
          interval = '1d';
          break;
        case '6M':
        case '6mo':
          period = '2y';
          interval = '1d';
          break;
        case '1Y':
        case '1y':
          period = '5y';
          interval = '1wk';
          break;
        case '5Y':
        case '5y':
          period = '10y';
          interval = '1mo';
          break;
        case 'MAX':
        case 'max':
          period = 'max';
          interval = '1mo';
          break;
        // Intraday timeframes
        case '1min':
          period = '1d';
          interval = '1m';
          break;
        case '5min':
          period = '5d';
          interval = '5m';
          break;
        case '15min':
          period = '1mo';
          interval = '15m';
          break;
        case '30min':
          period = '1mo';
          interval = '30m';
          break;
        case '1h':
          period = '3mo';
          interval = '1h';
          break;
        case '4h':
          period = '6mo';
          interval = '1h'; // yfinance doesn't have 4h, use 1h
          break;
      }

      const response = await fetch(
        `${API_BASE}/api/v1/market-data/historical/${symbol}?` +
        `period=${period}&interval=${interval}`
      );

      if (!response.ok) {
        console.warn('Failed to fetch historical data, using mock data');
        // Use mock data as fallback
        const mockData = generateMockData(timeframe);
        setMarketData(mockData);
        return;
      }
      
      const data = await response.json();
      
      // Format data for the chart - handle the nested data structure
      const rawData = data.data || data;
      const formattedData: MarketData[] = rawData.map((item: any, index: number) => ({
        time: item.time ? Math.floor(new Date(item.time).getTime() / 1000) as Time : 
               item.Date ? Math.floor(new Date(item.Date).getTime() / 1000) as Time :
               (Math.floor(Date.now() / 1000) - (rawData.length - index) * getTimeIncrement(timeframe)) as Time,
        open: item.Open || item.open,
        high: item.High || item.high,
        low: item.Low || item.low,
        close: item.Close || item.close,
        volume: item.Volume || item.volume,
      }));
      
      // Sort by time and remove duplicates
      const uniqueData = formattedData
        .sort((a, b) => (a.time as number) - (b.time as number))
        .filter((item, index, array) => {
          if (index === 0) return true;
          return (item.time as number) !== (array[index - 1].time as number);
        });

      // Only update if we got valid data
      if (uniqueData.length > 0) {
        setMarketData(uniqueData);
        console.log('Real data loaded:', uniqueData.length, 'points');
      }
      
    } catch (err) {
      console.error('Error fetching historical data:', err);
      // Keep using mock data on error
    } finally {
      setIsLoading(false);
    }
  }, [symbol, timeframe]);

  // Fetch AI predictions
  const fetchPredictions = useCallback(async () => {
    if (!marketData.length || !showPrediction) return;
    
    setIsPredicting(true);
    
    try {
      // Calculate prediction periods based on timeframe
      const periodsMap: { [key: string]: number } = {
        '1min': 60,    // 1 hour ahead
        '5min': 48,    // 4 hours ahead
        '15min': 96,   // 1 day ahead
        '30min': 96,   // 2 days ahead
        '1h': 168,     // 1 week ahead
        '4h': 42,      // 1 week ahead
        '1d': 7,       // 1 week ahead
        '1w': 4,       // 1 month ahead
      };

      const periods = periodsMap[timeframe] || 24;
      
      const response = await fetch(
        `${API_BASE}/api/v1/ai-analysis/multi-period-prediction/${symbol}?` +
        `timeframe=${timeframe}&periods=${periods}`
      );

      if (!response.ok) throw new Error('Failed to fetch predictions');
      
      const data = await response.json();
      
      // Format prediction data
      const lastCandle = marketData[marketData.length - 1];
      const predictions: PredictionData[] = data.data.predictions.map((pred: any, index: number) => {
        const timeIncrement = getTimeIncrement(timeframe);
        const predTime = (lastCandle.time as number) + (timeIncrement * (index + 1));
        
        return {
          time: predTime as Time,
          value: pred.price,
          upperBound: pred.upper_bound,
          lowerBound: pred.lower_bound,
          confidence: pred.confidence,
        };
      });

      setPredictionData(predictions);
      setPredictionConfidence(data.data.overall_confidence || 0);
      
    } catch (err) {
      console.error('Error fetching predictions:', err);
    } finally {
      setIsPredicting(false);
    }
  }, [marketData, symbol, timeframe, showPrediction]);

  // Generate mock data when API fails
  const generateMockData = (tf: string): MarketData[] => {
    const now = Math.floor(Date.now() / 1000);
    const increment = getTimeIncrement(tf);
    const dataPoints = 100;
    
    const mockData: MarketData[] = [];
    let currentPrice = 474.50; // SPY approximate price
    
    for (let i = dataPoints; i > 0; i--) {
      const time = (now - (i * increment)) as Time;
      const volatility = 0.002; // 0.2% volatility
      
      const open = currentPrice;
      const change = (Math.random() - 0.5) * volatility * currentPrice;
      const close = currentPrice + change;
      const high = Math.max(open, close) * (1 + Math.random() * volatility);
      const low = Math.min(open, close) * (1 - Math.random() * volatility);
      const volume = Math.floor(1000000 + Math.random() * 4000000);
      
      mockData.push({
        time,
        open,
        high,
        low,
        close,
        volume,
      });
      
      currentPrice = close;
    }
    
    return mockData;
  };

  // Helper function to get time increment based on timeframe
  const getTimeIncrement = (tf: string): number => {
    const increments: { [key: string]: number } = {
      '1min': 60,
      '5min': 300,
      '15min': 900,
      '30min': 1800,
      '1h': 3600,
      '4h': 14400,
      '1d': 86400,
      '1w': 604800,
    };
    return increments[tf] || 900; // Default to 15 min
  };

  // Initialize chart with extended features
  useEffect(() => {
    console.log('Chart initialization:', { 
      hasContainer: !!chartContainerRef.current, 
      dataLength: marketData.length,
      containerSize: chartContainerRef.current ? {
        width: chartContainerRef.current.clientWidth,
        height: chartContainerRef.current.clientHeight
      } : null
    });
    
    if (!chartContainerRef.current || !marketData.length) return;

    const containerWidth = chartContainerRef.current.clientWidth || 800;
    const containerHeight = chartContainerRef.current.clientHeight || 400;
    const chartHeight = typeof height === 'string' ? containerHeight : height;
    
    console.log('Creating chart with dimensions:', { containerWidth, chartHeight });
    
    const chart = createChart(chartContainerRef.current, {
      width: containerWidth,
      height: chartHeight,
      layout: {
        background: { type: ColorType.Solid, color: '#0a0a0a' },
        textColor: '#d1d4dc',
        fontSize: 10,
      },
      grid: {
        vertLines: { color: 'rgba(42, 46, 57, 0.1)' },
        horzLines: { color: 'rgba(42, 46, 57, 0.1)' },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          color: 'rgba(255, 215, 0, 0.3)',
          width: 1,
          style: LineStyle.Solid,
        },
        horzLine: {
          color: 'rgba(255, 215, 0, 0.3)',
          width: 1,
          style: LineStyle.Solid,
        },
      },
      rightPriceScale: {
        borderColor: 'rgba(197, 203, 206, 0.1)',
        scaleMargins: { top: 0.1, bottom: 0.2 },
      },
      timeScale: {
        borderColor: 'rgba(197, 203, 206, 0.1)',
        timeVisible: true,
        secondsVisible: false,
      },
    });

    chartRef.current = chart;

    // Add candlestick series
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#00D4AA',
      downColor: '#FF4757',
      borderUpColor: '#00D4AA',
      borderDownColor: '#FF4757',
      wickUpColor: '#00D4AA',
      wickDownColor: '#FF4757',
    });

    candlestickSeries.setData(marketData);
    candlestickSeriesRef.current = candlestickSeries;

    // Add volume series
    const volumeSeries = chart.addHistogramSeries({
      color: 'rgba(255, 215, 0, 0.1)',
      priceFormat: { type: 'volume' },
      priceScaleId: '',
    });

    const volumeData = marketData.map(item => ({
      time: item.time,
      value: item.volume,
      color: item.close >= item.open ? 'rgba(0, 212, 170, 0.1)' : 'rgba(255, 71, 87, 0.1)',
    }));

    volumeSeries.setData(volumeData);
    volumeSeriesRef.current = volumeSeries;

    // Add prediction visualization if available
    if (predictionData.length > 0 && showPrediction) {
      // Main prediction line
      const predictionLine = chart.addLineSeries({
        color: '#FFD700',
        lineWidth: 3,
        lineStyle: LineStyle.Dashed,
        title: 'AI Prediction',
        crosshairMarkerVisible: false,
      });

      const predictionLineData = predictionData.map(p => ({
        time: p.time,
        value: p.value,
      }));

      // Connect to last real price
      const lastCandle = marketData[marketData.length - 1];
      predictionLineData.unshift({
        time: lastCandle.time,
        value: lastCandle.close,
      });

      predictionLine.setData(predictionLineData);
      predictionLineRef.current = predictionLine;

      // Upper confidence bound
      const upperBound = chart.addLineSeries({
        color: 'rgba(255, 215, 0, 0.3)',
        lineWidth: 1,
        lineStyle: LineStyle.Dotted,
        crosshairMarkerVisible: false,
      });

      const upperBoundData = predictionData.map(p => ({
        time: p.time,
        value: p.upperBound,
      }));
      upperBoundData.unshift({
        time: lastCandle.time,
        value: lastCandle.close,
      });

      upperBound.setData(upperBoundData);
      upperBoundLineRef.current = upperBound;

      // Lower confidence bound
      const lowerBound = chart.addLineSeries({
        color: 'rgba(255, 215, 0, 0.3)',
        lineWidth: 1,
        lineStyle: LineStyle.Dotted,
        crosshairMarkerVisible: false,
      });

      const lowerBoundData = predictionData.map(p => ({
        time: p.time,
        value: p.lowerBound,
      }));
      lowerBoundData.unshift({
        time: lastCandle.time,
        value: lastCandle.close,
      });

      lowerBound.setData(lowerBoundData);
      lowerBoundLineRef.current = lowerBound;

      // Add markers for key prediction points
      const markers = predictionData
        .filter((_, index) => index % Math.ceil(predictionData.length / 5) === 0)
        .map(p => ({
          time: p.time,
          position: 'inBar' as SeriesMarkerPosition,
          color: '#FFD700',
          shape: 'circle' as SeriesMarkerShape,
          text: `${(p.confidence * 100).toFixed(0)}%`,
        }));

      predictionLine.setMarkers(markers);
    }

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current) {
        const newHeight = typeof height === 'string' ? chartContainerRef.current.clientHeight : height;
        chart.applyOptions({ 
          width: chartContainerRef.current.clientWidth,
          height: newHeight
        });
      }
    };

    window.addEventListener('resize', handleResize);
    chart.timeScale().fitContent();

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [marketData, predictionData, height, symbol, showPrediction]);

  // WebSocket for real-time updates
  useEffect(() => {
    if (!symbol || !marketData.length || !chartRef.current) return;
    
    // Generate unique client ID
    const clientId = `chart_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const ws = new WebSocket(`${WS_BASE}/ws/market/${clientId}`);
    
    ws.onopen = () => {
      console.log('WebSocket connected for live data');
      // Subscribe to this symbol
      ws.send(JSON.stringify({ action: 'subscribe', symbol }));
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'price_update' && data.symbol === symbol) {
        // Update live price on chart
        const currentTime = Math.floor(Date.now() / 1000) as Time;
        const priceUpdate = {
          time: currentTime,
          value: data.data.price
        };
        
        // Update candlestick series with live price
        if (candlestickSeriesRef.current && marketData.length > 0) {
          const lastCandle = marketData[marketData.length - 1];
          const updatedCandle = {
            ...lastCandle,
            time: currentTime,
            close: data.data.price,
            high: Math.max(lastCandle.high, data.data.price),
            low: Math.min(lastCandle.low, data.data.price)
          };
          
          try {
            candlestickSeriesRef.current.update(updatedCandle);
          } catch (e) {
            console.error('Error updating live price:', e);
          }
        }
        
        // Update live quote display
        setLiveQuote({
          symbol: data.symbol,
          price: data.data.price,
          change: data.data.change,
          changePercent: data.data.changePercent,
          volume: data.data.volume,
          timestamp: Date.now()
        });
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
    };

    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: 'unsubscribe', symbol }));
        ws.close();
      }
    };
  }, [symbol, marketData]);

  // Fetch data on mount and symbol change
  useEffect(() => {
    fetchHistoricalData();
  }, [fetchHistoricalData]);

  // Fetch predictions after historical data is loaded
  useEffect(() => {
    if (marketData.length > 0) {
      fetchPredictions();
    }
  }, [marketData, fetchPredictions]);

  return (
    <Box sx={{ width: '100%', height: '100%', position: 'relative', backgroundColor: '#0a0a0a' }}>
      {/* Live Quote Display */}
      {liveQuote && (
        <Box sx={{
          position: 'absolute',
          top: 10,
          left: 10,
          zIndex: 10,
          backgroundColor: 'rgba(10, 10, 10, 0.9)',
          borderRadius: 1,
          p: 1.5,
          border: '1px solid rgba(255, 215, 0, 0.2)',
        }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
            <Typography sx={{ fontSize: '0.75rem', color: 'rgba(255, 255, 255, 0.6)' }}>
              {symbol}
            </Typography>
            <Chip 
              label="LIVE" 
              size="small" 
              sx={{ 
                height: 16,
                fontSize: '0.6rem',
                backgroundColor: '#00D4AA',
                color: '#000',
                fontWeight: 600
              }} 
            />
          </Box>
          <Typography sx={{ fontSize: '1.5rem', fontWeight: 700, color: '#fff' }}>
            ${liveQuote.price.toFixed(2)}
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            {liveQuote.changePercent >= 0 ? (
              <TrendingUp sx={{ fontSize: 14, color: '#00D4AA' }} />
            ) : (
              <TrendingDown sx={{ fontSize: 14, color: '#FF4757' }} />
            )}
            <Typography sx={{ 
              fontSize: '0.875rem', 
              color: liveQuote.changePercent >= 0 ? '#00D4AA' : '#FF4757',
              fontWeight: 600
            }}>
              {liveQuote.changePercent >= 0 ? '+' : ''}{liveQuote.changePercent.toFixed(2)}%
            </Typography>
          </Box>
        </Box>
      )}

      {/* Prediction Status Indicator */}
      {showPrediction && (
        <Box sx={{
          position: 'absolute',
          top: 10,
          right: 10,
          zIndex: 10,
          display: 'flex',
          alignItems: 'center',
          gap: 1,
          backgroundColor: 'rgba(10, 10, 10, 0.9)',
          borderRadius: 1,
          p: 1,
          border: '1px solid rgba(255, 215, 0, 0.2)',
        }}>
          <Psychology sx={{ fontSize: 16, color: '#FFD700' }} />
          <Box>
            <Typography sx={{ fontSize: '0.65rem', color: 'rgba(255, 255, 255, 0.6)' }}>
              AI PREDICTION
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <Typography sx={{ fontSize: '0.8rem', color: '#FFD700', fontWeight: 600 }}>
                {(predictionConfidence * 100).toFixed(0)}% Confidence
              </Typography>
              {isPredicting && (
                <CircularProgress size={10} sx={{ color: '#FFD700' }} />
              )}
            </Box>
          </Box>
        </Box>
      )}

      {/* Loading State */}
      {isLoading ? (
        <Box sx={{ 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center', 
          height: '100%',
          minHeight: '400px',
          backgroundColor: '#1a1a1a'
        }}>
          <Box sx={{ textAlign: 'center' }}>
            <CircularProgress sx={{ color: '#FFD700', mb: 2 }} size={60} />
            <Typography sx={{ color: '#FFD700', fontSize: '1.2rem' }}>
              Loading {symbol} chart...
            </Typography>
          </Box>
        </Box>
      ) : error ? (
        <Alert severity="error" sx={{ m: 2 }}>{error}</Alert>
      ) : marketData.length === 0 ? (
        <Box sx={{ p: 4, textAlign: 'center' }}>
          <Typography sx={{ color: '#FFD700' }}>No data available</Typography>
        </Box>
      ) : (
        <>
          {/* Chart Container */}
          <Box 
            ref={chartContainerRef} 
            sx={{ 
              width: '100%', 
              height: '100%', 
              minHeight: '400px',
              position: 'relative',
              backgroundColor: '#0a0a0a',
            }} 
          />
          
          {/* Prediction Loading Indicator */}
          {isPredicting && (
            <LinearProgress 
              sx={{ 
                position: 'absolute',
                bottom: 0,
                left: 0,
                right: 0,
                height: 2,
                '& .MuiLinearProgress-bar': {
                  backgroundColor: '#FFD700',
                }
              }} 
            />
          )}
        </>
      )}
    </Box>
  );
};

export default PredictiveAIChart;