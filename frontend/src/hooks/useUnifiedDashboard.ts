/**
 * useUnifiedDashboard Hook
 * Custom hook for accessing unified dashboard data and WebSocket updates
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import unifiedDashboardService, { 
  UnifiedAnalysis, 
  ActiveSignal, 
  MarketOverview,
  PortfolioRisk 
} from '../services/unifiedDashboardService';

interface UseUnifiedDashboardOptions {
  autoRefresh?: boolean;
  refreshInterval?: number;
  enableWebSocket?: boolean;
}

export const useUnifiedDashboard = (options: UseUnifiedDashboardOptions = {}) => {
  const {
    autoRefresh = true,
    refreshInterval = 30000, // 30 seconds
    enableWebSocket = true,
  } = options;

  // State
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<UnifiedAnalysis | null>(null);
  const [activeSignals, setActiveSignals] = useState<ActiveSignal[]>([]);
  const [marketOverview, setMarketOverview] = useState<MarketOverview | null>(null);
  const [portfolioRisk, setPortfolioRisk] = useState<PortfolioRisk | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  // WebSocket ref
  const wsRef = useRef<WebSocket | null>(null);
  const clientIdRef = useRef<string>(`client-${Date.now()}`);

  // Get comprehensive analysis for a symbol
  const getAnalysis = useCallback(async (symbol: string) => {
    setLoading(true);
    setError(null);
    try {
      const data = await unifiedDashboardService.getComprehensiveAnalysis(symbol);
      setAnalysis(data);
      return data;
    } catch (err: any) {
      setError(err.message || 'Failed to fetch analysis');
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  // Get active signals
  const fetchActiveSignals = useCallback(async (minConfidence: number = 0.6) => {
    try {
      const data = await unifiedDashboardService.getActiveSignals(minConfidence);
      setActiveSignals(data.signals);
      return data.signals;
    } catch (err: any) {
      console.error('Error fetching active signals:', err);
      return [];
    }
  }, []);

  // Get market overview
  const fetchMarketOverview = useCallback(async () => {
    try {
      const data = await unifiedDashboardService.getMarketOverview();
      setMarketOverview(data);
      return data;
    } catch (err: any) {
      console.error('Error fetching market overview:', err);
      return null;
    }
  }, []);

  // Get portfolio risk
  const fetchPortfolioRisk = useCallback(async (positions: Array<{ symbol: string; value: number }>) => {
    try {
      const data = await unifiedDashboardService.getPortfolioRisk(positions);
      setPortfolioRisk(data);
      return data;
    } catch (err: any) {
      console.error('Error fetching portfolio risk:', err);
      return null;
    }
  }, []);

  // Handle WebSocket messages
  const handleWebSocketMessage = useCallback((data: any) => {
    switch (data.type) {
      case 'connection':
        setIsConnected(true);
        console.log('Dashboard connected:', data);
        break;
      
      case 'update':
        // Handle real-time updates
        if (data.data?.signals) {
          setActiveSignals(prevSignals => {
            // Merge new signals with existing ones
            const newSignals = data.data.signals;
            const existingIds = new Set(prevSignals.map(s => `${s.source}-${s.symbol}-${s.timestamp}`));
            const uniqueNewSignals = newSignals.filter((s: ActiveSignal) => 
              !existingIds.has(`${s.source}-${s.symbol}-${s.timestamp}`)
            );
            return [...uniqueNewSignals, ...prevSignals].slice(0, 50); // Keep last 50
          });
        }
        break;
      
      case 'symbol_data':
        // Handle symbol-specific updates
        if (data.symbol && analysis?.symbol === data.symbol) {
          setAnalysis(prev => prev ? {
            ...prev,
            market_data: {
              ...prev.market_data,
              ...data.data,
            }
          } : null);
        }
        break;
      
      case 'pong':
        // Keep-alive response
        break;
      
      default:
        console.log('Unknown WebSocket message type:', data.type);
    }
  }, [analysis]);

  // Subscribe to symbols
  const subscribeToSymbols = useCallback((symbols: string[]) => {
    if (wsRef.current) {
      unifiedDashboardService.subscribeToSymbol(wsRef.current, symbols);
    }
  }, []);

  // Unsubscribe from symbols
  const unsubscribeFromSymbols = useCallback((symbols: string[]) => {
    if (wsRef.current) {
      unifiedDashboardService.unsubscribeFromSymbol(wsRef.current, symbols);
    }
  }, []);

  // Setup WebSocket connection
  useEffect(() => {
    if (!enableWebSocket) return;

    const connectWebSocket = () => {
      try {
        wsRef.current = unifiedDashboardService.connectWebSocket(
          clientIdRef.current,
          handleWebSocketMessage
        );

        // Send periodic ping to keep connection alive
        const pingInterval = setInterval(() => {
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type: 'ping' }));
          }
        }, 30000);

        return () => {
          clearInterval(pingInterval);
        };
      } catch (err) {
        console.error('Failed to connect WebSocket:', err);
      }
    };

    const cleanup = connectWebSocket();

    return () => {
      cleanup?.();
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      setIsConnected(false);
    };
  }, [enableWebSocket, handleWebSocketMessage]);

  // Auto-refresh data
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      fetchActiveSignals();
      fetchMarketOverview();
    }, refreshInterval);

    // Initial fetch
    fetchActiveSignals();
    fetchMarketOverview();

    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval, fetchActiveSignals, fetchMarketOverview]);

  return {
    // State
    loading,
    error,
    analysis,
    activeSignals,
    marketOverview,
    portfolioRisk,
    isConnected,
    
    // Actions
    getAnalysis,
    fetchActiveSignals,
    fetchMarketOverview,
    fetchPortfolioRisk,
    subscribeToSymbols,
    unsubscribeFromSymbols,
  };
};