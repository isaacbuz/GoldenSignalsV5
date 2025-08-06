/**
 * Unified Dashboard Service
 * Connects to the backend unified dashboard API
 */

import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

export interface UnifiedAnalysis {
  symbol: string;
  timestamp: string;
  analyses: {
    market_regime?: any;
    technical_analysis?: any;
    volatility?: any;
    sentiment?: any;
    liquidity?: any;
    risk?: any;
    arbitrage?: any;
    execution_plan?: any;
  };
  unified_signal: {
    action: string;
    signal_value: number;
    confidence: number;
    contributors: Array<{
      agent: string;
      contribution: number;
    }>;
    suggested_quantity?: number;
  };
  market_data: {
    price: number;
    volume: number;
    change_percent: number;
  };
}

export interface ActiveSignal {
  source: string;
  symbol: string;
  signal: string;
  strength: number;
  timestamp: string;
  metadata: any;
}

export interface MarketOverview {
  indices: Record<string, {
    price: number;
    change_percent: number;
  }>;
  market_regime: string;
  overall_sentiment: {
    signal: string;
    fear_greed_index: number;
  };
  volatility_regime: string;
  key_metrics: {
    vix: number;
    dollar_index: number;
    bond_yields: Record<string, number>;
    commodity_prices: Record<string, number>;
  };
  timestamp: string;
}

export interface PortfolioRisk {
  portfolio_risk_score: number;
  portfolio_var_95: number;
  unique_risks: string[];
  recommendations: string[];
  position_count: number;
  total_value: number;
}

class UnifiedDashboardService {
  /**
   * Get comprehensive analysis for a symbol
   */
  async getComprehensiveAnalysis(
    symbol: string,
    includeRealtime: boolean = true,
    includePredictions: boolean = true,
    includeArbitrage: boolean = false
  ): Promise<UnifiedAnalysis> {
    try {
      const response = await axios.post(
        `${API_BASE_URL}/dashboard/analyze/${symbol}`,
        null,
        {
          params: {
            include_realtime: includeRealtime,
            include_predictions: includePredictions,
            include_arbitrage: includeArbitrage,
          },
        }
      );
      return response.data;
    } catch (error) {
      console.error('Error fetching comprehensive analysis:', error);
      throw error;
    }
  }

  /**
   * Get active signals across all agents
   */
  async getActiveSignals(
    minConfidence: number = 0.6,
    signalTypes?: string[]
  ): Promise<{ total_signals: number; signals: ActiveSignal[]; timestamp: string }> {
    try {
      const response = await axios.get(`${API_BASE_URL}/dashboard/signals`, {
        params: {
          min_confidence: minConfidence,
          signal_types: signalTypes,
        },
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching active signals:', error);
      throw error;
    }
  }

  /**
   * Get market overview
   */
  async getMarketOverview(): Promise<MarketOverview> {
    try {
      const response = await axios.get(`${API_BASE_URL}/dashboard/market/overview`);
      return response.data;
    } catch (error) {
      console.error('Error fetching market overview:', error);
      throw error;
    }
  }

  /**
   * Get portfolio risk analysis
   */
  async getPortfolioRisk(positions: Array<{ symbol: string; value: number }>): Promise<PortfolioRisk> {
    try {
      const response = await axios.get(`${API_BASE_URL}/dashboard/portfolio/risk`, {
        params: {
          positions: JSON.stringify(positions),
        },
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching portfolio risk:', error);
      throw error;
    }
  }

  /**
   * Get performance metrics for all agents
   */
  async getPerformanceMetrics(): Promise<Record<string, any>> {
    try {
      const response = await axios.get(`${API_BASE_URL}/dashboard/performance`);
      return response.data;
    } catch (error) {
      console.error('Error fetching performance metrics:', error);
      throw error;
    }
  }

  /**
   * Connect to WebSocket for real-time updates
   */
  connectWebSocket(clientId: string, onMessage: (data: any) => void): WebSocket {
    const wsUrl = API_BASE_URL.replace('http', 'ws').replace('/api/v1', '');
    const ws = new WebSocket(`${wsUrl}/dashboard/ws/${clientId}`);

    ws.onopen = () => {
      console.log('Dashboard WebSocket connected');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('Dashboard WebSocket error:', error);
    };

    ws.onclose = () => {
      console.log('Dashboard WebSocket disconnected');
      // Implement reconnection logic if needed
    };

    return ws;
  }

  /**
   * Subscribe to symbol updates via WebSocket
   */
  subscribeToSymbol(ws: WebSocket, symbols: string[]) {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: 'subscribe',
          symbols: symbols,
        })
      );
    }
  }

  /**
   * Unsubscribe from symbol updates
   */
  unsubscribeFromSymbol(ws: WebSocket, symbols: string[]) {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: 'unsubscribe',
          symbols: symbols,
        })
      );
    }
  }
}

export default new UnifiedDashboardService();