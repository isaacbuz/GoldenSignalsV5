import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface MarketData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  marketCap: number;
  dayHigh: number;
  dayLow: number;
  open: number;
  previousClose: number;
  timestamp: string;
}

interface MarketStatus {
  is_open: boolean;
  status: string;
  current_time: string;
  market_open: string;
  market_close: string;
  indices?: Record<string, MarketData>;
}

interface MarketDataState {
  quotes: Record<string, MarketData>;
  historicalData: any[];
  loading: boolean;
  error: string | null;
  selectedSymbol: string;
  watchlist: string[];
  marketStatus: MarketStatus | null;
  lastUpdate: string | null;
}

const initialState: MarketDataState = {
  quotes: {},
  historicalData: [],
  loading: false,
  error: null,
  selectedSymbol: 'AAPL',
  watchlist: ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'SPY', 'QQQ'],
  marketStatus: null,
  lastUpdate: null,
};

// Async thunks
export const fetchQuote = createAsyncThunk(
  'marketData/fetchQuote',
  async (symbol: string) => {
    const response = await axios.get(`${API_BASE_URL}/api/v1/market-data/quote/${symbol}`);
    return response.data;
  }
);

export const fetchHistoricalData = createAsyncThunk(
  'marketData/fetchHistoricalData',
  async ({ symbol, period = '1d', interval = '5m' }: { symbol: string; period?: string; interval?: string }) => {
    const response = await axios.get(
      `${API_BASE_URL}/api/v1/market-data/historical/${symbol}?period=${period}&interval=${interval}`
    );
    return response.data;
  }
);

export const fetchMultipleQuotes = createAsyncThunk(
  'marketData/fetchMultipleQuotes',
  async (symbols: string[]) => {
    const response = await axios.post(`${API_BASE_URL}/api/v1/market-data/quotes`, { symbols });
    return response.data;
  }
);

export const fetchMarketStatus = createAsyncThunk(
  'marketData/fetchMarketStatus',
  async () => {
    const response = await axios.get(`${API_BASE_URL}/api/v1/market-data/status`);
    return response.data;
  }
);

export const searchSymbols = createAsyncThunk(
  'marketData/searchSymbols',
  async (query: string) => {
    const response = await axios.get(`${API_BASE_URL}/api/v1/market-data/search?query=${query}`);
    return response.data;
  }
);

const marketDataSlice = createSlice({
  name: 'marketData',
  initialState,
  reducers: {
    setSelectedSymbol: (state, action: PayloadAction<string>) => {
      state.selectedSymbol = action.payload;
    },
    updateQuote: (state, action: PayloadAction<MarketData>) => {
      state.quotes[action.payload.symbol] = action.payload;
      state.lastUpdate = new Date().toISOString();
    },
    addToWatchlist: (state, action: PayloadAction<string>) => {
      if (!state.watchlist.includes(action.payload)) {
        state.watchlist.push(action.payload);
      }
    },
    removeFromWatchlist: (state, action: PayloadAction<string>) => {
      state.watchlist = state.watchlist.filter(symbol => symbol !== action.payload);
    },
    clearError: (state) => {
      state.error = null;
    },
    setMarketStatus: (state, action: PayloadAction<MarketStatus>) => {
      state.marketStatus = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      // Fetch quote
      .addCase(fetchQuote.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchQuote.fulfilled, (state, action) => {
        state.loading = false;
        state.quotes[action.payload.symbol] = action.payload;
      })
      .addCase(fetchQuote.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch quote';
      })
      // Fetch historical data
      .addCase(fetchHistoricalData.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchHistoricalData.fulfilled, (state, action) => {
        state.loading = false;
        state.historicalData = action.payload.data;
      })
      .addCase(fetchHistoricalData.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch historical data';
      })
      // Fetch multiple quotes
      .addCase(fetchMultipleQuotes.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchMultipleQuotes.fulfilled, (state, action) => {
        state.loading = false;
        Object.entries(action.payload).forEach(([symbol, quote]) => {
          if (quote) {
            state.quotes[symbol] = quote as MarketData;
          }
        });
        state.lastUpdate = new Date().toISOString();
      })
      .addCase(fetchMultipleQuotes.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch quotes';
      })
      // Fetch market status
      .addCase(fetchMarketStatus.fulfilled, (state, action) => {
        state.marketStatus = action.payload;
      });
  },
});

export const { 
  setSelectedSymbol, 
  updateQuote, 
  addToWatchlist, 
  removeFromWatchlist, 
  clearError,
  setMarketStatus 
} = marketDataSlice.actions;
export default marketDataSlice.reducer;