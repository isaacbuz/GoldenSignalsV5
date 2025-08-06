import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface Signal {
  id: string;
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  price: number;
  target_price?: number;
  stop_loss?: number;
  take_profit?: number;
  risk_level: string;
  reasoning: string;
  consensus_strength: number;
  agents_consensus: Record<string, any>;
  status: string;
  created_at: string;
  expires_at: string;
  pnl?: number;
  pnl_percentage?: number;
}

interface SignalsState {
  signals: Signal[];
  analytics: any;
  loading: boolean;
  error: string | null;
  filter: {
    symbol?: string;
    status?: string;
    action?: string;
    minConfidence?: number;
  };
}

const initialState: SignalsState = {
  signals: [],
  analytics: null,
  loading: false,
  error: null,
  filter: {},
};

// Async thunks
export const fetchSignals = createAsyncThunk(
  'signals/fetchSignals',
  async (params: {
    symbol?: string;
    status?: string;
    action?: string;
    min_confidence?: number;
    limit?: number;
  }) => {
    const queryParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        queryParams.append(key, value.toString());
      }
    });
    
    const response = await axios.get(`${API_BASE_URL}/api/v1/signals?${queryParams}`);
    return response.data;
  }
);

export const fetchSignalAnalytics = createAsyncThunk(
  'signals/fetchAnalytics',
  async ({ symbol, days = 30 }: { symbol?: string; days?: number }) => {
    const params = new URLSearchParams();
    if (symbol) params.append('symbol', symbol);
    params.append('days', days.toString());
    
    const response = await axios.get(`${API_BASE_URL}/api/v1/signals/analytics?${params}`);
    return response.data;
  }
);

export const createSignal = createAsyncThunk(
  'signals/createSignal',
  async (signalData: any) => {
    const response = await axios.post(`${API_BASE_URL}/api/v1/signals`, signalData);
    return response.data;
  }
);

const signalsSlice = createSlice({
  name: 'signals',
  initialState,
  reducers: {
    updateFilter: (state, action: PayloadAction<SignalsState['filter']>) => {
      state.filter = { ...state.filter, ...action.payload };
    },
    addSignal: (state, action: PayloadAction<Signal>) => {
      state.signals.unshift(action.payload);
    },
    updateSignal: (state, action: PayloadAction<Signal>) => {
      const index = state.signals.findIndex(s => s.id === action.payload.id);
      if (index !== -1) {
        state.signals[index] = action.payload;
      }
    },
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      // Fetch signals
      .addCase(fetchSignals.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchSignals.fulfilled, (state, action) => {
        state.loading = false;
        state.signals = action.payload.signals;
      })
      .addCase(fetchSignals.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch signals';
      })
      // Fetch analytics
      .addCase(fetchSignalAnalytics.fulfilled, (state, action) => {
        state.analytics = action.payload;
      })
      // Create signal
      .addCase(createSignal.fulfilled, (state, action) => {
        state.signals.unshift(action.payload);
      });
  },
});

export const { updateFilter, addSignal, updateSignal, clearError } = signalsSlice.actions;
export default signalsSlice.reducer;