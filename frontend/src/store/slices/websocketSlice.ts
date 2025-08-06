import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface WebSocketState {
  connected: boolean;
  connecting: boolean;
  error: string | null;
  subscriptions: string[];
  lastMessage: any;
}

const initialState: WebSocketState = {
  connected: false,
  connecting: false,
  error: null,
  subscriptions: [],
  lastMessage: null,
};

const websocketSlice = createSlice({
  name: 'websocket',
  initialState,
  reducers: {
    connect: (state) => {
      state.connecting = true;
      state.error = null;
    },
    connected: (state) => {
      state.connected = true;
      state.connecting = false;
      state.error = null;
    },
    disconnect: (state) => {
      state.connected = false;
      state.connecting = false;
      state.subscriptions = [];
    },
    connectionError: (state, action: PayloadAction<string>) => {
      state.connected = false;
      state.connecting = false;
      state.error = action.payload;
    },
    subscribe: (state, action: PayloadAction<string>) => {
      if (!state.subscriptions.includes(action.payload)) {
        state.subscriptions.push(action.payload);
      }
    },
    unsubscribe: (state, action: PayloadAction<string>) => {
      state.subscriptions = state.subscriptions.filter(s => s !== action.payload);
    },
    messageReceived: (state, action: PayloadAction<any>) => {
      state.lastMessage = action.payload;
    },
  },
});

export const {
  connect,
  connected,
  disconnect,
  connectionError,
  subscribe,
  unsubscribe,
  messageReceived,
} = websocketSlice.actions;

export default websocketSlice.reducer;