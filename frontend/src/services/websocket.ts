import { store } from '../store/store';
import { 
  connect as wsConnect, 
  connected, 
  disconnect, 
  connectionError, 
  messageReceived 
} from '../store/slices/websocketSlice';
import { updateQuote } from '../store/slices/marketDataSlice';
import { addSignal, updateSignal } from '../store/slices/signalsSlice';

class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectInterval: number = 5000;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private url: string;

  constructor() {
    const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';
    // Generate a unique client ID for this connection
    const clientId = `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    this.url = `${wsUrl}/ws/market/${clientId}`;
  }

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return;
    }

    store.dispatch(wsConnect());

    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        console.log('WebSocket connected');
        store.dispatch(connected());
        
        // Clear any reconnect timer
        if (this.reconnectTimer) {
          clearTimeout(this.reconnectTimer);
          this.reconnectTimer = null;
        }
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          store.dispatch(messageReceived(data));
          
          // Handle different message types
          switch (data.type) {
            case 'price_update':
              store.dispatch(updateQuote(data.data));
              break;
              
            case 'signal_update':
              store.dispatch(addSignal(data.data));
              break;
              
            case 'signal_status':
              store.dispatch(updateSignal(data.data));
              break;
              
            case 'heartbeat':
              // Handle heartbeat
              this.sendMessage({ type: 'heartbeat' });
              break;
              
            default:
              console.log('Unknown message type:', data.type);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        store.dispatch(connectionError('WebSocket error occurred'));
      };

      this.ws.onclose = () => {
        console.log('WebSocket disconnected');
        store.dispatch(disconnect());
        this.scheduleReconnect();
      };
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      store.dispatch(connectionError(error.message));
      this.scheduleReconnect();
    }
  }

  disconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  sendMessage(message: any) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected, message not sent:', message);
    }
  }

  subscribe(symbol: string) {
    this.sendMessage({
      action: 'subscribe',
      symbol
    });
  }

  unsubscribe(symbol: string) {
    this.sendMessage({
      action: 'unsubscribe',
      symbol
    });
  }

  private scheduleReconnect() {
    if (this.reconnectTimer) {
      return;
    }

    console.log(`Scheduling reconnect in ${this.reconnectInterval}ms`);
    
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, this.reconnectInterval);
  }
}

// Create singleton instance
const wsService = new WebSocketService();

export default wsService;