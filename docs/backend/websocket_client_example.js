/**
 * WebSocket Client Example for GoldenSignalsAI
 * 
 * This example shows how to connect to the orchestrated WebSocket
 * and receive real-time signals with agent activities
 */

class GoldenSignalsWebSocket {
    constructor(url = 'ws://localhost:8000/ws/signals') {
        this.url = url;
        this.ws = null;
        this.subscriptions = new Set();
        this.reconnectInterval = 5000;
        this.shouldReconnect = true;
        this.messageHandlers = new Map();
    }

    // Connect to WebSocket
    connect(token = null) {
        const wsUrl = token ? `${this.url}?token=${token}` : this.url;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('✅ Connected to GoldenSignals WebSocket');
            
            // Re-subscribe to all symbols
            this.subscriptions.forEach(symbol => {
                this.subscribe(symbol);
            });
            
            // Start heartbeat
            this.startHeartbeat();
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };
        
        this.ws.onerror = (error) => {
            console.error('❌ WebSocket error:', error);
        };
        
        this.ws.onclose = () => {
            console.log('🔌 WebSocket disconnected');
            this.stopHeartbeat();
            
            if (this.shouldReconnect) {
                setTimeout(() => this.connect(token), this.reconnectInterval);
            }
        };
    }

    // Handle incoming messages
    handleMessage(data) {
        const { type } = data;
        
        // Call registered handlers
        const handlers = this.messageHandlers.get(type) || [];
        handlers.forEach(handler => handler(data));
        
        // Default handling
        switch (type) {
            case 'connected':
                console.log('Connected:', data.message);
                break;
                
            case 'subscribed':
                console.log(`Subscribed to ${data.symbol}`);
                break;
                
            case 'price_update':
                this.handlePriceUpdate(data.data);
                break;
                
            case 'agent_update':
                this.handleAgentUpdate(data.data);
                break;
                
            case 'signal_update':
                this.handleSignalUpdate(data.data);
                break;
                
            case 'decision_update':
                this.handleDecisionUpdate(data.data);
                break;
                
            case 'alert':
                this.handleAlert(data);
                break;
                
            case 'heartbeat':
                // Heartbeat received
                break;
                
            default:
                console.log('Unknown message type:', type, data);
        }
    }

    // Subscribe to a symbol
    subscribe(symbol) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'subscribe',
                symbol: symbol
            }));
        }
        this.subscriptions.add(symbol);
    }

    // Unsubscribe from a symbol
    unsubscribe(symbol) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'unsubscribe',
                symbol: symbol
            }));
        }
        this.subscriptions.delete(symbol);
    }

    // Trigger analysis for a symbol
    analyzeSymbol(symbol) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'analyze',
                symbol: symbol
            }));
        }
    }

    // Get orchestrator status
    getStatus() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'get_status'
            }));
        }
    }

    // Register message handler
    on(messageType, handler) {
        if (!this.messageHandlers.has(messageType)) {
            this.messageHandlers.set(messageType, []);
        }
        this.messageHandlers.get(messageType).push(handler);
    }

    // Remove message handler
    off(messageType, handler) {
        const handlers = this.messageHandlers.get(messageType) || [];
        const index = handlers.indexOf(handler);
        if (index > -1) {
            handlers.splice(index, 1);
        }
    }

    // Default handlers
    handlePriceUpdate(data) {
        console.log(`💰 ${data.symbol}: $${data.price.toFixed(2)} (Vol: ${data.volume.toLocaleString()})`);
    }

    handleAgentUpdate(data) {
        console.log(`🤖 ${data.agent}: ${data.signal} (${(data.confidence * 100).toFixed(0)}%)`);
    }

    handleSignalUpdate(data) {
        console.log('🎯 SIGNAL GENERATED!');
        console.log(`   Symbol: ${data.symbol}`);
        console.log(`   Action: ${data.action}`);
        console.log(`   Confidence: ${(data.confidence * 100).toFixed(0)}%`);
        console.log(`   Price: $${data.price.toFixed(2)}`);
        
        const consensus = data.agents_consensus;
        console.log(`   Agents: ${consensus.completed_agents}/${consensus.total_agents}`);
        console.log(`   Consensus: ${consensus.consensus_strength}`);
    }

    handleDecisionUpdate(data) {
        console.log(`📋 Decision for ${data.symbol}: ${data.decision.action}`);
        if (data.decision.reason) {
            console.log(`   Reason: ${data.decision.reason}`);
        }
    }

    handleAlert(data) {
        console.log(`ℹ️  ${data.severity.toUpperCase()}: ${data.message}`);
    }

    // Heartbeat management
    startHeartbeat() {
        this.heartbeatInterval = setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'heartbeat' }));
            }
        }, 30000); // 30 seconds
    }

    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
        }
    }

    // Disconnect
    disconnect() {
        this.shouldReconnect = false;
        if (this.ws) {
            this.ws.close();
        }
    }
}

// Example usage:
const client = new GoldenSignalsWebSocket();

// Register custom handlers
client.on('signal_update', (data) => {
    // Update UI with new signal
    updateSignalUI(data.data);
});

client.on('agent_update', (data) => {
    // Show agent activity
    showAgentActivity(data.data);
});

client.on('price_update', (data) => {
    // Update price display
    updatePriceDisplay(data.data);
});

// Connect to WebSocket
client.connect();

// Subscribe to symbols
client.subscribe('AAPL');
client.subscribe('NVDA');
client.subscribe('TSLA');

// Trigger analysis
client.analyzeSymbol('AAPL');

// Example UI update functions (implement these in your app)
function updateSignalUI(signal) {
    // Update your signal display
    console.log('Update signal UI:', signal);
}

function showAgentActivity(activity) {
    // Show agent processing status
    console.log('Agent activity:', activity);
}

function updatePriceDisplay(priceData) {
    // Update price in UI
    console.log('Price update:', priceData);
}

// React Hook Example
/**
 * React hook for GoldenSignals WebSocket
 * 
 * Usage:
 * const { signals, agentActivities, subscribe, analyze } = useGoldenSignalsWS();
 */
function useGoldenSignalsWS() {
    const [connected, setConnected] = React.useState(false);
    const [signals, setSignals] = React.useState([]);
    const [agentActivities, setAgentActivities] = React.useState([]);
    const clientRef = React.useRef(null);
    
    React.useEffect(() => {
        const client = new GoldenSignalsWebSocket();
        clientRef.current = client;
        
        // Register handlers
        client.on('connected', () => setConnected(true));
        client.on('close', () => setConnected(false));
        
        client.on('signal_update', (data) => {
            setSignals(prev => [...prev, data.data]);
        });
        
        client.on('agent_update', (data) => {
            setAgentActivities(prev => [...prev, data.data]);
        });
        
        // Connect
        client.connect();
        
        // Cleanup
        return () => {
            client.disconnect();
        };
    }, []);
    
    const subscribe = (symbol) => {
        if (clientRef.current) {
            clientRef.current.subscribe(symbol);
        }
    };
    
    const analyze = (symbol) => {
        if (clientRef.current) {
            clientRef.current.analyzeSymbol(symbol);
        }
    };
    
    return {
        connected,
        signals,
        agentActivities,
        subscribe,
        analyze
    };
}

// Export for use in other modules
export { GoldenSignalsWebSocket, useGoldenSignalsWS };