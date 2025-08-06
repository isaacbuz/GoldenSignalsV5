# WebSocket Orchestration Documentation

## Overview

The GoldenSignalsAI WebSocket Orchestration system provides real-time signal generation with live agent activity streaming. It integrates the multi-agent orchestration system with WebSocket broadcasting to deliver instant updates on:

- Agent processing activities
- Signal generation with consensus details
- Real-time price updates
- Trading decisions
- System alerts

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   WebSocket     │────▶│    WebSocket     │────▶│     Agent       │
│    Clients      │     │   Orchestrator   │     │  Orchestrator   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                           │
                               ▼                           ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │   WebSocket      │     │   AI Agents     │
                        │    Manager        │     │  (FinGPT, etc)  │
                        └──────────────────┘     └─────────────────┘
```

## Features

### 1. Real-Time Agent Activity Streaming
- See each agent's analysis in real-time
- Monitor agent confidence levels
- Track consensus building

### 2. Intelligent Signal Generation
- Multi-agent consensus with Byzantine Fault Tolerance
- RAG-augmented signals for enhanced accuracy
- Confidence scoring and reasoning transparency

### 3. Market Monitoring
- Automatic analysis on significant price changes (1%+)
- Continuous monitoring of key symbols
- Real-time price and volume updates

### 4. Room-Based Subscriptions
- Subscribe to specific symbols
- Efficient message routing
- Scalable architecture

## WebSocket Endpoints

### Main Orchestrated Endpoint
```
ws://localhost:8000/ws/signals
```

Features:
- Multi-symbol subscriptions
- On-demand analysis
- Full agent activity streaming
- Authentication support

### Symbol-Specific Endpoint
```
ws://localhost:8000/ws/symbols/{symbol}
```

Features:
- Auto-subscribe to symbol
- Automatic initial analysis
- Focused updates for single symbol

## Message Types

### Client to Server

#### Subscribe to Symbol
```json
{
  "type": "subscribe",
  "symbol": "AAPL"
}
```

#### Trigger Analysis
```json
{
  "type": "analyze",
  "symbol": "AAPL"
}
```

#### Get Status
```json
{
  "type": "get_status"
}
```

### Server to Client

#### Agent Update
```json
{
  "type": "agent_update",
  "data": {
    "symbol": "AAPL",
    "agent": "TechnicalAnalyst",
    "signal": "BUY",
    "confidence": 0.85,
    "timestamp": "2024-01-01T00:00:00Z"
  }
}
```

#### Signal Update
```json
{
  "type": "signal_update",
  "data": {
    "symbol": "AAPL",
    "signal_id": "550e8400-e29b-41d4-a716-446655440000",
    "action": "BUY",
    "confidence": 0.90,
    "price": 150.00,
    "agents_consensus": {
      "total_agents": 5,
      "completed_agents": 5,
      "consensus_strength": "STRONG",
      "agent_details": {
        "agent_1": {
          "name": "FinGPT",
          "signal": "BUY",
          "confidence": 0.92
        }
      }
    },
    "timestamp": "2024-01-01T00:00:00Z",
    "metadata": {
      "reasoning": ["Strong technical breakout", "Positive sentiment"],
      "target_price": 160.00,
      "stop_loss": 145.00
    }
  }
}
```

## Usage Examples

### JavaScript/Browser
```javascript
// Create WebSocket client
const ws = new WebSocket('ws://localhost:8000/ws/signals');

// Handle connection
ws.onopen = () => {
  console.log('Connected to GoldenSignals');
  
  // Subscribe to AAPL
  ws.send(JSON.stringify({
    type: 'subscribe',
    symbol: 'AAPL'
  }));
  
  // Trigger analysis
  ws.send(JSON.stringify({
    type: 'analyze',
    symbol: 'AAPL'
  }));
};

// Handle messages
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch(data.type) {
    case 'agent_update':
      console.log(`Agent ${data.data.agent}: ${data.data.signal}`);
      break;
      
    case 'signal_update':
      console.log(`Signal: ${data.data.action} with ${data.data.confidence} confidence`);
      break;
  }
};
```

### Python Client
```python
import asyncio
import json
import websockets

async def client():
    uri = "ws://localhost:8000/ws/signals"
    
    async with websockets.connect(uri) as websocket:
        # Subscribe to symbol
        await websocket.send(json.dumps({
            "type": "subscribe",
            "symbol": "NVDA"
        }))
        
        # Listen for updates
        async for message in websocket:
            data = json.loads(message)
            print(f"Received: {data['type']}")
```

## Testing

Run the test script to verify WebSocket functionality:
```bash
cd backend
python test_websocket_orchestration.py
```

This will:
1. Connect to the WebSocket
2. Subscribe to symbols
3. Trigger analysis
4. Display real-time updates
5. Show message statistics

## Configuration

### Environment Variables
```env
# WebSocket settings
WS_HEARTBEAT_INTERVAL=30
WS_MESSAGE_QUEUE_SIZE=1000
WS_MAX_CONNECTIONS=1000

# Agent settings
AGENT_MIN_CONFIDENCE=0.6
AGENT_CONSENSUS_THRESHOLD=0.7
SIGNAL_GENERATION_INTERVAL=30
```

### Agent Configuration
Agents can be configured in the orchestrator:
```python
orchestrator.min_agents_for_signal = 3  # Minimum agents for consensus
orchestrator.consensus_threshold = 0.7  # 70% agreement required
```

## Performance Considerations

### Scalability
- Room-based subscriptions minimize unnecessary broadcasts
- Async processing for all agent activities
- Message queuing for reliable delivery

### Optimization Tips
1. Subscribe only to needed symbols
2. Use symbol-specific endpoint for single symbol monitoring
3. Implement client-side throttling for UI updates
4. Handle reconnection logic gracefully

## Monitoring

Check WebSocket status:
```bash
curl http://localhost:8000/ws/status
```

Response:
```json
{
  "websocket_manager": {
    "total_connections": 10,
    "active_connections": 8,
    "messages_sent": 1523,
    "rooms": {
      "AAPL": 5,
      "NVDA": 3
    }
  },
  "orchestrator": {
    "running": true,
    "active_analyses": 2,
    "active_agents": 5
  }
}
```

## Troubleshooting

### Common Issues

1. **Connection Drops**
   - Implement reconnection logic
   - Check heartbeat configuration
   - Monitor network stability

2. **Missing Updates**
   - Ensure proper subscription
   - Check agent configuration
   - Verify market data sources

3. **High Latency**
   - Monitor agent execution times
   - Check database performance
   - Review network configuration

### Debug Mode
Enable debug logging:
```python
import logging
logging.getLogger("websocket_orchestrator").setLevel(logging.DEBUG)
```

## Security

### Authentication
For authenticated connections:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/signals?token=YOUR_JWT_TOKEN');
```

### Rate Limiting
- Heartbeat required every 60 seconds
- Analysis requests limited per client
- Subscription limits enforced

## Future Enhancements

1. **Binary Protocol Support**
   - MessagePack for efficient data transfer
   - Protobuf for structured messages

2. **Advanced Features**
   - Historical signal replay
   - Custom alert conditions
   - Portfolio-wide monitoring

3. **Performance**
   - Redis pub/sub for horizontal scaling
   - WebSocket compression
   - Edge caching for price data