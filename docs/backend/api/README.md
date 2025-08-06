# API Documentation

GoldenSignalsAI REST API provides comprehensive endpoints for market data, AI agents, and trading signals.

## Base URL

```
Production: https://api.goldensignals.ai/api/v1
Development: http://localhost:8000/api/v1
```

## Authentication

### JWT Bearer Token
All protected endpoints require JWT authentication:

```bash
# Get access token
curl -X POST "/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=password"

# Use token in requests  
curl -X GET "/market/price/AAPL" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### API Response Format

All API responses follow this structure:

```json
{
  "status": "success|error",
  "data": { /* response data */ },
  "message": "Human readable message",
  "timestamp": "2024-01-01T12:00:00Z",
  "request_id": "uuid-here"
}
```

## Rate Limits

| Endpoint Category | Limit | Window |
|------------------|-------|---------|
| Market Data | 100 req/min | Per user |
| Agent Analysis | 50 req/min | Per user |
| Authentication | 10 req/min | Per IP |
| WebSocket | 1000 msgs/min | Per connection |

## Error Handling

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 422 | Validation Error |
| 429 | Rate Limited |
| 500 | Internal Error |

### Error Response Format

```json
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid symbol format",
    "details": {
      "field": "symbol",
      "value": "invalid_symbol"
    }
  },
  "timestamp": "2024-01-01T12:00:00Z",
  "request_id": "uuid-here"
}
```

## Endpoints Overview

### Authentication Endpoints
- `POST /auth/register` - User registration
- `POST /auth/login` - User login
- `POST /auth/refresh` - Refresh token
- `GET /auth/me` - Current user info
- `PUT /auth/change-password` - Change password

### Market Data Endpoints
- `GET /market/price/{symbol}` - Current price
- `GET /market/historical/{symbol}` - Historical data
- `POST /market/quotes` - Multiple quotes
- `GET /market/search` - Symbol search
- `GET /market/sectors` - Sector performance

### AI Agents Endpoints
- `GET /agents` - List available agents
- `GET /agents/{agent_id}/status` - Agent status
- `POST /agents/{agent_id}/analyze` - Request analysis
- `GET /agents/{agent_id}/metrics` - Agent metrics
- `POST /agents/consensus` - Multi-agent consensus

### Trading Signals Endpoints
- `GET /signals` - List user signals
- `POST /signals` - Create signal
- `GET /signals/{signal_id}` - Signal details
- `PUT /signals/{signal_id}` - Update signal
- `DELETE /signals/{signal_id}` - Delete signal

### WebSocket Endpoints
- `WS /ws/market` - Real-time market data
- `WS /ws/signals` - Trading signals stream
- `WS /ws/agents` - Agent updates

## Request/Response Examples

### Get Current Price

**Request:**
```bash
GET /market/price/AAPL
Authorization: Bearer YOUR_TOKEN
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "symbol": "AAPL",
    "price": 150.25,
    "change": 2.50,
    "change_percent": 1.69,
    "volume": 75000000,
    "bid": 150.20,
    "ask": 150.30,
    "day_high": 152.00,
    "day_low": 148.50,
    "timestamp": "2024-01-01T16:00:00Z"
  }
}
```

### Generate AI Signal

**Request:**
```bash
POST /agents/technical_analysis/analyze
Authorization: Bearer YOUR_TOKEN
Content-Type: application/json

{
  "symbol": "AAPL",
  "timeframe": "1d",
  "parameters": {
    "indicators": ["RSI", "MACD", "BB"],
    "lookback_period": 20
  }
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "symbol": "AAPL",
    "signal": "BUY",
    "confidence": 0.85,
    "reasoning": [
      "RSI oversold recovery (30 → 45)",
      "MACD bullish crossover",
      "Price above Bollinger middle band"
    ],
    "indicators": {
      "rsi": 45.2,
      "macd": 0.45,
      "bollinger_position": 0.65
    },
    "price_target": 160.00,
    "stop_loss": 145.00,
    "risk_reward_ratio": 2.1,
    "timestamp": "2024-01-01T16:05:00Z"
  }
}
```

### Multi-Agent Consensus

**Request:**
```bash
POST /agents/consensus
Authorization: Bearer YOUR_TOKEN
Content-Type: application/json

{
  "symbols": ["AAPL", "TSLA"],
  "timeframe": "1d",
  "agents": ["technical_analysis", "sentiment_analysis", "volatility"]
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "AAPL": {
      "consensus_signal": "BUY",
      "consensus_confidence": 0.78,
      "agent_signals": {
        "technical_analysis": {
          "signal": "BUY",
          "confidence": 0.85,
          "weight": 0.4
        },
        "sentiment_analysis": {
          "signal": "BUY",
          "confidence": 0.72,
          "weight": 0.3
        },
        "volatility": {
          "signal": "HOLD",
          "confidence": 0.65,
          "weight": 0.3
        }
      },
      "risk_assessment": "MEDIUM",
      "position_size": 0.15
    }
  }
}
```

## Pagination

For endpoints returning lists, use pagination parameters:

```bash
GET /signals?page=2&limit=50&sort=created_at&order=desc
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "items": [...],
    "pagination": {
      "page": 2,
      "limit": 50,
      "total": 1250,
      "pages": 25,
      "has_next": true,
      "has_prev": true
    }
  }
}
```

## Filtering and Sorting

### Market Data Filtering
```bash
GET /market/historical/AAPL?period=1y&interval=1d&start=2023-01-01&end=2023-12-31
```

### Signal Filtering
```bash
GET /signals?symbol=AAPL&action=BUY&confidence_min=0.7&date_from=2024-01-01
```

## WebSocket API

### Connection
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/market');

// Authentication
ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'YOUR_JWT_TOKEN'
  }));
};
```

### Subscription Management
```javascript
// Subscribe to symbol updates
ws.send(JSON.stringify({
  type: 'subscribe',
  symbol: 'AAPL'
}));

// Unsubscribe
ws.send(JSON.stringify({
  type: 'unsubscribe', 
  symbol: 'AAPL'
}));
```

### Message Types
- `price_update` - Real-time price changes
- `signal_update` - New trading signals
- `agent_update` - Individual agent analysis
- `decision_update` - Final trading decisions
- `alert` - System alerts and notifications

## SDK and Client Libraries

### Python Client
```python
from goldensignals import Client

client = Client(
    base_url="https://api.goldensignals.ai/api/v1",
    api_key="your_api_key"
)

# Get current price
price = await client.market.get_price("AAPL")

# Generate signal
signal = await client.agents.analyze(
    agent="technical_analysis",
    symbol="AAPL",
    timeframe="1d"
)
```

### JavaScript/TypeScript Client
```typescript
import { GoldenSignalsClient } from '@goldensignals/client';

const client = new GoldenSignalsClient({
  baseURL: 'https://api.goldensignals.ai/api/v1',
  apiKey: 'your_api_key'
});

// WebSocket connection
const ws = client.websocket.connect();
ws.subscribe('AAPL', (update) => {
  console.log('Price update:', update);
});
```

## Testing

### Postman Collection
Import our Postman collection for easy API testing:
- [Download Collection](../assets/goldensignals-api.postman_collection.json)

### API Testing Environment
```bash
# Set environment variables
export API_BASE_URL="http://localhost:8000/api/v1"
export API_TOKEN="your_test_token"

# Run API tests
python -m pytest tests/api/ -v
```

## OpenAPI Specification

Full OpenAPI 3.0 specification available at:
- **Interactive Docs**: `/docs` (Swagger UI)
- **ReDoc**: `/redoc` (Alternative documentation)
- **JSON Schema**: `/openapi.json`

## Support

- **GitHub Issues**: [Report bugs](https://github.com/goldensignals/api/issues)
- **Discord**: [Join community](https://discord.gg/goldensignals)
- **Email**: api-support@goldensignals.ai

## Changelog

### v1.0.0 (Current)
- Initial API release
- Market data endpoints
- AI agent integration
- WebSocket support
- Authentication system