# Quick Start: Minimal Working Implementation

To get a basic working version, here's what needs to be implemented immediately:

## 1. Install Missing Dependencies

```bash
# Backend
cd backend
pip install yfinance pandas numpy websockets python-jose[cryptography] \
    sqlalchemy asyncpg redis chromadb openai python-multipart

# Frontend  
cd ../frontend
npm install @emotion/react @emotion/styled @mui/material \
    lightweight-charts axios socket.io-client @reduxjs/toolkit react-redux
```

## 2. Critical Files to Implement

### Backend Priority Files:

1. **Database Models** (`backend/models/base.py`)
```python
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./goldensignals.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
```

2. **Market Data Service** (Update existing)
```python
import yfinance as yf
import pandas as pd

async def fetch_market_data(symbol: str, period: str = "1d"):
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period)
    return data.to_dict('records')
```

3. **WebSocket Manager** (Update existing)
```python
from fastapi import WebSocket
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)
```

4. **Basic Signal Service** (Update existing)
```python
async def generate_basic_signal(symbol: str, data: pd.DataFrame):
    # Simple moving average crossover
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    latest = data.iloc[-1]
    if latest['SMA_20'] > latest['SMA_50']:
        return {"symbol": symbol, "signal": "BUY", "confidence": 0.7}
    else:
        return {"symbol": symbol, "signal": "SELL", "confidence": 0.7}
```

### Frontend Priority Files:

1. **Update App.tsx**
```tsx
import React, { useEffect, useState } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { Container, Box, Typography } from '@mui/material';
import TradingChart from './components/charts/TradingChart';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
  },
});

function App() {
  const [marketData, setMarketData] = useState(null);
  
  useEffect(() => {
    // Fetch initial data
    fetch('http://localhost:8000/api/v1/market-data/AAPL')
      .then(res => res.json())
      .then(data => setMarketData(data));
      
    // Setup WebSocket
    const ws = new WebSocket('ws://localhost:8000/ws');
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setMarketData(data);
    };
    
    return () => ws.close();
  }, []);

  return (
    <ThemeProvider theme={darkTheme}>
      <Container>
        <Box sx={{ my: 4 }}>
          <Typography variant="h3" component="h1" gutterBottom>
            GoldenSignals AI
          </Typography>
          <TradingChart data={marketData} />
        </Box>
      </Container>
    </ThemeProvider>
  );
}

export default App;
```

2. **Implement TradingChart.tsx**
```tsx
import React, { useEffect, useRef } from 'react';
import { createChart, IChartApi } from 'lightweight-charts';

interface TradingChartProps {
  data: any;
}

const TradingChart: React.FC<TradingChartProps> = ({ data }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (chartContainerRef.current && !chartRef.current) {
      chartRef.current = createChart(chartContainerRef.current, {
        width: chartContainerRef.current.clientWidth,
        height: 400,
        layout: {
          background: { color: '#1e1e1e' },
          textColor: '#d1d4dc',
        },
      });

      const candlestickSeries = chartRef.current.addCandlestickSeries();
      
      // Add sample data
      if (data) {
        candlestickSeries.setData(data);
      }
    }
  }, [data]);

  return <div ref={chartContainerRef} />;
};

export default TradingChart;
```

## 3. Minimal API Routes

Add these to `backend/app.py`:

```python
@app.get("/api/v1/market-data/{symbol}")
async def get_market_data(symbol: str):
    data = await fetch_market_data(symbol)
    return {"symbol": symbol, "data": data}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Send market updates every 5 seconds
            await asyncio.sleep(5)
            data = await fetch_market_data("AAPL")
            await manager.broadcast({"type": "market_update", "data": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

## 4. Run the Application

1. Start Backend:
```bash
cd backend
python app.py
```

2. Start Frontend:
```bash
cd frontend
npm run dev
```

## What This Gives You

- ✅ Basic market data fetching
- ✅ Simple WebSocket for real-time updates
- ✅ Basic chart display
- ✅ Simple signal generation
- ✅ Working RAG/MCP structure (not fully connected yet)

## Next Steps

1. Connect RAG to actual embeddings
2. Implement real MCP tools
3. Add authentication
4. Implement proper agents
5. Add more chart features
6. Create signal dashboard