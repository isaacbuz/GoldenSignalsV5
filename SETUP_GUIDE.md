# GoldenSignalsAI Setup Guide

## ✅ All Issues Fixed!

This project has been fully updated with all the critical issues resolved:

### 🎯 What's Been Fixed

1. **✅ Database Layer**
   - PostgreSQL connection configured
   - SQLAlchemy models implemented (User, MarketData, Trading, Portfolio)
   - Alembic migrations set up
   - Database initialization script ready

2. **✅ Authentication System**
   - JWT authentication implemented
   - User registration/login endpoints
   - Protected routes with security middleware
   - API key generation for programmatic access

3. **✅ Market Data Integration**
   - YFinance service fully integrated
   - Real-time price quotes working
   - Historical data fetching implemented
   - Market status and search functionality

4. **✅ WebSocket Implementation**
   - Full WebSocket manager with rooms/subscriptions
   - Real-time price streaming
   - Signal broadcasting
   - Automatic reconnection handling

5. **✅ Trading Agents**
   - MA Crossover Agent (working)
   - Sentiment Analysis Agent (working)
   - Risk Management Agent (position sizing)
   - All agents generate actionable signals

6. **✅ Frontend Integration**
   - Charts connected to real backend data
   - WebSocket service configured
   - Redux store for state management
   - Real-time updates working

7. **✅ Security Fixes**
   - CORS properly configured (production-ready)
   - Environment variables for secrets
   - Input validation with Pydantic schemas
   - Rate limiting configuration

8. **✅ Testing**
   - Comprehensive test suite added
   - Authentication tests
   - Market data tests
   - Trading agent tests

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- PostgreSQL 14+ (or use Docker)
- Redis (optional, for caching)

### 1. Database Setup

```bash
# Using Docker (recommended)
docker run --name goldensignals-db \
  -e POSTGRES_USER=goldensignals \
  -e POSTGRES_PASSWORD=goldensignals_pass \
  -e POSTGRES_DB=goldensignals \
  -p 5432:5432 \
  -d postgres:15-alpine

# Or install PostgreSQL locally and create database
createdb goldensignals
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env

# Edit .env and add your API keys (optional but recommended):
# - OPENAI_API_KEY or ANTHROPIC_API_KEY for AI features
# - ALPHA_VANTAGE_API_KEY or FINNHUB_API_KEY for additional market data

# Run database migrations
alembic upgrade head

# Start the backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### 4. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/v1/health/detailed

## 📊 Features Now Working

### Market Data
- Real-time stock quotes
- Historical price data with multiple timeframes
- Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- Market status and indices tracking
- Symbol search functionality

### Trading Signals
- Multi-agent consensus system
- Technical analysis (MA crossovers, momentum)
- Sentiment analysis (fear/greed indicators)
- Risk-adjusted position sizing
- Entry/exit price calculations

### Portfolio Management
- Position tracking
- P&L calculations
- Risk metrics (Sharpe ratio, max drawdown)
- Order management system
- Performance analytics

### Real-time Updates
- WebSocket price streaming
- Live signal notifications
- Portfolio updates
- Market alerts

## 🧪 Running Tests

```bash
cd backend

# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_auth.py
```

## 🔒 Security Notes

1. **Change Default Secrets**: The `.env.example` file contains placeholder secrets. Generate your own:
   ```python
   import secrets
   print(secrets.token_urlsafe(32))
   ```

2. **API Keys**: While the system works without external API keys (using yfinance), adding them enables:
   - More reliable data sources
   - Higher rate limits
   - Additional features (AI analysis, premium data)

3. **Production Deployment**: 
   - Use environment-specific `.env` files
   - Enable HTTPS with SSL certificates
   - Configure proper CORS origins
   - Use a production database (not SQLite)
   - Enable rate limiting

## 📁 Project Structure

```
backend/
├── alembic/          # Database migrations
├── api/              # API routes
│   └── routes/       # Organized by feature
├── core/             # Core configuration
├── models/           # SQLAlchemy models
├── schemas/          # Pydantic validation
├── services/         # Business logic
├── tests/            # Test suite
└── app.py           # Main application

frontend/
├── src/
│   ├── components/   # React components
│   ├── services/     # API & WebSocket
│   ├── store/        # Redux state
│   └── pages/        # Page components
└── package.json

agents/
├── technical/        # Technical analysis agents
├── sentiment/        # Market sentiment agents
└── risk/            # Risk management agents
```

## 🎯 What You Can Do Now

1. **View Live Market Data**: Search and track any stock symbol
2. **Generate Trading Signals**: Get AI-powered buy/sell recommendations
3. **Track Portfolio**: Monitor positions and P&L in real-time
4. **Analyze Charts**: Professional TradingView-style charts with indicators
5. **Receive Alerts**: Real-time notifications for price changes and signals
6. **Backtest Strategies**: Test trading strategies on historical data
7. **Manage Risk**: Automatic position sizing and stop-loss calculations

## 🐛 Troubleshooting

### Database Connection Issues
```bash
# Check PostgreSQL is running
pg_isready -h localhost -p 5432

# Reset database if needed
alembic downgrade base
alembic upgrade head
```

### Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000
# Kill it
kill -9 <PID>
```

### WebSocket Not Connecting
- Check backend is running on port 8000
- Verify CORS settings in backend
- Check browser console for errors

### Missing Dependencies
```bash
# Backend
pip install -r requirements.txt --upgrade

# Frontend
rm -rf node_modules package-lock.json
npm install
```

## 📈 Performance Optimizations

The system is now optimized with:
- Database connection pooling
- Redis caching (when available)
- Async/await throughout
- Efficient WebSocket broadcasting
- Pagination on data endpoints
- Rate limiting protection

## 🎉 Ready to Trade!

Your GoldenSignalsAI platform is now fully functional with:
- ✅ Real market data
- ✅ AI-powered signals
- ✅ Professional charts
- ✅ Secure authentication
- ✅ Real-time updates
- ✅ Production-ready code

Start the application and begin trading with confidence!