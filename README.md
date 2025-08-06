# GoldenSignalsAI V5 - Advanced AI-Powered Trading Intelligence Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Node.js 18+](https://img.shields.io/badge/node.js-18+-green.svg)](https://nodejs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-orange.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18+-blue.svg)](https://reactjs.org/)

A state-of-the-art AI-powered trading platform integrating **30+ external data sources**, **20+ specialized AI agents**, advanced machine learning models, and comprehensive alternative data analytics. Built with clean architecture principles, featuring RAG (Retrieval-Augmented Generation), MCP (Model Context Protocol), and autonomous trading orchestration.

## 🌟 Key Capabilities

### 🤖 AI Agent Ecosystem (20+ Specialized Agents)

#### **Core Trading Agents**
- **Technical Analysis Agent**: Advanced pattern recognition, multi-timeframe analysis
- **Sentiment Analysis Agent**: News, social media, and market sentiment aggregation
- **Risk Management Agent**: Portfolio risk assessment, position sizing, hedging strategies
- **Smart Execution Agent**: Optimal order routing, slippage minimization
- **Volatility Agent**: Options analytics, volatility surface modeling, VIX correlations

#### **Market Intelligence Agents**
- **Options Flow Intelligence**: Real-time unusual options activity detection
- **Gamma Exposure Agent**: Market maker positioning, GEX levels calculation
- **Liquidity Prediction Agent**: Order book depth analysis, liquidity forecasting
- **Market Regime Agent**: Bull/bear/sideways market detection, regime transitions
- **Arbitrage Detection Agent**: Cross-market, statistical, and triangular arbitrage

#### **Alternative Data Agents**
- **News Sentiment Agent**: Multi-source news with credibility weighting (Bloomberg, Reuters, WSJ)
- **Social Sentiment Agent**: Reddit WSB analysis, Twitter trends, StockTwits, meme stock detection
- **Weather Impact Agent**: Commodity correlations, agricultural impacts, energy demand
- **Commodity Data Agent**: Oil, gold, copper trends affecting equity sectors
- **Economic Indicator Agent**: Government data integration (FRED, BLS, Treasury)

#### **Master Orchestrators**
- **Alternative Data Master Agent**: Orchestrates all alternative data sources with adaptive weighting
- **Meta Consensus Agent**: Combines signals from all agents using ensemble methods
- **Unified Orchestrator**: Central AI brain managing all system components

### 📊 Data Integration (30+ APIs)

#### **Market Data Providers**
- Polygon.io, Alpha Vantage, Finnhub, TwelveData
- Real-time quotes, historical data, options chains

#### **Alternative Data Sources**
- **News**: NewsAPI, Marketaux, Finnhub News
- **Social**: Reddit API, Twitter API, StockTwits
- **Weather**: OpenWeatherMap, NOAA
- **Commodities**: Quandl, EIA, USDA
- **Crypto**: CoinGecko, Etherscan
- **Economic**: FRED, BLS, US Treasury, ECB, World Bank, OECD

#### **Government Data (Free)**
- Federal Reserve Economic Data (FRED)
- Bureau of Labor Statistics (BLS)
- US Treasury Direct (bond yields, auction data)
- Energy Information Administration (EIA)

### 🧠 Machine Learning Pipeline

#### **Advanced Models**
- **LSTM Networks**: Time series prediction, sequential pattern learning
- **Transformer Models**: Attention-based market prediction
- **Ensemble Models**: Random forests, gradient boosting, model stacking
- **AutoML Integration**: Automated model selection and hyperparameter tuning

#### **Training & Optimization**
- Online learning with real-time model updates
- Backtesting framework with walk-forward analysis
- Feature engineering pipeline (200+ technical indicators)
- Cross-validation with market regime awareness

### 🔄 Real-Time Processing

#### **WebSocket Streams**
- Market data streaming (quotes, trades, order book)
- Agent signal broadcasting
- Portfolio updates and alerts
- Multi-client synchronization

#### **Event-Driven Architecture**
- Pub/sub messaging for agent communication
- Event sourcing for audit trails
- CQRS pattern for read/write optimization

### 📈 Professional Trading Interface

#### **Advanced Charting**
- **AIHybridChart**: TradingView-style with D3.js overlays
- AI pattern recognition overlays
- Multi-timeframe analysis
- Volume profile, market profile
- Custom indicators and drawing tools

#### **Risk Analytics Dashboard**
- Real-time P&L tracking
- VaR and stress testing
- Portfolio Greeks (options)
- Correlation matrices
- Exposure heatmaps

## 🚀 Quick Start

### Prerequisites

```bash
# System Requirements
- Python 3.11+
- Node.js 18+
- PostgreSQL 14+ or SQLite
- Redis 7+ (optional, for caching)
- 8GB+ RAM recommended
- 10GB+ disk space
```

### Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/GoldenSignalsV5.git
cd GoldenSignalsV5

# 2. Set up Python environment
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.template .env
# Edit .env and add your API keys (see API Keys section below)

# 4. Initialize database
python -m alembic upgrade head

# 5. Set up frontend
cd ../frontend
npm install
```

### 🔑 API Keys Configuration

The platform integrates with 30+ data providers. Essential keys for basic operation:

```bash
# backend/.env

# === CRITICAL SECURITY ===
SECRET_KEY=<generate-with-script>  # python -c "import secrets; print(secrets.token_urlsafe(32))"
JWT_SECRET_KEY=<generate-with-script>

# === MARKET DATA (need at least one) ===
POLYGON_API_KEY=your_key_here
ALPHA_VANTAGE_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here

# === AI/LLM (recommended) ===
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# === ALTERNATIVE DATA (optional but powerful) ===
# News & Sentiment
NEWSAPI_KEY=your_key_here
MARKETAUX_KEY=your_key_here

# Social Media
REDDIT_CLIENT_ID=your_id_here
REDDIT_CLIENT_SECRET=your_secret_here
TWITTER_BEARER_TOKEN=your_token_here
STOCKTWITS_TOKEN=your_token_here

# Weather (for commodity correlations)
OPENWEATHER_API_KEY=your_key_here

# Commodities & Economic
QUANDL_API_KEY=your_key_here
FRED_API_KEY=your_key_here  # Free from St. Louis Fed
EIA_API_KEY=your_key_here   # Free from US Energy Dept
```

### 🏃 Run the Application

```bash
# Start everything with one command
./start.sh

# Or start components separately:

# Backend API (from backend directory)
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Frontend (from frontend directory)
npm run dev

# Access the application
# Frontend: http://localhost:3000
# API Docs: http://localhost:8000/docs
# WebSocket: ws://localhost:8000/ws
```

## 📁 Project Architecture

```
GoldenSignalsV5/
├── backend/
│   ├── agents/                    # 20+ Specialized AI Agents
│   │   ├── base.py               # Base agent framework
│   │   ├── technical_analysis_agent.py
│   │   ├── sentiment_analysis_agent.py
│   │   ├── risk_management_agent.py
│   │   ├── options_flow_intelligence.py
│   │   ├── gamma_exposure_agent.py
│   │   ├── volatility_agent.py
│   │   ├── news_sentiment_agent.py      # NEW
│   │   ├── social_sentiment_agent.py    # NEW
│   │   ├── weather_impact_agent.py      # NEW
│   │   ├── commodity_data_agent.py      # NEW
│   │   └── alternative_data_master_agent.py  # NEW
│   │
│   ├── api/
│   │   ├── routes/               # 25+ API endpoint modules
│   │   │   ├── market_data.py
│   │   │   ├── signals.py
│   │   │   ├── alternative_data.py  # NEW: 12 endpoints
│   │   │   ├── government_data.py   # NEW: 8 endpoints
│   │   │   └── ...
│   │   └── websocket/            # Real-time streaming
│   │
│   ├── services/
│   │   ├── alternative_data_service.py  # NEW: 30+ API integrations
│   │   ├── market_data_unified.py       # Unified data aggregation
│   │   ├── enhanced_data_aggregator.py  # Cross-source validation
│   │   └── ...
│   │
│   ├── ml/                       # Machine Learning Pipeline
│   │   ├── models/              # LSTM, Transformer, Ensemble
│   │   ├── training/            # Training pipelines
│   │   └── features/            # Feature engineering
│   │
│   ├── core/
│   │   ├── orchestrator.py      # Unified system orchestrator
│   │   ├── langgraph_orchestrator.py  # LangGraph integration
│   │   └── ai_brain.py         # Central AI coordination
│   │
│   └── tests/                   # Comprehensive test suite
│       ├── test_volatility_agent.py
│       └── test_alternative_data.py
│
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── layout/ProfessionalLayout.tsx
│       │   └── charts/AIHybridChart.tsx
│       ├── pages/               # Trading interfaces
│       └── services/            # API integrations
│
└── docs/
    ├── API_REFERENCE.md         # Complete API documentation
    ├── AGENT_GUIDE.md          # Agent capabilities & usage
    └── DATA_SOURCES.md         # Data provider setup guide
```

## 🧪 Testing

```bash
# Run all tests
cd backend
pytest

# Run specific test categories
pytest tests/agents/              # Agent framework tests
pytest tests/test_volatility_agent.py  # Volatility agent tests
pytest tests/test_alternative_data.py  # Alternative data tests

# Run with coverage
pytest --cov=. --cov-report=html

# Run integration tests
pytest tests/integration/ -v

# Performance benchmarks
python scripts/benchmark_agents.py
```

## 📊 API Endpoints

### Core Trading APIs
- `GET /api/v1/market/{symbol}` - Real-time market data
- `GET /api/v1/signals/{symbol}` - Trading signals
- `POST /api/v1/agents/analyze` - Run agent analysis
- `WebSocket /ws` - Real-time streaming

### Alternative Data APIs (NEW)
- `GET /api/alternative/data/comprehensive` - All alternative data sources
- `GET /api/alternative/news/sentiment` - News sentiment analysis
- `GET /api/alternative/social/sentiment` - Social media sentiment
- `GET /api/alternative/weather/impact` - Weather impact on markets
- `GET /api/alternative/commodity/data` - Commodity market data
- `GET /api/alternative/analysis/{symbol}/master` - Master AI analysis

### Government Data APIs (NEW)
- `GET /api/v1/government/fred/series` - FRED economic data
- `GET /api/v1/government/treasury/yields` - Treasury yield curve
- `GET /api/v1/government/bls/employment` - Employment statistics
- `GET /api/v1/government/economic/calendar` - Economic events

### Machine Learning APIs
- `POST /api/v1/ml/train` - Train ML models
- `GET /api/v1/ml/predict/{symbol}` - Get predictions
- `GET /api/v1/ml/performance` - Model performance metrics

## 🎯 Use Cases

### 1. **Meme Stock Detection**
The Social Sentiment Agent monitors Reddit WSB, Twitter, and StockTwits for:
- Unusual mention spikes
- Rocket emoji patterns 🚀🌙💎🙌
- "Diamond hands", "YOLO", "to the moon" phrases
- Viral content velocity tracking

### 2. **Weather-Driven Commodity Trading**
Weather Impact Agent correlates:
- Hurricanes → Energy stocks (refineries, offshore drilling)
- Droughts → Agricultural futures and food companies
- Cold snaps → Natural gas demand and utilities

### 3. **Options Flow Intelligence**
- Detect unusual options activity before price moves
- Track "smart money" institutional flows
- Identify hedging vs. directional bets
- Calculate dealer gamma exposure (GEX)

### 4. **Cross-Asset Arbitrage**
- Commodity-equity correlations (oil→energy stocks)
- Currency-commodity relationships
- Bond-equity rotation signals
- Crypto-traditional market divergences

## 🚦 Performance Metrics

- **Latency**: <50ms API response time
- **Throughput**: 10,000+ signals/second
- **Data Sources**: 30+ integrated APIs
- **Agent Decisions**: <100ms per agent
- **ML Inference**: <200ms per prediction
- **WebSocket Clients**: 1000+ concurrent
- **Historical Data**: 10+ years backtesting

## 🛡️ Security Features

- JWT-based authentication
- API key encryption at rest
- Rate limiting per endpoint
- SQL injection prevention
- XSS protection
- CORS configuration
- Audit logging
- Secret rotation support

## 📈 Roadmap

### Phase 1: Foundation ✅
- Core architecture
- Base agents
- API structure
- Database models

### Phase 2: Data Integration ✅
- 30+ API integrations
- Alternative data sources
- Government data feeds
- Real-time streaming

### Phase 3: AI Enhancement (Current)
- Advanced ML models
- Agent orchestration
- Pattern recognition
- Predictive analytics

### Phase 4: Production Features (Planned)
- Paper trading mode
- Live trading integration
- Mobile app
- Cloud deployment
- Enterprise features

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution
- New data source integrations
- Additional AI agents
- ML model improvements
- Frontend enhancements
- Documentation
- Test coverage

## 📚 Documentation

- **[API Reference](docs/API_REFERENCE.md)** - Complete API documentation
- **[Agent Guide](docs/AGENT_GUIDE.md)** - AI agent capabilities
- **[Data Sources](docs/DATA_SOURCES.md)** - Provider setup guides
- **[ML Pipeline](docs/ML_PIPELINE.md)** - Machine learning documentation
- **[Architecture](docs/ARCHITECTURE.md)** - System design overview

## 🆘 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/GoldenSignalsV5/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/GoldenSignalsV5/discussions)
- **Email**: support@goldensignals.ai

## ⚖️ License

[MIT License](LICENSE) - See LICENSE file for details

## 🙏 Acknowledgments

- Financial data providers for API access
- Open source community for amazing tools
- Contributors and testers
- Research papers and academic resources

---

**Disclaimer**: This software is for educational and research purposes only. Not financial advice. Trading involves risk of loss. Always do your own research and consult with financial professionals.

---

Built with ❤️ by the GoldenSignals Team | Powered by AI 🤖