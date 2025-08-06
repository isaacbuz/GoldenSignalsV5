# GoldenSignalsAI - AI-Powered Trading Platform

A production-ready trading platform with AI-driven signal generation, real-time market data, and advanced analytics.

## 🚀 Features

### Core Capabilities
- **Real-time Market Data**: Live quotes, historical data, and technical indicators
- **AI Signal Generation**: Multi-agent system with weighted consensus
- **RAG Intelligence**: Context-aware signal augmentation using historical patterns
- **MCP Integration**: Standardized AI model interactions
- **WebSocket Streaming**: Real-time updates for prices and signals
- **Multi-tier Caching**: Memory → Redis → Database architecture
- **Professional UI**: React + Material-UI with TradingView-style charts

### Technical Architecture
- **Backend**: FastAPI with async/await, SQLAlchemy, Redis
- **Frontend**: React, Redux Toolkit, Material-UI, WebSocket
- **AI/ML**: Custom agents, RAG engine, MCP servers
- **Infrastructure**: Docker-ready, structured logging, health monitoring

## 📋 Prerequisites

- Python 3.8+
- Node.js 16+
- Redis (optional but recommended)
- PostgreSQL (production) or SQLite (development)

## 🛠️ Installation

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd GoldenSignalsAI_Clean

# Run the start script
./start.sh
```

The start script will:
- Check prerequisites
- Create virtual environments
- Install dependencies
- Start Redis if available
- Launch backend and frontend servers
- Display health status

### Manual Installation

#### Backend Setup

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env with your configuration

# Run migrations (if using PostgreSQL)
alembic upgrade head

# Start the server
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# Environment
ENVIRONMENT=development
DEBUG=true

# Database
DATABASE_URL=sqlite+aiosqlite:///./goldensignals.db
# For PostgreSQL: postgresql+asyncpg://user:pass@localhost/goldensignals

# Redis
REDIS_URL=redis://localhost:6379

# API Keys (optional)
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here

# Market Data Providers (optional)
ALPHA_VANTAGE_API_KEY=your-key-here
POLYGON_API_KEY=your-key-here
```

## 📖 API Documentation

Once running, access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Key Endpoints

#### Market Data
- `GET /api/v1/market-data/quote/{symbol}` - Get real-time quote
- `GET /api/v1/market-data/historical/{symbol}` - Get historical data
- `GET /api/v1/market-data/indicators/{symbol}` - Get technical indicators

#### Signals
- `GET /api/v1/signals` - List signals with filtering
- `POST /api/v1/signals` - Create new signal
- `GET /api/v1/signals/analytics` - Get signal analytics

#### Agents
- `POST /api/v1/agents/analyze/{symbol}` - Run AI analysis
- `GET /api/v1/agents/performance` - Get agent performance metrics
- `POST /api/v1/agents/performance/rebalance` - Rebalance agent weights

#### MCP Tools
- `GET /api/v2/mcp/servers` - List available MCP servers
- `POST /api/v2/mcp/servers/{server}/execute` - Execute MCP tool
- `GET /api/v2/mcp/tools` - List all available tools

#### WebSocket
- `ws://localhost:8000/ws` - Real-time market data and signals

## 🏗️ Architecture

### Backend Structure
```
backend/
├── agents/           # AI trading agents
│   ├── base.py      # Base agent framework
│   ├── orchestrator.py # Agent coordination
│   └── technical/   # Technical analysis agents
├── api/             # API routes
│   └── routes/      # Endpoint definitions
├── core/            # Core utilities
│   ├── config.py    # Settings management
│   ├── database.py  # Database configuration
│   ├── dependencies.py # Dependency injection
│   ├── errors.py    # Error handling
│   └── logging.py   # Structured logging
├── mcp/             # Model Context Protocol
│   ├── servers/     # MCP server implementations
│   └── client/      # MCP client
├── models/          # Database models
├── rag/             # RAG engine
│   └── core/        # RAG implementation
├── services/        # Business logic
└── app.py          # Main application
```

### Frontend Structure
```
frontend/
├── src/
│   ├── components/  # React components
│   │   └── charts/  # Trading charts
│   ├── pages/       # Page components
│   ├── services/    # API services
│   ├── store/       # Redux store
│   │   └── slices/  # Redux slices
│   └── App.tsx      # Main app component
```

## 🧪 Testing

```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

## 📊 Monitoring

### Health Checks
- Basic: http://localhost:8000/api/v1/health
- Detailed: http://localhost:8000/api/v1/health/detailed

### Logs
- Backend: `logs/backend.log`
- Frontend: `logs/frontend.log`
- Errors: `logs/errors.log`

## 🚀 Production Deployment

### Docker Deployment

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# Scale workers
docker-compose scale worker=3
```

### Environment Variables for Production

```env
ENVIRONMENT=production
DEBUG=false
DATABASE_URL=postgresql+asyncpg://user:pass@db/goldensignals
REDIS_URL=redis://redis:6379
SECRET_KEY=your-secret-key-here
```

## 📝 Development

### Adding New Agents

1. Create agent class inheriting from `BaseAgent`
2. Implement `analyze()` and `get_required_data_types()`
3. Register in orchestrator
4. Add tests

### Adding New MCP Tools

1. Define tool in MCP server
2. Implement tool logic
3. Add to tool registry
4. Document usage

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- TradingView for chart inspiration
- FastAPI for the excellent framework
- The open-source community

## 📞 Support

- Documentation: See `/docs` folder
- Issues: GitHub Issues
- Email: support@goldensignalsai.com

---

Built with ❤️ by the GoldenSignalsAI Team