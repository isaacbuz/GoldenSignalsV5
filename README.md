# GoldenSignalsAI V5 - AI-Powered Trading Platform

A modern AI-powered trading platform built with clean architecture principles, featuring RAG (Retrieval-Augmented Generation), MCP (Model Context Protocol), and autonomous trading agents.

## 🚀 Quick Start

```bash
# Start the application
./start.sh

# Start in production mode
./start.sh --prod

# Stop the application
./stop.sh
```

## 📁 Project Structure

```
GoldenSignalsAI_V5/
├── backend/                 # FastAPI backend
│   ├── api/                # API routes and WebSocket endpoints
│   ├── agents/            # Trading agents and orchestrator
│   ├── core/              # Core configuration and dependencies
│   ├── services/          # Business logic services
│   ├── models/            # SQLAlchemy data models
│   ├── tests/             # Comprehensive test suite
│   │   ├── agents/        # Agent framework tests
│   │   ├── agents_tests/  # Specific agent tests
│   │   ├── api/           # API endpoint tests
│   │   ├── services/      # Service layer tests
│   │   ├── ml_tests/      # ML model tests
│   │   ├── websocket_tests/ # WebSocket tests
│   │   └── integration/   # Integration tests
│   ├── scripts/           # Maintenance and utility scripts
│   ├── logs/              # Application logs
│   └── app.py            # Main FastAPI application
├── frontend/              # React TypeScript frontend
│   └── src/              # Source code with professional layout
├── docs/                  # Consolidated documentation
│   ├── backend/          # Backend documentation
│   ├── project-planning/ # Project planning documents
│   └── README.md         # Documentation index
├── config/               # Configuration files
└── scripts/              # Build and deployment scripts
```

## 📚 Documentation

All documentation is in the `docs/` folder:

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture overview
- **[docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md)** - Current implementation status
- **[docs/CLAUDE_IMPLEMENTATION_PROMPT.md](docs/CLAUDE_IMPLEMENTATION_PROMPT.md)** - Implementation guide for Claude
- **[docs/CLAUDE_DETAILED_IMPLEMENTATION_GUIDE.md](docs/CLAUDE_DETAILED_IMPLEMENTATION_GUIDE.md)** - Detailed 20-day implementation plan
- **[docs/QUICK_START_IMPLEMENTATION.md](docs/QUICK_START_IMPLEMENTATION.md)** - Quick start for minimal implementation

## 🏗️ Implementation Status

**Current State**: ~15% Complete (Framework/Structure only)

### ✅ Implemented
- Clean architecture structure
- RAG system framework
- MCP server framework
- API route structure
- Basic React setup

### ❌ Not Yet Implemented
- Database connections
- Market data integration
- AI/ML models
- WebSocket streaming
- Trading logic
- Frontend components
- Authentication

See [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md) for full details.

## 🧪 Testing

The project includes a comprehensive test suite organized by functionality:

```bash
# Run all tests
cd backend && python run_tests.py

# Run specific test categories
pytest tests/agents_tests/     # Agent tests
pytest tests/api_tests/        # API tests
pytest tests/services_tests/   # Service tests
pytest tests/ml_tests/         # ML model tests
pytest tests/websocket_tests/  # WebSocket tests

# Run with coverage
pytest --cov=. --cov-report=html
```

See **[backend/tests/README.md](backend/tests/README.md)** for detailed testing documentation.

## 🛠️ For Developers

### Using Claude for Implementation

1. Provide Claude with:
   - The implementation prompt: `docs/CLAUDE_IMPLEMENTATION_PROMPT.md`
   - The detailed guide: `docs/CLAUDE_DETAILED_IMPLEMENTATION_GUIDE.md`
   - Access to archive: `goldensignals_full_archive_20250802_173153.tar.gz`

2. Example prompt:
   ```
   Please implement Day 1 tasks from the CLAUDE_DETAILED_IMPLEMENTATION_GUIDE.md,
   using the archive as reference for best practices.
   ```

### Manual Implementation

Follow the day-by-day guide in `docs/CLAUDE_DETAILED_IMPLEMENTATION_GUIDE.md`:
- Days 1-3: Core Infrastructure
- Days 4-6: Market Data & Real-time
- Days 7-10: AI/ML Integration
- Days 11-13: Trading Logic
- Days 14-17: Frontend
- Days 18-20: Testing & Deployment

## 🎯 Key Features (When Complete)

- **RAG System**: Context-aware market intelligence
- **MCP Server**: Standardized AI tool interface
- **Trading Agents**: Autonomous analysis and decisions
- **Real-time Data**: WebSocket streaming
- **Professional Charts**: TradingView-style interface
- **Risk Management**: Position sizing and protection
- **Backtesting**: Historical strategy validation

## 📋 Requirements

- Python 3.9+
- Node.js 18+
- PostgreSQL 14+
- Redis 7+

## 🚦 Getting Started

1. Clone the repository
2. Review the architecture: `docs/ARCHITECTURE.md`
3. Check implementation status: `docs/IMPLEMENTATION_STATUS.md`
4. Follow the implementation guide or use Claude
5. Run tests as you build
6. Deploy with Docker

## 📄 License

[Your License Here]

---

For detailed implementation instructions, see the [docs/](docs/) folder.
