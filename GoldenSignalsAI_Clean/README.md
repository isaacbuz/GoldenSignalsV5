# GoldenSignalsAI - Clean Architecture

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
GoldenSignalsAI_Clean/
├── backend/                 # FastAPI backend
│   ├── api/                # API routes (RAG, MCP)
│   ├── core/              # Core configuration
│   ├── services/          # Business logic
│   │   ├── rag/          # RAG implementation
│   │   ├── mcp/          # MCP server
│   │   └── orchestrator.py # Central orchestrator
│   ├── models/            # Data models
│   └── app.py            # Main application
├── frontend/              # React frontend
│   └── src/              # Source code
├── agents/               # Trading agents
├── tests/                # Test suite
├── scripts/              # Utility scripts
├── config/               # Configuration files
└── docs/                 # Documentation
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