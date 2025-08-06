# GoldenSignalsAI Architecture

## Overview

GoldenSignalsAI is an AI-powered trading platform built with a clean, modular architecture that integrates:
- **RAG (Retrieval-Augmented Generation)** for context-aware market intelligence
- **MCP (Model Context Protocol)** for standardized AI model interactions
- **Agentic Workflow** for autonomous trading analysis

## Core Components

### 1. RAG System (`backend/services/rag/`)

The RAG system provides context-aware market intelligence by:
- **Vector Store**: Stores and indexes market documents, news, and trading rules
- **Retriever**: Finds relevant context for queries using similarity search
- **Generator**: Produces trading insights using retrieved context

**Key Features:**
- Document ingestion for market data, news, research
- Similarity-based retrieval
- Context-aware insight generation
- Source tracking for transparency

**API Endpoints:**
- `POST /api/v1/rag/query` - Query for trading insights
- `POST /api/v1/rag/ingest` - Add new documents
- `GET /api/v1/rag/status` - System status

### 2. MCP Server (`backend/services/mcp/`)

The MCP server provides a standardized interface for AI operations:
- **Tools Registry**: Manages available AI tools
- **Message Protocol**: Handles standardized requests/responses
- **Batch Processing**: Executes multiple tools concurrently

**Available Tools:**
- `get_market_data` - Fetch real-time market data
- `generate_signal` - Generate AI trading signals
- `analyze_portfolio` - Portfolio analysis and recommendations

**API Endpoints:**
- `GET /api/v1/mcp/tools` - List available tools
- `POST /api/v1/mcp/execute` - Execute a single tool
- `POST /api/v1/mcp/batch` - Execute multiple tools

### 3. Trading Orchestrator (`backend/services/orchestrator.py`)

The orchestrator is the central hub that coordinates:
- RAG system for market context
- MCP server for AI tools
- Trading agents for specialized analysis
- Signal generation pipeline

**Workflow:**
1. Receives market update
2. Enriches with RAG context
3. Executes MCP tools for data/analysis
4. Runs through specialized agents
5. Generates consolidated trading signal

### 4. Trading Agents (`agents/`)

Specialized agents for different aspects of trading:
- **Technical Analysis** - Chart patterns, indicators
- **Sentiment Analysis** - News, social media sentiment
- **Risk Management** - Position sizing, stop-loss
- **Portfolio Optimization** - Asset allocation
- **Orchestrator Agent** - Coordinates other agents

Each agent can:
- Access RAG system for context
- Use MCP tools for analysis
- Contribute to signal generation

## Data Flow

```
Market Data → Orchestrator → RAG Context
                ↓
            MCP Tools ← → Trading Agents
                ↓
        Signal Generation
                ↓
            Frontend UI
```

## API Structure

```
/api/v1/
├── /health          # System health check
├── /rag/            # RAG operations
│   ├── /query       # Query for insights
│   ├── /ingest      # Add documents
│   └── /status      # RAG status
├── /mcp/            # MCP operations
│   ├── /tools       # List tools
│   ├── /execute     # Execute tool
│   └── /batch       # Batch execution
├── /signals/        # Trading signals (TODO)
├── /agents/         # Agent management (TODO)
└── /market-data/    # Market data (TODO)
```

## Frontend Architecture

- **React + TypeScript** for type safety
- **Single App.tsx** - No duplicate implementations
- **Unified Chart Component** - Consolidated charting
- **Material-UI** for consistent design
- **WebSocket** for real-time updates

## Key Design Principles

1. **Modular**: Each component has a single responsibility
2. **Extensible**: Easy to add new agents, tools, or data sources
3. **Context-Aware**: RAG provides relevant market context
4. **Standardized**: MCP ensures consistent AI interactions
5. **Autonomous**: Agents can operate independently
6. **Real-time**: WebSocket support for live updates

## Getting Started

1. Start the backend:
   ```bash
   cd backend
   python app.py
   ```

2. Start the frontend:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

3. Access the API docs: http://localhost:8000/docs

## Future Enhancements

- Additional MCP tools for advanced analysis
- More specialized trading agents
- Enhanced RAG with larger knowledge base
- Real-time model fine-tuning
- Production deployment with Kubernetes