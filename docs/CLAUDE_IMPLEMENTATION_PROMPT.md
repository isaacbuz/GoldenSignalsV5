# Claude Implementation Prompt for GoldenSignalsAI

## Context Setting

You are tasked with transforming the GoldenSignalsAI_Clean project from a basic structural framework into a professional, production-ready AI-powered trading platform. The project currently has the architecture in place but lacks implementation. You have access to a comprehensive archive (goldensignals_full_archive_20250802_173153.tar.gz) containing the previous implementation with working code that you should reference for best practices and functionality.

## Project Overview

**Current State**: Basic framework with RAG (Retrieval-Augmented Generation), MCP (Model Context Protocol), and Agentic architecture structure, but no actual implementation.

**Target State**: A fully functional, professional-grade trading platform that leverages AI for market analysis, signal generation, and autonomous trading decisions.

## Implementation Requirements

### Phase 1: Core Infrastructure Implementation

1. **Database Layer**
   - Implement PostgreSQL with SQLAlchemy ORM from the archived project's `src/core/database.py` and models
   - Set up Alembic migrations with proper versioning
   - Implement connection pooling and async database operations
   - Add Redis caching layer referencing `src/core/redis_manager.py`

2. **Configuration System**
   - Port the configuration management from `src/config/` in the archive
   - Implement environment-based settings (dev, staging, prod)
   - Add AWS integration config from `src/config/aws_config.py`
   - Set up proper secret management

3. **Authentication & Security**
   - Implement JWT-based authentication from the archived `src/api/auth/`
   - Add role-based access control (RBAC)
   - Implement API rate limiting and request validation
   - Add comprehensive error handling middleware

### Phase 2: Market Data & Real-time Systems

1. **Market Data Service**
   - Port the working implementation from archived `src/services/market_data_service.py`
   - Integrate yfinance with proper error handling and retries
   - Implement the candle normalizer from `frontend/src/utils/candleNormalizer.ts`
   - Add multi-timeframe data support (1m, 5m, 15m, 1h, 1d)

2. **WebSocket Implementation**
   - Implement the production WebSocket manager from archived `src/websocket/scalable_manager.py`
   - Add connection pooling and automatic reconnection
   - Implement real-time price streaming with buffering
   - Add WebSocket authentication and room-based broadcasting

3. **Caching Strategy**
   - Implement three-tier caching (Memory L1, Redis L2, Database L3)
   - Port cache invalidation strategies from the archive
   - Add TTL-based cache management

### Phase 3: AI/ML Integration

1. **RAG System Implementation**
   - Complete the RAG system using patterns from archived `src/rag/` implementations
   - Integrate ChromaDB or Pinecone for vector storage
   - Implement OpenAI embeddings with fallback to local models
   - Add document processing pipeline for:
     - Market news ingestion
     - Technical analysis reports
     - Historical pattern storage
     - Trading rule knowledge base

2. **MCP Server Enhancement**
   - Implement all MCP tools based on archived `mcp_servers/` directory
   - Add these essential tools:
     - Market data fetching with technical indicators
     - Multi-agent signal generation
     - Portfolio optimization
     - Risk assessment
     - Backtesting interface
   - Implement tool chaining and parallel execution

3. **ML Models Integration**
   - Port the ML prediction models from `src/ml/models/`
   - Implement the transformer-based price prediction
   - Add the pattern recognition system
   - Include sentiment analysis from news/social media

### Phase 4: Trading Agents Implementation

1. **Agent System**
   Reference the archived `agents/` directory to implement:
   - **Technical Analysis Agent**: 
     - RSI, MACD, Bollinger Bands, Moving Averages
     - Chart pattern recognition (Head & Shoulders, Triangles, etc.)
     - Support/Resistance level detection
   - **Sentiment Analysis Agent**:
     - News sentiment scoring
     - Social media analysis
     - Market fear/greed index
   - **Risk Management Agent**:
     - Position sizing algorithms
     - Stop-loss/Take-profit optimization
     - Portfolio heat mapping
   - **ML Prediction Agent**:
     - Time series forecasting
     - Volatility prediction
     - Trend classification
   - **Orchestrator Agent**:
     - Multi-agent consensus mechanism
     - Confidence weight adjustment
     - Signal aggregation logic

2. **Signal Generation**
   - Implement the signal service from archived `src/services/signal_service.py`
   - Add multi-timeframe analysis
   - Include confidence scoring algorithm
   - Implement signal validation and backtesting

### Phase 5: Frontend Implementation

1. **Chart Component**
   - Port the UnifiedChart functionality into the clean TradingChart component
   - Implement using TradingView Lightweight Charts
   - Add these features from the archive:
     - Real-time candlestick updates
     - Multiple technical indicators overlay
     - Drawing tools
     - Multi-timeframe support
     - Volume profile
     - AI signal markers

2. **Dashboard Components**
   - Implement the signal dashboard showing:
     - Active signals with confidence scores
     - Agent consensus visualization
     - P&L tracking
     - Risk metrics
   - Add portfolio management interface
   - Include backtesting results viewer

3. **State Management**
   - Implement Redux Toolkit for global state
   - Add WebSocket middleware for real-time updates
   - Include persistent user preferences

### Phase 6: Advanced Features

1. **Backtesting Engine**
   - Port the backtesting system from archived `src/domain/backtesting/`
   - Add walk-forward optimization
   - Implement Monte Carlo simulation
   - Include transaction cost modeling

2. **Performance Optimization**
   - Implement the optimizations from Week 5 of the archived project
   - Add query optimization from `src/infrastructure/database/enhanced_query_optimizer.py`
   - Include bundle optimization for frontend
   - Add service worker for offline capability

3. **Monitoring & Observability**
   - Implement comprehensive logging system
   - Add Prometheus metrics
   - Include error tracking (Sentry integration)
   - Add performance monitoring dashboard

## Best Practices to Follow

1. **Code Quality**
   - Use type hints throughout Python code
   - Implement comprehensive error handling
   - Add detailed docstrings
   - Follow PEP 8 and ESLint standards

2. **Testing**
   - Achieve >80% test coverage
   - Include unit, integration, and e2e tests
   - Add performance benchmarks
   - Implement continuous testing in CI/CD

3. **Security**
   - Never expose API keys or secrets
   - Implement request signing for sensitive operations
   - Add audit logging for all trading actions
   - Use parameterized queries to prevent SQL injection

4. **Performance**
   - Implement async operations throughout
   - Use connection pooling for all external services
   - Add circuit breakers for external API calls
   - Optimize database queries with proper indexing

## Implementation Order

1. Start with database and configuration (foundation)
2. Implement market data and WebSocket (data flow)
3. Complete RAG and MCP systems (AI infrastructure)
4. Implement one agent fully as a template
5. Build out remaining agents
6. Create frontend with basic chart
7. Add advanced features iteratively

## Validation Checklist

- [ ] Can fetch and display real-time market data
- [ ] WebSocket streams updates without drops
- [ ] RAG system provides contextual insights
- [ ] MCP tools execute reliably
- [ ] Agents generate actionable signals
- [ ] Frontend displays professional trading interface
- [ ] System handles errors gracefully
- [ ] Performance meets <100ms API response time
- [ ] All critical paths have test coverage
- [ ] Documentation is comprehensive

## Expected Outcome

A production-ready trading platform that:
- Processes real-time market data with <50ms latency
- Generates AI-powered trading signals with explainable reasoning
- Provides professional-grade charting and analysis tools
- Handles concurrent users with horizontal scaling
- Maintains 99.9% uptime with proper error recovery
- Offers a smooth, responsive user experience

Use the archived project as a reference for implementation details, but improve upon it by:
- Removing duplications
- Optimizing performance bottlenecks
- Enhancing error handling
- Improving code organization
- Adding missing test coverage

The goal is not just to port the old code, but to create a superior, clean implementation that represents the best of both architectures.