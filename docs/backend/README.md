# GoldenSignalsAI Documentation

Complete documentation for the GoldenSignalsAI trading system.

## Documentation Structure

### 📚 Architecture
- [`/architecture/README.md`](architecture/README.md) - System architecture overview
- [`/architecture/design-patterns.md`](architecture/design-patterns.md) - Design patterns used
- [`/architecture/data-flow.md`](architecture/data-flow.md) - Data flow diagrams
- [`/architecture/scalability.md`](architecture/scalability.md) - Scalability considerations

### 🤖 AI Agents
- [`/agents/README.md`](agents/README.md) - AI agents overview
- [`/agents/technical-analysis.md`](agents/technical-analysis.md) - Technical analysis agent
- [`/agents/sentiment-analysis.md`](agents/sentiment-analysis.md) - Sentiment analysis agent
- [`/agents/volatility-agent.md`](agents/volatility-agent.md) - Volatility analysis agent
- [`/agents/orchestration.md`](agents/orchestration.md) - Agent orchestration

### 🔌 API Documentation
- [`/api/README.md`](api/README.md) - API overview and authentication
- [`/api/market-data.md`](api/market-data.md) - Market data endpoints
- [`/api/agents.md`](api/agents.md) - AI agents endpoints
- [`/api/signals.md`](api/signals.md) - Trading signals endpoints
- [`/api/websockets.md`](api/websockets.md) - WebSocket documentation

### 🔧 Services
- [`/services/README.md`](services/README.md) - Services overview
- [`/services/data-aggregation.md`](services/data-aggregation.md) - Data aggregation service
- [`/services/websocket-manager.md`](services/websocket-manager.md) - WebSocket management
- [`/services/rag-system.md`](services/rag-system.md) - RAG knowledge system
- [`/services/mcp-server.md`](services/mcp-server.md) - Model Context Protocol

### 🗄️ Database
- [`/database/README.md`](database/README.md) - Database overview
- [`/database/models.md`](database/models.md) - Data models
- [`/database/migrations.md`](database/migrations.md) - Migration guide
- [`/database/performance.md`](database/performance.md) - Performance optimization

### 🔐 Security
- [`/security/README.md`](security/README.md) - Security overview
- [`/security/authentication.md`](security/authentication.md) - Authentication & authorization
- [`/security/api-security.md`](security/api-security.md) - API security measures
- [`/security/secrets-management.md`](security/secrets-management.md) - Secrets management

### 📈 Monitoring
- [`/monitoring/README.md`](monitoring/README.md) - Monitoring overview
- [`/monitoring/metrics.md`](monitoring/metrics.md) - System metrics
- [`/monitoring/logging.md`](monitoring/logging.md) - Logging strategy
- [`/monitoring/performance.md`](monitoring/performance.md) - Performance monitoring

### 🚀 Deployment
- [`/deployment/README.md`](deployment/README.md) - Deployment overview
- [`/deployment/docker.md`](deployment/docker.md) - Docker deployment
- [`/deployment/kubernetes.md`](deployment/kubernetes.md) - Kubernetes deployment
- [`/deployment/environment.md`](deployment/environment.md) - Environment configuration

### 💻 Development
- [`/development/README.md`](development/README.md) - Development guide
- [`/development/setup.md`](development/setup.md) - Local development setup
- [`/development/testing.md`](development/testing.md) - Testing strategy
- [`/development/contributing.md`](development/contributing.md) - Contribution guidelines

## Quick Start

1. **System Overview**: Start with [`/architecture/README.md`](architecture/README.md)
2. **API Usage**: Check [`/api/README.md`](api/README.md) for API documentation
3. **Development**: Follow [`/development/setup.md`](development/setup.md) for local setup
4. **Deployment**: Use [`/deployment/README.md`](deployment/README.md) for production deployment

## Documentation Standards

- All documentation uses Markdown format
- Code examples include language specification for syntax highlighting
- API documentation follows OpenAPI 3.0 standards
- Architecture diagrams use Mermaid syntax where possible
- Each module includes usage examples and troubleshooting guides

## Last Updated

Documentation last updated: {{ current_date }}
Version: 5.0.0