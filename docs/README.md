# GoldenSignalsAI - Clean Architecture

A modern AI-powered trading platform built with FastAPI, React, and advanced ML agents.

## Architecture

### Backend (FastAPI)
- **API Routes**: RESTful endpoints for market data, signals, and agent management
- **WebSocket**: Real-time data streaming
- **Services**: Market data fetching, signal generation, agent orchestration
- **RAG System**: Retrieval-Augmented Generation for market insights
- **MCP Servers**: Model Context Protocol for AI interactions

### Frontend (React + TypeScript)
- **Trading Charts**: Real-time market visualization
- **Signal Dashboard**: AI-generated trading signals
- **Portfolio Management**: Track and optimize positions

### AI Agents
- **Technical Analysis**: Chart patterns and indicators
- **Sentiment Analysis**: News and social media sentiment
- **Risk Management**: Position sizing and stop-loss
- **Portfolio Optimization**: Asset allocation
- **Orchestrator**: Coordinates all agents

## Quick Start

```bash
./start.sh
```

## Project Structure

```
├── backend/
│   ├── api/          # API endpoints
│   ├── core/         # Core configuration
│   ├── services/     # Business logic
│   ├── models/       # Data models
│   └── app.py        # Main application
├── frontend/
│   ├── src/          # React source
│   └── public/       # Static assets
├── agents/           # Trading agents
├── tests/            # Test suite
└── docs/             # Documentation
```
