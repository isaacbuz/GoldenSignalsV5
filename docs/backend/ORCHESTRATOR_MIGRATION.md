# Orchestrator Consolidation Guide

## Overview
We have consolidated multiple orchestrator implementations into a single, unified LangGraph-based orchestrator.

## Previous Orchestrators (Now Deprecated)
1. **agents/orchestrator.py** - Original agent orchestrator
2. **services/orchestrator.py** - Trading orchestrator
3. **agents/meta_signal_orchestrator.py** - Meta signal orchestrator
4. **services/websocket_orchestrator.py** - WebSocket integration orchestrator

## New Unified Orchestrator
**Location**: `core/langgraph_orchestrator.py`

### Key Features:
- **LangGraph-based**: Uses LangGraph for sophisticated workflow management
- **God AI Integration**: Implements the central AI brain architecture
- **Multi-phase Analysis**: 8-phase comprehensive analysis pipeline
- **WebSocket Support**: Built-in real-time broadcasting
- **RAG Integration**: Enhanced with proper LLM-powered RAG
- **Unified Market Data**: Uses consolidated market data service

## Migration Steps

### 1. Update Imports
```python
# Old
from agents.orchestrator import AgentOrchestrator
from services.orchestrator import TradingOrchestrator
from agents.meta_signal_orchestrator import MetaSignalOrchestrator

# New
from core.langgraph_orchestrator import langgraph_orchestrator
```

### 2. Update API Routes
All orchestrator endpoints should now use the unified orchestrator:

```python
# In api/routes/agents.py
from core.langgraph_orchestrator import langgraph_orchestrator

@router.post("/analyze/{symbol}")
async def analyze_symbol(symbol: str):
    signal = await langgraph_orchestrator.analyze(symbol)
    return signal
```

### 3. WebSocket Integration
The new orchestrator has built-in WebSocket support:

```python
# WebSocket updates are automatic
signal = await langgraph_orchestrator.analyze(symbol, thread_id=client_id)
# Broadcasting happens internally
```

### 4. Agent Registration
Register agents with the new orchestrator:

```python
from core.langgraph_orchestrator import langgraph_orchestrator

# Register your agents
langgraph_orchestrator.register_agent(my_agent)
```

## Feature Mapping

| Old Feature | New Implementation |
|------------|-------------------|
| Agent consensus | Phase 3: Agent Analysis |
| Pattern detection | Phase 2: Pattern Detection |
| Risk assessment | Phase 6: Risk Assessment |
| Signal aggregation | Phase 8: Signal Generation |
| WebSocket broadcasting | Built-in at each phase |
| Meta analysis | God AI Decision (Phase 5) |

## Benefits of Consolidation

1. **Single Source of Truth**: One orchestrator to maintain
2. **Consistent Behavior**: All analysis follows same pipeline
3. **Better Performance**: Optimized parallel execution
4. **Enhanced Features**: LangGraph provides advanced workflow capabilities
5. **Easier Testing**: Single component to test
6. **Reduced Complexity**: Fewer circular dependencies

## Backward Compatibility

To maintain backward compatibility during migration:

```python
# Create compatibility wrapper
class AgentOrchestrator:
    """Compatibility wrapper for old orchestrator"""
    
    async def analyze_market(self, market_data):
        symbol = market_data.get("symbol")
        return await langgraph_orchestrator.analyze(symbol)
```

## Configuration

The unified orchestrator uses these environment variables:
- `OPENAI_API_KEY` - For GPT models
- `ANTHROPIC_API_KEY` - For Claude models
- Market data API keys (Polygon, Alpha Vantage, etc.)

## Testing

Run tests for the new orchestrator:
```bash
pytest tests/test_langgraph_orchestrator.py
```

## Deprecation Timeline

1. **Phase 1** (Immediate): New features use unified orchestrator
2. **Phase 2** (1 week): Migrate existing routes
3. **Phase 3** (2 weeks): Remove old orchestrators
4. **Phase 4** (3 weeks): Clean up unused dependencies

## Support

For questions about migration:
1. Check this guide
2. Review the LangGraph orchestrator code
3. Run the migration validation script: `python scripts/validate_orchestrator_migration.py`