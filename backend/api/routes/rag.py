"""
RAG API Routes
Endpoints for interacting with the RAG system
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

# TODO: Fix imports after resolving circular dependencies
# from services.rag.rag_system import RAGSystem
# from services.orchestrator import orchestrator
orchestrator = None  # Temporary placeholder

router = APIRouter(prefix="/rag", tags=["RAG"])

class RAGQuery(BaseModel):
    """RAG query request"""
    query: str
    filters: Optional[Dict[str, Any]] = None
    max_results: int = 5

class RAGDocument(BaseModel):
    """Document for ingestion"""
    id: Optional[str] = None
    content: str
    metadata: Dict[str, Any] = {}

class RAGIngestionRequest(BaseModel):
    """Batch document ingestion request"""
    documents: List[RAGDocument]

class RAGResponse(BaseModel):
    """RAG query response"""
    query: str
    analysis: str
    recommendation: str
    risk_factors: List[str]
    confidence: float
    sources: List[str]
    timestamp: str

@router.post("/query", response_model=RAGResponse)
async def query_rag_system(request: RAGQuery):
    """
    Query the RAG system for trading insights
    
    This endpoint allows you to ask questions about:
    - Market conditions
    - Trading strategies
    - Risk analysis
    - Historical patterns
    """
    try:
        # Check if orchestrator is available
        if not orchestrator:
            raise HTTPException(status_code=503, detail="RAG system not available")
            
        # Get RAG system from orchestrator
        rag_system = orchestrator.rag_system
        
        # Process query
        insights = await rag_system.process_query(request.query)
        
        # Format response
        response = RAGResponse(
            query=request.query,
            analysis=insights.get("analysis", ""),
            recommendation=insights.get("recommendation", "HOLD"),
            risk_factors=insights.get("risk_factors", []),
            confidence=insights.get("confidence", 0.5),
            sources=insights.get("sources", []),
            timestamp=insights.get("timestamp", datetime.now().isoformat())
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")

@router.post("/ingest")
async def ingest_documents(request: RAGIngestionRequest):
    """
    Ingest documents into the RAG system
    
    Use this to add:
    - Market news
    - Research reports  
    - Trading rules
    - Historical data
    """
    try:
        # Prepare documents
        documents_data = []
        for doc in request.documents:
            doc_data = {
                "id": doc.id or f"doc_{datetime.now().timestamp()}",
                "content": doc.content,
                "metadata": doc.metadata
            }
            documents_data.append(doc_data)
            
        # Ingest into RAG system
        await orchestrator.rag_system.ingest_market_data(documents_data)
        
        return {
            "status": "success",
            "documents_ingested": len(documents_data),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {str(e)}")

@router.get("/status")
async def get_rag_status():
    """Get RAG system status"""
    try:
        vector_store = orchestrator.rag_system.vector_store
        
        return {
            "status": "active",
            "documents_count": len(vector_store.documents),
            "last_query": None,  # TODO: Track last query
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get RAG status: {str(e)}")

@router.post("/examples")
async def get_example_queries():
    """Get example RAG queries for testing"""
    
    examples = [
        {
            "query": "What is the current trading setup for AAPL?",
            "description": "Get comprehensive analysis for a specific symbol"
        },
        {
            "query": "What are the key risk factors in the current market?",
            "description": "Understand market-wide risks"
        },
        {
            "query": "Explain the momentum trading strategy",
            "description": "Learn about specific trading strategies"
        },
        {
            "query": "What technical indicators suggest a bullish trend?",
            "description": "Technical analysis insights"
        }
    ]
    
    return {"examples": examples}