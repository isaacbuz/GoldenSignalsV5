"""
RAG API Routes
Endpoints for interacting with the RAG system
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

from services.rag_service import rag_service

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
        # Query the RAG service
        results = await rag_service.query(
            query=request.query,
            k=request.max_results,
            filters=request.filters,
            use_llm=True
        )
        
        if not results:
            raise HTTPException(status_code=404, detail="No relevant information found")
        
        # Extract the generated response and sources
        if results and "response" in results[0]:
            # LLM-generated response
            response_text = results[0]["response"]
            sources = results[0].get("sources", [])
            
            # Simple analysis extraction (could be enhanced with NLP)
            response = RAGResponse(
                query=request.query,
                analysis=response_text,
                recommendation="HOLD",  # Default, could parse from response
                risk_factors=["Market volatility", "Limited data availability"],
                confidence=0.7,  # Default confidence
                sources=[source.get("id", "unknown") for source in sources],
                timestamp=datetime.now().isoformat()
            )
        else:
            # Fallback response from raw documents
            response = RAGResponse(
                query=request.query,
                analysis=f"Found {len(results)} relevant documents related to your query.",
                recommendation="HOLD",
                risk_factors=["Insufficient analysis data"],
                confidence=0.5,
                sources=[result.get("id", "unknown") for result in results],
                timestamp=datetime.now().isoformat()
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
        # Prepare documents for RAG service
        documents_data = []
        for doc in request.documents:
            doc_data = {
                "id": doc.id or f"doc_{datetime.now().timestamp()}",
                "content": doc.content,
                "metadata": doc.metadata,
                "source": doc.metadata.get("source", "user_input"),
                "document_type": doc.metadata.get("document_type", "general")
            }
            documents_data.append(doc_data)
            
        # Ingest into RAG system
        ingested_count = await rag_service.ingest_documents(
            documents_data, 
            collection_name="market_intelligence"
        )
        
        return {
            "status": "success",
            "documents_ingested": ingested_count,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {str(e)}")

@router.get("/status")
async def get_rag_status():
    """Get RAG system status"""
    try:
        # Get metrics from RAG service
        metrics = rag_service.get_metrics()
        
        return {
            "status": "active",
            "metrics": metrics,
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