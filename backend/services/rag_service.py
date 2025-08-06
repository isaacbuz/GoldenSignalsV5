"""
Enhanced RAG Service with Proper LLM Integration
Implements production-ready RAG with multiple LLM providers and vector databases
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
import numpy as np
from collections import defaultdict

# LLM Providers
import openai
from anthropic import AsyncAnthropic

# Vector Database
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

# Internal imports
from core.config import Settings
from core.logging import logger

logger = logger
settings = Settings()


class EmbeddingModel(Enum):
    """Available embedding models"""
    SENTENCE_TRANSFORMER = "sentence-transformers/all-MiniLM-L6-v2"
    OPENAI_ADA = "text-embedding-ada-002"
    OPENAI_3_SMALL = "text-embedding-3-small"
    OPENAI_3_LARGE = "text-embedding-3-large"


class LLMProvider(Enum):
    """Available LLM providers"""
    OPENAI_GPT4 = "gpt-4-turbo-preview"
    OPENAI_GPT35 = "gpt-3.5-turbo"
    ANTHROPIC_CLAUDE = "claude-3-opus-20240229"
    ANTHROPIC_CLAUDE_SONNET = "claude-3-sonnet-20240229"


@dataclass
class Document:
    """Enhanced document with comprehensive metadata"""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "unknown"
    symbol: Optional[str] = None
    document_type: str = "general"  # news, analysis, report, social, technical
    relevance_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "symbol": self.symbol,
            "document_type": self.document_type,
            "relevance_score": self.relevance_score
        }


@dataclass
class RAGContext:
    """Context for RAG generation"""
    query: str
    documents: List[Document]
    market_data: Optional[Dict[str, Any]] = None
    user_preferences: Optional[Dict[str, Any]] = None
    max_tokens: int = 2000
    temperature: float = 0.7


class EnhancedRAGService:
    """
    Production-ready RAG service with real LLM integration
    Supports multiple embedding models and LLM providers
    """
    
    def __init__(self):
        # Initialize LLM clients
        self.openai_client = None
        self.anthropic_client = None
        
        if settings.OPENAI_API_KEY:
            openai.api_key = settings.OPENAI_API_KEY
            self.openai_client = openai
            logger.info("OpenAI client initialized")
        
        if settings.ANTHROPIC_API_KEY:
            self.anthropic_client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
            logger.info("Anthropic client initialized")
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(EmbeddingModel.SENTENCE_TRANSFORMER.value)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client(ChromaSettings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./vector_db",
            anonymized_telemetry=False
        ))
        
        # Get or create collections
        self.market_collection = self.chroma_client.get_or_create_collection(
            name="market_intelligence",
            metadata={"description": "Market data and analysis"}
        )
        
        self.news_collection = self.chroma_client.get_or_create_collection(
            name="news_sentiment",
            metadata={"description": "News and sentiment data"}
        )
        
        # Simple cache implementation (could be enhanced with Redis)
        self.cache = {}
        
        # Performance metrics
        self.metrics = {
            "queries_processed": 0,
            "documents_ingested": 0,
            "cache_hits": 0,
            "llm_calls": 0,
            "average_response_time": 0.0
        }
        
        logger.info("Enhanced RAG Service initialized")
    
    async def query(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        use_llm: bool = True,
        provider: LLMProvider = LLMProvider.OPENAI_GPT35
    ) -> List[Dict[str, Any]]:
        """
        Query the RAG system
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            filters: Optional filters for document retrieval
            use_llm: Whether to use LLM for response generation
            provider: LLM provider to use
            
        Returns:
            List of relevant contexts or generated response
        """
        start_time = datetime.now()
        
        # Check cache
        cache_key = f"rag:{hashlib.md5(f'{query}{k}{filters}'.encode()).hexdigest()}"
        cached = self.cache.get(cache_key)
        if cached:
            self.metrics["cache_hits"] += 1
            return cached
        
        try:
            # Step 1: Generate query embedding
            query_embedding = await self._generate_embedding(query)
            
            # Step 2: Retrieve relevant documents
            documents = await self._retrieve_documents(
                query_embedding,
                k=k,
                filters=filters
            )
            
            if not documents:
                logger.warning(f"No documents found for query: {query}")
                return []
            
            # Step 3: Generate response using LLM (if enabled)
            if use_llm and (self.openai_client or self.anthropic_client):
                context = RAGContext(
                    query=query,
                    documents=documents
                )
                
                response = await self._generate_response(context, provider)
                result = [{
                    "response": response,
                    "sources": [doc.to_dict() for doc in documents],
                    "provider": provider.value
                }]
            else:
                # Return raw documents
                result = [doc.to_dict() for doc in documents]
            
            # Cache result (simple in-memory cache)
            self.cache[cache_key] = result
            
            # Update metrics
            self.metrics["queries_processed"] += 1
            elapsed = (datetime.now() - start_time).total_seconds()
            self.metrics["average_response_time"] = (
                (self.metrics["average_response_time"] * (self.metrics["queries_processed"] - 1) + elapsed) /
                self.metrics["queries_processed"]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return []
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            # Use sentence transformer (local, fast)
            embedding = self.embedding_model.encode(text).tolist()
            return embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []
    
    async def _retrieve_documents(
        self,
        query_embedding: List[float],
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Retrieve relevant documents from vector store"""
        try:
            # Build where clause for filters
            where_clause = {}
            if filters:
                if "symbol" in filters:
                    where_clause["symbol"] = filters["symbol"]
                if "document_type" in filters:
                    where_clause["document_type"] = filters["document_type"]
                if "source" in filters:
                    where_clause["source"] = filters["source"]
            
            # Query market collection
            market_results = self.market_collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_clause if where_clause else None
            )
            
            # Query news collection
            news_results = self.news_collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_clause if where_clause else None
            )
            
            # Combine and rank results
            documents = []
            
            # Process market results
            if market_results["documents"]:
                for i in range(len(market_results["documents"][0])):
                    doc = Document(
                        id=market_results["ids"][0][i],
                        content=market_results["documents"][0][i],
                        metadata=market_results["metadatas"][0][i] if market_results["metadatas"] else {},
                        relevance_score=1 - market_results["distances"][0][i],  # Convert distance to similarity
                        source="market_intelligence",
                        document_type=market_results["metadatas"][0][i].get("document_type", "analysis") if market_results["metadatas"] else "analysis"
                    )
                    documents.append(doc)
            
            # Process news results
            if news_results["documents"]:
                for i in range(len(news_results["documents"][0])):
                    doc = Document(
                        id=news_results["ids"][0][i],
                        content=news_results["documents"][0][i],
                        metadata=news_results["metadatas"][0][i] if news_results["metadatas"] else {},
                        relevance_score=1 - news_results["distances"][0][i],
                        source="news_sentiment",
                        document_type="news"
                    )
                    documents.append(doc)
            
            # Sort by relevance and take top k
            documents.sort(key=lambda x: x.relevance_score, reverse=True)
            return documents[:k]
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []
    
    async def _generate_response(
        self,
        context: RAGContext,
        provider: LLMProvider
    ) -> str:
        """Generate response using LLM"""
        try:
            # Prepare context documents
            context_text = "\n\n".join([
                f"[Source: {doc.source}]\n{doc.content}"
                for doc in context.documents
            ])
            
            # Create prompt
            system_prompt = """You are an expert financial analyst and trading advisor. 
            Analyze the provided context and answer the query with specific, actionable insights.
            Focus on:
            1. Market trends and patterns
            2. Trading opportunities and risks
            3. Technical and fundamental factors
            4. Confidence levels and reasoning
            
            Be concise but comprehensive. Cite sources when making claims."""
            
            user_prompt = f"""Context Documents:
            {context_text}
            
            Market Query: {context.query}
            
            Provide a comprehensive analysis and trading recommendation."""
            
            # Generate response based on provider
            if provider in [LLMProvider.OPENAI_GPT4, LLMProvider.OPENAI_GPT35] and self.openai_client:
                response = await self._generate_openai_response(
                    system_prompt,
                    user_prompt,
                    provider.value,
                    context.max_tokens,
                    context.temperature
                )
            elif provider in [LLMProvider.ANTHROPIC_CLAUDE, LLMProvider.ANTHROPIC_CLAUDE_SONNET] and self.anthropic_client:
                response = await self._generate_anthropic_response(
                    system_prompt,
                    user_prompt,
                    provider.value,
                    context.max_tokens,
                    context.temperature
                )
            else:
                # Fallback to simple concatenation
                response = self._generate_fallback_response(context)
            
            self.metrics["llm_calls"] += 1
            return response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return self._generate_fallback_response(context)
    
    async def _generate_openai_response(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate response using OpenAI"""
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
    
    async def _generate_anthropic_response(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate response using Anthropic"""
        try:
            response = await self.anthropic_client.messages.create(
                model=model,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise
    
    def _generate_fallback_response(self, context: RAGContext) -> str:
        """Generate fallback response without LLM"""
        response_parts = [
            f"Analysis for: {context.query}",
            "\nRelevant Information Found:"
        ]
        
        for i, doc in enumerate(context.documents[:3], 1):
            response_parts.append(
                f"\n{i}. [{doc.source}] {doc.content[:200]}..."
            )
        
        response_parts.append(
            f"\n\nBased on {len(context.documents)} relevant documents."
        )
        
        return "\n".join(response_parts)
    
    async def ingest_documents(
        self,
        documents: List[Dict[str, Any]],
        collection_name: str = "market_intelligence"
    ) -> int:
        """
        Ingest documents into the vector store
        
        Args:
            documents: List of document dictionaries
            collection_name: Target collection name
            
        Returns:
            Number of documents ingested
        """
        try:
            # Select collection
            collection = (
                self.market_collection if collection_name == "market_intelligence"
                else self.news_collection
            )
            
            # Prepare documents for ingestion
            ids = []
            contents = []
            embeddings = []
            metadatas = []
            
            for doc_data in documents:
                # Generate ID if not provided
                doc_id = doc_data.get("id", hashlib.md5(
                    f"{doc_data.get('content', '')}_{datetime.now().isoformat()}".encode()
                ).hexdigest())
                
                # Generate embedding
                embedding = await self._generate_embedding(doc_data.get("content", ""))
                
                if embedding:
                    ids.append(doc_id)
                    contents.append(doc_data.get("content", ""))
                    embeddings.append(embedding)
                    
                    # Prepare metadata
                    metadata = {
                        "timestamp": datetime.now().isoformat(),
                        "source": doc_data.get("source", "unknown"),
                        "symbol": doc_data.get("symbol"),
                        "document_type": doc_data.get("document_type", "general")
                    }
                    metadata.update(doc_data.get("metadata", {}))
                    metadatas.append(metadata)
            
            if ids:
                # Add to collection
                collection.add(
                    ids=ids,
                    documents=contents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                
                self.metrics["documents_ingested"] += len(ids)
                logger.info(f"Ingested {len(ids)} documents into {collection_name}")
                
                return len(ids)
            
            return 0
            
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            return 0
    
    async def update_market_context(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        analysis_results: Dict[str, Any]
    ):
        """Update RAG with latest market context"""
        try:
            documents = []
            
            # Create market snapshot document
            market_doc = {
                "id": f"market_{symbol}_{datetime.now().timestamp()}",
                "content": f"""Market Update for {symbol}:
                Price: ${market_data.get('price', 'N/A')}
                Volume: {market_data.get('volume', 'N/A')}
                Change: {market_data.get('change_percent', 'N/A')}%
                RSI: {market_data.get('rsi', 'N/A')}
                MACD: {market_data.get('macd', 'N/A')}
                Market Cap: ${market_data.get('market_cap', 'N/A')}
                PE Ratio: {market_data.get('pe_ratio', 'N/A')}
                """,
                "symbol": symbol,
                "document_type": "market_snapshot",
                "source": "real_time_data",
                "metadata": market_data
            }
            documents.append(market_doc)
            
            # Create analysis document
            if analysis_results:
                analysis_doc = {
                    "id": f"analysis_{symbol}_{datetime.now().timestamp()}",
                    "content": f"""Analysis for {symbol}:
                    Signal: {analysis_results.get('signal', 'N/A')}
                    Confidence: {analysis_results.get('confidence', 'N/A')}
                    Pattern: {analysis_results.get('pattern', 'N/A')}
                    Risk Level: {analysis_results.get('risk_level', 'N/A')}
                    Recommendation: {analysis_results.get('recommendation', 'N/A')}
                    """,
                    "symbol": symbol,
                    "document_type": "analysis",
                    "source": "ai_analysis",
                    "metadata": analysis_results
                }
                documents.append(analysis_doc)
            
            # Ingest documents
            await self.ingest_documents(documents)
            
        except Exception as e:
            logger.error(f"Failed to update market context: {e}")
    
    async def semantic_search(
        self,
        query: str,
        collection: str = "all",
        limit: int = 10
    ) -> List[Document]:
        """Perform semantic search across collections"""
        try:
            query_embedding = await self._generate_embedding(query)
            
            all_results = []
            
            if collection in ["all", "market_intelligence"]:
                market_results = self.market_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=limit
                )
                all_results.extend(self._parse_chroma_results(market_results, "market_intelligence"))
            
            if collection in ["all", "news_sentiment"]:
                news_results = self.news_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=limit
                )
                all_results.extend(self._parse_chroma_results(news_results, "news_sentiment"))
            
            # Sort by relevance
            all_results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return all_results[:limit]
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def _parse_chroma_results(self, results: Dict, source: str) -> List[Document]:
        """Parse ChromaDB results into Document objects"""
        documents = []
        
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                doc = Document(
                    id=results["ids"][0][i],
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    relevance_score=1 - results["distances"][0][i],
                    source=source
                )
                documents.append(doc)
        
        return documents
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get RAG service metrics"""
        return {
            **self.metrics,
            "collections": {
                "market_intelligence": self.market_collection.count(),
                "news_sentiment": self.news_collection.count()
            },
            "providers_available": {
                "openai": self.openai_client is not None,
                "anthropic": self.anthropic_client is not None
            }
        }
    
    async def clear_old_documents(self, days: int = 30):
        """Clear documents older than specified days"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Clear from market collection
            self.market_collection.delete(
                where={"timestamp": {"$lt": cutoff_date}}
            )
            
            # Clear from news collection
            self.news_collection.delete(
                where={"timestamp": {"$lt": cutoff_date}}
            )
            
            logger.info(f"Cleared documents older than {days} days")
            
        except Exception as e:
            logger.error(f"Failed to clear old documents: {e}")


# Singleton instance
rag_service = EnhancedRAGService()