"""
RAG (Retrieval-Augmented Generation) Core Engine
Provides context-aware signal generation using historical data and market knowledge
"""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import numpy as np
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from redis import asyncio as aioredis

from core.config import settings
from core.logging import get_logger
from core.database import get_db
from models.signal import Signal as SignalModel

logger = get_logger(__name__)


class DocumentType(str, Enum):
    """Types of documents in the RAG system"""
    MARKET_DATA = "market_data"
    SIGNAL_HISTORY = "signal_history"
    NEWS_ARTICLE = "news_article"
    TECHNICAL_PATTERN = "technical_pattern"
    ECONOMIC_INDICATOR = "economic_indicator"
    RESEARCH_REPORT = "research_report"


@dataclass
class Document:
    """Document stored in RAG system"""
    id: str
    type: DocumentType
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class RAGContext:
    """Context retrieved from RAG system"""
    query: str
    retrieved_documents: List[Document]
    relevance_scores: List[float]
    metadata: Dict[str, Any]
    timestamp: datetime
    
    def get_top_k(self, k: int = 3) -> List[Document]:
        """Get top k most relevant documents"""
        sorted_docs = sorted(
            zip(self.retrieved_documents, self.relevance_scores),
            key=lambda x: x[1],
            reverse=True
        )
        return [doc for doc, _ in sorted_docs[:k]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "query": self.query,
            "num_documents": len(self.retrieved_documents),
            "avg_relevance": np.mean(self.relevance_scores) if self.relevance_scores else 0,
            "top_documents": [
                {
                    "type": doc.type,
                    "metadata": doc.metadata,
                    "relevance": score
                }
                for doc, score in zip(self.retrieved_documents[:3], self.relevance_scores[:3])
            ],
            "timestamp": self.timestamp.isoformat()
        }


class RAGEngine:
    """
    Core RAG Engine for enhanced signal generation
    
    Features:
    - Vector similarity search for historical patterns
    - Context-aware signal augmentation
    - Multi-source data integration
    - Caching for performance
    """
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.embeddings = {}  # In-memory storage (use vector DB in production)
        self.documents = {}   # Document storage
        self.redis_client = None
        self.initialized = False
        
        # Configuration
        self.similarity_threshold = 0.7
        self.max_cache_age = timedelta(hours=1)
        self.cache_ttl = 3600  # 1 hour
        
        logger.info(f"RAG Engine initialized with embedding dimension: {embedding_dim}")
    
    async def initialize(self):
        """Initialize RAG components"""
        if self.initialized:
            return
            
        logger.info("Initializing RAG engine components...")
        
        # Initialize Redis for caching
        try:
            self.redis_client = await aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Connected to Redis for RAG caching")
        except Exception as e:
            logger.warning(f"Redis connection failed, using in-memory cache only: {e}")
            self.redis_client = None
        
        # Load existing embeddings if any
        await self._load_embeddings()
        
        self.initialized = True
        logger.info("RAG engine initialization complete")
    
    async def _load_embeddings(self):
        """Load existing embeddings from storage"""
        # In production, load from vector database
        # For now, start with empty embeddings
        logger.info("Starting with fresh embeddings")
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Add documents to RAG system
        
        Args:
            documents: List of documents with content and metadata
            
        Returns:
            Number of documents added
        """
        added_count = 0
        
        for doc_data in documents:
            try:
                # Create document
                doc = Document(
                    id=doc_data.get("id", f"doc_{datetime.utcnow().timestamp()}_{added_count}"),
                    type=DocumentType(doc_data.get("type", DocumentType.MARKET_DATA)),
                    content=doc_data["content"],
                    metadata=doc_data.get("metadata", {})
                )
                
                # Generate embedding
                embedding = await self._generate_embedding(doc.content)
                doc.embedding = embedding
                
                # Store document and embedding
                self.documents[doc.id] = doc
                self.embeddings[doc.id] = embedding
                
                # Cache in Redis if available
                if self.redis_client:
                    await self._cache_document(doc)
                
                added_count += 1
                
            except Exception as e:
                logger.error(f"Failed to add document: {e}")
                continue
        
        logger.info(f"Added {added_count}/{len(documents)} documents to RAG system")
        return added_count
    
    async def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text
        
        In production, use a real embedding model like:
        - sentence-transformers
        - OpenAI embeddings
        - Custom trained embeddings
        """
        # Mock embedding for now
        # In production, use actual embedding model
        np.random.seed(hash(text) % 2**32)
        return np.random.rand(self.embedding_dim)
    
    async def _cache_document(self, doc: Document):
        """Cache document in Redis"""
        try:
            cache_key = f"rag:doc:{doc.id}"
            cache_data = {
                "type": doc.type,
                "content": doc.content,
                "metadata": json.dumps(doc.metadata),
                "timestamp": doc.timestamp.isoformat()
            }
            await self.redis_client.hset(cache_key, mapping=cache_data)
            await self.redis_client.expire(cache_key, self.cache_ttl)
        except Exception as e:
            logger.error(f"Failed to cache document: {e}")
    
    async def retrieve_context(
        self, 
        query: str, 
        k: int = 5,
        doc_types: Optional[List[DocumentType]] = None
    ) -> RAGContext:
        """
        Retrieve relevant context for query
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            doc_types: Filter by document types
            
        Returns:
            RAG context with retrieved documents
        """
        # Check cache first
        cached_context = await self._get_cached_context(query, k, doc_types)
        if cached_context:
            return cached_context
        
        # Generate query embedding
        query_embedding = await self._generate_embedding(query)
        
        # Search for similar documents
        results = await self._search_similar(query_embedding, k, doc_types)
        
        # Create context
        context = RAGContext(
            query=query,
            retrieved_documents=results["documents"],
            relevance_scores=results["scores"],
            metadata={
                "retrieval_time": datetime.utcnow(),
                "num_searched": len(self.embeddings),
                "doc_types": [t.value for t in (doc_types or [])]
            },
            timestamp=datetime.utcnow()
        )
        
        # Cache context
        await self._cache_context(context)
        
        return context
    
    async def _search_similar(
        self, 
        query_embedding: np.ndarray, 
        k: int,
        doc_types: Optional[List[DocumentType]] = None
    ) -> Dict[str, Any]:
        """Search for similar documents using cosine similarity"""
        similarities = []
        
        for doc_id, embedding in self.embeddings.items():
            doc = self.documents.get(doc_id)
            if not doc:
                continue
                
            # Filter by document type if specified
            if doc_types and doc.type not in doc_types:
                continue
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, embedding)
            
            if similarity >= self.similarity_threshold:
                similarities.append((doc, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top k
        top_k = similarities[:k]
        
        if not top_k:
            # Return empty results if no similar documents found
            return {"documents": [], "scores": []}
        
        documents = [doc for doc, _ in top_k]
        scores = [score for _, score in top_k]
        
        return {"documents": documents, "scores": scores}
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)
    
    async def _get_cached_context(
        self, 
        query: str, 
        k: int,
        doc_types: Optional[List[DocumentType]] = None
    ) -> Optional[RAGContext]:
        """Get cached context if available"""
        if not self.redis_client:
            return None
            
        try:
            cache_key = f"rag:context:{hash(query)}:{k}:{doc_types}"
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                # Reconstruct context from cache
                # This is simplified - in production, properly deserialize
                logger.debug(f"Retrieved context from cache for query: {query[:50]}...")
                return None  # Simplified for now
                
        except Exception as e:
            logger.error(f"Failed to get cached context: {e}")
            
        return None
    
    async def _cache_context(self, context: RAGContext):
        """Cache context in Redis"""
        if not self.redis_client:
            return
            
        try:
            cache_key = f"rag:context:{hash(context.query)}:..."
            # Simplified caching - in production, properly serialize
            await self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(context.to_dict())
            )
        except Exception as e:
            logger.error(f"Failed to cache context: {e}")
    
    async def generate_augmented_signal(
        self, 
        base_signal: Dict[str, Any], 
        context: RAGContext
    ) -> Dict[str, Any]:
        """
        Generate signal augmented with RAG context
        
        Args:
            base_signal: Original signal from agents
            context: Retrieved RAG context
            
        Returns:
            Augmented signal with context-aware adjustments
        """
        augmented_signal = base_signal.copy()
        
        # Add RAG context
        augmented_signal["rag_context"] = context.to_dict()
        
        # Calculate context-based confidence adjustment
        if context.relevance_scores:
            avg_relevance = np.mean(context.relevance_scores)
            max_relevance = max(context.relevance_scores)
            
            # Adjust confidence based on context quality
            confidence_multiplier = 1.0
            
            if max_relevance > 0.9:  # Very relevant historical context
                confidence_multiplier = 1.15
                augmented_signal["rag_reasoning"] = "Strong historical pattern match found"
            elif max_relevance > 0.8:  # Good historical context
                confidence_multiplier = 1.1
                augmented_signal["rag_reasoning"] = "Good historical pattern correlation"
            elif max_relevance < 0.6:  # Weak context
                confidence_multiplier = 0.9
                augmented_signal["rag_reasoning"] = "Limited historical context available"
            
            # Apply adjustment
            original_confidence = augmented_signal.get("confidence", 0.5)
            augmented_signal["confidence"] = min(
                original_confidence * confidence_multiplier,
                0.95  # Cap at 95%
            )
            augmented_signal["confidence_adjustment"] = confidence_multiplier
        
        # Extract insights from top documents
        top_docs = context.get_top_k(3)
        if top_docs:
            insights = []
            for doc in top_docs:
                if doc.type == DocumentType.SIGNAL_HISTORY:
                    insights.append(f"Historical signal: {doc.metadata.get('outcome', 'unknown')}")
                elif doc.type == DocumentType.TECHNICAL_PATTERN:
                    insights.append(f"Pattern: {doc.metadata.get('pattern_name', 'unknown')}")
                elif doc.type == DocumentType.NEWS_ARTICLE:
                    insights.append(f"News sentiment: {doc.metadata.get('sentiment', 'neutral')}")
            
            augmented_signal["rag_insights"] = insights
        
        # Add pattern recognition
        if self._detect_patterns(context):
            augmented_signal["patterns_detected"] = True
            augmented_signal["pattern_confidence"] = 0.8
        
        return augmented_signal
    
    def _detect_patterns(self, context: RAGContext) -> bool:
        """Detect if retrieved context contains significant patterns"""
        # Simple pattern detection
        # In production, use more sophisticated pattern matching
        pattern_docs = [
            doc for doc in context.retrieved_documents 
            if doc.type == DocumentType.TECHNICAL_PATTERN
        ]
        return len(pattern_docs) >= 2
    
    async def add_signal_feedback(
        self, 
        signal_id: str, 
        outcome: Dict[str, Any]
    ):
        """
        Add signal outcome as feedback to improve future retrievals
        
        Args:
            signal_id: ID of the signal
            outcome: Signal outcome data
        """
        # Create feedback document
        feedback_doc = {
            "id": f"feedback_{signal_id}",
            "type": DocumentType.SIGNAL_HISTORY,
            "content": f"Signal {signal_id} outcome: {outcome.get('result', 'unknown')}",
            "metadata": {
                "signal_id": signal_id,
                "outcome": outcome.get("result"),
                "profit": outcome.get("profit"),
                "was_correct": outcome.get("was_correct"),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        await self.add_documents([feedback_doc])
        logger.info(f"Added feedback for signal {signal_id}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get RAG engine statistics"""
        doc_type_counts = {}
        for doc in self.documents.values():
            doc_type_counts[doc.type] = doc_type_counts.get(doc.type, 0) + 1
        
        return {
            "total_documents": len(self.documents),
            "total_embeddings": len(self.embeddings),
            "document_types": doc_type_counts,
            "embedding_dimension": self.embedding_dim,
            "initialized": self.initialized
        }