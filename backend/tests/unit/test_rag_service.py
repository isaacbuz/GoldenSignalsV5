"""
Unit Tests for RAG Service
Comprehensive testing of RAG functionality
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
import json

from services.rag_service import EnhancedRAGService, Document, RAGContext, LLMProvider


class TestEnhancedRAGService:
    """Test suite for EnhancedRAGService"""
    
    @pytest.fixture
    def rag_service(self):
        """Create RAG service instance for testing"""
        with patch('services.rag_service.SentenceTransformer'), \
             patch('services.rag_service.chromadb.Client'), \
             patch('services.rag_service.Settings'):
            service = EnhancedRAGService()
            service.cache = {}  # Reset cache for each test
            return service
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing"""
        return [
            Document(
                id="doc1",
                content="AAPL stock is showing bullish momentum with strong volume",
                source="market_intelligence",
                document_type="analysis",
                timestamp=datetime.now()
            ),
            Document(
                id="doc2", 
                content="Tesla earnings beat expectations, stock surging",
                source="news_sentiment",
                document_type="news",
                timestamp=datetime.now()
            )
        ]
    
    @pytest.mark.asyncio
    async def test_query_with_cache_hit(self, rag_service):
        """Test query with cache hit"""
        # Setup
        query = "AAPL analysis"
        cached_result = [{"response": "Cached response"}]
        rag_service.cache["rag:test_key"] = cached_result
        
        with patch('hashlib.md5') as mock_md5:
            mock_md5.return_value.hexdigest.return_value = "test_key"
            
            result = await rag_service.query(query)
            
            assert result == cached_result
            assert rag_service.metrics["cache_hits"] == 1
    
    @pytest.mark.asyncio
    async def test_query_with_llm_response(self, rag_service, sample_documents):
        """Test query with LLM response generation"""
        # Setup
        rag_service.openai_client = Mock()
        
        # Mock embedding generation
        with patch.object(rag_service, '_generate_embedding', return_value=[0.1, 0.2, 0.3]):
            # Mock document retrieval
            with patch.object(rag_service, '_retrieve_documents', return_value=sample_documents):
                # Mock LLM response
                with patch.object(rag_service, '_generate_response', 
                                return_value="AAPL shows strong bullish signals"):
                    
                    result = await rag_service.query("AAPL analysis", use_llm=True)
                    
                    assert len(result) == 1
                    assert "response" in result[0]
                    assert "sources" in result[0]
                    assert result[0]["response"] == "AAPL shows strong bullish signals"
                    assert len(result[0]["sources"]) == 2
    
    @pytest.mark.asyncio
    async def test_query_without_llm(self, rag_service, sample_documents):
        """Test query returning raw documents"""
        # Mock embedding and retrieval
        with patch.object(rag_service, '_generate_embedding', return_value=[0.1, 0.2, 0.3]):
            with patch.object(rag_service, '_retrieve_documents', return_value=sample_documents):
                
                result = await rag_service.query("AAPL analysis", use_llm=False)
                
                assert len(result) == 2
                assert result[0]["id"] == "doc1"
                assert result[1]["id"] == "doc2"
    
    @pytest.mark.asyncio
    async def test_generate_embedding(self, rag_service):
        """Test embedding generation"""
        # Mock sentence transformer
        mock_embedding = [0.1, 0.2, 0.3, 0.4]
        rag_service.embedding_model = Mock()
        rag_service.embedding_model.encode.return_value = Mock()
        rag_service.embedding_model.encode.return_value.tolist.return_value = mock_embedding
        
        result = await rag_service._generate_embedding("test text")
        
        assert result == mock_embedding
        rag_service.embedding_model.encode.assert_called_once_with("test text")
    
    @pytest.mark.asyncio
    async def test_retrieve_documents(self, rag_service):
        """Test document retrieval from vector store"""
        # Mock ChromaDB results
        mock_results = {
            "documents": [["doc1 content", "doc2 content"]],
            "ids": [["doc1", "doc2"]],
            "distances": [[0.1, 0.2]],
            "metadatas": [[{"document_type": "analysis"}, {"document_type": "news"}]]
        }
        
        rag_service.market_collection = Mock()
        rag_service.market_collection.query.return_value = mock_results
        
        rag_service.news_collection = Mock()
        rag_service.news_collection.query.return_value = {"documents": [[]], "ids": [[]], "distances": [[]], "metadatas": [[]]}
        
        query_embedding = [0.1, 0.2, 0.3]
        result = await rag_service._retrieve_documents(query_embedding, k=5)
        
        assert len(result) == 2
        assert result[0].id == "doc1"
        assert result[0].content == "doc1 content"
        assert result[0].relevance_score == 0.9  # 1 - 0.1
        assert result[0].document_type == "analysis"
    
    @pytest.mark.asyncio
    async def test_generate_openai_response(self, rag_service):
        """Test OpenAI response generation"""
        # Mock OpenAI client
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "AI generated response"
        
        rag_service.openai_client = Mock()
        rag_service.openai_client.chat = Mock()
        rag_service.openai_client.chat.completions = Mock()
        rag_service.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = mock_response
            
            result = await rag_service._generate_openai_response(
                "system prompt", "user prompt", "gpt-3.5-turbo", 2000, 0.7
            )
            
            assert result == "AI generated response"
    
    @pytest.mark.asyncio
    async def test_generate_anthropic_response(self, rag_service):
        """Test Anthropic response generation"""
        # Mock Anthropic client
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Claude generated response"
        
        rag_service.anthropic_client = Mock()
        rag_service.anthropic_client.messages = Mock()
        rag_service.anthropic_client.messages.create = AsyncMock(return_value=mock_response)
        
        result = await rag_service._generate_anthropic_response(
            "system prompt", "user prompt", "claude-3-sonnet-20240229", 2000, 0.7
        )
        
        assert result == "Claude generated response"
    
    def test_generate_fallback_response(self, rag_service, sample_documents):
        """Test fallback response generation"""
        context = RAGContext(
            query="test query",
            documents=sample_documents
        )
        
        result = rag_service._generate_fallback_response(context)
        
        assert "test query" in result
        assert "2 relevant documents" in result
        assert "AAPL stock" in result or "Tesla earnings" in result
    
    @pytest.mark.asyncio
    async def test_ingest_documents(self, rag_service):
        """Test document ingestion"""
        # Mock collection
        rag_service.market_collection = Mock()
        rag_service.market_collection.add = Mock()
        
        # Mock embedding generation
        with patch.object(rag_service, '_generate_embedding', return_value=[0.1, 0.2, 0.3]):
            
            documents = [
                {
                    "id": "test_doc",
                    "content": "Test content",
                    "source": "test",
                    "document_type": "analysis"
                }
            ]
            
            result = await rag_service.ingest_documents(documents)
            
            assert result == 1
            rag_service.market_collection.add.assert_called_once()
            assert rag_service.metrics["documents_ingested"] == 1
    
    @pytest.mark.asyncio
    async def test_update_market_context(self, rag_service):
        """Test market context update"""
        with patch.object(rag_service, 'ingest_documents', return_value=2) as mock_ingest:
            
            market_data = {"price": 150.0, "volume": 1000000, "change_percent": 2.5}
            analysis_results = {"signal": "BUY", "confidence": 0.8}
            
            await rag_service.update_market_context("AAPL", market_data, analysis_results)
            
            mock_ingest.assert_called_once()
            args = mock_ingest.call_args[0][0]
            assert len(args) == 2  # market doc + analysis doc
            assert args[0]["symbol"] == "AAPL"
            assert "Price: $150.0" in args[0]["content"]
    
    @pytest.mark.asyncio
    async def test_semantic_search(self, rag_service, sample_documents):
        """Test semantic search across collections"""
        # Mock embedding
        with patch.object(rag_service, '_generate_embedding', return_value=[0.1, 0.2, 0.3]):
            # Mock parse results
            with patch.object(rag_service, '_parse_chroma_results', return_value=sample_documents):
                
                # Mock collections
                rag_service.market_collection = Mock()
                rag_service.market_collection.query.return_value = {"test": "data"}
                rag_service.news_collection = Mock()
                rag_service.news_collection.query.return_value = {"test": "data"}
                
                result = await rag_service.semantic_search("AAPL", collection="all", limit=5)
                
                assert len(result) == 4  # 2 docs from each collection
                assert result[0].relevance_score >= result[1].relevance_score  # Sorted by relevance
    
    def test_get_metrics(self, rag_service):
        """Test metrics retrieval"""
        # Mock collections
        rag_service.market_collection = Mock()
        rag_service.market_collection.count.return_value = 100
        rag_service.news_collection = Mock() 
        rag_service.news_collection.count.return_value = 50
        
        # Set some metrics
        rag_service.metrics["queries_processed"] = 10
        rag_service.metrics["documents_ingested"] = 25
        
        result = rag_service.get_metrics()
        
        assert result["queries_processed"] == 10
        assert result["documents_ingested"] == 25
        assert result["collections"]["market_intelligence"] == 100
        assert result["collections"]["news_sentiment"] == 50
        assert "providers_available" in result
    
    @pytest.mark.asyncio
    async def test_clear_old_documents(self, rag_service):
        """Test clearing old documents"""
        # Mock collections
        rag_service.market_collection = Mock()
        rag_service.market_collection.delete = Mock()
        rag_service.news_collection = Mock()
        rag_service.news_collection.delete = Mock()
        
        await rag_service.clear_old_documents(days=30)
        
        rag_service.market_collection.delete.assert_called_once()
        rag_service.news_collection.delete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, rag_service):
        """Test error handling in various scenarios"""
        # Test embedding generation error
        with patch.object(rag_service.embedding_model, 'encode', side_effect=Exception("Model error")):
            result = await rag_service._generate_embedding("test")
            assert result == []
        
        # Test query error handling
        with patch.object(rag_service, '_generate_embedding', side_effect=Exception("Query error")):
            result = await rag_service.query("test query")
            assert result == []
        
        # Test document ingestion error
        with patch.object(rag_service, '_generate_embedding', side_effect=Exception("Ingest error")):
            result = await rag_service.ingest_documents([{"content": "test"}])
            assert result == 0


@pytest.mark.integration
class TestRAGServiceIntegration:
    """Integration tests for RAG service"""
    
    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Test complete RAG pipeline"""
        # This would test with actual models and databases
        # Skip if external dependencies not available
        pytest.skip("Requires external dependencies - run manually")
    
    @pytest.mark.asyncio 
    async def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        # This would benchmark query response times
        pytest.skip("Performance test - run manually")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])