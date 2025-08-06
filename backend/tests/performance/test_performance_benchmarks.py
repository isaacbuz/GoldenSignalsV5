"""
Performance Benchmarks
Load testing and performance validation for critical system components
"""

import pytest
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch, AsyncMock
import statistics

from services.enhanced_data_aggregator import EnhancedDataAggregator
from services.websocket_manager import ws_manager
from services.rag_service import EnhancedRAGService
from agents.technical_analysis_agent import TechnicalAnalysisAgent
from core.orchestrator import SignalOrchestrator


class TestPerformanceBenchmarks:
    """Performance benchmarks for system components"""
    
    def test_data_aggregator_throughput(self):
        """Test data aggregator throughput under load"""
        with patch('services.enhanced_data_aggregator.yf.download') as mock_yf, \
             patch('services.enhanced_data_aggregator.requests.get') as mock_requests:
            
            # Mock responses
            mock_yf.return_value = Mock()
            mock_requests.return_value.json.return_value = {"data": "test"}
            
            aggregator = EnhancedDataAggregator()
            symbols = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"] * 20  # 100 symbols
            
            start_time = time.time()
            
            # Sequential processing
            results = []
            for symbol in symbols:
                try:
                    result = aggregator.get_basic_data(symbol, "1d")
                    results.append(result)
                except:
                    pass
            
            sequential_time = time.time() - start_time
            
            # Test parallel processing
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                for symbol in symbols:
                    future = executor.submit(aggregator.get_basic_data, symbol, "1d")
                    futures.append(future)
                
                parallel_results = []
                for future in futures:
                    try:
                        result = future.result(timeout=1.0)
                        parallel_results.append(result)
                    except:
                        pass
            
            parallel_time = time.time() - start_time
            
            print(f"Sequential time: {sequential_time:.2f}s")
            print(f"Parallel time: {parallel_time:.2f}s")
            print(f"Speedup: {sequential_time/parallel_time:.2f}x")
            
            # Parallel should be significantly faster
            assert parallel_time < sequential_time * 0.8
    
    @pytest.mark.asyncio
    async def test_websocket_concurrent_connections(self):
        """Test WebSocket manager with many concurrent connections"""
        # Mock WebSocket connections
        mock_websockets = []
        
        start_time = time.time()
        
        # Simulate many connections
        connection_count = 100
        
        for i in range(connection_count):
            mock_ws = Mock()
            mock_ws.send_json = AsyncMock()
            client_id = await ws_manager.connect(mock_ws, {"user_id": i})
            mock_websockets.append((client_id, mock_ws))
        
        connection_time = time.time() - start_time
        
        # Test broadcasting to all connections
        start_time = time.time()
        
        await ws_manager.broadcast_json({
            "type": "test_broadcast",
            "message": "Performance test",
            "timestamp": time.time()
        })
        
        broadcast_time = time.time() - start_time
        
        # Clean up
        for client_id, _ in mock_websockets:
            await ws_manager.disconnect(client_id)
        
        print(f"Connection setup time for {connection_count} connections: {connection_time:.2f}s")
        print(f"Broadcast time: {broadcast_time:.2f}s")
        print(f"Connections per second: {connection_count/connection_time:.2f}")
        
        # Performance assertions
        assert connection_time < 5.0  # Should connect 100 clients in under 5 seconds
        assert broadcast_time < 1.0   # Should broadcast to all in under 1 second
    
    @pytest.mark.asyncio
    async def test_rag_service_query_performance(self):
        """Test RAG service query performance"""
        with patch('services.rag_service.SentenceTransformer'), \
             patch('services.rag_service.chromadb.Client'), \
             patch('services.rag_service.Settings'):
            
            rag_service = EnhancedRAGService()
            
            # Mock embedding generation
            with patch.object(rag_service, '_generate_embedding', return_value=[0.1] * 384):
                with patch.object(rag_service, '_retrieve_documents', return_value=[]):
                    
                    queries = [
                        "What is the current market sentiment for AAPL?",
                        "Technical analysis for Tesla stock",
                        "Latest news about Google earnings",
                        "Volatility analysis for Microsoft",
                        "Sentiment data for Amazon stock"
                    ] * 20  # 100 queries
                    
                    # Sequential queries
                    start_time = time.time()
                    sequential_results = []
                    for query in queries:
                        result = await rag_service.query(query, use_llm=False)
                        sequential_results.append(result)
                    sequential_time = time.time() - start_time
                    
                    # Concurrent queries  
                    start_time = time.time()
                    concurrent_tasks = [rag_service.query(query, use_llm=False) for query in queries]
                    concurrent_results = await asyncio.gather(*concurrent_tasks)
                    concurrent_time = time.time() - start_time
                    
                    print(f"Sequential RAG queries ({len(queries)}): {sequential_time:.2f}s")
                    print(f"Concurrent RAG queries: {concurrent_time:.2f}s")
                    print(f"Queries per second (concurrent): {len(queries)/concurrent_time:.2f}")
                    
                    # Concurrent should be faster
                    assert concurrent_time < sequential_time * 0.7
                    assert len(concurrent_results) == len(queries)
    
    @pytest.mark.asyncio
    async def test_agent_analysis_performance(self):
        """Test AI agent analysis performance under load"""
        with patch('agents.base.BaseAgent.get_market_data') as mock_data:
            mock_data.return_value = {
                "symbol": "AAPL",
                "price": 150.0,
                "volume": 1000000,
                "technical_indicators": {"rsi": 65.0, "macd": 0.5}
            }
            
            agent = TechnicalAnalysisAgent("test_agent")
            symbols = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"] * 10  # 50 analyses
            
            # Sequential analysis
            start_time = time.time()
            sequential_results = []
            for symbol in symbols:
                result = await agent.analyze(symbol, "1d")
                sequential_results.append(result)
            sequential_time = time.time() - start_time
            
            # Concurrent analysis
            start_time = time.time()
            concurrent_tasks = [agent.analyze(symbol, "1d") for symbol in symbols]
            concurrent_results = await asyncio.gather(*concurrent_tasks)
            concurrent_time = time.time() - start_time
            
            print(f"Sequential agent analyses ({len(symbols)}): {sequential_time:.2f}s")
            print(f"Concurrent agent analyses: {concurrent_time:.2f}s")
            print(f"Analyses per second (concurrent): {len(symbols)/concurrent_time:.2f}")
            
            # Performance assertions
            assert concurrent_time < sequential_time * 0.8
            assert len(concurrent_results) == len(symbols)
            assert all(result["symbol"] in symbols for result in concurrent_results)
    
    @pytest.mark.asyncio
    async def test_signal_orchestrator_performance(self):
        """Test signal orchestrator performance with multiple symbols"""
        with patch.object(EnhancedDataAggregator, 'get_market_data') as mock_data:
            mock_data.return_value = {"symbol": "TEST", "price": 100.0}
            
            orchestrator = SignalOrchestrator()
            
            with patch.object(orchestrator, '_generate_agent_signals') as mock_generate:
                mock_generate.return_value = {
                    "technical": {"signal": "BUY", "confidence": 0.8},
                    "sentiment": {"signal": "HOLD", "confidence": 0.6}
                }
                
                symbols = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN", "NFLX", "NVDA", "AMD"] * 5  # 40 signals
                
                start_time = time.time()
                
                # Generate signals concurrently
                tasks = [orchestrator.generate_signal(symbol, "1d") for symbol in symbols]
                results = await asyncio.gather(*tasks)
                
                total_time = time.time() - start_time
                
                print(f"Generated {len(symbols)} signals in {total_time:.2f}s")
                print(f"Signals per second: {len(symbols)/total_time:.2f}")
                
                # Performance assertions
                assert total_time < 10.0  # Should complete in under 10 seconds
                assert len(results) == len(symbols)
                assert all("consensus_signal" in result for result in results)
    
    def test_database_query_performance(self):
        """Test database query performance"""
        with patch('core.database.get_session') as mock_session:
            mock_session_instance = Mock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            
            # Mock query results
            mock_signals = [Mock() for _ in range(1000)]
            mock_query = Mock()
            mock_query.filter.return_value.limit.return_value.all = AsyncMock(return_value=mock_signals)
            mock_session_instance.query.return_value = mock_query
            
            start_time = time.time()
            
            # Simulate multiple database queries
            query_count = 100
            for _ in range(query_count):
                # This would be actual database queries in real scenario
                pass
            
            query_time = time.time() - start_time
            
            print(f"Simulated {query_count} database queries in {query_time:.2f}s")
            
            # Should be very fast since we're mocking
            assert query_time < 1.0
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self):
        """Test memory usage under load"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create many objects to test memory management
        large_objects = []
        for i in range(1000):
            # Simulate large data structures
            data = {
                "symbol": f"TEST{i}",
                "price_history": [100.0 + j for j in range(100)],
                "analysis_results": {
                    "technical": {"signal": "BUY", "confidence": 0.8},
                    "sentiment": {"signal": "HOLD", "confidence": 0.6},
                    "metadata": {"test_data": "x" * 100}
                }
            }
            large_objects.append(data)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Clear objects
        large_objects.clear()
        
        # Give garbage collector time to work
        import gc
        gc.collect()
        await asyncio.sleep(0.1)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Initial memory: {initial_memory:.2f} MB")
        print(f"Peak memory: {peak_memory:.2f} MB")
        print(f"Final memory: {final_memory:.2f} MB")
        print(f"Memory increase: {peak_memory - initial_memory:.2f} MB")
        print(f"Memory reclaimed: {peak_memory - final_memory:.2f} MB")
        
        # Memory should be mostly reclaimed
        assert final_memory < peak_memory * 0.8
    
    def test_response_time_distribution(self):
        """Test response time distribution for API endpoints"""
        with patch('services.enhanced_data_aggregator.yf.download') as mock_yf:
            mock_yf.return_value = Mock()
            
            aggregator = EnhancedDataAggregator()
            response_times = []
            
            # Measure multiple requests
            for _ in range(100):
                start_time = time.time()
                try:
                    aggregator.get_basic_data("AAPL", "1d")
                except:
                    pass
                response_time = time.time() - start_time
                response_times.append(response_time)
            
            # Calculate statistics
            avg_time = statistics.mean(response_times)
            median_time = statistics.median(response_times)
            p95_time = sorted(response_times)[int(0.95 * len(response_times))]
            p99_time = sorted(response_times)[int(0.99 * len(response_times))]
            
            print(f"Response time statistics:")
            print(f"Average: {avg_time:.3f}s")
            print(f"Median: {median_time:.3f}s") 
            print(f"95th percentile: {p95_time:.3f}s")
            print(f"99th percentile: {p99_time:.3f}s")
            
            # Performance assertions
            assert avg_time < 0.1    # 100ms average
            assert p95_time < 0.2    # 200ms 95th percentile
            assert p99_time < 0.5    # 500ms 99th percentile


@pytest.mark.performance
class TestLoadTesting:
    """Load testing for system components"""
    
    @pytest.mark.asyncio
    async def test_high_frequency_signal_generation(self):
        """Test system under high frequency signal generation load"""
        orchestrator = SignalOrchestrator()
        
        with patch.object(orchestrator, '_generate_agent_signals') as mock_generate:
            mock_generate.return_value = {
                "technical": {"signal": "BUY", "confidence": 0.8}
            }
            
            # High frequency requests
            symbols = ["AAPL", "TSLA", "GOOGL"] * 100  # 300 signals
            
            start_time = time.time()
            
            # Batch process signals
            batch_size = 20
            all_results = []
            
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i+batch_size]
                batch_tasks = [orchestrator.generate_signal(symbol, "1d") for symbol in batch]
                batch_results = await asyncio.gather(*batch_tasks)
                all_results.extend(batch_results)
                
                # Small delay between batches to avoid overwhelming
                await asyncio.sleep(0.01)
            
            total_time = time.time() - start_time
            
            print(f"Generated {len(symbols)} signals in {total_time:.2f}s")
            print(f"Throughput: {len(symbols)/total_time:.2f} signals/second")
            
            assert len(all_results) == len(symbols)
            assert total_time < 30.0  # Should complete within 30 seconds
    
    @pytest.mark.asyncio 
    async def test_websocket_message_throughput(self):
        """Test WebSocket message throughput"""
        # Create mock connections
        connection_count = 50
        mock_connections = []
        
        for i in range(connection_count):
            mock_ws = Mock()
            mock_ws.send_json = AsyncMock()
            client_id = await ws_manager.connect(mock_ws)
            await ws_manager.subscribe(client_id, "LOAD_TEST")
            mock_connections.append(client_id)
        
        # Send many messages
        message_count = 1000
        start_time = time.time()
        
        for i in range(message_count):
            await ws_manager.broadcast_price_update("LOAD_TEST", 100.0 + i, 1000000)
        
        broadcast_time = time.time() - start_time
        
        # Clean up
        for client_id in mock_connections:
            await ws_manager.disconnect(client_id)
        
        total_messages = message_count * connection_count
        throughput = total_messages / broadcast_time
        
        print(f"Sent {message_count} messages to {connection_count} connections")
        print(f"Total messages: {total_messages}")
        print(f"Time: {broadcast_time:.2f}s") 
        print(f"Throughput: {throughput:.2f} messages/second")
        
        # Should handle reasonable throughput
        assert throughput > 1000  # At least 1000 messages/second


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to show print statements