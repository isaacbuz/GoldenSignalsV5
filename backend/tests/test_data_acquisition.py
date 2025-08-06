"""
Comprehensive tests for data acquisition services
Tests all data sources, fallback mechanisms, and error handling
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# from services.market_data_service import MarketDataService  # TODO: Update to unified service
from services.live_data_provider import LiveDataProvider
from services.enhanced_data_aggregator import EnhancedDataAggregator
from models.market_data import LiveQuote


class TestMarketDataService:
    """Test the primary market data service"""
    
    @pytest.fixture
    def market_service(self, mock_redis):
        """Create market data service with mocked Redis"""
        with patch('services.market_data_service.get_redis_client', return_value=mock_redis):
            return MarketDataService()
    
    @pytest.mark.asyncio
    async def test_get_quote_success(self, market_service, mock_yfinance):
        """Test successful quote retrieval"""
        with patch('yfinance.Ticker', return_value=mock_yfinance.Ticker()):
            quote = await market_service.get_quote("AAPL")
            
            assert quote is not None
            assert quote["symbol"] == "AAPL"
            assert quote["price"] == 150.25
            assert quote["volume"] == 75000000
            assert "timestamp" in quote
    
    @pytest.mark.asyncio
    async def test_get_quote_with_cache(self, market_service, mock_redis):
        """Test quote retrieval with caching"""
        # First call - cache miss
        mock_redis.get.return_value = None
        
        with patch('yfinance.Ticker') as mock_ticker:
            ticker = MagicMock()
            ticker.info = {'regularMarketPrice': 150.25}
            mock_ticker.return_value = ticker
            
            quote1 = await market_service.get_quote("AAPL")
            assert mock_redis.set.called
            
            # Second call - cache hit
            cached_data = '{"symbol": "AAPL", "price": 150.25}'
            mock_redis.get.return_value = cached_data.encode()
            
            quote2 = await market_service.get_quote("AAPL")
            assert quote2["price"] == 150.25
    
    @pytest.mark.asyncio
    async def test_get_quote_fallback_on_error(self, market_service):
        """Test fallback behavior when primary source fails"""
        with patch('yfinance.Ticker', side_effect=Exception("Network error")):
            quote = await market_service.get_quote("AAPL")
            assert quote is None  # Should return None on failure
    
    @pytest.mark.asyncio
    async def test_get_historical_data(self, market_service):
        """Test historical data retrieval"""
        # Create mock DataFrame
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        df = pd.DataFrame({
            'Open': np.random.uniform(140, 160, 30),
            'High': np.random.uniform(140, 160, 30),
            'Low': np.random.uniform(140, 160, 30),
            'Close': np.random.uniform(140, 160, 30),
            'Volume': np.random.randint(50000000, 100000000, 30)
        }, index=dates)
        
        with patch('yfinance.Ticker') as mock_ticker:
            ticker = MagicMock()
            ticker.history.return_value = df
            mock_ticker.return_value = ticker
            
            historical = await market_service.get_historical_data("AAPL", period="1mo")
            
            assert historical is not None
            assert len(historical) == 30
            assert all(col in historical.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
    
    @pytest.mark.asyncio
    async def test_get_intraday_data(self, market_service):
        """Test intraday data retrieval"""
        # Create mock intraday DataFrame
        times = pd.date_range(start=datetime.now().replace(hour=9, minute=30), 
                             periods=390, freq='1min')
        df = pd.DataFrame({
            'Open': np.random.uniform(149, 151, 390),
            'High': np.random.uniform(149, 151, 390),
            'Low': np.random.uniform(149, 151, 390),
            'Close': np.random.uniform(149, 151, 390),
            'Volume': np.random.randint(100000, 500000, 390)
        }, index=times)
        
        with patch('yfinance.download', return_value=df):
            intraday = await market_service.get_intraday_data("AAPL")
            
            assert intraday is not None
            assert len(intraday) > 0
            assert all(col in intraday.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])


class TestLiveDataProvider:
    """Test the live data provider with multiple sources"""
    
    @pytest.fixture
    def live_provider(self):
        """Create live data provider"""
        return LiveDataProvider()
    
    @pytest.mark.asyncio
    async def test_get_live_quote_yfinance(self, live_provider):
        """Test live quote from yfinance"""
        with patch('yfinance.Ticker') as mock_ticker:
            ticker = MagicMock()
            ticker.info = {
                'regularMarketPrice': 150.25,
                'bid': 150.20,
                'ask': 150.30,
                'bidSize': 100,
                'askSize': 100,
                'regularMarketVolume': 75000000
            }
            mock_ticker.return_value = ticker
            
            quote = await live_provider.get_live_quote("AAPL")
            
            assert isinstance(quote, LiveQuote)
            assert quote.symbol == "AAPL"
            assert quote.price == 150.25
            assert quote.bid == 150.20
            assert quote.ask == 150.30
            assert quote.source == "yfinance"
    
    @pytest.mark.asyncio
    async def test_get_live_quote_finnhub_fallback(self, live_provider, mock_finnhub_client):
        """Test fallback to Finnhub when yfinance fails"""
        # Mock yfinance failure
        with patch('yfinance.Ticker', side_effect=Exception("yfinance error")):
            # Mock Finnhub success
            with patch('finnhub.Client', return_value=mock_finnhub_client):
                quote = await live_provider.get_live_quote("AAPL")
                
                assert isinstance(quote, LiveQuote)
                assert quote.price == 150.25
                assert quote.source == "finnhub"
    
    @pytest.mark.asyncio
    async def test_get_live_quote_all_sources_fail(self, live_provider):
        """Test behavior when all data sources fail"""
        with patch('yfinance.Ticker', side_effect=Exception("yfinance error")):
            with patch('finnhub.Client', side_effect=Exception("finnhub error")):
                with patch('aiohttp.ClientSession') as mock_session:
                    mock_session.return_value.__aenter__.return_value.get.side_effect = Exception("API error")
                    
                    quote = await live_provider.get_live_quote("AAPL")
                    assert quote is None
    
    @pytest.mark.asyncio
    async def test_get_aggregated_quote(self, live_provider):
        """Test aggregated quote from multiple sources"""
        # Mock multiple successful sources
        quotes = {
            'yfinance': LiveQuote(
                symbol="AAPL", price=150.25, bid=150.20, ask=150.30,
                volume=75000000, source="yfinance", timestamp=datetime.now()
            ),
            'finnhub': LiveQuote(
                symbol="AAPL", price=150.30, bid=150.25, ask=150.35,
                volume=75000000, source="finnhub", timestamp=datetime.now()
            )
        }
        
        with patch.object(live_provider, '_get_quote_from_yfinance', return_value=quotes['yfinance']):
            with patch.object(live_provider, '_get_quote_from_finnhub', return_value=quotes['finnhub']):
                aggregated = await live_provider.get_aggregated_quote("AAPL")
                
                assert aggregated is not None
                assert "consensus_price" in aggregated
                assert "sources" in aggregated
                assert len(aggregated["sources"]) >= 2
                # Consensus price should be average
                assert aggregated["consensus_price"] == pytest.approx(150.275, 0.01)


class TestEnhancedDataAggregator:
    """Test the enhanced data aggregator"""
    
    @pytest.fixture
    async def aggregator(self):
        """Create enhanced data aggregator"""
        aggregator = EnhancedDataAggregator()
        await aggregator.initialize()
        yield aggregator
        await aggregator.close()
    
    @pytest.mark.asyncio
    async def test_get_comprehensive_market_data(self, aggregator, mock_market_data):
        """Test comprehensive market data aggregation"""
        with patch.object(aggregator.market_service, 'get_quote', return_value=mock_market_data):
            with patch.object(aggregator.live_provider, 'get_aggregated_quote', 
                            return_value={"consensus_price": 150.25, "sources": ["yfinance", "finnhub"]}):
                
                data = await aggregator.get_comprehensive_market_data("AAPL")
                
                assert data is not None
                assert data["symbol"] == "AAPL"
                assert "price" in data
                assert "consensus_price" in data
                assert "sources" in data
                assert "metadata" in data
    
    @pytest.mark.asyncio
    async def test_get_market_data_batch(self, aggregator, sample_symbols):
        """Test batch market data retrieval"""
        mock_quotes = []
        for symbol in sample_symbols:
            mock_quotes.append({
                "symbol": symbol,
                "price": 100 + len(symbol) * 10,  # Different price per symbol
                "volume": 50000000
            })
        
        with patch.object(aggregator.market_service, 'get_quotes', return_value=mock_quotes):
            batch_data = await aggregator.get_market_data_batch(sample_symbols)
            
            assert len(batch_data) == len(sample_symbols)
            for symbol, data in batch_data.items():
                assert data["symbol"] == symbol
                assert "price" in data
    
    @pytest.mark.asyncio
    async def test_data_source_fallback_chain(self, aggregator):
        """Test complete fallback chain for data sources"""
        symbol = "AAPL"
        
        # Simulate progressive failures
        with patch.object(aggregator.market_service, 'get_quote', side_effect=[None, None, {"price": 150.25}]):
            with patch.object(aggregator.live_provider, 'get_live_quote', 
                            return_value=LiveQuote(symbol=symbol, price=150.30, source="finnhub", timestamp=datetime.now())):
                
                # First call should fallback to live provider
                data = await aggregator.get_comprehensive_market_data(symbol)
                assert data is not None
                assert data["price"] == 150.30
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, aggregator, sample_symbols):
        """Test rate limiting for API calls"""
        # This test ensures we don't exceed rate limits
        import time
        
        start_time = time.time()
        
        # Make multiple rapid requests
        tasks = []
        for _ in range(10):
            for symbol in sample_symbols[:2]:  # Use only 2 symbols
                tasks.append(aggregator.get_comprehensive_market_data(symbol))
        
        with patch.object(aggregator.market_service, 'get_quote', return_value={"price": 150.25}):
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        elapsed_time = time.time() - start_time
        
        # Should have some rate limiting in place
        assert elapsed_time > 0.1  # At least some delay
        # Check that we got results (not all errors)
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) > 0


class TestDataValidation:
    """Test data validation and sanitization"""
    
    @pytest.mark.asyncio
    async def test_validate_quote_data(self):
        """Test quote data validation"""
        from services.data_validator import validate_quote
        
        # Valid quote
        valid_quote = {
            "symbol": "AAPL",
            "price": 150.25,
            "volume": 75000000,
            "timestamp": datetime.now().isoformat()
        }
        assert validate_quote(valid_quote) is True
        
        # Invalid quotes
        invalid_quotes = [
            {"symbol": "AAPL", "price": -10},  # Negative price
            {"symbol": "AAPL", "price": 0},     # Zero price
            {"symbol": "", "price": 150.25},    # Empty symbol
            {"price": 150.25},                  # Missing symbol
            {"symbol": "AAPL"},                 # Missing price
        ]
        
        for quote in invalid_quotes:
            assert validate_quote(quote) is False
    
    @pytest.mark.asyncio
    async def test_sanitize_market_data(self):
        """Test market data sanitization"""
        from services.data_validator import sanitize_market_data
        
        raw_data = {
            "symbol": "  AAPL  ",  # Whitespace
            "price": "150.25",     # String instead of float
            "volume": "75,000,000", # Formatted number
            "extra_field": "remove_me",
            "timestamp": None
        }
        
        sanitized = sanitize_market_data(raw_data)
        
        assert sanitized["symbol"] == "AAPL"
        assert isinstance(sanitized["price"], float)
        assert sanitized["price"] == 150.25
        assert isinstance(sanitized["volume"], int)
        assert sanitized["volume"] == 75000000
        assert "extra_field" not in sanitized
        assert "timestamp" in sanitized  # Should add timestamp if missing


@pytest.mark.performance
class TestDataAcquisitionPerformance:
    """Performance tests for data acquisition"""
    
    @pytest.mark.asyncio
    async def test_quote_retrieval_performance(self, performance_timer):
        """Test quote retrieval performance"""
        service = MarketDataService()
        
        performance_timer.start()
        
        # Get quotes for 10 symbols
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", 
                  "META", "AMZN", "NFLX", "AMD", "INTC"]
        
        with patch('yfinance.Ticker') as mock_ticker:
            ticker = MagicMock()
            ticker.info = {'regularMarketPrice': 150.25}
            mock_ticker.return_value = ticker
            
            tasks = [service.get_quote(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks)
        
        performance_timer.stop()
        
        # Should complete within reasonable time
        assert performance_timer.elapsed() < 2.0  # 2 seconds for 10 quotes
        assert all(r is not None for r in results)
    
    @pytest.mark.asyncio
    async def test_concurrent_data_requests(self):
        """Test handling of concurrent data requests"""
        aggregator = EnhancedDataAggregator()
        await aggregator.initialize()
        
        # Create 50 concurrent requests
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"] * 10
        
        with patch.object(aggregator.market_service, 'get_quote', 
                         return_value={"price": 150.25}):
            
            tasks = [aggregator.get_comprehensive_market_data(s) for s in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Should handle all requests without errors
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) == len(symbols)
        
        await aggregator.close()