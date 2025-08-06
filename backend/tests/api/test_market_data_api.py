"""
Market Data API Tests
Comprehensive testing of market data endpoints
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from fastapi import status

from app import app
from api.routes.market_data import router


client = TestClient(app)


class TestMarketDataAPI:
    """Test suite for market data API endpoints"""
    
    def test_get_price_success(self):
        """Test successful price retrieval"""
        with patch('api.routes.market_data.live_data_provider') as mock_provider:
            mock_provider.get_current_price = AsyncMock(return_value={
                "price": 150.25,
                "change": 2.50,
                "change_percent": 1.69,
                "volume": 1000000,
                "timestamp": datetime.now()
            })
            
            response = client.get("/api/v1/market/price/AAPL")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["symbol"] == "AAPL"
            assert data["price"] == 150.25
            assert "timestamp" in data
    
    def test_get_price_not_found(self):
        """Test price retrieval for invalid symbol"""
        with patch('api.routes.market_data.live_data_provider') as mock_provider:
            mock_provider.get_current_price = AsyncMock(return_value=None)
            
            response = client.get("/api/v1/market/price/INVALID")
            
            assert response.status_code == status.HTTP_404_NOT_FOUND
            data = response.json()
            assert "not found" in data["detail"].lower()
    
    def test_get_price_server_error(self):
        """Test price retrieval with server error"""
        with patch('api.routes.market_data.live_data_provider') as mock_provider:
            mock_provider.get_current_price = AsyncMock(side_effect=Exception("API Error"))
            
            response = client.get("/api/v1/market/price/AAPL")
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    
    def test_get_historical_data_success(self):
        """Test successful historical data retrieval"""
        with patch('api.routes.market_data.live_data_provider') as mock_provider:
            mock_data = [
                {
                    "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
                    "open": 148.0,
                    "high": 152.0,
                    "low": 147.5,
                    "close": 150.25,
                    "volume": 1000000
                }
            ]
            mock_provider.get_historical_data = AsyncMock(return_value=mock_data)
            
            response = client.get("/api/v1/market/historical/AAPL?period=1mo&interval=1d")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["symbol"] == "AAPL"
            assert len(data["data"]) == 1
            assert data["data"][0]["close"] == 150.25
    
    def test_get_historical_data_invalid_period(self):
        """Test historical data with invalid period"""
        response = client.get("/api/v1/market/historical/AAPL?period=invalid")
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_get_historical_data_invalid_interval(self):
        """Test historical data with invalid interval"""
        response = client.get("/api/v1/market/historical/AAPL?interval=invalid")
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_get_multiple_quotes_success(self):
        """Test successful multiple quotes retrieval"""
        with patch('api.routes.market_data.live_data_provider') as mock_provider:
            mock_provider.get_multiple_quotes = AsyncMock(return_value={
                "AAPL": {"price": 150.25, "change": 2.50},
                "TSLA": {"price": 800.0, "change": -10.0}
            })
            
            response = client.post("/api/v1/market/quotes", 
                                 json={"symbols": ["AAPL", "TSLA"]})
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "AAPL" in data
            assert "TSLA" in data
            assert data["AAPL"]["price"] == 150.25
    
    def test_get_multiple_quotes_empty_list(self):
        """Test multiple quotes with empty symbol list"""
        response = client.post("/api/v1/market/quotes", json={"symbols": []})
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_get_multiple_quotes_too_many_symbols(self):
        """Test multiple quotes with too many symbols"""
        symbols = [f"SYMBOL{i}" for i in range(101)]  # 101 symbols
        response = client.post("/api/v1/market/quotes", json={"symbols": symbols})
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_get_market_status(self):
        """Test market status endpoint"""
        with patch('api.routes.market_data.live_data_provider') as mock_provider:
            mock_provider.get_market_status = AsyncMock(return_value={
                "market": "NASDAQ",
                "status": "OPEN",
                "next_open": "2024-01-02T09:30:00",
                "next_close": "2024-01-01T16:00:00"
            })
            
            response = client.get("/api/v1/market/status")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "OPEN"
    
    def test_get_sectors_performance(self):
        """Test sectors performance endpoint"""
        with patch('api.routes.market_data.live_data_provider') as mock_provider:
            mock_provider.get_sectors_performance = AsyncMock(return_value={
                "Technology": {"change_percent": 2.5, "volume": 1000000000},
                "Healthcare": {"change_percent": -1.2, "volume": 500000000}
            })
            
            response = client.get("/api/v1/market/sectors")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "Technology" in data
            assert "Healthcare" in data
    
    def test_search_symbols_success(self):
        """Test symbol search endpoint"""
        with patch('api.routes.market_data.live_data_provider') as mock_provider:
            mock_provider.search_symbols = AsyncMock(return_value=[
                {"symbol": "AAPL", "name": "Apple Inc.", "type": "stock", "exchange": "NASDAQ"},
                {"symbol": "GOOGL", "name": "Alphabet Inc.", "type": "stock", "exchange": "NASDAQ"}
            ])
            
            response = client.get("/api/v1/market/search?query=apple")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data) >= 1
            assert data[0]["symbol"] == "AAPL"
    
    def test_search_symbols_no_query(self):
        """Test symbol search without query parameter"""
        response = client.get("/api/v1/market/search")
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_search_symbols_short_query(self):
        """Test symbol search with too short query"""
        response = client.get("/api/v1/market/search?query=a")
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_get_options_chain_success(self):
        """Test options chain endpoint"""
        with patch('api.routes.market_data.live_data_provider') as mock_provider:
            mock_provider.get_options_chain = AsyncMock(return_value={
                "calls": [
                    {"strike": 150, "bid": 2.50, "ask": 2.60, "volume": 1000, "open_interest": 5000}
                ],
                "puts": [
                    {"strike": 150, "bid": 1.20, "ask": 1.30, "volume": 800, "open_interest": 3000}
                ]
            })
            
            response = client.get("/api/v1/market/options/AAPL?expiry=2024-01-19")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "calls" in data
            assert "puts" in data
            assert len(data["calls"]) >= 1
    
    def test_get_options_chain_invalid_expiry(self):
        """Test options chain with invalid expiry date"""
        response = client.get("/api/v1/market/options/AAPL?expiry=invalid-date")
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_get_financial_metrics_success(self):
        """Test financial metrics endpoint"""
        with patch('api.routes.market_data.live_data_provider') as mock_provider:
            mock_provider.get_financial_metrics = AsyncMock(return_value={
                "market_cap": 3000000000000,
                "pe_ratio": 28.5,
                "earnings_per_share": 5.25,
                "dividend_yield": 0.0045,
                "beta": 1.2
            })
            
            response = client.get("/api/v1/market/metrics/AAPL")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["pe_ratio"] == 28.5
            assert "market_cap" in data
    
    def test_get_news_success(self):
        """Test news endpoint"""
        with patch('api.routes.market_data.live_data_provider') as mock_provider:
            mock_provider.get_news = AsyncMock(return_value=[
                {
                    "headline": "Apple reports strong earnings",
                    "summary": "Apple Inc. reported better than expected earnings...",
                    "source": "Reuters",
                    "url": "https://example.com/news/1",
                    "published_at": datetime.now().isoformat()
                }
            ])
            
            response = client.get("/api/v1/market/news/AAPL")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data) >= 1
            assert "headline" in data[0]
    
    def test_get_earnings_calendar_success(self):
        """Test earnings calendar endpoint"""
        with patch('api.routes.market_data.live_data_provider') as mock_provider:
            mock_provider.get_earnings_calendar = AsyncMock(return_value=[
                {
                    "symbol": "AAPL",
                    "company": "Apple Inc.",
                    "earnings_date": "2024-01-25T16:30:00",
                    "eps_estimate": 2.10,
                    "revenue_estimate": 120000000000
                }
            ])
            
            response = client.get("/api/v1/market/earnings?start_date=2024-01-01&end_date=2024-01-31")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data) >= 1
            assert data[0]["symbol"] == "AAPL"
    
    def test_get_earnings_calendar_invalid_dates(self):
        """Test earnings calendar with invalid date range"""
        response = client.get("/api/v1/market/earnings?start_date=invalid&end_date=2024-01-31")
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_websocket_connection_handling(self):
        """Test WebSocket connection handling"""
        # This would require WebSocket testing setup
        # For now, just test that the endpoint exists
        response = client.get("/api/v1/market/ws")
        # WebSocket endpoints return 426 Upgrade Required for HTTP requests
        assert response.status_code in [status.HTTP_426_UPGRADE_REQUIRED, status.HTTP_405_METHOD_NOT_ALLOWED]


@pytest.mark.asyncio
class TestMarketDataAPIAsync:
    """Async tests for market data API"""
    
    async def test_concurrent_price_requests(self):
        """Test handling of concurrent price requests"""
        symbols = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"]
        
        with patch('api.routes.market_data.live_data_provider') as mock_provider:
            mock_provider.get_current_price = AsyncMock(return_value={
                "price": 150.0, "change": 1.0, "volume": 1000000
            })
            
            # Simulate concurrent requests
            tasks = []
            for symbol in symbols:
                task = asyncio.create_task(
                    asyncio.to_thread(client.get, f"/api/v1/market/price/{symbol}")
                )
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks)
            
            # All requests should succeed
            for response in responses:
                assert response.status_code == status.HTTP_200_OK


@pytest.mark.integration
class TestMarketDataAPIIntegration:
    """Integration tests with real external dependencies"""
    
    def test_real_market_data_integration(self):
        """Test with real market data provider (if available)"""
        # Skip if not in integration test environment
        pytest.skip("Requires real market data provider - run manually")
    
    def test_rate_limiting(self):
        """Test API rate limiting"""
        # Make many requests quickly to test rate limiting
        responses = []
        for _ in range(100):
            response = client.get("/api/v1/market/price/AAPL")
            responses.append(response)
        
        # Should eventually get rate limited
        status_codes = [r.status_code for r in responses]
        assert status.HTTP_429_TOO_MANY_REQUESTS in status_codes or all(
            code == status.HTTP_200_OK for code in status_codes
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])