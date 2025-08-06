"""
Tests for market data endpoints
"""

import pytest
from fastapi.testclient import TestClient
from app import app
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime

client = TestClient(app)


class TestMarketData:
    """Test market data endpoints"""
    
    @patch('services.market_data_service.market_data_service.get_quote')
    async def test_get_quote(self, mock_get_quote):
        """Test getting a single quote"""
        mock_get_quote.return_value = {
            "symbol": "AAPL",
            "price": 150.25,
            "previousClose": 149.50,
            "change": 0.75,
            "changePercent": 0.5,
            "volume": 75000000,
            "timestamp": datetime.now().isoformat()
        }
        
        response = client.get("/api/v1/market-data/quote/AAPL")
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "AAPL"
        assert data["price"] == 150.25
        assert "timestamp" in data
    
    @patch('services.market_data_service.market_data_service.get_quotes')
    async def test_get_multiple_quotes(self, mock_get_quotes):
        """Test getting multiple quotes"""
        mock_get_quotes.return_value = [
            {
                "symbol": "AAPL",
                "price": 150.25,
                "changePercent": 0.5
            },
            {
                "symbol": "GOOGL",
                "price": 2800.50,
                "changePercent": -0.3
            }
        ]
        
        response = client.post(
            "/api/v1/market-data/quotes",
            json={"symbols": ["AAPL", "GOOGL"]}
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert any(q["symbol"] == "AAPL" for q in data)
        assert any(q["symbol"] == "GOOGL" for q in data)
    
    def test_get_quotes_no_symbols(self):
        """Test getting quotes with no symbols"""
        response = client.post(
            "/api/v1/market-data/quotes",
            json={"symbols": []}
        )
        assert response.status_code == 422  # Validation error
    
    def test_get_quotes_too_many_symbols(self):
        """Test getting quotes with too many symbols"""
        symbols = [f"SYM{i}" for i in range(51)]  # 51 symbols
        response = client.post(
            "/api/v1/market-data/quotes",
            json={"symbols": symbols}
        )
        assert response.status_code == 400
        assert "Maximum 50 symbols" in response.json()["detail"]
    
    @patch('services.market_data_service.market_data_service.get_historical_data')
    async def test_get_historical_data(self, mock_get_historical):
        """Test getting historical data"""
        # Create mock DataFrame
        df = pd.DataFrame({
            'Open': [150.0, 151.0],
            'High': [152.0, 153.0],
            'Low': [149.0, 150.0],
            'Close': [151.0, 152.0],
            'Volume': [1000000, 1100000]
        }, index=pd.date_range('2024-01-01', periods=2))
        
        mock_get_historical.return_value = df
        
        response = client.get(
            "/api/v1/market-data/historical/AAPL",
            params={"period": "1mo", "interval": "1d"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "AAPL"
        assert data["period"] == "1mo"
        assert data["interval"] == "1d"
        assert len(data["data"]) == 2
    
    @patch('services.market_data_service.market_data_service.get_market_status')
    async def test_get_market_status(self, mock_get_status):
        """Test getting market status"""
        mock_get_status.return_value = {
            "market_open": True,
            "current_time": datetime.now().isoformat(),
            "indices": {
                "S&P 500": {"price": 4500.00, "change": 25.00},
                "Dow Jones": {"price": 35000.00, "change": 150.00}
            }
        }
        
        response = client.get("/api/v1/market-data/market-status")
        assert response.status_code == 200
        data = response.json()
        assert "market_open" in data
        assert "current_time" in data
        assert "indices" in data
    
    @patch('services.market_data_service.market_data_service.search_symbols')
    async def test_search_symbols(self, mock_search):
        """Test symbol search"""
        mock_search.return_value = [
            {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ"},
            {"symbol": "APLE", "name": "Apple Hospitality REIT", "exchange": "NYSE"}
        ]
        
        response = client.get(
            "/api/v1/market-data/search",
            params={"query": "apple", "limit": 10}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "apple"
        assert len(data["results"]) == 2
        assert data["count"] == 2
    
    def test_search_symbols_empty_query(self):
        """Test symbol search with empty query"""
        response = client.get(
            "/api/v1/market-data/search",
            params={"query": "", "limit": 10}
        )
        assert response.status_code == 422  # Validation error
    
    def test_search_symbols_invalid_limit(self):
        """Test symbol search with invalid limit"""
        response = client.get(
            "/api/v1/market-data/search",
            params={"query": "apple", "limit": 100}  # Over max limit
        )
        assert response.status_code == 422  # Validation error
    
    @patch('services.market_data_service.market_data_service.get_cache_stats')
    def test_get_cache_stats(self, mock_cache_stats):
        """Test getting cache statistics"""
        mock_cache_stats.return_value = {
            "cached_items": 10,
            "cache_keys": ["quote_AAPL", "quote_GOOGL"],
            "ttl_seconds": 60
        }
        
        response = client.get("/api/v1/market-data/cache-stats")
        assert response.status_code == 200
        data = response.json()
        assert data["cached_items"] == 10
        assert "cache_keys" in data
        assert data["ttl_seconds"] == 60