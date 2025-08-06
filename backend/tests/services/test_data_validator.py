"""
Test cases for DataValidator service
"""

import pytest
from datetime import datetime

from services.data_validator import DataValidator


class TestDataValidator:
    """Test DataValidator functionality"""
    
    def test_validate_quote_valid(self):
        """Test quote validation with valid data"""
        quote = {
            "symbol": "AAPL",
            "price": 150.0,
            "volume": 1000000,
            "bid": 149.95,
            "ask": 150.05,
            "high": 151.0,
            "low": 149.0
        }
        
        assert DataValidator.validate_quote(quote) is True
    
    def test_validate_quote_missing_required_fields(self):
        """Test quote validation with missing required fields"""
        # Missing price
        quote = {"symbol": "AAPL"}
        assert DataValidator.validate_quote(quote) is False
        
        # Missing symbol
        quote = {"price": 150.0}
        assert DataValidator.validate_quote(quote) is False
        
        # None values
        quote = {"symbol": "AAPL", "price": None}
        assert DataValidator.validate_quote(quote) is False
    
    def test_validate_quote_invalid_symbol(self):
        """Test quote validation with invalid symbol"""
        # Empty symbol
        quote = {"symbol": "", "price": 150.0}
        assert DataValidator.validate_quote(quote) is False
        
        # None symbol
        quote = {"symbol": None, "price": 150.0}
        assert DataValidator.validate_quote(quote) is False
        
        # Non-string symbol
        quote = {"symbol": 123, "price": 150.0}
        assert DataValidator.validate_quote(quote) is False
    
    def test_validate_quote_invalid_price(self):
        """Test quote validation with invalid price"""
        # Zero price
        quote = {"symbol": "AAPL", "price": 0}
        assert DataValidator.validate_quote(quote) is False
        
        # Negative price
        quote = {"symbol": "AAPL", "price": -10.0}
        assert DataValidator.validate_quote(quote) is False
        
        # Non-numeric price
        quote = {"symbol": "AAPL", "price": "not_a_number"}
        assert DataValidator.validate_quote(quote) is False
    
    def test_validate_quote_optional_fields(self):
        """Test quote validation with optional fields"""
        # Valid optional fields
        quote = {
            "symbol": "AAPL",
            "price": 150.0,
            "volume": 1000000,
            "change": -2.5,  # Can be negative
            "change_percent": -1.6  # Can be negative
        }
        assert DataValidator.validate_quote(quote) is True
        
        # Invalid optional numeric fields
        quote = {
            "symbol": "AAPL", 
            "price": 150.0,
            "volume": "invalid"
        }
        assert DataValidator.validate_quote(quote) is False
    
    def test_sanitize_market_data_symbol(self):
        """Test market data sanitization for symbols"""
        data = {"symbol": "  aapl  "}
        sanitized = DataValidator.sanitize_market_data(data)
        assert sanitized["symbol"] == "AAPL"
        
        # Remove invalid characters
        data = {"symbol": "A@A#P$L%"}
        sanitized = DataValidator.sanitize_market_data(data)
        assert sanitized["symbol"] == "AAPL"
    
    def test_sanitize_market_data_prices(self):
        """Test market data sanitization for price fields"""
        data = {
            "symbol": "AAPL",
            "price": "150.50",
            "bid": "$149.95",
            "ask": "150,05",  # European format
            "high": 151.0,
            "volume": "1,000,000"
        }
        
        sanitized = DataValidator.sanitize_market_data(data)
        
        assert sanitized["price"] == 150.50
        assert sanitized["bid"] == 149.95
        assert sanitized["ask"] == 150.05
        assert sanitized["high"] == 151.0
        assert sanitized["volume"] == 1000000
    
    def test_sanitize_market_data_invalid_values(self):
        """Test market data sanitization with invalid values"""
        data = {
            "symbol": "AAPL",
            "price": "not_a_number",
            "volume": "invalid_volume"
        }
        
        sanitized = DataValidator.sanitize_market_data(data)
        
        # Invalid price should be skipped
        assert "price" not in sanitized
        
        # Invalid volume should default to 0
        assert sanitized["volume"] == 0
    
    def test_sanitize_market_data_none_values(self):
        """Test market data sanitization with None values"""
        data = {
            "symbol": "AAPL",
            "price": None,
            "volume": None,
            "bid": 149.95
        }
        
        sanitized = DataValidator.sanitize_market_data(data)
        
        assert "price" not in sanitized
        assert sanitized["volume"] == 0
        assert sanitized["bid"] == 149.95
    
    def test_sanitize_market_data_empty_input(self):
        """Test market data sanitization with empty input"""
        sanitized = DataValidator.sanitize_market_data({})
        assert isinstance(sanitized, dict)
        assert len(sanitized) == 0
    
    def test_sanitize_market_data_comprehensive(self):
        """Test comprehensive market data sanitization"""
        data = {
            "symbol": "  tsla  ",
            "price": "$250.75",
            "volume": "5,500,000",
            "bid": "250.70",
            "ask": "250.80",
            "high": "$255.00",
            "low": "248.50",
            "previous_close": "252,25",
            "change": "-1.50",
            "change_percent": "-0.59",
            "invalid_field": "should_be_ignored"
        }
        
        sanitized = DataValidator.sanitize_market_data(data)
        
        assert sanitized["symbol"] == "TSLA"
        assert sanitized["price"] == 250.75
        assert sanitized["volume"] == 5500000
        assert sanitized["bid"] == 250.70
        assert sanitized["ask"] == 250.80
        assert sanitized["high"] == 255.00
        assert sanitized["low"] == 248.50
        assert sanitized["previous_close"] == 252.25
        
        # Should not include invalid_field
        assert "invalid_field" not in sanitized