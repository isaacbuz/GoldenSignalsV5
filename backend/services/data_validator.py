"""
Data validation and sanitization service
Ensures data quality and consistency across the platform
"""

from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import re
from decimal import Decimal, InvalidOperation


class DataValidator:
    """Validates and sanitizes market data"""
    
    @staticmethod
    def validate_quote(quote: Dict[str, Any]) -> bool:
        """
        Validate quote data
        
        Args:
            quote: Quote dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Required fields
        required_fields = ["symbol", "price"]
        
        # Check required fields exist
        for field in required_fields:
            if field not in quote or quote[field] is None:
                return False
        
        # Validate symbol
        if not isinstance(quote["symbol"], str) or not quote["symbol"].strip():
            return False
        
        # Validate price
        try:
            price = float(quote["price"])
            if price <= 0:
                return False
        except (TypeError, ValueError):
            return False
        
        # Validate optional numeric fields
        numeric_fields = ["volume", "bid", "ask", "high", "low", "change", "change_percent"]
        for field in numeric_fields:
            if field in quote and quote[field] is not None:
                try:
                    value = float(quote[field])
                    # Volume can be 0, but prices should be positive
                    if field != "volume" and field not in ["change", "change_percent"] and value < 0:
                        return False
                except (TypeError, ValueError):
                    return False
        
        return True
    
    @staticmethod
    def sanitize_market_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize and normalize market data
        
        Args:
            data: Raw market data
            
        Returns:
            Sanitized market data
        """
        sanitized = {}
        
        # Symbol - uppercase and strip
        if "symbol" in data:
            symbol = str(data["symbol"]).strip().upper()
            # Remove invalid characters
            symbol = re.sub(r'[^A-Z0-9\-\.]', '', symbol)
            sanitized["symbol"] = symbol
        
        # Price fields
        price_fields = ["price", "bid", "ask", "high", "low", "open", "close", 
                       "previous_close", "target_price", "stop_loss"]
        
        for field in price_fields:
            if field in data and data[field] is not None:
                try:
                    # Handle formatted strings
                    value = str(data[field]).replace(',', '').replace('$', '')
                    sanitized[field] = float(value)
                except (TypeError, ValueError):
                    # Skip invalid price values
                    continue
        
        # Volume - handle formatted numbers
        if "volume" in data and data["volume"] is not None:
            try:
                volume = str(data["volume"]).replace(',', '')
                sanitized["volume"] = int(float(volume))
            except (TypeError, ValueError):
                sanitized["volume"] = 0
        
        # Percentage fields
        percent_fields = ["change_percent", "confidence"]
        for field in percent_fields:
            if field in data and data[field] is not None:
                try:
                    value = str(data[field]).replace('%', '')
                    sanitized[field] = float(value)
                except (TypeError, ValueError):
                    # Skip invalid price values
                    continue
        
        # Timestamp
        if "timestamp" not in data or data["timestamp"] is None:
            sanitized["timestamp"] = datetime.now().isoformat()
        else:
            sanitized["timestamp"] = data["timestamp"]
        
        # Copy other fields
        for key, value in data.items():
            if key not in sanitized and value is not None:
                sanitized[key] = value
        
        return sanitized
    
    @staticmethod
    def validate_signal(signal: Dict[str, Any]) -> bool:
        """
        Validate trading signal
        
        Args:
            signal: Signal data to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Required fields
        required = ["symbol", "action", "confidence"]
        
        for field in required:
            if field not in signal or signal[field] is None:
                return False
        
        # Validate action
        valid_actions = ["BUY", "SELL", "HOLD"]
        if signal["action"] not in valid_actions:
            return False
        
        # Validate confidence
        try:
            confidence = float(signal["confidence"])
            if not 0 <= confidence <= 1:
                return False
        except (TypeError, ValueError):
            return False
        
        # Validate symbol
        if not isinstance(signal["symbol"], str) or not signal["symbol"].strip():
            return False
        
        return True
    
    @staticmethod
    def validate_historical_data(data: Union[List, Dict]) -> bool:
        """
        Validate historical data structure
        
        Args:
            data: Historical data (DataFrame as dict or list)
            
        Returns:
            True if valid, False otherwise
        """
        if not data:
            return False
        
        # If it's a list of records
        if isinstance(data, list):
            if len(data) == 0:
                return False
            
            # Check first record
            required_fields = ["date", "close"]
            first_record = data[0]
            
            for field in required_fields:
                if field not in first_record:
                    return False
        
        # If it's a dict (from DataFrame)
        elif isinstance(data, dict):
            # Should have date index and price columns
            if "Close" not in data and "close" not in data:
                return False
        
        return True
    
    @staticmethod
    def sanitize_user_input(input_str: str, input_type: str = "general") -> str:
        """
        Sanitize user input to prevent injection attacks
        
        Args:
            input_str: User input string
            input_type: Type of input (symbol, query, general)
            
        Returns:
            Sanitized string
        """
        if not input_str:
            return ""
        
        # Convert to string and strip
        cleaned = str(input_str).strip()
        
        if input_type == "symbol":
            # Stock symbols: alphanumeric, dash, dot only
            cleaned = re.sub(r'[^A-Za-z0-9\-\.]', '', cleaned).upper()
            # Limit length
            cleaned = cleaned[:10]
        
        elif input_type == "query":
            # Search queries: alphanumeric, space, basic punctuation
            cleaned = re.sub(r'[^A-Za-z0-9\s\-\.\,]', '', cleaned)
            # Limit length
            cleaned = cleaned[:100]
        
        else:
            # General: remove potential SQL/script injections
            # Remove SQL keywords
            sql_pattern = r'\b(DROP|DELETE|INSERT|UPDATE|SELECT|UNION|EXEC|SCRIPT)\b'
            cleaned = re.sub(sql_pattern, '', cleaned, flags=re.IGNORECASE)
            
            # Remove script tags
            cleaned = re.sub(r'<[^>]*>', '', cleaned)
            
            # Limit length
            cleaned = cleaned[:500]
        
        return cleaned
    
    @staticmethod
    def validate_api_response(response: Dict[str, Any], expected_fields: List[str]) -> bool:
        """
        Validate API response structure
        
        Args:
            response: API response dictionary
            expected_fields: List of expected fields
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(response, dict):
            return False
        
        for field in expected_fields:
            if field not in response:
                return False
        
        return True
    
    @staticmethod
    def normalize_price(price: Any) -> Optional[float]:
        """
        Normalize price to float
        
        Args:
            price: Price in various formats
            
        Returns:
            Normalized price or None
        """
        if price is None:
            return None
        
        try:
            # Handle Decimal
            if isinstance(price, Decimal):
                return float(price)
            
            # Handle string formats
            if isinstance(price, str):
                # Remove currency symbols and commas
                price = price.replace('$', '').replace(',', '').strip()
            
            return float(price)
        
        except (TypeError, ValueError, InvalidOperation):
            return None
    
    @staticmethod
    def validate_websocket_message(message: Dict[str, Any]) -> bool:
        """
        Validate WebSocket message format
        
        Args:
            message: WebSocket message
            
        Returns:
            True if valid, False otherwise
        """
        # Must have type
        if "type" not in message:
            return False
        
        # Validate based on type
        msg_type = message["type"]
        
        if msg_type == "subscribe":
            return "symbol" in message
        
        elif msg_type == "analyze":
            return "symbol" in message
        
        elif msg_type in ["heartbeat", "get_status"]:
            return True
        
        # Unknown type
        return False


# Create singleton instance
data_validator = DataValidator()


# Convenience functions
def validate_quote(quote: Dict[str, Any]) -> bool:
    """Validate quote data"""
    return data_validator.validate_quote(quote)


def sanitize_market_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize market data"""
    return data_validator.sanitize_market_data(data)


def validate_signal(signal: Dict[str, Any]) -> bool:
    """Validate trading signal"""
    return data_validator.validate_signal(signal)


def sanitize_user_input(input_str: str, input_type: str = "general") -> str:
    """Sanitize user input"""
    return data_validator.sanitize_user_input(input_str, input_type)