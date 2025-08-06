"""
Tests for trading agents
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from agents.technical.ma_crossover_agent import MACrossoverAgent
from agents.sentiment.simple_sentiment_agent import SimpleSentimentAgent
from unittest.mock import patch, MagicMock


class TestMACrossoverAgent:
    """Test MA Crossover Agent"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.agent = MACrossoverAgent()
        
        # Create sample price data
        dates = pd.date_range(end=datetime.now(), periods=250, freq='D')
        prices = 100 + np.cumsum(np.random.randn(250) * 2)
        
        self.sample_data = pd.DataFrame({
            'Close': prices,
            'Open': prices - np.random.rand(250),
            'High': prices + np.random.rand(250) * 2,
            'Low': prices - np.random.rand(250) * 2,
            'Volume': np.random.randint(1000000, 10000000, 250)
        }, index=dates)
    
    @patch('yfinance.Ticker')
    def test_generate_signal_golden_cross(self, mock_ticker):
        """Test signal generation for golden cross"""
        # Create data with golden cross pattern
        dates = pd.date_range(end=datetime.now(), periods=250, freq='D')
        prices = np.concatenate([
            np.linspace(100, 90, 150),  # Downtrend
            np.linspace(90, 110, 100)   # Uptrend causing golden cross
        ])
        
        data = pd.DataFrame({
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, 250)
        }, index=dates)
        
        mock_ticker.return_value.history.return_value = data
        
        signal = self.agent.generate_signal("AAPL")
        
        assert signal["metadata"]["agent"] == "ma_crossover_agent"
        assert signal["metadata"]["symbol"] == "AAPL"
        assert "confidence" in signal
        assert signal["confidence"] >= 0 and signal["confidence"] <= 1
    
    @patch('yfinance.Ticker')
    def test_generate_signal_death_cross(self, mock_ticker):
        """Test signal generation for death cross"""
        # Create data with death cross pattern
        dates = pd.date_range(end=datetime.now(), periods=250, freq='D')
        prices = np.concatenate([
            np.linspace(100, 110, 150),  # Uptrend
            np.linspace(110, 90, 100)    # Downtrend causing death cross
        ])
        
        data = pd.DataFrame({
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, 250)
        }, index=dates)
        
        mock_ticker.return_value.history.return_value = data
        
        signal = self.agent.generate_signal("AAPL")
        
        assert signal["metadata"]["agent"] == "ma_crossover_agent"
        # Death cross should generate SELL or HOLD signal
        assert signal["action"] in ["SELL", "HOLD"]
    
    @patch('yfinance.Ticker')
    def test_generate_signal_insufficient_data(self, mock_ticker):
        """Test signal generation with insufficient data"""
        # Create minimal data
        data = pd.DataFrame({
            'Close': [100, 101, 102],
            'Volume': [1000000, 1100000, 1200000]
        })
        
        mock_ticker.return_value.history.return_value = data
        
        signal = self.agent.generate_signal("AAPL")
        
        assert signal["action"] == "HOLD"
        assert signal["confidence"] == 0.0
        assert "error" in signal["metadata"]
    
    @patch('yfinance.Ticker')
    def test_generate_signal_error_handling(self, mock_ticker):
        """Test error handling in signal generation"""
        mock_ticker.side_effect = Exception("API Error")
        
        signal = self.agent.generate_signal("AAPL")
        
        assert signal["action"] == "HOLD"
        assert signal["confidence"] == 0.0
        assert "error" in signal["metadata"]


class TestSentimentAgent:
    """Test Sentiment Analysis Agent"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.agent = SimpleSentimentAgent()
    
    @patch('yfinance.Ticker')
    def test_generate_signal_extreme_fear(self, mock_ticker):
        """Test signal generation in extreme fear conditions"""
        # Create data simulating extreme fear (oversold)
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        prices = np.linspace(100, 70, 100)  # Strong downtrend
        
        data = pd.DataFrame({
            'Close': prices,
            'Open': prices + 1,
            'High': prices + 2,
            'Low': prices - 1,
            'Volume': np.linspace(10000000, 20000000, 100)  # Increasing volume
        }, index=dates)
        
        mock_ticker.return_value.history.return_value = data
        mock_ticker.return_value.info = {
            'marketCap': 1000000000,
            'trailingPE': 10  # Low PE suggesting fear
        }
        
        signal = self.agent.generate_signal("AAPL")
        
        assert signal["metadata"]["agent"] == "sentiment_agent"
        # Extreme fear often leads to contrarian BUY signal
        assert "sentiment_score" in signal["metadata"]["indicators"]
    
    @patch('yfinance.Ticker')
    def test_generate_signal_extreme_greed(self, mock_ticker):
        """Test signal generation in extreme greed conditions"""
        # Create data simulating extreme greed (overbought)
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        prices = np.linspace(70, 100, 100)  # Strong uptrend
        
        data = pd.DataFrame({
            'Close': prices,
            'Open': prices - 1,
            'High': prices + 1,
            'Low': prices - 2,
            'Volume': np.linspace(20000000, 10000000, 100)  # Decreasing volume
        }, index=dates)
        
        mock_ticker.return_value.history.return_value = data
        mock_ticker.return_value.info = {
            'marketCap': 1000000000,
            'trailingPE': 50  # High PE suggesting greed
        }
        
        signal = self.agent.generate_signal("AAPL")
        
        assert signal["metadata"]["agent"] == "sentiment_agent"
        assert "rsi" in signal["metadata"]["indicators"]
        assert "volatility_ratio" in signal["metadata"]["indicators"]
    
    @patch('yfinance.Ticker')
    def test_calculate_rsi(self, mock_ticker):
        """Test RSI calculation"""
        # Create data for RSI testing
        prices = pd.Series([44, 44.25, 44.5, 43.75, 44.65, 45.12, 45.84, 46.08, 45.89, 46.03,
                           45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64])
        
        rsi = self.agent._calculate_rsi(prices, period=14)
        
        # RSI should be between 0 and 100
        assert 0 <= rsi <= 100
    
    @patch('yfinance.Ticker')
    def test_generate_signal_neutral_sentiment(self, mock_ticker):
        """Test signal generation in neutral sentiment"""
        # Create sideways market data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        prices = 100 + np.random.randn(100) * 2  # Random walk around 100
        
        data = pd.DataFrame({
            'Close': prices,
            'Open': prices + np.random.randn(100) * 0.5,
            'High': prices + abs(np.random.randn(100)),
            'Low': prices - abs(np.random.randn(100)),
            'Volume': np.random.randint(5000000, 15000000, 100)
        }, index=dates)
        
        mock_ticker.return_value.history.return_value = data
        mock_ticker.return_value.info = {
            'marketCap': 1000000000,
            'trailingPE': 20  # Neutral PE
        }
        
        signal = self.agent.generate_signal("AAPL")
        
        assert signal["metadata"]["agent"] == "sentiment_agent"
        # Neutral sentiment usually leads to HOLD
        assert signal["action"] in ["HOLD", "BUY", "SELL"]
        assert 0 <= signal["confidence"] <= 1