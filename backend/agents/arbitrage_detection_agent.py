"""
Arbitrage Detection Agent V5
Comprehensive arbitrage opportunity detection across multiple strategies
Enhanced from archive with V5 architecture patterns
"""

import asyncio
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
import uuid

import numpy as np
import pandas as pd
from scipy import stats

from core.logging import get_logger

logger = get_logger(__name__)


class ArbitrageType(Enum):
    """Types of arbitrage opportunities"""
    CROSS_EXCHANGE = "cross_exchange"          # Price differences across exchanges
    TRIANGULAR = "triangular"                  # Currency/crypto triangular arbitrage
    STATISTICAL = "statistical"                # Mean reversion pairs trading
    MERGER = "merger"                          # M&A arbitrage
    ETF_NAV = "etf_nav"                       # ETF vs underlying assets
    OPTIONS_PUT_CALL = "options_put_call"     # Put-call parity violations
    FUTURES_SPOT = "futures_spot"             # Futures-spot basis trade
    CROSS_ASSET = "cross_asset"               # Related assets divergence


class ArbitrageSignal(Enum):
    """Arbitrage trading signals"""
    STRONG_OPPORTUNITY = "strong_opportunity"
    OPPORTUNITY = "opportunity"
    POTENTIAL = "potential"
    MONITORING = "monitoring"
    NO_OPPORTUNITY = "no_opportunity"


@dataclass
class ArbitrageOpportunity:
    """Individual arbitrage opportunity"""
    opportunity_id: str
    type: ArbitrageType
    symbol_buy: str
    symbol_sell: str
    venue_buy: str
    venue_sell: str
    price_buy: float
    price_sell: float
    spread: float
    spread_percentage: float
    volume_available: float
    expected_profit: float
    execution_time_window: int  # seconds
    confidence: float
    risks: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'opportunity_id': self.opportunity_id,
            'type': self.type.value,
            'symbol_buy': self.symbol_buy,
            'symbol_sell': self.symbol_sell,
            'venue_buy': self.venue_buy,
            'venue_sell': self.venue_sell,
            'price_buy': self.price_buy,
            'price_sell': self.price_sell,
            'spread': self.spread,
            'spread_percentage': self.spread_percentage,
            'volume_available': self.volume_available,
            'expected_profit': self.expected_profit,
            'execution_time_window': self.execution_time_window,
            'confidence': self.confidence,
            'risks': self.risks
        }


@dataclass
class StatisticalPair:
    """Statistical arbitrage pair"""
    symbol_1: str
    symbol_2: str
    correlation: float
    cointegration_score: float
    half_life: float  # Mean reversion half-life in days
    z_score: float
    hedge_ratio: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ArbitrageMetrics:
    """Aggregated arbitrage metrics"""
    total_opportunities: int
    avg_spread_percentage: float
    total_expected_profit: float
    best_opportunity_profit: float
    opportunities_by_type: Dict[str, int]
    avg_confidence: float
    execution_urgency: str  # immediate, short-term, monitoring
    market_efficiency_score: float  # 0-1, higher = more efficient
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ArbitrageAnalysis:
    """Complete arbitrage analysis result"""
    timestamp: datetime
    opportunities: List[ArbitrageOpportunity]
    statistical_pairs: List[StatisticalPair]
    metrics: ArbitrageMetrics
    signal: ArbitrageSignal
    signal_strength: float
    recommended_actions: List[str]
    market_conditions: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'opportunities': [o.to_dict() for o in self.opportunities],
            'statistical_pairs': [p.to_dict() for p in self.statistical_pairs],
            'metrics': self.metrics.to_dict(),
            'signal': self.signal.value,
            'signal_strength': self.signal_strength,
            'recommended_actions': self.recommended_actions,
            'market_conditions': self.market_conditions
        }


class ArbitrageDetectionAgent:
    """
    V5 Arbitrage Detection Agent
    Comprehensive arbitrage opportunity detection and analysis
    """
    
    def __init__(self):
        """Initialize the arbitrage detection agent"""
        # Configuration
        self.min_spread_percentage = 0.5  # 0.5% minimum spread
        self.min_volume = 1000.0  # Minimum USD volume
        self.max_execution_time = 60  # seconds
        
        # Fees and costs
        self.exchange_fees = {
            'default': 0.001,  # 0.1%
            'binance': 0.001,
            'coinbase': 0.005,
            'kraken': 0.0026,
            'nasdaq': 0.0005,
            'nyse': 0.0005
        }
        
        self.slippage_estimate = 0.001  # 0.1% slippage
        self.transfer_time = {
            'crypto': 600,  # 10 minutes
            'fiat': 86400,  # 1 day
            'stock': 172800  # 2 days (T+2 settlement)
        }
        
        # Statistical arbitrage parameters
        self.lookback_period = 60  # days
        self.z_score_threshold = 2.0
        self.min_correlation = 0.7
        self.cointegration_pvalue = 0.05
        
        # Risk parameters
        self.max_position_size = 10000.0  # USD
        self.max_opportunities = 10
        self.confidence_threshold = 0.6
        
        # Cache and history
        self.price_cache = {}
        self.opportunity_history = deque(maxlen=1000)
        self.pair_statistics = {}
        
        # Performance tracking
        self.detection_times = []
        self.successful_arbitrages = []
        
        logger.info("Arbitrage Detection Agent V5 initialized")
    
    async def analyze(self, market_data: Dict[str, Any]) -> ArbitrageAnalysis:
        """
        Perform comprehensive arbitrage analysis
        
        Args:
            market_data: Market data including prices from multiple sources
        
        Returns:
            Complete arbitrage analysis
        """
        try:
            start_time = datetime.now()
            
            # Detect different types of arbitrage opportunities
            opportunities = []
            
            # 1. Cross-exchange arbitrage
            cross_exchange_opps = await self._detect_cross_exchange_arbitrage(market_data)
            opportunities.extend(cross_exchange_opps)
            
            # 2. Statistical arbitrage pairs
            statistical_pairs = self._detect_statistical_arbitrage(market_data)
            stat_opps = self._convert_pairs_to_opportunities(statistical_pairs)
            opportunities.extend(stat_opps)
            
            # 3. Triangular arbitrage (for crypto)
            if market_data.get('asset_type') == 'crypto':
                triangular_opps = self._detect_triangular_arbitrage(market_data)
                opportunities.extend(triangular_opps)
            
            # 4. ETF-NAV arbitrage
            etf_opps = self._detect_etf_nav_arbitrage(market_data)
            opportunities.extend(etf_opps)
            
            # 5. Options put-call parity
            options_opps = self._detect_options_arbitrage(market_data)
            opportunities.extend(options_opps)
            
            # 6. Futures-spot basis trade
            futures_opps = self._detect_futures_spot_arbitrage(market_data)
            opportunities.extend(futures_opps)
            
            # Filter and rank opportunities
            valid_opportunities = self._filter_opportunities(opportunities)
            ranked_opportunities = self._rank_opportunities(valid_opportunities)
            
            # Calculate metrics
            metrics = self._calculate_metrics(ranked_opportunities)
            
            # Assess market conditions
            market_conditions = self._assess_market_conditions(market_data, metrics)
            
            # Generate signal
            signal, signal_strength = self._generate_signal(metrics, ranked_opportunities)
            
            # Generate recommendations
            recommended_actions = self._generate_recommendations(
                ranked_opportunities, signal, market_conditions
            )
            
            # Create analysis result
            analysis = ArbitrageAnalysis(
                timestamp=datetime.now(),
                opportunities=ranked_opportunities[:self.max_opportunities],
                statistical_pairs=statistical_pairs[:10],
                metrics=metrics,
                signal=signal,
                signal_strength=signal_strength,
                recommended_actions=recommended_actions,
                market_conditions=market_conditions
            )
            
            # Store in history
            for opp in ranked_opportunities[:5]:
                self.opportunity_history.append(opp)
            
            # Track performance
            calc_time = (datetime.now() - start_time).total_seconds()
            self.detection_times.append(calc_time)
            
            logger.info(f"Arbitrage analysis complete: {len(ranked_opportunities)} opportunities found")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Arbitrage analysis failed: {str(e)}")
            raise
    
    async def _detect_cross_exchange_arbitrage(self, market_data: Dict[str, Any]) -> List[ArbitrageOpportunity]:
        """Detect cross-exchange arbitrage opportunities"""
        opportunities = []
        
        try:
            # Get prices from different exchanges
            exchange_prices = market_data.get('exchange_prices', {})
            
            for symbol, prices_by_exchange in exchange_prices.items():
                exchanges = list(prices_by_exchange.keys())
                
                for i in range(len(exchanges)):
                    for j in range(i + 1, len(exchanges)):
                        exchange_1 = exchanges[i]
                        exchange_2 = exchanges[j]
                        
                        price_1 = prices_by_exchange[exchange_1]['price']
                        price_2 = prices_by_exchange[exchange_2]['price']
                        volume_1 = prices_by_exchange[exchange_1].get('volume', self.min_volume)
                        volume_2 = prices_by_exchange[exchange_2].get('volume', self.min_volume)
                        
                        # Identify arbitrage direction
                        if price_1 < price_2:
                            buy_exchange, sell_exchange = exchange_1, exchange_2
                            buy_price, sell_price = price_1, price_2
                            available_volume = min(volume_1, volume_2)
                        else:
                            buy_exchange, sell_exchange = exchange_2, exchange_1
                            buy_price, sell_price = price_2, price_1
                            available_volume = min(volume_2, volume_1)
                        
                        # Calculate spread
                        spread = sell_price - buy_price
                        spread_percentage = (spread / buy_price) * 100
                        
                        # Check if profitable after fees
                        buy_fee = self.exchange_fees.get(buy_exchange, self.exchange_fees['default'])
                        sell_fee = self.exchange_fees.get(sell_exchange, self.exchange_fees['default'])
                        total_fees = (buy_fee + sell_fee) * 100
                        
                        if spread_percentage > total_fees + self.slippage_estimate * 100:
                            # Calculate expected profit
                            trade_volume = min(available_volume, self.max_position_size / buy_price)
                            gross_profit = spread * trade_volume
                            fee_cost = (buy_price * buy_fee + sell_price * sell_fee) * trade_volume
                            slippage_cost = (buy_price + sell_price) * self.slippage_estimate * trade_volume
                            expected_profit = gross_profit - fee_cost - slippage_cost
                            
                            if expected_profit > 0:
                                opportunity = ArbitrageOpportunity(
                                    opportunity_id=str(uuid.uuid4()),
                                    type=ArbitrageType.CROSS_EXCHANGE,
                                    symbol_buy=symbol,
                                    symbol_sell=symbol,
                                    venue_buy=buy_exchange,
                                    venue_sell=sell_exchange,
                                    price_buy=buy_price,
                                    price_sell=sell_price,
                                    spread=spread,
                                    spread_percentage=spread_percentage,
                                    volume_available=trade_volume,
                                    expected_profit=expected_profit,
                                    execution_time_window=30,  # 30 seconds for crypto
                                    confidence=0.8,
                                    risks=['execution_risk', 'slippage_risk', 'transfer_delay']
                                )
                                opportunities.append(opportunity)
        
        except Exception as e:
            logger.error(f"Cross-exchange arbitrage detection failed: {str(e)}")
        
        return opportunities
    
    def _detect_statistical_arbitrage(self, market_data: Dict[str, Any]) -> List[StatisticalPair]:
        """Detect statistical arbitrage pairs"""
        pairs = []
        
        try:
            price_history = market_data.get('price_history', {})
            
            if not price_history:
                return pairs
            
            # Convert to DataFrame
            df = pd.DataFrame(price_history)
            
            if len(df) < self.lookback_period:
                return pairs
            
            # Get recent data
            recent_df = df.tail(self.lookback_period)
            returns = recent_df.pct_change().dropna()
            
            symbols = list(recent_df.columns)
            
            # Find cointegrated pairs
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    symbol_1 = symbols[i]
                    symbol_2 = symbols[j]
                    
                    # Calculate correlation
                    correlation = returns[symbol_1].corr(returns[symbol_2])
                    
                    if correlation < self.min_correlation:
                        continue
                    
                    # Test for cointegration
                    from statsmodels.tsa.stattools import coint
                    score, pvalue, _ = coint(recent_df[symbol_1], recent_df[symbol_2])
                    
                    if pvalue < self.cointegration_pvalue:
                        # Calculate hedge ratio
                        hedge_ratio = np.polyfit(recent_df[symbol_2], recent_df[symbol_1], 1)[0]
                        
                        # Calculate spread
                        spread = recent_df[symbol_1] - hedge_ratio * recent_df[symbol_2]
                        
                        # Calculate z-score
                        spread_mean = spread.mean()
                        spread_std = spread.std()
                        current_spread = spread.iloc[-1]
                        z_score = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0
                        
                        # Calculate half-life
                        spread_lag = spread.shift(1)
                        spread_ret = spread - spread_lag
                        spread_lag2 = spread_lag - spread_mean
                        
                        try:
                            # Use OLS to estimate mean reversion speed
                            model = np.polyfit(spread_lag2.dropna(), spread_ret.dropna(), 1)
                            half_life = -np.log(2) / model[0] if model[0] < 0 else np.inf
                        except:
                            half_life = np.inf
                        
                        if abs(z_score) > self.z_score_threshold and half_life < 30:
                            pair = StatisticalPair(
                                symbol_1=symbol_1,
                                symbol_2=symbol_2,
                                correlation=correlation,
                                cointegration_score=score,
                                half_life=half_life,
                                z_score=z_score,
                                hedge_ratio=hedge_ratio
                            )
                            pairs.append(pair)
        
        except Exception as e:
            logger.error(f"Statistical arbitrage detection failed: {str(e)}")
        
        return pairs
    
    def _detect_triangular_arbitrage(self, market_data: Dict[str, Any]) -> List[ArbitrageOpportunity]:
        """Detect triangular arbitrage in crypto markets"""
        opportunities = []
        
        try:
            # Example: BTC/USD -> ETH/BTC -> ETH/USD
            pairs = market_data.get('crypto_pairs', {})
            
            if 'BTC/USD' in pairs and 'ETH/BTC' in pairs and 'ETH/USD' in pairs:
                btc_usd = pairs['BTC/USD']
                eth_btc = pairs['ETH/BTC']
                eth_usd = pairs['ETH/USD']
                
                # Calculate implied ETH/USD through BTC
                implied_eth_usd = btc_usd * eth_btc
                
                # Check for arbitrage
                spread = eth_usd - implied_eth_usd
                spread_percentage = abs(spread / eth_usd) * 100
                
                if spread_percentage > self.min_spread_percentage:
                    # Determine trade direction
                    if implied_eth_usd < eth_usd:
                        # Buy ETH via BTC, sell ETH directly
                        trade_path = "USD -> BTC -> ETH -> USD"
                    else:
                        # Buy ETH directly, sell via BTC
                        trade_path = "USD -> ETH -> BTC -> USD"
                    
                    opportunity = ArbitrageOpportunity(
                        opportunity_id=str(uuid.uuid4()),
                        type=ArbitrageType.TRIANGULAR,
                        symbol_buy="ETH",
                        symbol_sell="ETH",
                        venue_buy="path_1",
                        venue_sell="path_2",
                        price_buy=min(implied_eth_usd, eth_usd),
                        price_sell=max(implied_eth_usd, eth_usd),
                        spread=abs(spread),
                        spread_percentage=spread_percentage,
                        volume_available=self.min_volume,
                        expected_profit=abs(spread) * (self.min_volume / eth_usd),
                        execution_time_window=10,  # Very short for triangular
                        confidence=0.7,
                        risks=['execution_speed', 'multiple_legs', 'slippage']
                    )
                    opportunities.append(opportunity)
        
        except Exception as e:
            logger.error(f"Triangular arbitrage detection failed: {str(e)}")
        
        return opportunities
    
    def _detect_etf_nav_arbitrage(self, market_data: Dict[str, Any]) -> List[ArbitrageOpportunity]:
        """Detect ETF vs NAV arbitrage opportunities"""
        opportunities = []
        
        try:
            etf_data = market_data.get('etf_data', {})
            
            for etf_symbol, data in etf_data.items():
                market_price = data.get('market_price', 0)
                nav = data.get('nav', 0)  # Net Asset Value
                
                if market_price > 0 and nav > 0:
                    premium_discount = ((market_price - nav) / nav) * 100
                    
                    if abs(premium_discount) > self.min_spread_percentage:
                        if premium_discount > 0:
                            # ETF trading at premium - sell ETF, buy underlying
                            action = "sell_etf_buy_underlying"
                        else:
                            # ETF trading at discount - buy ETF, sell underlying
                            action = "buy_etf_sell_underlying"
                        
                        opportunity = ArbitrageOpportunity(
                            opportunity_id=str(uuid.uuid4()),
                            type=ArbitrageType.ETF_NAV,
                            symbol_buy=etf_symbol if premium_discount < 0 else "underlying",
                            symbol_sell=etf_symbol if premium_discount > 0 else "underlying",
                            venue_buy="market",
                            venue_sell="market",
                            price_buy=min(market_price, nav),
                            price_sell=max(market_price, nav),
                            spread=abs(market_price - nav),
                            spread_percentage=abs(premium_discount),
                            volume_available=data.get('volume', self.min_volume),
                            expected_profit=abs(market_price - nav) * self.min_volume,
                            execution_time_window=300,  # 5 minutes
                            confidence=0.6,
                            risks=['creation_redemption_lag', 'tracking_error']
                        )
                        opportunities.append(opportunity)
        
        except Exception as e:
            logger.error(f"ETF-NAV arbitrage detection failed: {str(e)}")
        
        return opportunities
    
    def _detect_options_arbitrage(self, market_data: Dict[str, Any]) -> List[ArbitrageOpportunity]:
        """Detect put-call parity violations"""
        opportunities = []
        
        try:
            options_data = market_data.get('options_data', {})
            
            for symbol, data in options_data.items():
                spot_price = data.get('spot_price', 0)
                
                for strike, option_data in data.get('strikes', {}).items():
                    call_price = option_data.get('call_price', 0)
                    put_price = option_data.get('put_price', 0)
                    time_to_expiry = option_data.get('time_to_expiry', 0)
                    risk_free_rate = market_data.get('risk_free_rate', 0.02)
                    
                    if all([spot_price, call_price, put_price, time_to_expiry]):
                        # Put-Call Parity: C - P = S - K * e^(-r*T)
                        pv_strike = strike * np.exp(-risk_free_rate * time_to_expiry)
                        theoretical_diff = spot_price - pv_strike
                        actual_diff = call_price - put_price
                        
                        parity_violation = actual_diff - theoretical_diff
                        violation_percentage = abs(parity_violation / spot_price) * 100
                        
                        if violation_percentage > self.min_spread_percentage:
                            if parity_violation > 0:
                                # Calls overpriced relative to puts
                                action = "sell_call_buy_put_buy_stock"
                            else:
                                # Puts overpriced relative to calls
                                action = "sell_put_buy_call_sell_stock"
                            
                            opportunity = ArbitrageOpportunity(
                                opportunity_id=str(uuid.uuid4()),
                                type=ArbitrageType.OPTIONS_PUT_CALL,
                                symbol_buy=f"{symbol}_options",
                                symbol_sell=f"{symbol}_options",
                                venue_buy="options_market",
                                venue_sell="options_market",
                                price_buy=min(call_price, put_price),
                                price_sell=max(call_price, put_price),
                                spread=abs(parity_violation),
                                spread_percentage=violation_percentage,
                                volume_available=option_data.get('volume', self.min_volume),
                                expected_profit=abs(parity_violation) * 100,  # Per contract
                                execution_time_window=60,
                                confidence=0.7,
                                risks=['early_exercise', 'dividends', 'liquidity']
                            )
                            opportunities.append(opportunity)
        
        except Exception as e:
            logger.error(f"Options arbitrage detection failed: {str(e)}")
        
        return opportunities
    
    def _detect_futures_spot_arbitrage(self, market_data: Dict[str, Any]) -> List[ArbitrageOpportunity]:
        """Detect futures-spot basis arbitrage"""
        opportunities = []
        
        try:
            futures_data = market_data.get('futures_data', {})
            
            for symbol, data in futures_data.items():
                spot_price = data.get('spot_price', 0)
                
                for expiry, futures_price in data.get('futures_prices', {}).items():
                    if spot_price > 0 and futures_price > 0:
                        basis = futures_price - spot_price
                        basis_percentage = (basis / spot_price) * 100
                        
                        # Calculate theoretical futures price (cost of carry)
                        time_to_expiry = data.get('time_to_expiry', {}).get(expiry, 0)
                        risk_free_rate = market_data.get('risk_free_rate', 0.02)
                        storage_cost = data.get('storage_cost', 0)
                        
                        theoretical_futures = spot_price * np.exp((risk_free_rate + storage_cost) * time_to_expiry)
                        
                        mispricing = futures_price - theoretical_futures
                        mispricing_percentage = abs(mispricing / spot_price) * 100
                        
                        if mispricing_percentage > self.min_spread_percentage:
                            if mispricing > 0:
                                # Futures overpriced - sell futures, buy spot
                                action = "sell_futures_buy_spot"
                            else:
                                # Futures underpriced - buy futures, sell spot
                                action = "buy_futures_sell_spot"
                            
                            opportunity = ArbitrageOpportunity(
                                opportunity_id=str(uuid.uuid4()),
                                type=ArbitrageType.FUTURES_SPOT,
                                symbol_buy=symbol if mispricing < 0 else f"{symbol}_futures",
                                symbol_sell=symbol if mispricing > 0 else f"{symbol}_futures",
                                venue_buy="spot_market" if mispricing < 0 else "futures_market",
                                venue_sell="futures_market" if mispricing > 0 else "spot_market",
                                price_buy=min(spot_price, futures_price),
                                price_sell=max(spot_price, futures_price),
                                spread=abs(mispricing),
                                spread_percentage=mispricing_percentage,
                                volume_available=data.get('volume', self.min_volume),
                                expected_profit=abs(mispricing) * (self.min_volume / spot_price),
                                execution_time_window=300,
                                confidence=0.65,
                                risks=['basis_risk', 'margin_requirements', 'roll_risk']
                            )
                            opportunities.append(opportunity)
        
        except Exception as e:
            logger.error(f"Futures-spot arbitrage detection failed: {str(e)}")
        
        return opportunities
    
    def _convert_pairs_to_opportunities(self, pairs: List[StatisticalPair]) -> List[ArbitrageOpportunity]:
        """Convert statistical pairs to arbitrage opportunities"""
        opportunities = []
        
        for pair in pairs:
            if abs(pair.z_score) > self.z_score_threshold:
                if pair.z_score > 0:
                    # Spread is high - sell symbol_1, buy symbol_2
                    action = f"sell_{pair.symbol_1}_buy_{pair.symbol_2}"
                else:
                    # Spread is low - buy symbol_1, sell symbol_2
                    action = f"buy_{pair.symbol_1}_sell_{pair.symbol_2}"
                
                # Estimate profit based on mean reversion
                expected_move = abs(pair.z_score) * 0.5  # Expect 50% mean reversion
                expected_profit = expected_move * self.min_volume * 0.01  # Rough estimate
                
                opportunity = ArbitrageOpportunity(
                    opportunity_id=str(uuid.uuid4()),
                    type=ArbitrageType.STATISTICAL,
                    symbol_buy=pair.symbol_1 if pair.z_score < 0 else pair.symbol_2,
                    symbol_sell=pair.symbol_1 if pair.z_score > 0 else pair.symbol_2,
                    venue_buy="market",
                    venue_sell="market",
                    price_buy=1.0,  # Normalized
                    price_sell=1.0 + abs(pair.z_score) * 0.01,
                    spread=abs(pair.z_score) * 0.01,
                    spread_percentage=abs(pair.z_score),
                    volume_available=self.min_volume,
                    expected_profit=expected_profit,
                    execution_time_window=int(pair.half_life * 86400),  # Convert days to seconds
                    confidence=0.5 + min(0.3, pair.correlation * 0.3),
                    risks=['model_risk', 'correlation_breakdown', 'execution_lag']
                )
                opportunities.append(opportunity)
        
        return opportunities
    
    def _filter_opportunities(self, opportunities: List[ArbitrageOpportunity]) -> List[ArbitrageOpportunity]:
        """Filter opportunities based on criteria"""
        filtered = []
        
        for opp in opportunities:
            # Check minimum profit
            if opp.expected_profit < 10:  # Minimum $10 profit
                continue
            
            # Check confidence
            if opp.confidence < self.confidence_threshold:
                continue
            
            # Check execution window
            if opp.execution_time_window > self.max_execution_time:
                continue
            
            # Check volume
            if opp.volume_available < self.min_volume:
                continue
            
            filtered.append(opp)
        
        return filtered
    
    def _rank_opportunities(self, opportunities: List[ArbitrageOpportunity]) -> List[ArbitrageOpportunity]:
        """Rank opportunities by profitability and confidence"""
        
        # Calculate score for each opportunity
        for opp in opportunities:
            # Score based on profit, confidence, and execution speed
            profit_score = min(1.0, opp.expected_profit / 1000)  # Normalize to $1000
            confidence_score = opp.confidence
            speed_score = 1.0 - min(1.0, opp.execution_time_window / self.max_execution_time)
            
            opp.score = profit_score * 0.5 + confidence_score * 0.3 + speed_score * 0.2
        
        # Sort by score
        ranked = sorted(opportunities, key=lambda x: x.score, reverse=True)
        
        return ranked
    
    def _calculate_metrics(self, opportunities: List[ArbitrageOpportunity]) -> ArbitrageMetrics:
        """Calculate aggregated metrics"""
        
        if not opportunities:
            return ArbitrageMetrics(
                total_opportunities=0,
                avg_spread_percentage=0.0,
                total_expected_profit=0.0,
                best_opportunity_profit=0.0,
                opportunities_by_type={},
                avg_confidence=0.0,
                execution_urgency="monitoring",
                market_efficiency_score=1.0
            )
        
        # Calculate metrics
        total_opportunities = len(opportunities)
        avg_spread = np.mean([opp.spread_percentage for opp in opportunities])
        total_profit = sum(opp.expected_profit for opp in opportunities)
        best_profit = max(opp.expected_profit for opp in opportunities)
        avg_confidence = np.mean([opp.confidence for opp in opportunities])
        
        # Count by type
        type_counts = {}
        for opp in opportunities:
            type_name = opp.type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        # Determine urgency
        min_window = min(opp.execution_time_window for opp in opportunities)
        if min_window < 60:
            urgency = "immediate"
        elif min_window < 300:
            urgency = "short-term"
        else:
            urgency = "monitoring"
        
        # Market efficiency score (fewer opportunities = more efficient)
        efficiency = 1.0 - min(1.0, total_opportunities / 20)
        
        return ArbitrageMetrics(
            total_opportunities=total_opportunities,
            avg_spread_percentage=avg_spread,
            total_expected_profit=total_profit,
            best_opportunity_profit=best_profit,
            opportunities_by_type=type_counts,
            avg_confidence=avg_confidence,
            execution_urgency=urgency,
            market_efficiency_score=efficiency
        )
    
    def _assess_market_conditions(self, market_data: Dict[str, Any], metrics: ArbitrageMetrics) -> Dict[str, Any]:
        """Assess current market conditions"""
        
        conditions = {
            'volatility': market_data.get('volatility', 'normal'),
            'liquidity': market_data.get('liquidity', 'normal'),
            'efficiency': 'high' if metrics.market_efficiency_score > 0.7 else 'low',
            'opportunity_density': 'high' if metrics.total_opportunities > 10 else 'low',
            'dominant_type': max(metrics.opportunities_by_type.items(), key=lambda x: x[1])[0] if metrics.opportunities_by_type else None
        }
        
        return conditions
    
    def _generate_signal(self, metrics: ArbitrageMetrics, opportunities: List[ArbitrageOpportunity]) -> Tuple[ArbitrageSignal, float]:
        """Generate trading signal"""
        
        if not opportunities:
            return ArbitrageSignal.NO_OPPORTUNITY, 0.0
        
        # Signal based on opportunity quality and quantity
        if metrics.total_opportunities >= 5 and metrics.avg_confidence > 0.7:
            signal = ArbitrageSignal.STRONG_OPPORTUNITY
            strength = min(1.0, metrics.avg_confidence * 1.2)
        elif metrics.total_opportunities >= 2 and metrics.avg_confidence > 0.6:
            signal = ArbitrageSignal.OPPORTUNITY
            strength = metrics.avg_confidence
        elif metrics.total_opportunities >= 1:
            signal = ArbitrageSignal.POTENTIAL
            strength = metrics.avg_confidence * 0.8
        else:
            signal = ArbitrageSignal.MONITORING
            strength = 0.3
        
        # Adjust for urgency
        if metrics.execution_urgency == "immediate":
            strength *= 1.2
        
        return signal, min(1.0, strength)
    
    def _generate_recommendations(self, opportunities: List[ArbitrageOpportunity], 
                                 signal: ArbitrageSignal, market_conditions: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if signal == ArbitrageSignal.STRONG_OPPORTUNITY:
            recommendations.append("Multiple high-confidence arbitrage opportunities detected - execute immediately")
        elif signal == ArbitrageSignal.OPPORTUNITY:
            recommendations.append("Viable arbitrage opportunities available - review and execute selectively")
        elif signal == ArbitrageSignal.POTENTIAL:
            recommendations.append("Potential arbitrage detected - monitor closely for confirmation")
        else:
            recommendations.append("No significant arbitrage opportunities - continue monitoring")
        
        # Specific opportunity recommendations
        if opportunities:
            best_opp = opportunities[0]
            recommendations.append(f"Best opportunity: {best_opp.type.value} with ${best_opp.expected_profit:.2f} expected profit")
            
            if best_opp.execution_time_window < 60:
                recommendations.append("⚡ Immediate execution required - opportunity may close quickly")
        
        # Type-specific recommendations
        type_counts = {}
        for opp in opportunities[:5]:
            type_counts[opp.type.value] = type_counts.get(opp.type.value, 0) + 1
        
        if type_counts:
            dominant_type = max(type_counts.items(), key=lambda x: x[1])[0]
            recommendations.append(f"Focus on {dominant_type} opportunities - highest frequency")
        
        # Market condition recommendations
        if market_conditions.get('volatility') == 'high':
            recommendations.append("High volatility - arbitrage opportunities may close faster than usual")
        
        if market_conditions.get('efficiency') == 'low':
            recommendations.append("Market inefficiency detected - favorable for arbitrage strategies")
        
        # Risk warnings
        if any(opp.confidence < 0.6 for opp in opportunities[:3]):
            recommendations.append("⚠️ Some opportunities have lower confidence - consider position sizing")
        
        return recommendations[:8]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        
        total_detections = len(self.opportunity_history)
        successful = len(self.successful_arbitrages)
        
        if total_detections == 0:
            success_rate = 0.0
        else:
            success_rate = successful / total_detections
        
        avg_detection_time = np.mean(self.detection_times) if self.detection_times else 0.0
        
        # Type distribution
        type_dist = {}
        for opp in self.opportunity_history:
            type_name = opp.type.value
            type_dist[type_name] = type_dist.get(type_name, 0) + 1
        
        return {
            'total_opportunities_detected': total_detections,
            'successful_arbitrages': successful,
            'success_rate': success_rate,
            'average_detection_time_seconds': avg_detection_time,
            'opportunity_type_distribution': type_dist,
            'supported_arbitrage_types': len(ArbitrageType),
            'cache_size': len(self.price_cache)
        }


# Create global instance
arbitrage_detection_agent = ArbitrageDetectionAgent()