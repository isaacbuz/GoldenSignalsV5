"""
Government Data Service
Integrates official US government economic data APIs including FRED, Treasury, and BLS
"""

import asyncio
import aiohttp
import logging
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


class GovernmentDataSource(Enum):
    """Available government data sources"""
    FRED = "fred"  # Federal Reserve Economic Data
    TREASURY = "treasury"  # US Treasury Department
    BLS = "bls"  # Bureau of Labor Statistics
    CENSUS = "census"  # US Census Bureau
    BEA = "bea"  # Bureau of Economic Analysis


@dataclass
class EconomicDataPoint:
    """Standardized economic data point"""
    indicator: str
    value: float
    timestamp: datetime
    source: GovernmentDataSource
    unit: str
    frequency: str  # daily, weekly, monthly, quarterly, annual
    metadata: Dict[str, Any]


@dataclass
class TreasuryYield:
    """Treasury yield data"""
    maturity: str  # 1M, 3M, 6M, 1Y, 2Y, 5Y, 10Y, 20Y, 30Y
    yield_value: float
    date: datetime
    change_daily: float
    change_weekly: float


@dataclass
class LaborStatistic:
    """Labor market statistics"""
    metric: str  # unemployment_rate, cpi, ppi, etc.
    value: float
    date: datetime
    seasonally_adjusted: bool
    year_over_year_change: float


class GovernmentDataService:
    """Service for fetching and processing government economic data"""
    
    def __init__(self):
        # API keys and endpoints
        self.fred_api_key = os.getenv('FRED_API_KEY')
        self.bls_api_key = os.getenv('BLS_API_KEY')  # Optional, provides enhanced access
        
        # API base URLs
        self.endpoints = {
            'fred': 'https://api.stlouisfed.org/fred',
            'treasury': 'https://api.fiscaldata.treasury.gov/services/api/fiscal_service',
            'bls': 'https://api.bls.gov/publicAPI',
            'census': 'https://api.census.gov/data',
            'bea': 'https://apps.bea.gov/api/data'
        }
        
        # Key economic indicators and their series IDs
        self.fred_series = {
            # Interest Rates
            'fed_funds_rate': 'DFF',
            '10_year_treasury': 'DGS10',
            '2_year_treasury': 'DGS2',
            '30_year_treasury': 'DGS30',
            'yield_curve_10_2': 'T10Y2Y',
            
            # Inflation
            'cpi': 'CPIAUCSL',
            'core_cpi': 'CPILFESL',
            'pce': 'PCEPI',
            'core_pce': 'PCEPILFE',
            'inflation_expectations_5y': 'T5YIE',
            
            # Employment
            'unemployment_rate': 'UNRATE',
            'nonfarm_payrolls': 'PAYEMS',
            'initial_jobless_claims': 'ICSA',
            'labor_force_participation': 'CIVPART',
            
            # GDP and Growth
            'real_gdp': 'GDPC1',
            'gdp_growth_rate': 'A191RL1Q225SBEA',
            'industrial_production': 'INDPRO',
            'retail_sales': 'RSXFS',
            
            # Money Supply
            'm2_money_supply': 'M2SL',
            'velocity_of_m2': 'M2V',
            
            # Housing
            'housing_starts': 'HOUST',
            'case_shiller_index': 'CSUSHPISA',
            'mortgage_rate_30y': 'MORTGAGE30US',
            
            # Consumer
            'consumer_sentiment': 'UMCSENT',
            'personal_savings_rate': 'PSAVERT',
            'consumer_credit': 'TOTALSL',
            
            # Business
            'durable_goods_orders': 'DGORDER',
            'ism_manufacturing': 'MANEMP',
            'capacity_utilization': 'TCU',
            
            # Market Indicators
            'vix': 'VIXCLS',
            'dollar_index': 'DTWEXBGS',
            'oil_price': 'DCOILWTICO',
            'gold_price': 'GOLDAMGBD228NLBM'
        }
        
        # BLS series codes
        self.bls_series = {
            'unemployment_rate': 'LNS14000000',
            'cpi_all_items': 'CUUR0000SA0',
            'cpi_core': 'CUUR0000SA0L1E',
            'ppi': 'WPUFD4',
            'employment_cost_index': 'CIU1010000000000A',
            'average_hourly_earnings': 'CES0500000003',
            'job_openings': 'JTS00000000JOL'
        }
        
        # Treasury data endpoints
        self.treasury_endpoints = {
            'daily_yields': '/v2/accounting/od/avg_interest_rates',
            'auction_results': '/v1/accounting/od/auction_results',
            'debt_to_penny': '/v2/accounting/od/debt_to_penny',
            'treasury_securities': '/v1/accounting/od/securities'
        }
        
        # Cache for frequently accessed data
        self.cache = {}
        self.cache_expiry = {}
        
    async def get_comprehensive_economic_data(self) -> Dict[str, Any]:
        """Fetch comprehensive economic data from all sources"""
        
        async with aiohttp.ClientSession() as session:
            # Fetch data from multiple sources in parallel
            tasks = [
                self._fetch_fred_indicators(session),
                self._fetch_treasury_yields(session),
                self._fetch_bls_data(session),
                self._fetch_treasury_debt_data(session)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            fred_data = results[0] if not isinstance(results[0], Exception) else {}
            treasury_yields = results[1] if not isinstance(results[1], Exception) else {}
            bls_data = results[2] if not isinstance(results[2], Exception) else {}
            debt_data = results[3] if not isinstance(results[3], Exception) else {}
            
            # Combine and analyze
            analysis = self._analyze_economic_conditions(
                fred_data, treasury_yields, bls_data, debt_data
            )
            
            return {
                'timestamp': datetime.utcnow(),
                'fred_indicators': fred_data,
                'treasury_yields': treasury_yields,
                'labor_statistics': bls_data,
                'debt_metrics': debt_data,
                'analysis': analysis,
                'market_implications': self._determine_market_implications(analysis)
            }
    
    async def _fetch_fred_indicators(
        self, 
        session: aiohttp.ClientSession
    ) -> Dict[str, Any]:
        """Fetch key indicators from FRED API"""
        
        if not self.fred_api_key:
            logger.warning("FRED API key not configured")
            return {}
        
        indicators = {}
        
        # Fetch priority indicators
        priority_series = [
            'fed_funds_rate', '10_year_treasury', 'unemployment_rate',
            'cpi', 'real_gdp', 'vix', 'dollar_index'
        ]
        
        for indicator_name in priority_series:
            series_id = self.fred_series.get(indicator_name)
            if not series_id:
                continue
                
            try:
                # Check cache
                cache_key = f"fred_{series_id}"
                if self._is_cache_valid(cache_key):
                    indicators[indicator_name] = self.cache[cache_key]
                    continue
                
                # Fetch from API
                params = {
                    'series_id': series_id,
                    'api_key': self.fred_api_key,
                    'file_type': 'json',
                    'limit': 10,
                    'sort_order': 'desc'
                }
                
                url = f"{self.endpoints['fred']}/series/observations?{urlencode(params)}"
                
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        observations = data.get('observations', [])
                        
                        if observations:
                            latest = observations[0]
                            previous = observations[1] if len(observations) > 1 else None
                            
                            indicator_data = {
                                'value': float(latest['value']),
                                'date': latest['date'],
                                'change': None,
                                'change_percent': None
                            }
                            
                            if previous and previous['value'] != '.':
                                prev_value = float(previous['value'])
                                indicator_data['change'] = indicator_data['value'] - prev_value
                                if prev_value != 0:
                                    indicator_data['change_percent'] = (
                                        (indicator_data['change'] / prev_value) * 100
                                    )
                            
                            indicators[indicator_name] = indicator_data
                            
                            # Cache the result
                            self._cache_data(cache_key, indicator_data, minutes=15)
                            
            except Exception as e:
                logger.error(f"Error fetching FRED data for {indicator_name}: {e}")
                
        return indicators
    
    async def _fetch_treasury_yields(
        self, 
        session: aiohttp.ClientSession
    ) -> Dict[str, TreasuryYield]:
        """Fetch current Treasury yields from US Treasury API"""
        
        yields = {}
        
        try:
            # Fetch recent treasury rates
            url = f"{self.endpoints['treasury']}{self.treasury_endpoints['daily_yields']}"
            params = {
                'filter': 'record_date:gte:' + (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                'sort': '-record_date',
                'page[size]': 100
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    records = data.get('data', [])
                    
                    if records:
                        # Process latest yields
                        latest_date_records = {}
                        
                        for record in records:
                            security_desc = record.get('security_desc', '')
                            
                            # Map security descriptions to standard maturities
                            maturity_map = {
                                'Treasury Bills': '3M',
                                'Treasury Notes': '10Y',
                                'Treasury Bonds': '30Y',
                                'Treasury Inflation-Protected': 'TIPS'
                            }
                            
                            for key, maturity in maturity_map.items():
                                if key in security_desc:
                                    rate = float(record.get('avg_interest_rate_amt', 0))
                                    date_str = record.get('record_date')
                                    
                                    yields[maturity] = TreasuryYield(
                                        maturity=maturity,
                                        yield_value=rate,
                                        date=datetime.strptime(date_str, '%Y-%m-%d'),
                                        change_daily=0,  # Would need historical data
                                        change_weekly=0
                                    )
                                    break
                                    
        except Exception as e:
            logger.error(f"Error fetching Treasury yields: {e}")
            
        return yields
    
    async def _fetch_bls_data(
        self, 
        session: aiohttp.ClientSession
    ) -> Dict[str, LaborStatistic]:
        """Fetch labor statistics from BLS API"""
        
        labor_stats = {}
        
        try:
            # Prepare request for multiple series
            series_ids = list(self.bls_series.values())[:5]  # Limit to avoid rate limits
            
            # Use v2 API if key is available, otherwise v1
            if self.bls_api_key:
                url = f"{self.endpoints['bls']}/v2/timeseries/data/"
                payload = {
                    'seriesid': series_ids,
                    'startyear': datetime.now().year - 1,
                    'endyear': datetime.now().year,
                    'registrationkey': self.bls_api_key
                }
            else:
                url = f"{self.endpoints['bls']}/v1/timeseries/data/"
                payload = {
                    'seriesid': series_ids,
                    'startyear': datetime.now().year - 1,
                    'endyear': datetime.now().year
                }
            
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('status') == 'REQUEST_SUCCEEDED':
                        for series in data.get('Results', {}).get('series', []):
                            series_id = series.get('seriesID')
                            series_data = series.get('data', [])
                            
                            if series_data:
                                latest = series_data[0]
                                
                                # Find the metric name
                                metric_name = next(
                                    (k for k, v in self.bls_series.items() if v == series_id),
                                    series_id
                                )
                                
                                # Calculate YoY change
                                yoy_change = 0
                                for point in series_data:
                                    if point['period'] == latest['period'] and \
                                       int(point['year']) == int(latest['year']) - 1:
                                        yoy_change = float(latest['value']) - float(point['value'])
                                        break
                                
                                labor_stats[metric_name] = LaborStatistic(
                                    metric=metric_name,
                                    value=float(latest['value']),
                                    date=datetime.strptime(
                                        f"{latest['year']}-{latest['period'][1:]}-01",
                                        '%Y-%m-%d'
                                    ),
                                    seasonally_adjusted=True,  # Most BLS data is SA
                                    year_over_year_change=yoy_change
                                )
                                
        except Exception as e:
            logger.error(f"Error fetching BLS data: {e}")
            
        return labor_stats
    
    async def _fetch_treasury_debt_data(
        self, 
        session: aiohttp.ClientSession
    ) -> Dict[str, Any]:
        """Fetch US debt data from Treasury API"""
        
        debt_data = {}
        
        try:
            # Fetch debt to the penny
            url = f"{self.endpoints['treasury']}{self.treasury_endpoints['debt_to_penny']}"
            params = {
                'sort': '-record_date',
                'page[size]': 10
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    records = data.get('data', [])
                    
                    if records:
                        latest = records[0]
                        
                        debt_data = {
                            'total_debt': float(latest.get('tot_pub_debt_out_amt', 0)),
                            'date': latest.get('record_date'),
                            'debt_held_by_public': float(latest.get('debt_held_public_amt', 0)),
                            'intragovernmental_holdings': float(latest.get('intragov_hold_amt', 0))
                        }
                        
                        # Calculate changes
                        if len(records) > 1:
                            previous = records[1]
                            debt_data['daily_change'] = (
                                debt_data['total_debt'] - 
                                float(previous.get('tot_pub_debt_out_amt', 0))
                            )
                            
        except Exception as e:
            logger.error(f"Error fetching Treasury debt data: {e}")
            
        return debt_data
    
    def _analyze_economic_conditions(
        self,
        fred_data: Dict,
        treasury_yields: Dict,
        bls_data: Dict,
        debt_data: Dict
    ) -> Dict[str, Any]:
        """Analyze economic conditions from multiple data sources"""
        
        analysis = {
            'yield_curve': self._analyze_yield_curve(fred_data, treasury_yields),
            'inflation_outlook': self._analyze_inflation(fred_data, bls_data),
            'employment_health': self._analyze_employment(fred_data, bls_data),
            'monetary_policy': self._analyze_monetary_policy(fred_data),
            'fiscal_health': self._analyze_fiscal_health(debt_data),
            'economic_regime': self._determine_economic_regime(fred_data, bls_data)
        }
        
        return analysis
    
    def _analyze_yield_curve(
        self, 
        fred_data: Dict, 
        treasury_yields: Dict
    ) -> Dict[str, Any]:
        """Analyze yield curve shape and implications"""
        
        ten_year = fred_data.get('10_year_treasury', {}).get('value')
        two_year = fred_data.get('2_year_treasury', {}).get('value')
        
        if ten_year and two_year:
            spread = ten_year - two_year
            
            if spread < 0:
                shape = 'inverted'
                implication = 'recession_warning'
            elif spread < 0.5:
                shape = 'flat'
                implication = 'slow_growth'
            elif spread < 1.5:
                shape = 'normal'
                implication = 'moderate_growth'
            else:
                shape = 'steep'
                implication = 'strong_growth_expectations'
                
            return {
                'shape': shape,
                'spread_10_2': spread,
                'implication': implication
            }
            
        return {'shape': 'unknown', 'spread_10_2': None, 'implication': 'insufficient_data'}
    
    def _analyze_inflation(
        self, 
        fred_data: Dict, 
        bls_data: Dict
    ) -> Dict[str, Any]:
        """Analyze inflation trends and outlook"""
        
        cpi = fred_data.get('cpi', {}).get('value')
        core_cpi = fred_data.get('core_cpi', {}).get('value')
        
        if cpi:
            if cpi > 4:
                outlook = 'high_inflation'
                fed_action = 'likely_hawkish'
            elif cpi > 2.5:
                outlook = 'above_target'
                fed_action = 'moderate_tightening'
            elif cpi > 1.5:
                outlook = 'at_target'
                fed_action = 'neutral'
            else:
                outlook = 'below_target'
                fed_action = 'likely_dovish'
                
            return {
                'current_cpi': cpi,
                'core_cpi': core_cpi,
                'outlook': outlook,
                'fed_action_likely': fed_action
            }
            
        return {'outlook': 'unknown', 'fed_action_likely': 'uncertain'}
    
    def _analyze_employment(
        self, 
        fred_data: Dict, 
        bls_data: Dict
    ) -> Dict[str, Any]:
        """Analyze employment conditions"""
        
        unemployment = fred_data.get('unemployment_rate', {}).get('value')
        
        if unemployment:
            if unemployment < 3.5:
                strength = 'very_strong'
                wage_pressure = 'high'
            elif unemployment < 4.5:
                strength = 'strong'
                wage_pressure = 'moderate'
            elif unemployment < 5.5:
                strength = 'moderate'
                wage_pressure = 'low'
            else:
                strength = 'weak'
                wage_pressure = 'none'
                
            return {
                'unemployment_rate': unemployment,
                'labor_market_strength': strength,
                'wage_pressure': wage_pressure
            }
            
        return {'labor_market_strength': 'unknown'}
    
    def _analyze_monetary_policy(self, fred_data: Dict) -> Dict[str, Any]:
        """Analyze monetary policy stance"""
        
        fed_funds = fred_data.get('fed_funds_rate', {}).get('value')
        m2_growth = fred_data.get('m2_money_supply', {}).get('change_percent')
        
        if fed_funds:
            if fed_funds > 5:
                stance = 'restrictive'
            elif fed_funds > 3:
                stance = 'neutral_to_restrictive'
            elif fed_funds > 1:
                stance = 'accommodative'
            else:
                stance = 'highly_accommodative'
                
            return {
                'fed_funds_rate': fed_funds,
                'policy_stance': stance,
                'm2_growth': m2_growth
            }
            
        return {'policy_stance': 'unknown'}
    
    def _analyze_fiscal_health(self, debt_data: Dict) -> Dict[str, Any]:
        """Analyze fiscal health indicators"""
        
        if debt_data:
            total_debt = debt_data.get('total_debt', 0)
            
            # Simple debt level assessment (would need GDP for debt/GDP ratio)
            if total_debt > 30_000_000_000_000:  # $30 trillion
                health = 'concerning'
            elif total_debt > 25_000_000_000_000:
                health = 'elevated'
            else:
                health = 'manageable'
                
            return {
                'total_debt': total_debt,
                'fiscal_health': health,
                'daily_change': debt_data.get('daily_change', 0)
            }
            
        return {'fiscal_health': 'unknown'}
    
    def _determine_economic_regime(
        self, 
        fred_data: Dict, 
        bls_data: Dict
    ) -> str:
        """Determine the current economic regime"""
        
        gdp_growth = fred_data.get('gdp_growth_rate', {}).get('value')
        inflation = fred_data.get('cpi', {}).get('value')
        unemployment = fred_data.get('unemployment_rate', {}).get('value')
        
        if gdp_growth and inflation and unemployment:
            if gdp_growth > 2 and inflation < 3 and unemployment < 4:
                return 'goldilocks'  # Ideal conditions
            elif gdp_growth > 2 and inflation > 4:
                return 'overheating'
            elif gdp_growth < 1 and inflation > 3:
                return 'stagflation'
            elif gdp_growth < 0:
                return 'recession'
            elif gdp_growth > 3:
                return 'expansion'
            else:
                return 'moderate_growth'
                
        return 'uncertain'
    
    def _determine_market_implications(self, analysis: Dict) -> Dict[str, Any]:
        """Determine market implications from economic analysis"""
        
        implications = {
            'equity_outlook': 'neutral',
            'bond_outlook': 'neutral',
            'dollar_outlook': 'neutral',
            'commodity_outlook': 'neutral',
            'sector_rotations': []
        }
        
        # Yield curve implications
        yield_curve = analysis.get('yield_curve', {})
        if yield_curve.get('shape') == 'inverted':
            implications['equity_outlook'] = 'bearish'
            implications['bond_outlook'] = 'bullish'
            implications['sector_rotations'].append('defensive_sectors')
            
        # Inflation implications
        inflation = analysis.get('inflation_outlook', {})
        if inflation.get('outlook') == 'high_inflation':
            implications['commodity_outlook'] = 'bullish'
            implications['bond_outlook'] = 'bearish'
            implications['sector_rotations'].append('commodities_and_energy')
            
        # Employment implications
        employment = analysis.get('employment_health', {})
        if employment.get('labor_market_strength') == 'very_strong':
            implications['equity_outlook'] = 'bullish'
            implications['dollar_outlook'] = 'bullish'
            
        # Economic regime implications
        regime = analysis.get('economic_regime')
        regime_implications = {
            'goldilocks': {
                'equity_outlook': 'very_bullish',
                'bond_outlook': 'neutral',
                'sectors': ['technology', 'consumer_discretionary']
            },
            'recession': {
                'equity_outlook': 'bearish',
                'bond_outlook': 'bullish',
                'sectors': ['utilities', 'consumer_staples', 'healthcare']
            },
            'stagflation': {
                'equity_outlook': 'bearish',
                'bond_outlook': 'bearish',
                'commodity_outlook': 'bullish',
                'sectors': ['energy', 'materials']
            },
            'expansion': {
                'equity_outlook': 'bullish',
                'sectors': ['financials', 'industrials', 'technology']
            }
        }
        
        if regime in regime_implications:
            regime_impl = regime_implications[regime]
            implications['equity_outlook'] = regime_impl.get('equity_outlook', implications['equity_outlook'])
            implications['bond_outlook'] = regime_impl.get('bond_outlook', implications['bond_outlook'])
            implications['commodity_outlook'] = regime_impl.get('commodity_outlook', implications['commodity_outlook'])
            implications['sector_rotations'].extend(regime_impl.get('sectors', []))
            
        return implications
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        
        if key in self.cache and key in self.cache_expiry:
            return datetime.utcnow() < self.cache_expiry[key]
        return False
    
    def _cache_data(self, key: str, data: Any, minutes: int = 15):
        """Cache data with expiration"""
        
        self.cache[key] = data
        self.cache_expiry[key] = datetime.utcnow() + timedelta(minutes=minutes)
    
    async def get_specific_indicator(
        self, 
        indicator_name: str, 
        source: GovernmentDataSource = GovernmentDataSource.FRED
    ) -> Optional[EconomicDataPoint]:
        """Fetch a specific economic indicator"""
        
        async with aiohttp.ClientSession() as session:
            if source == GovernmentDataSource.FRED:
                data = await self._fetch_fred_indicators(session)
                if indicator_name in data:
                    return EconomicDataPoint(
                        indicator=indicator_name,
                        value=data[indicator_name]['value'],
                        timestamp=datetime.strptime(data[indicator_name]['date'], '%Y-%m-%d'),
                        source=source,
                        unit='percent' if 'rate' in indicator_name else 'index',
                        frequency='daily',
                        metadata=data[indicator_name]
                    )
            elif source == GovernmentDataSource.BLS:
                data = await self._fetch_bls_data(session)
                if indicator_name in data:
                    stat = data[indicator_name]
                    return EconomicDataPoint(
                        indicator=indicator_name,
                        value=stat.value,
                        timestamp=stat.date,
                        source=source,
                        unit='index',
                        frequency='monthly',
                        metadata={'seasonally_adjusted': stat.seasonally_adjusted}
                    )
                    
        return None


# Create global instance
government_data_service = GovernmentDataService()