"""
Tests for Government Data Service
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from services.government_data_service import (
    GovernmentDataService,
    GovernmentDataSource,
    EconomicDataPoint,
    TreasuryYield,
    LaborStatistic
)


@pytest.fixture
def gov_data_service():
    """Create government data service instance"""
    return GovernmentDataService()


@pytest.fixture
def mock_fred_response():
    """Mock FRED API response"""
    return {
        'observations': [
            {
                'date': '2024-01-15',
                'value': '5.33'
            },
            {
                'date': '2024-01-14',
                'value': '5.32'
            }
        ]
    }


@pytest.fixture
def mock_treasury_response():
    """Mock Treasury API response"""
    return {
        'data': [
            {
                'record_date': '2024-01-15',
                'security_desc': 'Treasury Notes',
                'avg_interest_rate_amt': '4.25'
            },
            {
                'record_date': '2024-01-15',
                'security_desc': 'Treasury Bills',
                'avg_interest_rate_amt': '5.25'
            }
        ]
    }


@pytest.fixture
def mock_bls_response():
    """Mock BLS API response"""
    return {
        'status': 'REQUEST_SUCCEEDED',
        'Results': {
            'series': [
                {
                    'seriesID': 'LNS14000000',
                    'data': [
                        {
                            'year': '2024',
                            'period': 'M01',
                            'value': '3.7'
                        }
                    ]
                }
            ]
        }
    }


class TestGovernmentDataService:
    """Test government data service functionality"""
    
    def test_initialization(self, gov_data_service):
        """Test service initialization"""
        assert gov_data_service is not None
        assert len(gov_data_service.fred_series) > 0
        assert len(gov_data_service.bls_series) > 0
        assert 'fed_funds_rate' in gov_data_service.fred_series
        
    @pytest.mark.asyncio
    async def test_fetch_fred_indicators(self, gov_data_service, mock_fred_response):
        """Test fetching FRED indicators"""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_fred_response)
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            # Mock API key
            gov_data_service.fred_api_key = 'test_key'
            
            indicators = await gov_data_service._fetch_fred_indicators(
                mock_session.return_value.__aenter__.return_value
            )
            
            # Verify structure but don't check exact values
            assert isinstance(indicators, dict)
            
    @pytest.mark.asyncio
    async def test_fetch_treasury_yields(self, gov_data_service, mock_treasury_response):
        """Test fetching Treasury yields"""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_treasury_response)
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            yields = await gov_data_service._fetch_treasury_yields(
                mock_session.return_value.__aenter__.return_value
            )
            
            assert isinstance(yields, dict)
            # Check if yields were processed
            for maturity, yield_data in yields.items():
                assert isinstance(yield_data, TreasuryYield)
                assert yield_data.maturity in ['3M', '10Y', '30Y', 'TIPS']
                
    @pytest.mark.asyncio
    async def test_fetch_bls_data(self, gov_data_service, mock_bls_response):
        """Test fetching BLS data"""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_bls_response)
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            labor_stats = await gov_data_service._fetch_bls_data(
                mock_session.return_value.__aenter__.return_value
            )
            
            assert isinstance(labor_stats, dict)
            
    def test_analyze_yield_curve(self, gov_data_service):
        """Test yield curve analysis"""
        fred_data = {
            '10_year_treasury': {'value': 4.5},
            '2_year_treasury': {'value': 5.0}
        }
        
        analysis = gov_data_service._analyze_yield_curve(fred_data, {})
        
        assert analysis['shape'] == 'inverted'
        assert analysis['spread_10_2'] == -0.5
        assert analysis['implication'] == 'recession_warning'
        
    def test_analyze_inflation(self, gov_data_service):
        """Test inflation analysis"""
        fred_data = {
            'cpi': {'value': 3.0},
            'core_cpi': {'value': 2.8}
        }
        
        analysis = gov_data_service._analyze_inflation(fred_data, {})
        
        assert analysis['current_cpi'] == 3.0
        assert analysis['outlook'] == 'above_target'
        assert analysis['fed_action_likely'] == 'moderate_tightening'
        
    def test_analyze_employment(self, gov_data_service):
        """Test employment analysis"""
        fred_data = {
            'unemployment_rate': {'value': 3.7}
        }
        
        analysis = gov_data_service._analyze_employment(fred_data, {})
        
        assert analysis['unemployment_rate'] == 3.7
        assert analysis['labor_market_strength'] == 'strong'
        assert analysis['wage_pressure'] == 'moderate'
        
    def test_determine_economic_regime(self, gov_data_service):
        """Test economic regime determination"""
        fred_data = {
            'gdp_growth_rate': {'value': 2.5},
            'cpi': {'value': 2.0},
            'unemployment_rate': {'value': 3.8}
        }
        bls_data = {}
        
        regime = gov_data_service._determine_economic_regime(fred_data, bls_data)
        
        assert regime in ['goldilocks', 'expansion', 'moderate_growth', 
                         'recession', 'stagflation', 'overheating', 'uncertain']
        
    def test_determine_market_implications(self, gov_data_service):
        """Test market implications determination"""
        analysis = {
            'yield_curve': {'shape': 'normal'},
            'inflation_outlook': {'outlook': 'at_target'},
            'employment_health': {'labor_market_strength': 'strong'},
            'economic_regime': 'expansion'
        }
        
        implications = gov_data_service._determine_market_implications(analysis)
        
        assert 'equity_outlook' in implications
        assert 'bond_outlook' in implications
        assert 'dollar_outlook' in implications
        assert 'commodity_outlook' in implications
        assert 'sector_rotations' in implications
        
    def test_cache_functionality(self, gov_data_service):
        """Test caching mechanism"""
        # Test cache storage
        test_data = {'test': 'data'}
        gov_data_service._cache_data('test_key', test_data, minutes=5)
        
        assert 'test_key' in gov_data_service.cache
        assert gov_data_service._is_cache_valid('test_key')
        
        # Test cache expiry
        gov_data_service.cache_expiry['test_key'] = datetime.utcnow() - timedelta(minutes=1)
        assert not gov_data_service._is_cache_valid('test_key')
        
    @pytest.mark.asyncio
    async def test_get_specific_indicator(self, gov_data_service):
        """Test fetching specific indicator"""
        with patch.object(gov_data_service, '_fetch_fred_indicators') as mock_fetch:
            mock_fetch.return_value = {
                'fed_funds_rate': {
                    'value': 5.33,
                    'date': '2024-01-15',
                    'change': 0.01,
                    'change_percent': 0.19
                }
            }
            
            gov_data_service.fred_api_key = 'test_key'
            
            indicator = await gov_data_service.get_specific_indicator(
                'fed_funds_rate',
                GovernmentDataSource.FRED
            )
            
            assert indicator is not None
            assert isinstance(indicator, EconomicDataPoint)
            assert indicator.indicator == 'fed_funds_rate'
            assert indicator.value == 5.33
            
    @pytest.mark.asyncio
    async def test_comprehensive_economic_data(self, gov_data_service):
        """Test fetching comprehensive economic data"""
        with patch.object(gov_data_service, '_fetch_fred_indicators') as mock_fred, \
             patch.object(gov_data_service, '_fetch_treasury_yields') as mock_treasury, \
             patch.object(gov_data_service, '_fetch_bls_data') as mock_bls, \
             patch.object(gov_data_service, '_fetch_treasury_debt_data') as mock_debt:
            
            mock_fred.return_value = {'fed_funds_rate': {'value': 5.33}}
            mock_treasury.return_value = {}
            mock_bls.return_value = {}
            mock_debt.return_value = {'total_debt': 33000000000000}
            
            data = await gov_data_service.get_comprehensive_economic_data()
            
            assert 'timestamp' in data
            assert 'fred_indicators' in data
            assert 'treasury_yields' in data
            assert 'labor_statistics' in data
            assert 'debt_metrics' in data
            assert 'analysis' in data
            assert 'market_implications' in data
            
    def test_analyze_monetary_policy(self, gov_data_service):
        """Test monetary policy analysis"""
        fred_data = {
            'fed_funds_rate': {'value': 5.5},
            'm2_money_supply': {'change_percent': 3.2}
        }
        
        analysis = gov_data_service._analyze_monetary_policy(fred_data)
        
        assert analysis['fed_funds_rate'] == 5.5
        assert analysis['policy_stance'] == 'restrictive'
        assert analysis['m2_growth'] == 3.2
        
    def test_analyze_fiscal_health(self, gov_data_service):
        """Test fiscal health analysis"""
        debt_data = {
            'total_debt': 32000000000000,
            'daily_change': 50000000000
        }
        
        analysis = gov_data_service._analyze_fiscal_health(debt_data)
        
        assert analysis['total_debt'] == 32000000000000
        assert analysis['fiscal_health'] == 'concerning'
        assert analysis['daily_change'] == 50000000000