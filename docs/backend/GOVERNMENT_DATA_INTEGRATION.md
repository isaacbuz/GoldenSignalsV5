# Government Data Integration Guide

## Overview

The GoldenSignalsAI platform integrates official US government economic data APIs to provide institutional-grade economic analysis for trading decisions. This integration provides access to over 840,000 economic time series from trusted government sources.

## Data Sources

### 1. FRED (Federal Reserve Economic Data)
- **Provider**: Federal Reserve Bank of St. Louis
- **API Key Required**: Yes (Free)
- **Get API Key**: https://fredaccount.stlouisfed.org/apikeys
- **Data Available**: 840,000+ economic time series
- **Update Frequency**: Real-time to monthly

**Key Indicators**:
- Federal Funds Rate (DFF)
- Treasury Yields (DGS2, DGS10, DGS30)
- Unemployment Rate (UNRATE)
- Consumer Price Index (CPIAUCSL)
- GDP Growth (GDPC1)
- VIX Volatility Index
- Dollar Index (DXY)
- Commodity Prices (Oil, Gold)

### 2. US Treasury Fiscal Data API
- **Provider**: US Department of Treasury
- **API Key Required**: No (Completely Free)
- **Documentation**: https://fiscaldata.treasury.gov/api-documentation/
- **Data Available**: Treasury securities, debt data, auction results
- **Update Frequency**: Daily

**Key Data**:
- Daily Treasury Yields
- Auction Results
- Debt to the Penny
- Treasury Securities Outstanding

### 3. BLS (Bureau of Labor Statistics) API
- **Provider**: US Bureau of Labor Statistics
- **API Key Required**: Optional (Enhanced features with registration)
- **Get API Key**: https://www.bls.gov/developers/home.htm
- **Data Available**: Employment, inflation, productivity data
- **Update Frequency**: Monthly

**Key Statistics**:
- Unemployment Rate
- Consumer Price Index (CPI)
- Producer Price Index (PPI)
- Employment Cost Index
- Job Openings (JOLTS)

## Setup

### 1. Obtain API Keys

```bash
# Add to backend/.env file
FRED_API_KEY=your_fred_api_key_here
BLS_API_KEY=your_bls_api_key_here  # Optional
```

### 2. Verify Installation

```python
# Test the service
from services.government_data_service import government_data_service

# Get comprehensive data
data = await government_data_service.get_comprehensive_economic_data()
print(data.keys())
# Output: ['timestamp', 'fred_indicators', 'treasury_yields', 'labor_statistics', 'debt_metrics', 'analysis', 'market_implications']
```

## API Endpoints

### Get All Economic Data
```
GET /api/government/economic-data
```
Returns comprehensive economic data from all sources.

### Get Specific Indicator
```
GET /api/government/indicators/{indicator_name}?source=fred
```
Returns specific economic indicator data.

### Get Treasury Yields
```
GET /api/government/treasury/yields
```
Returns current US Treasury yields for all maturities.

### Get Labor Statistics
```
GET /api/government/bls/labor-stats
```
Returns unemployment, CPI, and other labor statistics.

### Get Market Regime Analysis
```
GET /api/government/market-regime
```
Determines current market regime based on economic conditions.

### Get Trading Signal
```
GET /api/government/analysis/{symbol}
```
Generates trading signal for a symbol based on government data.

## Usage Examples

### Python Client Example

```python
import aiohttp
import asyncio

async def get_economic_data():
    async with aiohttp.ClientSession() as session:
        # Get comprehensive data
        async with session.get('http://localhost:8000/api/government/economic-data') as resp:
            data = await resp.json()
            print(f"Fed Funds Rate: {data['data']['fred_indicators']['fed_funds_rate']['value']}%")
            print(f"Market Regime: {data['data']['analysis']['economic_regime']}")
        
        # Get trading signal
        async with session.get('http://localhost:8000/api/government/analysis/SPY') as resp:
            signal = await resp.json()
            print(f"Signal: {signal['signal']['action']}")
            print(f"Confidence: {signal['signal']['confidence']:.2%}")

asyncio.run(get_economic_data())
```

### JavaScript/Frontend Example

```javascript
// Get economic indicators
fetch('/api/government/economic-data')
  .then(res => res.json())
  .then(data => {
    console.log('Economic Data:', data);
    
    // Display key indicators
    const indicators = data.data.fred_indicators;
    console.log(`Unemployment: ${indicators.unemployment_rate.value}%`);
    console.log(`Inflation (CPI): ${indicators.cpi.value}%`);
    console.log(`10Y Treasury: ${indicators['10_year_treasury'].value}%`);
  });

// Get market regime
fetch('/api/government/market-regime')
  .then(res => res.json())
  .then(data => {
    console.log(`Market Regime: ${data.regime}`);
    console.log(`Yield Curve Shape: ${data.yield_curve.shape}`);
    console.log(`Inflation Outlook: ${data.inflation_outlook.outlook}`);
  });
```

## Market Regime Detection

The system identifies the following market regimes:

### 1. **Goldilocks** (Ideal Conditions)
- GDP Growth: 2-3.5%
- Inflation: 1.5-2.5%
- Unemployment: 3.5-4.5%
- **Favored Sectors**: Technology, Consumer Discretionary, Financials

### 2. **Recession**
- GDP Growth: < 0.5%
- Unemployment: > 5%
- Inverted Yield Curve
- **Favored Sectors**: Utilities, Healthcare, Consumer Staples

### 3. **Stagflation**
- GDP Growth: 0-1.5%
- Inflation: > 4%
- **Favored Sectors**: Energy, Materials, Utilities

### 4. **Overheating**
- GDP Growth: > 3.5%
- Inflation: > 3.5%
- Unemployment: < 3.5%
- **Favored Sectors**: Energy, Materials, Industrials

## Trading Signal Generation

The Government Data Agent analyzes economic indicators to generate trading signals:

1. **Indicator Analysis**: Evaluates each economic indicator against thresholds
2. **Regime Detection**: Identifies current market regime
3. **Sector Impact**: Calculates impact on different market sectors
4. **Signal Generation**: Produces BUY/HOLD/SELL signals with confidence levels
5. **Risk Assessment**: Identifies economic risks

### Signal Confidence Factors

- **Data Completeness**: More indicators = higher confidence
- **Indicator Agreement**: Consistent signals across indicators
- **Regime Clarity**: Clear economic regime identification
- **Data Freshness**: Recent data updates

## Key Economic Relationships

### Interest Rates Impact
- **Rising Rates**: 
  - Negative: Technology, Real Estate, Utilities
  - Positive: Financials (banks), Dollar
  
### Inflation Impact
- **High Inflation**:
  - Positive: Energy, Materials, Commodities
  - Negative: Bonds, Technology, Consumer Discretionary

### Unemployment Impact
- **Rising Unemployment**:
  - Negative: Consumer sectors, Financials
  - Positive: Defensive sectors (Utilities, Consumer Staples)

### Yield Curve
- **Inverted** (10Y < 2Y): Recession warning
- **Steep** (10Y >> 2Y): Growth expectations
- **Flat**: Economic uncertainty

## Performance Optimization

### Caching Strategy
- Economic indicators cached for 15 minutes
- Treasury data cached for 5 minutes
- Labor statistics cached for 1 hour

### Rate Limits
- **FRED API**: 120 requests per minute
- **BLS API**: 
  - Without key: 25 requests per day
  - With key: 500 requests per day
- **Treasury API**: No limits

### Best Practices
1. Use batch requests when possible
2. Cache responses appropriately
3. Implement fallback mechanisms
4. Monitor API usage
5. Handle errors gracefully

## Troubleshooting

### Common Issues

1. **No FRED API Key**
   - Error: "FRED API key not configured"
   - Solution: Add FRED_API_KEY to .env file

2. **BLS Rate Limit**
   - Error: "BLS rate limit exceeded"
   - Solution: Register for API key or reduce request frequency

3. **Stale Data**
   - Issue: Old economic data
   - Solution: Check data timestamps, clear cache if needed

4. **Missing Indicators**
   - Issue: Some indicators not available
   - Solution: Check API status, use fallback data sources

## Advanced Features

### Custom Indicator Sets

```python
# Get specific indicators
indicators = await government_data_service.get_specific_indicator(
    'fed_funds_rate',
    GovernmentDataSource.FRED
)
```

### Historical Analysis

```python
# Analyze historical patterns
analysis = government_data_service._analyze_economic_conditions(
    fred_data, treasury_yields, bls_data, debt_data
)
```

### Sector Rotation

```python
# Get sector recommendations based on economic conditions
from agents.government_data_agent import government_data_agent

signal = await government_data_agent.analyze({'symbol': 'XLF'})  # Financials ETF
print(signal.features['sector_recommendations'])
```

## Integration with Other Services

The government data integrates seamlessly with:
- **Enhanced Data Aggregator**: Provides economic context
- **Economic Indicator Agent**: Analyzes for trading signals
- **Risk Management**: Identifies macro risks
- **Market Regime Agent**: Determines market conditions

## Future Enhancements

Planned additions:
- European Central Bank (ECB) data
- Bank of England (BoE) data
- World Bank indicators
- IMF economic forecasts
- Congressional Budget Office (CBO) projections
- More granular regional Fed data

## Resources

- [FRED API Documentation](https://fred.stlouisfed.org/docs/api/fred/)
- [Treasury Fiscal Data API](https://fiscaldata.treasury.gov/api-documentation/)
- [BLS API Documentation](https://www.bls.gov/developers/)
- [Economic Calendar](https://www.marketwatch.com/economy-politics/calendar)
- [Federal Reserve Economic Data](https://fred.stlouisfed.org/)