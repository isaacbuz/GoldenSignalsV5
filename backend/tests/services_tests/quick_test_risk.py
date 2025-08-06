#!/usr/bin/env python3
"""Quick test for Risk Management Agent"""

import asyncio
from agents.risk_management_agent import risk_management_agent, RiskLevel

async def quick_test():
    print('🔒 Testing Risk Management Agent V5...')
    
    # Test portfolio data
    portfolio_data = {
        'name': 'Test Risk Portfolio',
        'cash': 50000,
        'type': 'mixed',
        'positions': [
            {
                'symbol': 'AAPL',
                'quantity': 100,
                'entry_price': 150.0,
                'current_price': 175.0,
                'position_type': 'long',
                'asset_class': 'equity',
                'sector': 'Technology'
            },
            {
                'symbol': 'GOOGL', 
                'quantity': 50,
                'entry_price': 2800.0,
                'current_price': 2750.0,
                'position_type': 'long',
                'asset_class': 'equity',
                'sector': 'Technology'
            }
        ]
    }
    
    # Test portfolio creation
    result = await risk_management_agent.create_portfolio(portfolio_data)
    portfolio_id = result['portfolio_id']
    
    print(f'Portfolio Created: {portfolio_id}')
    print(f'Total Value: ${result["total_value"]:,.0f}')
    print(f'Leverage: {result["leverage"]:.2f}x')
    
    # Test risk assessment
    assessment = await risk_management_agent.assess_portfolio_risk(portfolio_id)
    
    print(f'Risk Level: {assessment.overall_risk_level.value.upper()}')
    print(f'Risk Score: {assessment.risk_score:.1f}/100')
    print(f'Volatility: {assessment.metrics["volatility"]*100:.1f}%')
    print(f'Sharpe Ratio: {assessment.metrics["sharpe_ratio"]:.2f}')
    print(f'Recommendations: {len(assessment.recommendations)}')
    
    # Test VaR calculation
    var_result = await risk_management_agent.calculate_var(portfolio_id, confidence_level=0.95)
    print(f'VaR (95%): {var_result["var"]*100:.2f}% (${var_result["var_amount"]:,.0f})')
    
    print('✅ Risk Management Agent V5 Test Passed!')

if __name__ == "__main__":
    asyncio.run(quick_test())