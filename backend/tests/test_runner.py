"""
Comprehensive Test Runner
Runs all test suites with coverage reporting and performance metrics
"""

import pytest
import sys
import os
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any
import coverage
import logging

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('tests/test_results.log')
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveTestRunner:
    """Comprehensive test runner with coverage and performance tracking"""
    
    def __init__(self):
        self.test_suites = {
            'unit': [
                'tests/unit/test_rag_service.py',
                'tests/unit/test_mcp_server.py',
                'tests/test_market_data.py',
                'tests/test_volatility_agent.py'
            ],
            'api': [
                'tests/api/test_market_data_api.py',
                'tests/api/test_agents_api.py', 
                'tests/api/test_auth_api.py'
            ],
            'integration': [
                'tests/integration/test_websocket_integration.py',
                'tests/integration/test_signal_generation_integration.py'
            ],
            'agents': [
                'tests/agents/',
                'tests/test_trading_agents.py'
            ],
            'services': [
                'tests/services/'
            ],
            'performance': [
                'tests/performance/test_performance_benchmarks.py',
                'tests/test_performance_monitoring.py'
            ]
        }
        
        self.coverage = coverage.Coverage(
            source=['api', 'agents', 'core', 'models', 'services'],
            omit=[
                '*/tests/*',
                '*/venv/*',
                '*/migrations/*',
                '*/__pycache__/*'
            ]
        )
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete test suite with coverage"""
        logger.info("Starting comprehensive test run...")
        
        # Start coverage tracking
        self.coverage.start()
        
        results = {
            'start_time': time.time(),
            'suites': {},
            'coverage': {},
            'summary': {}
        }
        
        # Run each test suite
        for suite_name, test_paths in self.test_suites.items():
            logger.info(f"Running {suite_name} tests...")
            suite_result = self._run_test_suite(suite_name, test_paths)
            results['suites'][suite_name] = suite_result
            
        # Stop coverage and generate report
        self.coverage.stop()
        self.coverage.save()
        
        # Generate coverage report
        coverage_result = self._generate_coverage_report()
        results['coverage'] = coverage_result
        
        # Calculate summary
        results['summary'] = self._calculate_summary(results)
        results['end_time'] = time.time()
        results['duration'] = results['end_time'] - results['start_time']
        
        # Generate reports
        self._generate_reports(results)
        
        return results
    
    def _run_test_suite(self, suite_name: str, test_paths: List[str]) -> Dict[str, Any]:
        """Run individual test suite"""
        start_time = time.time()
        
        # Filter existing paths
        existing_paths = []
        for path in test_paths:
            if os.path.exists(path):
                existing_paths.append(path)
            else:
                logger.warning(f"Test path not found: {path}")
        
        if not existing_paths:
            logger.warning(f"No test files found for suite: {suite_name}")
            return {
                'status': 'skipped',
                'reason': 'no_test_files',
                'duration': 0,
                'tests_run': 0,
                'failures': 0,
                'errors': 0
            }
        
        try:
            # Run pytest on the paths
            result = pytest.main([
                '-v',
                '--tb=short',
                '--disable-warnings',
                f'--junitxml=tests/results/{suite_name}_results.xml',
                *existing_paths
            ])
            
            duration = time.time() - start_time
            
            return {
                'status': 'passed' if result == 0 else 'failed',
                'exit_code': result,
                'duration': duration,
                'paths': existing_paths
            }
            
        except Exception as e:
            logger.error(f"Error running {suite_name} tests: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'duration': time.time() - start_time,
                'paths': existing_paths
            }
    
    def _generate_coverage_report(self) -> Dict[str, Any]:
        """Generate coverage report"""
        try:
            # Generate console report
            coverage_report = []
            self.coverage.report(file=coverage_report)
            
            # Generate HTML report
            html_dir = 'tests/coverage_html'
            os.makedirs(html_dir, exist_ok=True)
            self.coverage.html_report(directory=html_dir)
            
            # Get coverage percentage
            total_coverage = self.coverage.report()
            
            return {
                'total_coverage': total_coverage,
                'html_report_dir': html_dir,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error generating coverage report: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _calculate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate test run summary"""
        total_suites = len(results['suites'])
        passed_suites = sum(1 for suite in results['suites'].values() 
                          if suite['status'] == 'passed')
        failed_suites = sum(1 for suite in results['suites'].values() 
                          if suite['status'] == 'failed')
        error_suites = sum(1 for suite in results['suites'].values() 
                         if suite['status'] == 'error')
        skipped_suites = sum(1 for suite in results['suites'].values() 
                           if suite['status'] == 'skipped')
        
        return {
            'total_suites': total_suites,
            'passed_suites': passed_suites,
            'failed_suites': failed_suites,
            'error_suites': error_suites,
            'skipped_suites': skipped_suites,
            'success_rate': (passed_suites / total_suites * 100) if total_suites > 0 else 0,
            'coverage_percentage': results['coverage'].get('total_coverage', 0)
        }
    
    def _generate_reports(self, results: Dict[str, Any]):
        """Generate test reports"""
        os.makedirs('tests/reports', exist_ok=True)
        
        # Generate JSON report
        import json
        with open('tests/reports/test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate human-readable report
        with open('tests/reports/test_summary.md', 'w') as f:
            f.write(self._generate_markdown_report(results))
        
        logger.info(f"Test reports generated in tests/reports/")
    
    def _generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Generate markdown test report"""
        summary = results['summary']
        duration_mins = results['duration'] / 60
        
        report = f"""# Test Results Report

## Summary
- **Total Test Suites**: {summary['total_suites']}
- **Passed**: {summary['passed_suites']}
- **Failed**: {summary['failed_suites']}
- **Errors**: {summary['error_suites']}
- **Skipped**: {summary['skipped_suites']}
- **Success Rate**: {summary['success_rate']:.1f}%
- **Coverage**: {summary['coverage_percentage']:.1f}%
- **Duration**: {duration_mins:.2f} minutes

## Test Suite Details

"""
        
        for suite_name, suite_result in results['suites'].items():
            status_emoji = {
                'passed': '✅',
                'failed': '❌', 
                'error': '💥',
                'skipped': '⏭️'
            }.get(suite_result['status'], '❓')
            
            report += f"""### {status_emoji} {suite_name.title()} Tests
- **Status**: {suite_result['status']}
- **Duration**: {suite_result['duration']:.2f}s
"""
            if 'paths' in suite_result:
                report += f"- **Paths**: {', '.join(suite_result['paths'])}\n"
            if suite_result['status'] == 'error':
                report += f"- **Error**: {suite_result.get('error', 'Unknown')}\n"
            
            report += "\n"
        
        report += f"""## Coverage Report
Coverage details available in: `{results['coverage'].get('html_report_dir', 'N/A')}`

Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return report

def main():
    """Main test runner entry point"""
    runner = ComprehensiveTestRunner()
    
    # Create necessary directories
    os.makedirs('tests/results', exist_ok=True)
    os.makedirs('tests/reports', exist_ok=True)
    
    # Run all tests
    results = runner.run_all_tests()
    
    # Print summary
    summary = results['summary']
    print(f"\n{'='*60}")
    print(f"TEST RUN COMPLETE")
    print(f"{'='*60}")
    print(f"Suites: {summary['passed_suites']}/{summary['total_suites']} passed")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Coverage: {summary['coverage_percentage']:.1f}%")
    print(f"Duration: {results['duration']:.2f}s")
    print(f"{'='*60}")
    
    # Exit with appropriate code
    sys.exit(0 if summary['failed_suites'] + summary['error_suites'] == 0 else 1)

if __name__ == "__main__":
    main()