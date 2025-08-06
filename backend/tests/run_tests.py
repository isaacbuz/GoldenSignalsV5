#!/usr/bin/env python3
"""
Comprehensive test runner for GoldenSignalsAI
Runs all tests with coverage and generates reports
"""

import subprocess
import argparse
from datetime import datetime
import json


class TestRunner:
    """Test runner with various options and reporting"""
    
    def __init__(self):
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "coverage": 0.0,
            "duration": 0.0
        }
    
    def run_tests(self, test_type="all", verbose=True, coverage=True, markers=None):
        """Run tests with specified options"""
        cmd = ["pytest"]
        
        # Add test type
        if test_type == "unit":
            cmd.extend(["-m", "unit"])
        elif test_type == "integration":
            cmd.extend(["-m", "integration"])
        elif test_type == "performance":
            cmd.extend(["-m", "performance"])
        elif test_type == "specific" and markers:
            cmd.extend(["-m", markers])
        
        # Add verbosity
        if verbose:
            cmd.append("-v")
        
        # Add coverage if requested
        if coverage:
            cmd.extend([
                "--cov=.",
                "--cov-report=html",
                "--cov-report=term-missing",
                "--cov-report=json"
            ])
        
        # Run tests
        print(f"Running command: {' '.join(cmd)}")
        print("-" * 80)
        
        start_time = datetime.now()
        result = subprocess.run(cmd, capture_output=False)
        duration = (datetime.now() - start_time).total_seconds()
        
        self.test_results["duration"] = duration
        
        # Parse coverage if available
        if coverage and os.path.exists("coverage.json"):
            with open("coverage.json", "r") as f:
                cov_data = json.load(f)
                self.test_results["coverage"] = cov_data.get("totals", {}).get("percent_covered", 0)
        
        return result.returncode == 0
    
    def run_specific_test_file(self, file_path, verbose=True):
        """Run a specific test file"""
        cmd = ["pytest", file_path]
        if verbose:
            cmd.append("-v")
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        return result.returncode == 0
    
    def run_test_suite(self):
        """Run complete test suite in order"""
        print("🧪 GoldenSignalsAI Test Suite")
        print("=" * 80)
        
        test_categories = [
            ("Unit Tests", "unit"),
            ("Integration Tests", "integration"),
            ("WebSocket Tests", "websocket"),
            ("Agent Tests", "agents"),
            ("Data Acquisition Tests", "data"),
            ("Performance Tests", "performance"),
        ]
        
        all_passed = True
        
        for name, marker in test_categories:
            print(f"\n📋 Running {name}...")
            print("-" * 40)
            
            passed = self.run_tests(test_type="specific", markers=marker, coverage=False)
            if not passed:
                all_passed = False
                print(f"❌ {name} failed!")
            else:
                print(f"✅ {name} passed!")
        
        # Run with coverage at the end
        print("\n📊 Running all tests with coverage...")
        print("-" * 80)
        self.run_tests(test_type="all", coverage=True)
        
        return all_passed
    
    def generate_report(self):
        """Generate test report"""
        print("\n" + "=" * 80)
        print("📊 TEST REPORT")
        print("=" * 80)
        
        print(f"Timestamp: {self.test_results['timestamp']}")
        print(f"Duration: {self.test_results['duration']:.2f} seconds")
        print(f"Coverage: {self.test_results['coverage']:.2f}%")
        
        # Save report
        with open("test_report.json", "w") as f:
            json.dump(self.test_results, f, indent=2)
        
        print("\nReports generated:")
        print("- HTML Coverage: htmlcov/index.html")
        print("- JSON Report: test_report.json")
        print("- XML Coverage: coverage.xml")
    
    def check_test_requirements(self):
        """Check if all test requirements are met"""
        print("🔍 Checking test requirements...")
        
        requirements = {
            "pytest": "pytest",
            "pytest-asyncio": "pytest-asyncio",
            "pytest-cov": "pytest-cov",
            "pytest-mock": "pytest-mock",
            "pytest-timeout": "pytest-timeout",
        }
        
        missing = []
        for name, package in requirements.items():
            try:
                __import__(package.replace("-", "_"))
                print(f"✅ {name} installed")
            except ImportError:
                print(f"❌ {name} missing")
                missing.append(package)
        
        if missing:
            print(f"\n⚠️  Missing packages: {', '.join(missing)}")
            print(f"Install with: pip install {' '.join(missing)}")
            return False
        
        return True


def main():
    """Main test runner entry point"""
    parser = argparse.ArgumentParser(description="GoldenSignalsAI Test Runner")
    parser.add_argument(
        "--type",
        choices=["all", "unit", "integration", "performance", "suite"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--file",
        help="Run specific test file"
    )
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Skip coverage reporting"
    )
    parser.add_argument(
        "--markers",
        help="Run tests with specific markers"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check requirements"
    )
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    # Check requirements first
    if not runner.check_test_requirements():
        if not args.check_only:
            print("\n❌ Test requirements not met!")
            sys.exit(1)
        sys.exit(0)
    
    if args.check_only:
        print("\n✅ All test requirements met!")
        sys.exit(0)
    
    # Run tests
    success = False
    
    if args.file:
        success = runner.run_specific_test_file(args.file)
    elif args.type == "suite":
        success = runner.run_test_suite()
    else:
        success = runner.run_tests(
            test_type=args.type,
            coverage=not args.no_coverage,
            markers=args.markers
        )
    
    # Generate report
    runner.generate_report()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()