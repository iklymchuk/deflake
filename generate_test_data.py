#!/usr/bin/env python3
"""
Generate realistic test data for deflake testing.

Usage:
    python3 generate_test_data.py
    
This creates 10 runs of test data in deflake_reports/ with intentional flakiness
in some tests to simulate real-world scenarios.

Output structure:
    deflake_reports/
    ├── pytest_run_01.json
    ├── pytest_run_02.json
    └── ... pytest_run_10.json

Each file contains test results that can be analyzed with:
    deflake --input ./deflake_reports/
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path


def generate_test_data(num_runs: int = 10):
    """
    Generate realistic test data with flakiness patterns.
    
    Args:
        num_runs: Number of test runs to generate (default 10)
    """
    
    report_dir = Path("deflake_reports")
    report_dir.mkdir(exist_ok=True)
    
    # Define test names
    tests = [
        "tests/test_login.py::test_valid_credentials",
        "tests/test_login.py::test_invalid_credentials",
        "tests/test_api.py::test_get_users",
        "tests/test_api.py::test_create_user",
        "tests/test_api.py::test_delete_user",
        "tests/test_ui.py::test_homepage_load",
    ]
    
    # Make these tests flaky (fail intermittently)
    flaky_tests = {
        "tests/test_login.py::test_valid_credentials": 0.35,  # 35% failure rate
        "tests/test_api.py::test_get_users": 0.40,            # 40% failure rate
    }
    
    start_time = datetime(2025, 1, 15, 10, 0, 0)
    
    print(f"🔄 Generating {num_runs} test runs...")
    print(f"📊 Flaky tests: {list(flaky_tests.keys())}\n")
    
    # Generate N runs
    for run_num in range(1, num_runs + 1):
        test_runs = []
        
        for test in tests:
            # Determine failure rate
            if test in flaky_tests:
                # Flaky: use configured failure rate
                failure_prob = flaky_tests[test]
            else:
                # Stable: 5% failure rate (rare)
                failure_prob = 0.05
            
            # Simulate multiple runs per test (1-3 failures per test execution)
            total_attempts = 3
            fail_count = sum(1 for _ in range(total_attempts) if random.random() < failure_prob)
            pass_count = total_attempts - fail_count
            
            test_run = {
                "test_id": test,
                "execution_id": f"pytest_run_{run_num:02d}",
                "timestamp": (start_time + timedelta(hours=run_num)).isoformat(),
                "build_number": 100 + run_num,
                "pass_count": pass_count,
                "fail_count": fail_count,
                "total_runs": total_attempts
            }
            test_runs.append(test_run)
        
        # Calculate statistics
        total_passed = sum(t['pass_count'] for t in test_runs)
        total_failed = sum(t['fail_count'] for t in test_runs)
        
        # Save to file
        data = {
            "test_runs": test_runs,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "run_number": run_num,
                "total_tests": len(tests),
                "total_runs": len(test_runs),
                "passed": sum(1 for t in test_runs if t['fail_count'] == 0),
                "failed": sum(1 for t in test_runs if t['fail_count'] > 0),
                "total_pass_count": total_passed,
                "total_fail_count": total_failed,
            }
        }
        
        output_file = report_dir / f"pytest_run_{run_num:02d}.json"
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Print progress
        status = "✅" if total_failed == 0 else "⚠️ "
        print(f"{status} Run {run_num:2d}: {total_passed} passes, {total_failed} fails (across {len(test_runs)} tests)")
    
    print(f"\n✅ Generated {num_runs} test runs in deflake_reports/")
    print(f"\n📋 Files created:")
    for i in range(1, num_runs + 1):
        file = report_dir / f"pytest_run_{i:02d}.json"
        if file.exists():
            size = file.stat().st_size
            print(f"   - {file.name} ({size} bytes)")
    
    print(f"\n💡 Next step: Analyze with:")
    print(f"   deflake --input ./deflake_reports/")
    print(f"\n   Or if using local dev:")
    print(f"   python3 main.py --input ./deflake_reports/")


if __name__ == "__main__":
    generate_test_data()
