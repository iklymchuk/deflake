# Statistical Flaky Test Detection System

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

> **A sophisticated, production-ready MVP for detecting flaky tests using statistical analysis and machine learning.**

This project demonstrates advanced test automation engineering principles, combining **Exponentially Weighted Moving Averages (EWMA)**, **Z-score outlier detection**, and **Isolation Forest machine learning** to identify non-deterministic test behavior in CI/CD pipelines.

---

## 🎯 Problem Statement

**Flaky tests** are tests that exhibit non-deterministic behavior—passing and failing intermittently without code changes. They erode confidence in test suites, waste developer time, and can mask real defects.

### Why This Matters
- **Developer Productivity**: Teams spend 15-30% of their time investigating flaky test failures ([Google Testing Blog](https://testing.googleblog.com/))
- **CI/CD Reliability**: Flaky tests delay deployments and reduce trust in automation
- **Test Suite Health**: Undetected flaky tests accumulate technical debt

This system provides **quantitative, data-driven detection** of flaky tests using historical test execution data.

---

## 🔬 Technical Approach

### 1. Exponentially Weighted Moving Average (EWMA)

EWMA gives more weight to recent test failures, detecting trends over time:

```
EWMAₜ = α × Failureₜ + (1 - α) × EWMAₜ₋₁

Where:
  - α (alpha) = smoothing factor (0 < α < 1)
  - Failureₜ = current failure rate (0 or 1)
  - EWMAₜ₋₁ = previous EWMA value
```

**Why EWMA?**
- Reacts quickly to recent changes while maintaining historical context
- Default α = 0.3 balances responsiveness vs. stability
- More robust than simple moving averages for irregular test patterns

### 2. Z-Score Statistical Outlier Detection

Z-score identifies tests whose failure patterns deviate significantly from normal:

```
Z = (X - μ) / σ

Where:
  - X = test's EWMA failure rate
  - μ = mean EWMA across all tests
  - σ = standard deviation of EWMA values
  - |Z| > threshold → flagged as flaky
```

**Interpretation:**
- `|Z| > 2.0` (default): ~95% confidence the test is an outlier
- `|Z| > 3.0`: ~99.7% confidence (stricter threshold)
- Adjustable via `--threshold` parameter

### 3. Isolation Forest (ML-Based Detection)

Scikit-learn's Isolation Forest detects anomalies by isolating outliers in multi-dimensional feature space:

**Features Used:**
- `failure_rate`: Raw pass/fail ratio
- `ewma_failure_rate`: Time-weighted failure trend
- `z_score`: Statistical deviation

**How It Works:**
1. Builds random decision trees that partition data
2. Anomalies are isolated faster (fewer splits needed)
3. Outputs binary classification: `is_flaky_ml`

**Hyperparameters:**
- `contamination=0.1`: Assumes 10% of tests are flaky
- `random_state=42`: Ensures reproducibility

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Ingestion                          │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  CSV/JSON → Pydantic Validation → TestRun Objects    │  │
│  └───────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Flakiness Detection Engine                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────┐  │
│  │  EWMA Calc      │→ │  Z-Score Calc   │→ │  Labeling  │  │
│  │  (Time Series)  │  │  (Outlier Det.) │  │  (Thresh.) │  │
│  └─────────────────┘  └─────────────────┘  └────────────┘  │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Isolation Forest (ML-Based Anomaly Detection)      │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Visualization & Reporting                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Console  │  │   JSON   │  │   CSV    │  │   HTML   │   │
│  │  Output  │  │ (CI/CD)  │  │ (Export) │  │ (Plotly) │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Module Overview

| Module | Responsibility | Key Classes/Functions |
|--------|----------------|----------------------|
| `data_ingestion.py` | Load and validate test data | `TestRun` (Pydantic model), `DataIngestion` |
| `flakiness_detector.py` | Statistical & ML analysis | `FlakinessDetector`, EWMA/Z-score/Isolation Forest |
| `visualizer.py` | Multi-format reporting | `Visualizer`, Plotly charts, JSON/CSV/HTML export |
| `config.py` | Configuration management | Pydantic config models, YAML loading |
| `main.py` | CLI orchestration | argparse, pipeline execution |

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.12+
pip (package manager)
```

### Installation

```bash
pip install deflake
```

Or from source:

```bash
git clone https://github.com/iklymchuk/deflake.git
cd deflake
pip install -e .
```

### Basic Usage

After installing `deflake`:

```bash
# Analyze CSV historical data
deflake --input data/test_history.csv

# Analyze pytest report directory
deflake --input ./deflake_reports/

# Adjust detection threshold (higher = stricter)
deflake --input data/test_history.csv --threshold 2.5

# Export results to custom directory
deflake --input data/test.csv --output-dir ./analysis

# Disable ML detection (use statistical only)
deflake --input data/test.csv --no-ml

# Export as multiple formats
deflake --input data/test.csv --json --csv --html

# Enable debug logging
deflake --input data/test.csv --log-level DEBUG

# View all options
deflake --help
```

### Sample Output

```bash
$ deflake --input data/test_history.csv

======================================================================
🔍 deflake - Statistical Flaky Test Detection
======================================================================

📂 Input: /Users/you/data/test_history.csv
📁 Output: /Users/you/deflake_reports

⏳ Loading and validating data...
✅ Loaded 150 test runs

⏳ Analyzing for flaky tests...
✅ Detection complete

📊 Results:
  Total tests analyzed: 50
  Flaky tests (statistical): 7
  Flaky tests (ML): 9

🔴 Top 10 Flaky Tests:
  1. test_login
     EWMA: 45.3% | Z-score: 2.81
  2. test_api_call  
     EWMA: 32.1% | Z-score: 2.12
  3. test_ui_render
     EWMA: 28.8% | Z-score: 1.95

⏳ Generating reports...
  ✅ JSON: /Users/you/deflake_reports/flaky_tests.json
  ✅ CSV: /Users/you/deflake_reports/flaky_tests.csv
  ✅ HTML: /Users/you/deflake_reports/flaky_tests.html

======================================================================
✨ deflake analysis complete!
======================================================================
```

---

## 📊 Input Sources

deflake accepts test data from three sources:

1. **CSV File** - Direct historical data
2. **JSON File** - Structured test results
3. **pytest Report Directory** - Aggregated results from test runs

See [REPORT_FORMAT.md](REPORT_FORMAT.md) for detailed format specifications.

### Data Format

#### Input CSV Structure

```csv
test_id,execution_id,timestamp,build_number,pass_count,fail_count,total_runs
test_login,exec_001,2025-01-15T10:30:00,1234,8,2,10
test_api_call,exec_002,2025-01-15T10:31:00,1234,10,0,10
test_login,exec_003,2025-01-15T11:00:00,1235,7,3,10
```

### Field Descriptions

| Field | Type | Description | Validation |
|-------|------|-------------|-----------|
| `test_id` | `str` | Unique test identifier | Required |
| `execution_id` | `str` | Unique execution identifier | Required |
| `timestamp` | `datetime` | Test execution time | ISO 8601 format |
| `build_number` | `int` | CI/CD build number | Positive integer |
| `pass_count` | `int` | Number of passed runs | ≥ 0 |
| `fail_count` | `int` | Number of failed runs | ≥ 0 |
| `total_runs` | `int` | Total test runs | `pass_count + fail_count` |

**Data Validation:** Pydantic models ensure data integrity with automatic validation.

---

## 🔧 Configuration

### config.yaml

```yaml
data:
  input_file: "data/sample_test_data.csv"
  input_format: "csv"  # csv or json

detection:
  alpha: 0.3          # EWMA smoothing factor (0 < α < 1)
  threshold: 2.0      # Z-score threshold for flaky labeling
  use_ml: true        # Enable Isolation Forest ML detection
  contamination: 0.1  # Expected proportion of flaky tests (ML)

reporting:
  output_dir: "./output"
  formats:
    - "console"  # Print to terminal
    - "json"     # CI/CD integration
    - "html"     # Interactive charts
    - "csv"      # Spreadsheet export
  top_n: 10      # Number of top flaky tests to report

logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### Parameter Tuning Guide

| Parameter | Low Value | Default | High Value | Use Case |
|-----------|-----------|---------|------------|----------|
| `alpha` | 0.1 | 0.3 | 0.5 | Low: Long-term trends; High: Recent failures matter more |
| `threshold` | 1.5 | 2.0 | 3.0 | Low: Catch more suspects; High: High confidence only |
| `contamination` | 0.05 | 0.1 | 0.2 | Expected % of flaky tests in suite |

---

## 📈 Real-World Usage Scenarios

### Scenario 1: CI/CD Integration (Jenkins/GitHub Actions)

**Objective:** Auto-flag flaky tests in pull requests

```bash
# .github/workflows/flaky-detection.yml
- name: Run Tests and Save Results
  run: |
    pytest tests/ --json=deflake_reports/pytest_run.json
    
- name: Analyze with deflake
  run: |
    pip install deflake
    deflake --input ./deflake_reports/ --output-dir ./reports
    
    # Parse JSON output for CI/CD decisions
    FLAKY_COUNT=$(jq '.summary.flaky_executions' reports/flaky_tests.json)
    
    if [ "$FLAKY_COUNT" -gt 5 ]; then
      echo "⚠️ Warning: $FLAKY_COUNT flaky tests detected!"
      exit 1
    fi
```

**Output:** `deflake_reports/flaky_tests.json` with structured data for CI/CD parsing

```json
{
  "summary": {
    "total_tests": 150,
    "total_executions": 1500,
    "flaky_executions": 7,
    "flaky_percentage": 0.47
  },
  "top_flaky_tests": [
    {
      "test_id": "test_login",
      "ewma_failure_rate": 0.453,
      "flaky_count": 2,
      "avg_failure_rate": 0.45
    }
  ]
}
```

### Scenario 2: Nightly Test Suite Health Report

**Objective:** Generate daily HTML reports for QA team review

```bash
# cron job: 0 2 * * * (runs at 2 AM daily)
# Script: daily_flake_report.sh

DATE=$(date +\%Y-\%m-\%d)
INPUT_FILE="/var/test_results/${DATE}.csv"
OUTPUT_DIR="/var/reports/daily/${DATE}"

# Run deflake analysis
deflake --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --threshold 1.8 \
        --html

# Email report
mail -s "Daily Flaky Test Report - $DATE" \
     qa-team@company.com \
     < "$OUTPUT_DIR/flaky_tests.html"
```

**Output:** Interactive HTML dashboard in `$OUTPUT_DIR/flaky_tests.html`

### Scenario 3: Post-Deployment Analysis

**Objective:** Compare flakiness before/after deployment

```bash
# Analyze pre-deployment data
deflake --input pre_deploy.csv --output-dir ./reports/before --json

# Analyze post-deployment data
deflake --input post_deploy.csv --output-dir ./reports/after --json

# Compare results
echo "=== Tests now flaky ==="
comm -13 <(jq -r '.top_flaky_tests[].test_id' reports/before/flaky_tests.json | sort) \
         <(jq -r '.top_flaky_tests[].test_id' reports/after/flaky_tests.json | sort)

echo "=== Tests now stable ==="
comm -23 <(jq -r '.top_flaky_tests[].test_id' reports/before/flaky_tests.json | sort) \
         <(jq -r '.top_flaky_tests[].test_id' reports/after/flaky_tests.json | sort)
```

---

## 🧪 Testing

### Run Test Suite

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=html

# Run specific test class
pytest tests/test_flakiness_detector.py::TestEWMACalculation -v

# Run parametrized tests only
pytest tests/ -v -k "parametrize"
```

### Test Coverage

```
tests/
├── conftest.py                    # Shared fixtures
├── test_data_ingestion.py         # Data validation tests
│   ├── TestTestRunModel          # Pydantic model validation
│   └── TestDataIngestion         # CSV/JSON loading
└── test_flakiness_detector.py    # Detection algorithm tests
    ├── TestEWMACalculation       # EWMA with various alphas
    ├── TestZScoreCalculation     # Z-score edge cases
    ├── TestFlakinessLabeling     # Threshold sensitivity
    ├── TestEdgeCases             # Empty data, single test
    └── TestMLDetection           # Isolation Forest
```

**Key Test Scenarios:**
- ✅ Edge cases: empty data, single test run, all-pass/all-fail tests
- ✅ Parametrized tests: multiple alpha values, thresholds, contamination levels
- ✅ Validation: Pydantic model constraints, data integrity checks
- ✅ Integration: End-to-end pipeline execution

---

## 📚 Advanced Topics

### When to Use Statistical vs. ML Detection

| Method | Best For | Limitations |
|--------|----------|-------------|
| **Statistical (Z-score)** | - Small datasets (<100 tests)<br>- Explainable decisions<br>- Real-time detection | - Assumes normal distribution<br>- Sensitive to outliers |
| **ML (Isolation Forest)** | - Large datasets (>100 tests)<br>- Complex patterns<br>- Multi-feature analysis | - Black-box model<br>- Requires tuning<br>- More computational overhead |

**Recommendation:** Use **both** (default) for maximum coverage—statistical for interpretability, ML for comprehensive detection.

### Handling Different Test Suite Sizes

| Suite Size | Recommended Settings |
|------------|---------------------|
| **Small (<50 tests)** | `alpha=0.5`, `threshold=2.5`, `use_ml=false` |
| **Medium (50-200 tests)** | `alpha=0.3`, `threshold=2.0`, `use_ml=true` |
| **Large (>200 tests)** | `alpha=0.2`, `threshold=1.8`, `contamination=0.15` |

### Continuous Improvement Workflow

```
1. Collect Data
   └─→ Run tests in CI/CD → Store results in CSV/DB
   
2. Detect Flaky Tests
   └─→ Run detection script → Review JSON/HTML reports
   
3. Quarantine & Fix
   └─→ Mark tests as @flaky → Investigate root causes
   
4. Monitor Trends
   └─→ Track flakiness over time → Adjust thresholds
   
5. Validate Fixes
   └─→ Re-run detection → Verify flakiness resolved
```

---

## 🐛 Troubleshooting

### Issue: "No flaky tests detected" but tests are clearly flaky

**Solution:**
1. Lower `threshold` parameter: `python main.py --threshold 1.5`
2. Increase `contamination` for ML: Edit `config.yaml` → `contamination: 0.2`
3. Check data quality: Ensure sufficient historical runs (≥10 per test)

### Issue: ModuleNotFoundError

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Issue: Pydantic validation errors

**Solution:**
- Verify CSV format matches expected schema (see Data Format section)
- Check for missing required fields: `test_id`, `execution_id`, `timestamp`
- Ensure `pass_count + fail_count = total_runs`

---

## 🛠️ Development

### Project Structure

```
flaky-test-detection/
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py      # Data loading & validation
│   ├── flakiness_detector.py  # Core detection algorithms
│   ├── visualizer.py          # Reporting & visualization
│   └── config.py              # Configuration management
├── tests/
│   ├── conftest.py            # Pytest fixtures
│   ├── test_data_ingestion.py
│   └── test_flakiness_detector.py
├── data/
│   └── sample_test_data.csv   # Example input
├── config.yaml                # Configuration file
├── main.py                    # CLI entry point
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

### Adding New Features

1. **New Detection Algorithm:**
   - Add method to `FlakinessDetector` class in `src/flakiness_detector.py`
   - Write tests in `tests/test_flakiness_detector.py`
   - Update `label_flaky()` to incorporate new algorithm

2. **New Export Format:**
   - Add method to `Visualizer` class in `src/visualizer.py`
   - Update `config.yaml` → `reporting.formats`
   - Handle in `main.py` CLI arguments

### Code Quality Standards

- **Type Hints:** All functions include type annotations
- **Docstrings:** Google-style docstrings for all public APIs
- **Logging:** Structured logging with configurable levels
- **Testing:** >80% code coverage target
- **Linting:** Follow PEP 8 style guidelines

---

## 📖 References & Further Reading

### Academic Papers
- [Understanding Flaky Tests: The Developer's Perspective](https://research.google/pubs/pub46394/) (Google Research)
- [An Empirical Study of Flaky Tests](https://dl.acm.org/doi/10.1145/2635868.2635920) (ACM)

### Statistical Methods
- [Exponentially Weighted Moving Average Control Charts](https://en.wikipedia.org/wiki/EWMA_chart)
- [Z-Score Standardization](https://en.wikipedia.org/wiki/Standard_score)

### Machine Learning
- [Isolation Forest Algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) (Scikit-learn)
- [Anomaly Detection Techniques](https://scikit-learn.org/stable/modules/outlier_detection.html)

### Industry Best Practices
- [Google Testing Blog: Flaky Tests at Google](https://testing.googleblog.com/2016/05/flaky-tests-at-google-and-how-we.html)
- [Microsoft: Dealing with Flaky Tests](https://devblogs.microsoft.com/devops/dealing-with-flaky-tests/)

---

## 🤝 Contributing

Contributions welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Write tests for new functionality
4. Ensure all tests pass: `pytest tests/ -v`
5. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 👤 Author

**Senior SDET Portfolio Project**

- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Name](https://linkedin.com/in/yourprofile)

---

## 🌟 Showcase Highlights

**Why This Project Demonstrates Senior SDET Skills:**

✅ **Statistical Rigor:** Implements industry-standard EWMA and Z-score techniques  
✅ **Machine Learning:** Integrates Isolation Forest for advanced anomaly detection  
✅ **Production-Ready:** CLI, configuration management, logging, error handling  
✅ **Comprehensive Testing:** Parametrized tests, edge cases, >80% coverage  
✅ **Real-World Applicability:** CI/CD integration examples, multiple export formats  
✅ **Code Quality:** Type hints, docstrings, modular architecture, Pydantic validation  
✅ **Documentation:** Detailed README with formulas, diagrams, troubleshooting  

**Perfect for demonstrating expertise in:**
- Test automation frameworks & tooling development
- Statistical analysis & data-driven QA
- Python best practices & software engineering
- CI/CD integration & DevOps collaboration
- Technical documentation & knowledge sharing

---

⭐ **If you find this project useful, please star the repository!**
---
<h4 style="text-align:center;">Built with ❤️ by Ivan Klymchuk</h4>