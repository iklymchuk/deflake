# Statistical Flaky Test Detection System

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

> **MVP for detecting flaky tests using statistical analysis and machine learning.**

---

## 🎯 Problem Statement

**Flaky tests** are tests that exhibit non-deterministic behavior—passing and failing intermittently without code changes. They erode confidence in test suites, waste developer time, and can mask real defects.

### Why This Matters
- **Developer Productivity**: Teams spend 15-30% of their time investigating flaky test failures ([Google Testing Blog](https://testing.googleblog.com/))
- **CI/CD Reliability**: Flaky tests delay deployments and reduce trust in automation
- **Test Suite Health**: Undetected flaky tests accumulate technical debt

This project provides **quantitative, data-driven detection** of flaky tests using historical test execution data.

---

## Technical Approach

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

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                     Data Ingestion                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  CSV/JSON → Pydantic Validation → TestRun Objects    │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────┬───────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────────┐
│              Flakiness Detection Engine                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────┐  │
│  │  EWMA Calc      │→ │  Z-Score Calc   │→ │  Labeling  │  │
│  │  (Time Series)  │  │  (Outlier Det.) │  │  (Thresh.) │  │
│  └─────────────────┘  └─────────────────┘  └────────────┘  │
│                                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Isolation Forest (ML-Based Anomaly Detection)      │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────┬───────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────────┐
│                  Visualization & Reporting                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Console  │  │   JSON   │  │   CSV    │  │   HTML   │    │
│  │  Output  │  │ (CI/CD)  │  │ (Export) │  │ (Plotly) │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└────────────────────────────────────────────────────────────┘
```

### Module Overview

| Module | Responsibility | Key Classes/Functions |
|--------|----------------|----------------------|
| `src/data_ingestion.py` | Load and validate test data | `TestRun` (Pydantic model), `DataIngestion` |
| `src/flakiness_detector.py` | Statistical & ML analysis | `FlakinessDetector`, EWMA/Z-score/Isolation Forest |
| `src/visualizer.py` | Multi-format reporting | `Visualizer`, Plotly charts, JSON/CSV/HTML export |
| `src/config.py` | Configuration management | Pydantic config models, YAML loading |
| `main.py` | CLI orchestration | argparse, pipeline execution |

---

## Quick Start

### Prerequisites

```bash
Python 3.12+
pip (package manager)
```

### Installation

```bash
git clone https://github.com/iklymchuk/deflake.git
cd deflake
pip install -e .
```

### Basic Usage

After installing `deflake`:

```bash
# Analyze CSV historical data
python3 main.py --input data/sample_test_data.csv

# Generate sample test data
python3 generate_test_data.py

# Analyze pytest report directory
python3 main.py --input ./deflake_reports/

# Adjust detection threshold (higher = stricter)
python3 main.py --input data/sample_test_data.csv --threshold 2.5

# Export results to custom directory
python3 main.py --input data/sample_test_data.csv --output-dir ./analysis

# Disable ML detection (use statistical only)
python3 main.py --input data/sample_test_data.csv --no-ml

# Enable debug logging
python3 main.py --input data/sample_test_data.csv --log-level DEBUG

```

### Sample Output

```bash
$ python3 main.py --input ./deflake_reports/ 

======================================================================
Starting Flaky Test Detection
======================================================================

Initialized DataIngestion with directory: ./deflake_reports/
Loading pytest reports from directory: ./deflake_reports/
Found 10 JSON files to process
Successfully loaded 60 test runs from directory
Analyzing test data for flakiness patterns...
Initializing FlakinessDetector with 60 test runs, alpha=0.3, threshold=2.0
Calculating EWMA with alpha=0.3
Calculating Z-scores for anomaly detection
Labeling flaky tests with Z-score threshold=0.1
Threshold method flagged 60 executions as flaky
Training Isolation Forest with contamination=0.1
ML method flagged 6 executions as flaky
Combined method: 60 executions labeled as flaky
Generating reports...
Initialized Visualizer
Generating console report for top 5 tests
Calculating top 5 flakiest tests

======================================================================
Flaky Test Detection Report
======================================================================

Total Tests Analyzed: 6
Total Test Executions: 60
Flaky Executions Detected: 60 (100.0%)

Top 5 Flakiest Tests:
----------------------------------------------------------------------
                                      test_id  avg_ewma_failure_rate  flaky_execution_count  avg_failure_rate
            tests/test_api.py::test_get_users               0.503885                     10          0.466667
  tests/test_login.py::test_valid_credentials               0.381833                     10          0.366667
         tests/test_ui.py::test_homepage_load               0.060823                     10          0.066667
tests/test_login.py::test_invalid_credentials               0.029412                     10          0.033333
          tests/test_api.py::test_delete_user               0.027731                     10          0.033333
======================================================================

Exporting results to JSON: output/results.json
Calculating top 10 flakiest tests
SON export complete: output/results.json
Exporting HTML report: output/report.html
Creating plot for top 5 flaky tests
Calculating top 5 flakiest tests
Calculating top 10 flakiest tests
HTML report complete: output/report.html
Creating plot for top 5 flaky tests
Calculating top 5 flakiest tests
Saved plot to output/flaky_tests_plot.html

======================================================================
Flaky Test Detection Complete
======================================================================
```

---

## Input Sources

deflake accepts test data from three sources:

1. **CSV File** - Direct historical data
2. **JSON File** - Structured test results
3. **pytest Report Directory** - Aggregated results from test runs

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

## Configuration

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

## Advanced Topics

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

---

## 📖 References & Further Reading

### Academic Papers
- [Understanding Flaky Tests: The Developer's Perspective](https://dl.acm.org/doi/10.1145/3338906.3338945) (ACM)
- [An Empirical Study of Flaky Tests](https://dl.acm.org/doi/10.1145/2635868.2635920) (ACM)

### Statistical Methods
- [Exponentially Weighted Moving Average Control Charts](https://en.wikipedia.org/wiki/EWMA_chart)
- [Z-Score Standardization](https://en.wikipedia.org/wiki/Standard_score)

### Machine Learning
- [Isolation Forest Algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) (Scikit-learn)
- [Anomaly Detection Techniques](https://scikit-learn.org/stable/modules/outlier_detection.html)

### Industry Best Practices
- [Google Testing Blog: Flaky Tests at Google](https://testing.googleblog.com/2016/05/flaky-tests-at-google-and-how-we.html)
- [Microsoft: Manage flaky tests](https://learn.microsoft.com/en-us/azure/devops/pipelines/test/flaky-test-management?view=azure-devops)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 👤 Author

- GitHub: [@iklymchuk](https://github.com/iklymchuk)
- LinkedIn: [Ivan Klymchuk](https://www.linkedin.com/in/iklymchuk/)

---

⭐ **If you find this project useful, please star the repository!**
---
<h4 style="text-align:center;">Built with ❤️ by Ivan Klymchuk</h4>