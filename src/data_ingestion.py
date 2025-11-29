"""
Data Ingestion and Validation Module.

This module provides classes for loading and validating test execution data
from CSV/JSON files using Pydantic for robust data validation.
"""

from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import List
from pathlib import Path
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)


class TestRun(BaseModel):
    """
    Model representing a single test execution run.

    Attributes:
        test_id: Unique identifier for the test
        execution_id: Unique identifier for this execution
        timestamp: When the test was executed
        build_number: CI/CD build number
        pass_count: Number of passing assertions/runs
        fail_count: Number of failing assertions/runs
        total_runs: Total number of runs (must equal pass + fail)
    """

    __test__ = False  # Prevent pytest from collecting as test class
    test_id: str
    execution_id: str
    timestamp: datetime
    build_number: int
    pass_count: int = Field(ge=0)
    fail_count: int = Field(ge=0)
    total_runs: int = Field(gt=0)

    @field_validator('total_runs')
    @classmethod
    def check_total_runs(cls, v: int, info) -> int:
        """
        Validate that total_runs equals pass_count + fail_count.

        Args:
            v: The total_runs value
            info: Validation context with other field values

        Returns:
            int: Validated total_runs value

        Raises:
            ValueError: If counts don't match
        """
        if 'pass_count' in info.data and 'fail_count' in info.data:
            if info.data['pass_count'] + info.data['fail_count'] != v:
                raise ValueError('pass_count + fail_count must equal total_runs')
        return v

    @property
    def pass_rate(self) -> float:
        """Calculate the pass rate (0.0 to 1.0)."""
        return self.pass_count / self.total_runs


class DataIngestion:
    """
    Handles loading and validation of test execution data.

    This class reads test data from multiple sources:
    - CSV files: standard format with columns (test_id, execution_id, timestamp, etc.)
    - pytest report directories: aggregated JSON reports from pytest runs
    - JSON files: single JSON file with test execution data

    The class automatically detects the input format and validates each record
    using the TestRun Pydantic model.
    """

    def __init__(self, input_path: str):
        """
        Initialize the data ingestion handler.

        Args:
            input_path: Path to input - can be:
                - CSV file: direct path to .csv file
                - Directory: pytest reports directory with JSON files
                - JSON file: direct path to .json file
        """
        self.input_path = input_path
        self.input_type = self._detect_input_type(input_path)
        logger.info(f"Initialized DataIngestion with {self.input_type}: {input_path}")

    @staticmethod
    def _detect_input_type(path: str) -> str:
        """
        Detect the type of input source.

        Args:
            path: Input path

        Returns:
            str: 'csv', 'json', or 'directory'
        """
        from pathlib import Path
        p = Path(path)
        
        if p.is_dir():
            return 'directory'
        elif p.suffix == '.csv':
            return 'csv'
        elif p.suffix == '.json':
            return 'json'
        else:
            # Try to detect by reading first line
            if p.exists():
                with open(p) as f:
                    first_line = f.readline()
                    if first_line.startswith('{') or first_line.startswith('['):
                        return 'json'
            return 'csv'  # Default assumption

    def load_data(self) -> List[TestRun]:
        """
        Load and validate test data from any supported source.

        Returns:
            List[TestRun]: List of validated test run objects

        Raises:
            FileNotFoundError: If input path doesn't exist
            ValueError: If data validation fails
        """
        try:
            if self.input_type == 'directory':
                return self._load_from_directory()
            elif self.input_type == 'json':
                return self._load_from_json()
            else:  # csv
                return self._load_from_csv()
        except FileNotFoundError:
            logger.error(f"Input not found: {self.input_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def _load_from_csv(self) -> List[TestRun]:
        """
        Load test data from CSV file.

        CSV format should have columns:
        test_id, execution_id, timestamp, build_number, pass_count, fail_count, total_runs

        Returns:
            List[TestRun]: List of validated test run objects
        """
        logger.info(f"Loading CSV data from {self.input_path}")
        df = pd.read_csv(self.input_path)
        
        # Ensure timestamp column is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        test_runs = []
        for idx, row in df.iterrows():
            try:
                test_run = TestRun(**row.to_dict())
                test_runs.append(test_run)
            except Exception as e:
                logger.warning(f"Skipping row {idx} due to validation error: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(test_runs)} test runs from CSV")
        return test_runs

    def _load_from_json(self) -> List[TestRun]:
        """
        Load test data from JSON file.

        JSON format should be array of test run objects:
        [
            {"test_id": "...", "execution_id": "...", ...},
            ...
        ]

        Returns:
            List[TestRun]: List of validated test run objects
        """
        import json
        
        logger.info(f"Loading JSON data from {self.input_path}")
        
        with open(self.input_path, 'r') as f:
            data = json.load(f)
        
        # Handle both single object and array
        if isinstance(data, dict):
            data = [data]
        
        test_runs = []
        for idx, item in enumerate(data):
            try:
                if 'timestamp' in item:
                    item['timestamp'] = pd.Timestamp(item['timestamp'])
                test_run = TestRun(**item)
                test_runs.append(test_run)
            except Exception as e:
                logger.warning(f"Skipping JSON item {idx} due to validation error: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(test_runs)} test runs from JSON")
        return test_runs

    def _load_from_directory(self) -> List[TestRun]:
        """
        Load test data from pytest reports directory.

        Expected directory structure:
            deflake_reports/
            ├── flaky_tests.json  (aggregated results)
            └── run_2025_01_15.json (individual run reports)

        Looks for JSON files and aggregates them into test runs.

        Returns:
            List[TestRun]: List of validated test run objects
        """
        from pathlib import Path
        import json
        
        logger.info(f"Loading pytest reports from directory: {self.input_path}")
        
        report_dir = Path(self.input_path)
        test_runs = []
        
        # Look for JSON files in directory
        json_files = list(report_dir.glob("*.json"))
        
        if not json_files:
            logger.warning(f"No JSON files found in {self.input_path}")
            return []
        
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Handle aggregated format
                if isinstance(data, dict) and 'test_runs' in data:
                    items = data['test_runs']
                elif isinstance(data, dict) and 'results' in data:
                    items = data['results']
                elif isinstance(data, list):
                    items = data
                else:
                    items = [data]
                
                # Process each item
                for item in items:
                    try:
                        if 'timestamp' in item and isinstance(item['timestamp'], str):
                            item['timestamp'] = pd.Timestamp(item['timestamp'])
                        test_run = TestRun(**item)
                        test_runs.append(test_run)
                    except Exception as e:
                        logger.debug(f"Skipping item in {json_file}: {e}")
                        continue
                
                logger.debug(f"Loaded {len(items)} items from {json_file.name}")
            
            except Exception as e:
                logger.warning(f"Error reading {json_file}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(test_runs)} test runs from directory")
        return test_runs