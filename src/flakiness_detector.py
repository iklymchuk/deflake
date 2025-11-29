"""
Flakiness Detection and Statistical Modeling Module.

This module implements statistical algorithms to detect flaky tests using:
- Exponentially Weighted Moving Average (EWMA) of failure rates
- Z-score anomaly detection
- Machine Learning (Isolation Forest) for advanced classification
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Optional
import logging
from .data_ingestion import TestRun

logger = logging.getLogger(__name__)


class FlakinessDetector:
    """
    Detects flaky tests using statistical and ML methods.

    This class analyzes historical test execution data to identify tests
    that fail intermittently without code changes (flaky tests).

    Attributes:
        df: DataFrame containing test execution data with calculated metrics
        alpha: EWMA smoothing factor
        threshold: Z-score threshold for flaky classification
    """

    def __init__(self, test_runs: List[TestRun], alpha: float = 0.3, threshold: float = 2.0):
        """
        Initialize the flakiness detector.

        Args:
            test_runs: List of validated test run objects
            alpha: EWMA smoothing factor (0 < α < 1), default 0.3
            threshold: Z-score threshold for flaky labeling, default 2.0
        """
        logger.info(f"Initializing FlakinessDetector with {len(test_runs)} test runs, alpha={alpha}, threshold={threshold}")
        self.alpha = alpha
        self.threshold = threshold
        
        # Handle empty test runs
        if not test_runs:
            self.df = pd.DataFrame()
            logger.warning("Initialized with empty test runs list")
            return
            
        self.df = pd.DataFrame([run.model_dump() for run in test_runs])
        self.df['pass_rate'] = self.df['pass_count'] / self.df['total_runs']
        self.df['failure_rate'] = 1 - self.df['pass_rate']
        self.df = self.df.sort_values(['test_id', 'timestamp'])
        logger.debug(f"Unique tests: {self.df['test_id'].nunique()}")

    def calculate_ewma_failure_rate(self, alpha: Optional[float] = None) -> pd.DataFrame:
        """
        Calculate Exponentially Weighted Moving Average of failure rate.

        EWMA gives more weight to recent failures, making it sensitive to
        emerging flakiness patterns. Formula:
        EWMA_t = α * value_t + (1-α) * EWMA_(t-1)

        Args:
            alpha: Smoothing factor (0-1). If None, uses instance alpha.
                  Higher = more weight to recent values

        Returns:
            pd.DataFrame: DataFrame with 'ewma_failure_rate' column added
        """
        if self.df.empty:
            logger.warning("Cannot calculate EWMA on empty DataFrame")
            return self.df
            
        alpha_value = alpha if alpha is not None else self.alpha
        logger.info(f"Calculating EWMA with alpha={alpha_value}")
        self.df['ewma_failure_rate'] = self.df.groupby('test_id')['failure_rate'].transform(
            lambda x: x.ewm(alpha=alpha_value, adjust=False).mean()
        )
        return self.df

    def calculate_z_score(self) -> pd.DataFrame:
        """
        Calculate Z-score for failure rates within each test.

        Z-score identifies statistical outliers: Z = (X - μ) / σ
        Values with |Z| > 2 are typically considered anomalies.

        Returns:
            pd.DataFrame: DataFrame with 'z_score' column added
        """
        if self.df.empty:
            logger.warning("Cannot calculate Z-score on empty DataFrame")
            return self.df
            
        logger.info("Calculating Z-scores for anomaly detection")
        self.df['z_score'] = self.df.groupby('test_id')['failure_rate'].transform(
            lambda x: stats.zscore(x, nan_policy='omit') if len(x) > 1 else 0
        )
        return self.df

    def label_flaky_threshold(self, threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Label tests as flaky using Z-score threshold rule.

        Args:
            threshold: Z-score threshold (absolute value). If None, uses instance threshold.
                      Tests with |Z| > threshold are labeled as flaky.

        Returns:
            pd.DataFrame: DataFrame with 'is_flaky_threshold' column added
        """
        if self.df.empty:
            logger.warning("Cannot label flaky tests on empty DataFrame")
            return self.df
            
        threshold_value = threshold if threshold is not None else self.threshold
        logger.info(f"Labeling flaky tests with Z-score threshold={threshold_value}")
        
        # Label as flaky if absolute Z-score exceeds threshold
        self.df['is_flaky_threshold'] = self.df['z_score'].abs() > threshold_value
        flaky_count = self.df['is_flaky_threshold'].sum()
        logger.info(f"Threshold method flagged {flaky_count} executions as flaky")
        return self.df

    def label_flaky_ml(self, contamination: float = 0.1) -> pd.DataFrame:
        """
        Label tests as flaky using Isolation Forest ML algorithm.

        Isolation Forest detects anomalies by isolating outliers in the
        feature space (EWMA, Z-score, failure_rate).

        Args:
            contamination: Expected proportion of outliers (0-0.5)

        Returns:
            pd.DataFrame: DataFrame with 'is_flaky_ml' column added
        """
        if self.df.empty:
            logger.warning("Cannot apply ML detection on empty DataFrame")
            self.df['is_flaky_ml'] = pd.Series(dtype=bool)
            return self.df
            
        if len(self.df) < 2:
            logger.warning("Insufficient data for ML detection (need at least 2 samples)")
            self.df['is_flaky_ml'] = False
            return self.df
            
        logger.info(f"Training Isolation Forest with contamination={contamination}")

        # Prepare features
        features = self.df[['ewma_failure_rate', 'z_score', 'failure_rate']].fillna(0)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Train Isolation Forest
        clf = IsolationForest(contamination=contamination, random_state=42)
        predictions = clf.fit_predict(features_scaled)

        # -1 = outlier (flaky), 1 = inlier (stable)
        self.df['is_flaky_ml'] = predictions == -1
        flaky_count = self.df['is_flaky_ml'].sum()
        logger.info(f"ML method flagged {flaky_count} executions as flaky")
        return self.df

    def label_flaky(self, threshold: Optional[float] = None, use_ml: bool = True,
                    contamination: float = 0.1) -> pd.DataFrame:
        """
        Label tests as flaky using combined statistical and ML approach.

        Args:
            threshold: Z-score threshold for statistical method. If None, uses instance threshold.
            use_ml: Whether to use ML-based detection
            contamination: Contamination parameter for Isolation Forest

        Returns:
            pd.DataFrame: DataFrame with 'is_flaky' column (combined result)
        """
        if self.df.empty:
            logger.warning("Cannot label flaky tests on empty DataFrame")
            return self.df
            
        # Statistical threshold labeling
        self.label_flaky_threshold(threshold)

        if use_ml:
            self.label_flaky_ml(contamination)
            # Combine: flaky if EITHER method flags it
            self.df['is_flaky'] = self.df['is_flaky_threshold'] | self.df['is_flaky_ml']
        else:
            self.df['is_flaky'] = self.df['is_flaky_threshold']
            self.df['is_flaky_ml'] = False  # Add column for consistency

        flaky_count = self.df['is_flaky'].sum()
        logger.info(f"Combined method: {flaky_count} executions labeled as flaky")
        return self.df

    def get_top_flaky_tests(self, n: int = 5) -> pd.DataFrame:
        """
        Get top N flakiest tests by average EWMA failure rate.

        Args:
            n: Number of top tests to return

        Returns:
            pd.DataFrame: Summary with test_id, avg EWMA, flaky execution count
        """
        logger.info(f"Calculating top {n} flakiest tests")
        flaky_summary = self.df.groupby('test_id').agg({
            'ewma_failure_rate': 'mean',
            'is_flaky': 'sum',
            'failure_rate': 'mean'
        }).reset_index()
        flaky_summary.columns = ['test_id', 'avg_ewma_failure_rate',
                                  'flaky_execution_count', 'avg_failure_rate']
        flaky_summary = flaky_summary.sort_values('avg_ewma_failure_rate', ascending=False)
        return flaky_summary.head(n)