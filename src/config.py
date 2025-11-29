"""
Configuration management for Flaky Test Detector.

This module handles loading and validating configuration from YAML files,
providing a centralized configuration object for the application.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """Configuration for data ingestion."""
    input_path: str
    format: str = Field(default="csv")

    @field_validator('format')
    @classmethod
    def validate_format(cls, v: str) -> str:
        if v not in ['csv', 'json']:
            raise ValueError(f"Unsupported format: {v}. Use 'csv' or 'json'")
        return v


class DetectionConfig(BaseModel):
    """Configuration for flakiness detection algorithms."""
    ewma_alpha: float = Field(default=0.3, ge=0.0, le=1.0)
    flaky_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    use_ml_model: bool = Field(default=True)
    ml_contamination: float = Field(default=0.1, ge=0.0, le=0.5)


class ReportingConfig(BaseModel):
    """Configuration for reporting and output."""
    top_n_tests: int = Field(default=5, ge=1)
    output_formats: List[str] = Field(default=["console", "json"])
    output_dir: str = Field(default="output")
    include_visualizations: bool = Field(default=True)

    @field_validator('output_formats')
    @classmethod
    def validate_formats(cls, v: List[str]) -> List[str]:
        valid_formats = {'console', 'json', 'html', 'csv'}
        for fmt in v:
            if fmt not in valid_formats:
                raise ValueError(f"Invalid format: {fmt}. Valid: {valid_formats}")
        return v


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    level: str = Field(default="INFO")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file: str = Field(default="flaky_detector.log")


class Config(BaseModel):
    """Main configuration container."""
    data: DataConfig
    detection: DetectionConfig
    reporting: ReportingConfig
    logging: LoggingConfig


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Config: Validated configuration object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    return Config(**config_data)
