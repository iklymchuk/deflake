"""
Command-Line Interface for Flaky Test Detector.

This module provides a CLI for running flaky test detection with configurable
parameters, supporting multiple output formats and logging levels.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from src.config import load_config, Config
from src.data_ingestion import DataIngestion
from src.flakiness_detector import FlakinessDetector
from src.visualizer import Visualizer


def setup_logging(config: Config) -> None:
    """
    Configure logging based on config settings.

    Args:
        config: Configuration object with logging settings
    """
    logging.basicConfig(
        level=config.logging.level,
        format=config.logging.format,
        handlers=[
            logging.FileHandler(config.logging.file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def run_detection(config: Config, cli_args: Optional[argparse.Namespace] = None) -> None:
    """
    Execute the flaky test detection pipeline.

    Args:
        config: Configuration object
        cli_args: Optional CLI arguments that override config
    """
    logger = logging.getLogger(__name__)
    logger.info("="*70)
    logger.info("Starting Flaky Test Detection")
    logger.info("="*70)

    # Override config with CLI args if provided
    if cli_args:
        if cli_args.input:
            config.data.input_path = cli_args.input
        if cli_args.threshold is not None:
            config.detection.flaky_threshold = cli_args.threshold
        if cli_args.output_dir:
            config.reporting.output_dir = cli_args.output_dir

    # Load data
    logger.info(f"Loading data from: {config.data.input_path}")
    ingestion = DataIngestion(config.data.input_path)
    test_runs = ingestion.load_data()

    # Detect flakiness
    logger.info("Analyzing test data for flakiness patterns...")
    detector = FlakinessDetector(test_runs)
    detector.calculate_ewma_failure_rate(alpha=config.detection.ewma_alpha)
    detector.calculate_z_score()
    detector.label_flaky(
        threshold=config.detection.flaky_threshold,
        use_ml=config.detection.use_ml_model,
        contamination=config.detection.ml_contamination
    )

    # Generate reports
    logger.info("Generating reports...")
    visualizer = Visualizer(detector)

    # Console output
    if "console" in config.reporting.output_formats:
        print(visualizer.generate_report(n=config.reporting.top_n_tests))

    # JSON export
    if "json" in config.reporting.output_formats:
        json_path = Path(config.reporting.output_dir) / "results.json"
        visualizer.export_json(str(json_path))

    # CSV export
    if "csv" in config.reporting.output_formats:
        csv_path = Path(config.reporting.output_dir) / "results.csv"
        visualizer.export_csv(str(csv_path))

    # HTML report
    if "html" in config.reporting.output_formats:
        html_path = Path(config.reporting.output_dir) / "report.html"
        visualizer.export_html_report(str(html_path))

    # Visualizations
    if config.reporting.include_visualizations:
        plot_path = Path(config.reporting.output_dir) / "flaky_tests_plot.html"
        fig = visualizer.plot_top_flaky_tests(
            n=config.reporting.top_n_tests,
            output_path=str(plot_path)
        )
        if "console" in config.reporting.output_formats:
            fig.show()

    logger.info("="*70)
    logger.info("Flaky Test Detection Complete")
    logger.info("="*70)


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Statistical Flaky Test Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python3 main.py

  # Specify custom input file
  python3 main.py --input data/my_tests.csv

  # Custom threshold and output directory
  python3 main.py --threshold 0.15 --output-dir results/

  # Use custom config file
  python3 main.py --config my_config.yaml
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to input data file (overrides config)'
    )

    parser.add_argument(
        '--threshold', '-t',
        type=float,
        help='Flakiness threshold 0-1 (overrides config)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for reports (overrides config)'
    )

    parser.add_argument(
        '--no-ml',
        action='store_true',
        help='Disable ML-based detection, use threshold only'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (overrides config)'
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config(args.config)

        # Override logging level if specified
        if args.log_level:
            config.logging.level = args.log_level

        # Override ML setting if specified
        if args.no_ml:
            config.detection.use_ml_model = False

        # Setup logging
        setup_logging(config)

        # Run detection
        run_detection(config, args)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()