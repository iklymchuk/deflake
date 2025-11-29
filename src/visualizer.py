"""
Visualization and Reporting Module.

This module generates visual reports and exports results in multiple formats
(console, JSON, HTML, CSV) for flaky test detection results.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict
import logging
from .flakiness_detector import FlakinessDetector

logger = logging.getLogger(__name__)


class Visualizer:
    """
    Handles visualization and report generation for flaky test results.

    Supports multiple output formats for CI/CD integration and analysis.
    """

    def __init__(self, detector: FlakinessDetector):
        """
        Initialize visualizer with detector results.

        Args:
            detector: FlakinessDetector instance with analyzed data
        """
        self.detector = detector
        logger.info("Initialized Visualizer")

    def plot_top_flaky_tests(self, n: int = 5, output_path: str = None) -> go.Figure:
        """
        Create interactive time-series plot of top flaky tests.

        Args:
            n: Number of top flaky tests to visualize
            output_path: Optional path to save HTML file

        Returns:
            plotly.graph_objects.Figure: Interactive plot
        """
        logger.info(f"Creating plot for top {n} flaky tests")
        top_tests = self.detector.get_top_flaky_tests(n)['test_id'].tolist()
        filtered_df = self.detector.df[self.detector.df['test_id'].isin(top_tests)]

        fig = px.line(
            filtered_df,
            x='build_number',
            y='failure_rate',
            color='test_id',
            title=f'Top {n} Flakiest Tests: Failure Rate Over Builds',
            labels={'failure_rate': 'Failure Rate', 'build_number': 'Build Number'},
            markers=True
        )

        fig.update_layout(
            hovermode='x unified',
            xaxis_title='Build Number',
            yaxis_title='Failure Rate',
            legend_title='Test ID'
        )

        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_file))
            logger.info(f"Saved plot to {output_path}")

        return fig

    def generate_report(self, n: int = 5) -> str:
        """
        Generate console text report of flaky test detection results.

        Args:
            n: Number of top tests to include

        Returns:
            str: Formatted text report
        """
        logger.info(f"Generating console report for top {n} tests")
        top_flaky = self.detector.get_top_flaky_tests(n)
        total_flaky = self.detector.df['is_flaky'].sum()
        total_tests = self.detector.df['test_id'].nunique()
        total_executions = len(self.detector.df)

        report = f"""
{'='*70}
Flaky Test Detection Report
{'='*70}

Total Tests Analyzed: {total_tests}
Total Test Executions: {total_executions}
Flaky Executions Detected: {total_flaky} ({total_flaky/total_executions*100:.1f}%)

Top {n} Flakiest Tests:
{'-'*70}
{top_flaky.to_string(index=False)}
{'='*70}
        """
        return report

    def export_json(self, output_path: str = "output/results.json") -> None:
        """
        Export results to JSON format for CI/CD integration.

        Args:
            output_path: Path to save JSON file
        """
        logger.info(f"Exporting results to JSON: {output_path}")
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        top_flaky = self.detector.get_top_flaky_tests(10)

        results = {
            "summary": {
                "total_tests": int(self.detector.df['test_id'].nunique()),
                "total_executions": len(self.detector.df),
                "flaky_executions": int(self.detector.df['is_flaky'].sum()),
                "flaky_percentage": float(self.detector.df['is_flaky'].mean() * 100)
            },
            "top_flaky_tests": top_flaky.to_dict(orient='records'),
            "all_results": self.detector.df.to_dict(orient='records')
        }

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"JSON export complete: {output_path}")

    def export_csv(self, output_path: str = "output/results.csv") -> None:
        """
        Export detailed results to CSV format.

        Args:
            output_path: Path to save CSV file
        """
        logger.info(f"Exporting results to CSV: {output_path}")
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        self.detector.df.to_csv(output_file, index=False)
        logger.info(f"CSV export complete: {output_path}")

    def export_html_report(self, output_path: str = "output/report.html") -> None:
        """
        Export HTML report with embedded visualizations.

        Args:
            output_path: Path to save HTML file
        """
        logger.info(f"Exporting HTML report: {output_path}")
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Create plot
        fig = self.plot_top_flaky_tests()

        # Get summary stats
        top_flaky = self.detector.get_top_flaky_tests(10)
        total_flaky = self.detector.df['is_flaky'].sum()
        total_tests = self.detector.df['test_id'].nunique()

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Flaky Test Detection Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                .summary {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Flaky Test Detection Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Total Tests:</strong> {total_tests}</p>
                <p><strong>Flaky Executions:</strong> {total_flaky}</p>
            </div>
            <h2>Top Flaky Tests</h2>
            {top_flaky.to_html()}
            <h2>Failure Rate Trends</h2>
            {fig.to_html(include_plotlyjs='cdn')}
        </body>
        </html>
        """

        with open(output_file, 'w') as f:
            f.write(html_content)

        logger.info(f"HTML report complete: {output_path}")