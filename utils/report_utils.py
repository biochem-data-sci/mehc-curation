"""Report generation utilities."""
import os
import pandas as pd
from typing import Dict, Any


class GetReport:
    """
    A class to generate reports and save them to a specified directory.

    Attributes:
        output_dir (str): The directory where reports will be saved.
        report_subdir_name (str): The name of the subdirectory for the report files.
    """

    def __init__(self, output_dir: str, report_subdir_name: str):
        """
        Initialize the GetReport class with output directory and subdirectory name.

        Args:
            output_dir (str): The directory where the report will be stored.
            report_subdir_name (str): The name of the subdirectory where reports will be saved.
                                    Examples: "deduplicate", "validate", "remove_mixtures"
        """
        self.output_dir = output_dir
        self.report_subdir_name = report_subdir_name

    def _ensure_directory_exists(self):
        """
        Ensure the report subdirectory exists, creating it if necessary.

        Returns:
            str: The path to the report subdirectory.
        """
        report_subdir_path = os.path.join(self.output_dir, self.report_subdir_name)
        if not os.path.exists(report_subdir_path):
            os.makedirs(report_subdir_path)
        return report_subdir_path

    def create_report_file(self, report_file_name: str, content: str):
        """
        Create a report file with the specified name and content.

        Args:
            report_file_name (str): The name of the report file to create.
            content (str): The content to write to the report file.
        """
        report_subdir_path = self._ensure_directory_exists()
        report_file_path = os.path.join(report_subdir_path, report_file_name)
        with open(report_file_path, "w", encoding="utf-8") as report_file:
            report_file.write(content)

    def create_csv_file(self, smiles_df: pd.DataFrame, csv_file_name: str):
        """
        Create a CSV file from a DataFrame.

        Args:
            smiles_df (pd.DataFrame): A DataFrame containing SMILES data to be saved.
            csv_file_name (str): The name of the CSV file to create.
        """
        report_subdir_path = self._ensure_directory_exists()
        csv_file_path = os.path.join(report_subdir_path, csv_file_name)
        smiles_df.to_csv(csv_file_path, index=False, encoding="utf-8")


def generate_summary_report(stats: Dict[str, Any], template_path: str = None) -> str:
    """
    Generate a summary report from statistics.
    
    Args:
        stats: Dictionary containing statistics
        template_path: Optional path to template file
        
    Returns:
        str: Formatted report content
    """
    if template_path and os.path.exists(template_path):
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()
        return template.format(**stats)
    else:
        # Basic fallback template
        report_lines = ["Processing Report", "=" * 50]
        for key, value in stats.items():
            report_lines.append(f"{key.replace('_', ' ').title()}: {value}")
        return "\n".join(report_lines)