"""Output management for cleaning operations."""
import pandas as pd
from typing import Optional


class CleaningOutputManager:
    """Manages output operations for cleaning."""
    
    def __init__(self, output_dir: Optional[str]):
        self.output_dir = output_dir
        self.get_output = output_dir is not None
    
    def save_step_outputs(self, config, cleaned_smi: pd.DataFrame, 
                         diff_data: pd.DataFrame,
                         formatted_report: str,
                         dup_idx_data: pd.DataFrame, get_report: bool):
        """Save outputs for a cleaning step."""
        from ...utils.report_utils import GetReport
        
        # Save main output
        if self.get_output:
            GetReport(
                output_dir=self.output_dir,
                report_subdir_name=config.report_subdir,
            ).create_csv_file(cleaned_smi, csv_file_name=config.output_csv)
        
        # Save detailed reports
        if get_report:
            report_gen = GetReport(
                output_dir=self.output_dir,
                report_subdir_name=config.report_subdir,
            )
            
            # Save report file
            report_gen.create_report_file(
                report_file_name=f"{config.step_name}_report.txt",
                content=formatted_report,
            )
            
            # Save duplicate data if available
            if len(dup_idx_data) > 0:
                report_gen.create_csv_file(
                    dup_idx_data, csv_file_name="duplicate_index_data.csv"
                )
    
    def save_complete_cleaning_outputs(self, smi_df: pd.DataFrame, 
                                       formatted_report: str,
                                       dup_idx_data: pd.DataFrame,
                                       get_report: bool, param_deduplicate: bool):
        """Save complete cleaning outputs."""
        from ...utils.report_utils import GetReport
        
        if self.get_output:
            GetReport(
                output_dir=self.output_dir,
                report_subdir_name="cleaning_salts_and_neutralizing_smiles",
            ).create_csv_file(smi_df, csv_file_name="post_cleaned_and_neutralized_smiles.csv")
        
        if get_report:
            GetReport(
                output_dir=self.output_dir,
                report_subdir_name="cleaning_salts_and_neutralizing_smiles",
            ).create_report_file(
                report_file_name="cleaning_salts_and_neutralizing_smiles_report.txt",
                content=formatted_report,
            )
            
            if param_deduplicate:
                GetReport(
                    output_dir=self.output_dir,
                    report_subdir_name="cleaning_salts_and_neutralizing_smiles",
                ).create_csv_file(
                    dup_idx_data, csv_file_name="duplicate_index_data.csv"
                )

