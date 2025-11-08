"""Output management for validation operations."""
import pandas as pd
from typing import Optional


class ValidationOutputManager:
    """Manages output operations for validation."""
    
    def __init__(self, output_dir: Optional[str]):
        self.output_dir = output_dir
        self.get_output = output_dir is not None
    
    def save_step_outputs(self, config, valid_smi: pd.DataFrame, 
                         invalid_smi: pd.Series, formatted_report: str,
                         dup_idx_data: pd.DataFrame, get_report: bool):
        """Save outputs for a validation step."""
        from ...utils.report_utils import GetReport
        
        # Save main output
        if self.get_output:
            GetReport(
                output_dir=self.output_dir,
                report_subdir_name=config.report_subdir,
            ).create_csv_file(valid_smi, csv_file_name=config.output_csv)
        
        # Save detailed reports
        if get_report:
            report_gen = GetReport(
                output_dir=self.output_dir,
                report_subdir_name=config.report_subdir,
            )
            
            # Save report file
            report_gen.create_report_file(
                report_file_name=f"{config.step_name}.txt",
                content=formatted_report,
            )
            
            # Save invalid SMILES
            report_gen.create_csv_file(
                invalid_smi, csv_file_name=config.invalid_csv
            )
            
            # Save duplicate data (preserve original bug for rm_mixture)
            if len(dup_idx_data) > 0:
                # Original bug: rm_mixture saves duplicates to wrong subdirectory
                dup_subdir = config.report_subdir
                if config.step_name == 'rm_mixture':
                    dup_subdir = "check_valid_smiles"
                
                GetReport(
                    output_dir=self.output_dir,
                    report_subdir_name=dup_subdir,
                ).create_csv_file(
                    dup_idx_data, csv_file_name="duplicate_index_data.csv"
                )
    
    def save_complete_validation_outputs(self, smi_df: pd.DataFrame, 
                                       formatted_report: str,
                                       dup_idx_data: pd.DataFrame,
                                       get_report: bool, param_deduplicate: bool):
        """Save complete validation outputs."""
        from ...utils.report_utils import GetReport
        
        if self.get_output:
            GetReport(
                output_dir=self.output_dir,
                report_subdir_name="complete_validation",
            ).create_csv_file(smi_df, csv_file_name="output_smiles.csv")
        
        if get_report:
            GetReport(
                output_dir=self.output_dir,
                report_subdir_name="complete_validation",
            ).create_report_file(
                report_file_name="complete_validation.txt",
                content=formatted_report,
            )
            
            if param_deduplicate:
                GetReport(
                    output_dir=self.output_dir,
                    report_subdir_name="complete_validation",
                ).create_csv_file(
                    dup_idx_data, csv_file_name="duplicate_index_data.csv"
                )