"""Output management for refinement operations."""
import pandas as pd
from typing import Optional


class RefinementOutputManager:
    """Manages output operations for refinement."""
    
    def __init__(self, output_dir: Optional[str], config):
        self.output_dir = output_dir
        self.config = config
        self.get_output = output_dir is not None
    
    def save_refinement_outputs(
        self,
        smi_df: pd.DataFrame,
        formatted_report: str,
        data_post_1st_stage_rm_dup: pd.DataFrame,
        data_post_2nd_stage_rm_dup: pd.DataFrame,
        data_post_3rd_stage_rm_dup: pd.DataFrame,
        get_report: bool,
    ):
        """Save all refinement outputs."""
        from ...utils.report_utils import GetReport
        
        # Save main output
        if self.get_output:
            GetReport(
                output_dir=self.output_dir,
                report_subdir_name=self.config.report_subdir,
            ).create_csv_file(smi_df, csv_file_name=self.config.output_csv)
        
        # Save detailed reports
        if get_report:
            report_gen = GetReport(
                output_dir=self.output_dir,
                report_subdir_name=self.config.report_subdir,
            )
            
            # Save report file
            report_gen.create_report_file(
                report_file_name=self.config.report_file,
                content=formatted_report,
            )
            
            # Save duplicate data files
            if not data_post_1st_stage_rm_dup.empty:
                report_gen.create_csv_file(
                    data_post_1st_stage_rm_dup,
                    csv_file_name=self.config.dup_1st_csv,
                )
            
            if not data_post_2nd_stage_rm_dup.empty:
                report_gen.create_csv_file(
                    data_post_2nd_stage_rm_dup,
                    csv_file_name=self.config.dup_2nd_csv,
                )
            
            if not data_post_3rd_stage_rm_dup.empty:
                report_gen.create_csv_file(
                    data_post_3rd_stage_rm_dup,
                    csv_file_name=self.config.dup_3rd_csv,
                )

