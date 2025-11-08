"""Cleaning pipeline - main interface preserving original API."""
import pandas as pd
from typing import Any

from .engine import CleaningEngine
from .template_manager import CleaningTemplateManager
from .output_manager import CleaningOutputManager
from ..config.cleaning_config import CleaningConfig
from ..utils.cleaning_utils import format_cleaning_return_values


class CleaningPipeline:
    """
    Main cleaning pipeline - preserves exact original CleaningStage API.
    """
    
    def __init__(self, smi_df: pd.DataFrame):
        self.smi_df = smi_df
        self.config = CleaningConfig()
        self.engine = CleaningEngine()
        self.template_manager = CleaningTemplateManager(self.config.template_dir)
    
    def cl_salt(
        self,
        validate: bool = True,
        output_dir: str = None,
        print_logs: bool = True,
        get_report: bool = False,
        get_diff: bool = False,
        param_deduplicate: bool = False,
        return_format_data: bool = False,
        n_cpu: int = -1,
        split_factor: int = 1,
        partial_dup_cols: list = None
    ) -> Any:
        """Clean salts - exact replica of original method."""
        return self._run_cleaning_step(
            'cl_salt', locals(), validate_first=validate
        )
    
    def neutralize(
        self,
        validate: bool = True,
        method: str = "boyle",
        output_dir: str = None,
        print_logs: bool = True,
        get_report: bool = False,
        get_diff: bool = False,
        param_deduplicate: bool = False,
        return_format_data: bool = False,
        n_cpu: int = -1,
        split_factor: int = 1,
        partial_dup_cols: list = None
    ) -> Any:
        """Neutralize SMILES - exact replica of original method."""
        return self._run_cleaning_step(
            'neutralize', locals(), validate_first=validate, method=method
        )
    
    def complete_cleaning(
        self,
        validate: bool = True,
        neutralizing_method: str = "boyle",
        output_dir: str = None,
        print_logs: bool = True,
        get_report: bool = False,
        param_deduplicate: bool = False,
        n_cpu: int = -1,
        split_factor: int = 1,
        partial_dup_cols: list = None
    ) -> pd.DataFrame:
        """Complete cleaning pipeline - exact replica of original method."""
        output_manager = CleaningOutputManager(output_dir)
        format_data = {}
        
        # Build initial template
        template_report = self.template_manager.build_complete_cleaning_template()
        
        # Run validation if requested
        if validate:
            from ...validation import ValidationStage
            self.smi_df, validation_format_data = ValidationStage(self.smi_df).validate_smi(
                print_logs=False,
                return_format_data=True,
                n_cpu=n_cpu,
                split_factor=split_factor,
            )
            format_data.update(validation_format_data)
            template_report = self.template_manager.add_validation_section(template_report)
        
        # Run salt cleaning
        self.smi_df, diff_after_cl_salt, _, cl_salt_format_data = self.engine.run_salt_cleaning(
            self.smi_df, n_cpu=n_cpu, split_factor=split_factor
        )
        format_data.update(cl_salt_format_data)
        template_report = self.template_manager.add_salt_cleaning_section(template_report)
        
        # Run neutralization
        self.smi_df, diff_after_neutralizing, neutralizing_format_data = self.engine.run_neutralization(
            self.smi_df, method=neutralizing_method, n_cpu=n_cpu, split_factor=split_factor
        )
        format_data.update(neutralizing_format_data)
        template_report = self.template_manager.add_neutralization_section(template_report)
        
        # Handle deduplication
        dup_idx_data = pd.DataFrame()
        if param_deduplicate:
            from ...utils.common import deduplicate
            
            self.smi_df, dup_idx_data, deduplicate_format_data = deduplicate(
                self.smi_df,
                validate=False,
                print_logs=False,
                show_dup_smi_and_idx=True,
                return_format_data=True,
                n_cpu=n_cpu,
                split_factor=split_factor,
                partial_dup_cols=partial_dup_cols
            )
            format_data.update(deduplicate_format_data)
            template_report = self.template_manager.add_deduplication_template(template_report)
            self.smi_df = self.smi_df.reset_index(drop=True)
        
        # Finalize template
        template_report = self.template_manager.finalize_template(template_report)
        formatted_report = template_report.format(**format_data)
        
        if print_logs:
            print(formatted_report)
        
        # Save outputs
        output_manager.save_complete_cleaning_outputs(
            self.smi_df, formatted_report, dup_idx_data, get_report, param_deduplicate
        )
        
        return self.smi_df
    
    def _run_cleaning_step(self, step_name: str, method_params: dict, 
                          validate_first: bool = True, method: str = None) -> Any:
        """Generic method to run cleaning steps - preserves exact original behavior."""
        # Remove 'self' from params
        params = {k: v for k, v in method_params.items() if k != 'self'}
        
        # Extract parameters
        output_dir = params.get('output_dir')
        print_logs = params.get('print_logs', True)
        get_report = params.get('get_report', False)
        get_diff = params.get('get_diff', False)
        param_deduplicate = params.get('param_deduplicate', False)
        return_format_data = params.get('return_format_data', False)
        n_cpu = params.get('n_cpu')
        split_factor = params.get('split_factor', 1)
        partial_dup_cols = params.get('partial_dup_cols')
        
        # Setup managers
        output_manager = CleaningOutputManager(output_dir)
        format_data = {}
        
        # Run pre-validation if needed
        if validate_first:
            from ...validation import ValidationStage
            self.smi_df, validate_format_data = ValidationStage(self.smi_df).validate_smi(
                print_logs=False,
                param_deduplicate=False,
                return_format_data=True,
                n_cpu=n_cpu,
                split_factor=split_factor,
            )
            format_data.update(validate_format_data)
        
        # Run main cleaning step
        if step_name == 'cl_salt':
            cleaned_smi, diff_data, missing_data, step_format_data = \
                self.engine.run_salt_cleaning(self.smi_df, n_cpu, split_factor)
            format_data.update(step_format_data)
        elif step_name == 'neutralize':
            if method is None:
                method = params.get('method', 'boyle')
            cleaned_smi, diff_data, step_format_data = \
                self.engine.run_neutralization(self.smi_df, method, n_cpu, split_factor)
            format_data.update(step_format_data)
            missing_data = None
        else:
            raise ValueError(f"Unknown cleaning step: {step_name}")
        
        # Update internal state
        self.smi_df = cleaned_smi
        
        # Build template
        template_report = self.template_manager.build_step_template(
            step_name, include_validation=validate_first
        )
        
        # Handle deduplication  
        dup_idx_data = pd.DataFrame()
        if param_deduplicate:
            from ...utils.common import deduplicate
            
            cleaned_smi, dup_idx_data, deduplicate_format_data = deduplicate(
                cleaned_smi,
                validate=False,
                print_logs=False,
                show_dup_smi_and_idx=True,
                return_format_data=True,
                n_cpu=n_cpu,
                split_factor=split_factor,
                partial_dup_cols=partial_dup_cols,
            )
            format_data.update(deduplicate_format_data)
            template_report = self.template_manager.add_deduplication_template(template_report)
            cleaned_smi = cleaned_smi.reset_index(drop=True)
            self.smi_df = cleaned_smi
        
        # Finalize template
        template_report = self.template_manager.finalize_template(template_report)
        formatted_report = template_report.format(**format_data)
        
        if print_logs:
            print(formatted_report)
        
        # Save outputs
        step_config = self.engine.step_configs[step_name]
        output_manager.save_step_outputs(
            step_config, cleaned_smi, diff_data, formatted_report, dup_idx_data, get_report
        )
        
        return format_cleaning_return_values(
            cleaned_smi, diff_data, format_data, get_diff, return_format_data
        )

