"""Normalization pipeline - main interface preserving original API."""
import pandas as pd
from typing import Any

from .engine import NormalizationEngine
from .template_manager import NormalizationTemplateManager
from .output_manager import NormalizationOutputManager
from ..config.normalization_config import NormalizationConfig
from ..utils.normalization_utils import format_normalization_return_values


class NormalizationPipeline:
    """
    Main normalization pipeline - preserves exact original NormalizationStage API.
    """
    
    def __init__(self, smi_df: pd.DataFrame):
        self.smi_df = smi_df
        self.config = NormalizationConfig()
        self.engine = NormalizationEngine()
        self.template_manager = NormalizationTemplateManager(self.config.template_dir)
    
    def detautomerize(
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
        """Detautomerize SMILES - exact replica of original method."""
        return self._run_normalization_step(
            'detautomerize', locals(), validate_first=validate
        )
    
    def destereoisomerize(
        self,
        validate: bool = True,
        output_dir: str = None,
        print_logs: bool = True,
        get_report: bool = False,
        get_diff: bool = False,
        param_deduplicate: bool = False,
        return_format_data: bool = False,
        n_cpu: int = -1,
        split_factor: int = None,
        partial_dup_cols: list = None
    ) -> Any:
        """Destereoisomerize SMILES - exact replica of original method."""
        return self._run_normalization_step(
            'destereoisomerize', locals(), validate_first=validate
        )
    
    def complete_normalization(
        self,
        validate: bool = True,
        output_dir: str = None,
        print_logs: bool = True,
        get_report: bool = False,
        param_deduplicate: bool = False,
        n_cpu: int = -1,
        split_factor: int = 1,
        partial_dup_cols: list = None
    ) -> pd.DataFrame:
        """Complete normalization pipeline - exact replica of original method."""
        output_manager = NormalizationOutputManager(output_dir)
        format_data = {}
        
        # Build initial template
        template_report = self.template_manager.build_complete_normalization_template()
        
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
        
        # Run destereoisomerization first (important order!)
        self.smi_df, diff_after_destereoisomerized, destereoisomerized_format_data = \
            self.engine.run_destereoisomerize(
                self.smi_df, n_cpu=n_cpu, split_factor=split_factor
            )
        format_data.update(destereoisomerized_format_data)
        template_report = self.template_manager.add_destereoisomerization_section(template_report)
        
        # Run detautomerization
        self.smi_df, diff_after_detautomerized, detautomerized_format_data = \
            self.engine.run_detautomerize(
                self.smi_df, n_cpu=n_cpu, split_factor=split_factor
            )
        format_data.update(detautomerized_format_data)
        template_report = self.template_manager.add_detautomerization_section(template_report)
        
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
        output_manager.save_complete_normalization_outputs(
            self.smi_df, formatted_report, dup_idx_data, get_report, param_deduplicate
        )
        
        return self.smi_df
    
    def _run_normalization_step(self, step_name: str, method_params: dict, 
                               validate_first: bool = True) -> Any:
        """Generic method to run normalization steps - preserves exact original behavior."""
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
        output_manager = NormalizationOutputManager(output_dir)
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
        
        # Run main normalization step
        if step_name == 'detautomerize':
            normalized_smi, diff_data, step_format_data = \
                self.engine.run_detautomerize(self.smi_df, n_cpu, split_factor)
            format_data.update(step_format_data)
        elif step_name == 'destereoisomerize':
            normalized_smi, diff_data, step_format_data = \
                self.engine.run_destereoisomerize(self.smi_df, n_cpu, split_factor)
            format_data.update(step_format_data)
        else:
            raise ValueError(f"Unknown normalization step: {step_name}")
        
        # Update internal state
        self.smi_df = normalized_smi
        
        # Build template
        template_report = self.template_manager.build_step_template(
            step_name, include_validation=validate_first
        )
        
        # Handle deduplication  
        dup_idx_data = pd.DataFrame()
        if param_deduplicate:
            from ...utils.common import deduplicate
            
            normalized_smi, dup_idx_data, deduplicate_format_data = deduplicate(
                normalized_smi,
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
            normalized_smi = normalized_smi.reset_index(drop=True)
            self.smi_df = normalized_smi
        
        # Finalize template
        template_report = self.template_manager.finalize_template(template_report)
        formatted_report = template_report.format(**format_data)
        
        if print_logs:
            print(formatted_report)
        
        # Save outputs
        step_config = self.engine.step_configs[step_name]
        output_manager.save_step_outputs(
            step_config, normalized_smi, diff_data, formatted_report, dup_idx_data, get_report
        )
        
        return format_normalization_return_values(
            normalized_smi, diff_data, format_data, get_diff, return_format_data
        )

