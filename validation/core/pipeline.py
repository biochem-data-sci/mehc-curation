"""Validation pipeline - main interface preserving original API."""
import pandas as pd
from typing import Any

from .engine import ValidationEngine
from .template_manager import ValidationTemplateManager
from .output_manager import ValidationOutputManager
from ..config.validation_config import ValidationConfig
from ..utils.validation_utils import format_return_values


class ValidationPipeline:
    """
    Main validation pipeline - preserves exact original ValidationStage API.
    """
    
    def __init__(self, smi_df: pd.DataFrame):
        self.smi_df = smi_df
        self.config = ValidationConfig()
        self.engine = ValidationEngine()
        self.template_manager = ValidationTemplateManager(self.config.template_dir)
    
    def validate_smi(
        self,
        output_dir: str = None,
        print_logs: bool = True,
        get_report: bool = False,
        get_invalid_smi_idx: bool = False,
        get_isomeric_smi: bool = True,
        param_deduplicate: bool = False,
        return_format_data: bool = False,
        n_cpu: int = -1,
        split_factor: int = 1,
        partial_dup_cols: list = None
    ) -> Any:
        """Validate SMILES - exact replica of original method."""
        return self._run_validation_step(
            'validate_smi', locals(), validate_first=False
        )
    
    def rm_mixture(
        self,
        validate: bool = True,
        output_dir: str = None,
        get_invalid_smi_idx: bool = False,
        print_logs: bool = True,
        get_report: bool = False,
        param_deduplicate: bool = False,
        return_format_data: bool = False,
        n_cpu: int = -1,
        split_factor: int = 1,
        partial_dup_cols: list = None
    ) -> Any:
        """Remove mixtures - exact replica of original method."""
        return self._run_validation_step(
            'rm_mixture', locals(), validate_first=validate
        )
    
    def rm_inorganic(
        self,
        validate: bool = True,
        output_dir: str = None,
        get_invalid_smi_idx: bool = False,
        print_logs: bool = True,
        get_report: bool = False,
        param_deduplicate: bool = False,
        return_format_data: bool = False,
        n_cpu: int = -1,
        split_factor: int = 1,
        partial_dup_cols: list = None
    ) -> Any:
        """Remove inorganics - exact replica of original method."""
        return self._run_validation_step(
            'rm_inorganic', locals(), validate_first=validate
        )
    
    def rm_organometallic(
        self,
        validate: bool = True,
        output_dir: str = None,
        get_invalid_smi_idx: bool = False,
        print_logs: bool = True,
        get_report: bool = False,
        param_deduplicate: bool = False,
        return_format_data: bool = False,
        n_cpu: int = -1,
        split_factor: int = 1,
        partial_dup_cols: list = None
    ) -> Any:
        """Remove organometallics - exact replica of original method."""
        return self._run_validation_step(
            'rm_organometallic', locals(), validate_first=validate
        )
    
    def complete_validation(
        self,
        validate: bool = True,
        output_dir: str = None,
        print_logs: bool = True,
        get_report: bool = False,
        param_deduplicate: bool = True,
        n_cpu: int = -1,
        split_factor: int = 1,
        partial_dup_cols: list = None
    ) -> pd.DataFrame:
        """Complete validation pipeline - exact replica of original method."""
        output_manager = ValidationOutputManager(output_dir)
        format_data = {}
        
        # Build initial template
        template_report = self.template_manager.build_complete_validation_template()
        
        # Run validation if requested
        if validate:
            self.smi_df, invalid_smi_idx, validation_format_data = self.validate_smi(
                print_logs=False,
                get_invalid_smi_idx=True,
                param_deduplicate=False,
                return_format_data=True,
                n_cpu=n_cpu,
                split_factor=split_factor,
            )
            format_data.update(validation_format_data)
            template_report = self.template_manager.add_validation_section(template_report)
        
        # Run mixture removal
        self.smi_df, mixture_idx, rm_mixture_format_data = self.rm_mixture(
            validate=False,
            print_logs=False,
            get_invalid_smi_idx=True,
            param_deduplicate=False,
            return_format_data=True,
            n_cpu=n_cpu,
            split_factor=split_factor,
        )
        format_data.update(rm_mixture_format_data)
        template_report = self.template_manager.add_mixture_section(template_report)
        
        # Run inorganic removal
        self.smi_df, inorganic_idx, rm_inorganic_format_data = self.rm_inorganic(
            validate=False,
            print_logs=False,
            get_invalid_smi_idx=True, 
            param_deduplicate=False,
            return_format_data=True,
            n_cpu=n_cpu,
            split_factor=split_factor,
        )
        format_data.update(rm_inorganic_format_data)
        template_report = self.template_manager.add_inorganic_section(template_report)
        
        # Run organometallic removal
        self.smi_df, organometallic_idx, rm_organometallic_format_data = self.rm_organometallic(
            validate=False,
            print_logs=False,
            get_invalid_smi_idx=True,
            param_deduplicate=False,
            return_format_data=True,
            n_cpu=n_cpu,
            split_factor=split_factor,
        )
        format_data.update(rm_organometallic_format_data)
        template_report = self.template_manager.add_organometallic_section(template_report)  # FIXED TYPO
        
        # Handle deduplication
        dup_idx_data = pd.DataFrame()
        if param_deduplicate:
            from ...utils.common import deduplicate
            
            self.smi_df, dup_idx_data, deduplicate_format_data = deduplicate(
                self.smi_df,
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
        output_manager.save_complete_validation_outputs(
            self.smi_df, formatted_report, dup_idx_data, get_report, param_deduplicate
        )
        
        return self.smi_df
    
    def _run_validation_step(self, step_name: str, method_params: dict, validate_first: bool = True) -> Any:
        """Generic method to run validation steps - preserves exact original behavior."""
        # Remove 'self' from params
        params = {k: v for k, v in method_params.items() if k != 'self'}
        
        # Extract parameters
        output_dir = params.get('output_dir')
        print_logs = params.get('print_logs', True)
        get_report = params.get('get_report', False)
        get_invalid_smi_idx = params.get('get_invalid_smi_idx', False)
        get_isomeric_smi = params.get('get_isomeric_smi', True)
        param_deduplicate = params.get('param_deduplicate', False)
        return_format_data = params.get('return_format_data', False)
        n_cpu = params.get('n_cpu')
        split_factor = params.get('split_factor', 1)
        partial_dup_cols = params.get('partial_dup_cols')
        
        # Setup managers
        output_manager = ValidationOutputManager(output_dir)
        format_data = {}
        
        # Run pre-validation if needed
        if validate_first and step_name != 'validate_smi':
            self.smi_df, validate_format_data = self.validate_smi(
                print_logs=False,
                param_deduplicate=False,
                return_format_data=True,
                n_cpu=n_cpu,
                split_factor=split_factor,
            )
            format_data.update(validate_format_data)
        
        # Run main validation
        valid_smi, invalid_smi, idx_of_invalid_smi, step_format_data = \
            self.engine.run_validation_step(self.smi_df, step_name, n_cpu, split_factor)
        format_data.update(step_format_data)
        
        # Apply SMILES normalization (only for validate_smi)
        if step_name == 'validate_smi':
            valid_smi = self.engine.normalize_smiles(valid_smi, get_isomeric_smi, n_cpu, split_factor)
        
        # Build template
        template_report = self.template_manager.build_step_template(
            step_name, include_validation=validate_first and step_name != 'validate_smi'
        )
        
        # Handle deduplication  
        dup_idx_data = pd.DataFrame()
        if param_deduplicate:
            from ...utils.common import deduplicate
            
            valid_smi, dup_idx_data, deduplicate_format_data = deduplicate(
                valid_smi,
                print_logs=False,
                show_dup_smi_and_idx=True,
                return_format_data=True,
                n_cpu=n_cpu,
                split_factor=split_factor,
                partial_dup_cols=partial_dup_cols,
            )
            format_data.update(deduplicate_format_data)
            template_report = self.template_manager.add_deduplication_template(template_report)
            valid_smi = valid_smi.reset_index(drop=True)
        
        # Finalize template
        template_report = self.template_manager.finalize_template(template_report)
        formatted_report = template_report.format(**format_data)
        
        if print_logs:
            print(formatted_report)
        
        # Save outputs
        step_config = self.engine.step_configs[step_name]
        output_manager.save_step_outputs(
            step_config, valid_smi, invalid_smi, formatted_report, dup_idx_data, get_report
        )
        
        return format_return_values(
            valid_smi, idx_of_invalid_smi, format_data, get_invalid_smi_idx, return_format_data
        )