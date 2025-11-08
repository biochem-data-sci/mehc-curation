"""Configuration settings for validation operations."""
import os
from dataclasses import dataclass
from typing import Dict, Optional, List


@dataclass
class ValidationStepConfig:
    """Configuration for individual validation steps."""
    step_name: str
    validation_method: str          # SMILESValidator method name
    valid_condition: str            # Filter condition for valid molecules
    invalid_condition: str          # Filter condition for invalid molecules
    template_file: str              # Template file name
    output_csv: str                 # Output CSV filename
    invalid_csv: str                # Invalid molecules CSV filename
    report_subdir: str              # Report subdirectory name
    format_keys: Dict[str, str]     # Keys for format data
    index_return_type: str = "index"  # "index" or "values" (for rm_inorganic bug)

    @classmethod
    def get_all_configs(cls) -> Dict[str, 'ValidationStepConfig']:
        """Get all validation step configurations."""
        return {
            'validate_smi': cls(
                step_name='validate_smi',
                validation_method='is_valid',
                valid_condition='==True',
                invalid_condition='==False',
                template_file='validity_check.txt',
                output_csv='valid_smiles.csv',
                invalid_csv='invalid_smiles.csv',
                report_subdir='check_valid_smiles',
                format_keys={
                    'input': 'validity_input',
                    'invalid': 'invalid',
                    'valid': 'valid'
                }
            ),
            'rm_mixture': cls(
                step_name='rm_mixture',
                validation_method='is_mixture',
                valid_condition='==False',  # Non-mixtures are valid
                invalid_condition='==True',  # Mixtures are invalid
                template_file='mixture_removal.txt',
                output_csv='non_mixtures.csv',
                invalid_csv='mixtures.csv',
                report_subdir='rm_mixture',
                format_keys={
                    'input': 'rm_mixture_input',
                    'invalid': 'mixture',
                    'valid': 'non_mixture'
                }
            ),
            'rm_inorganic': cls(
                step_name='rm_inorganic',
                validation_method='is_inorganic',
                valid_condition='==False',  # Organics are valid
                invalid_condition='==True',  # Inorganics are invalid
                template_file='inorganic_removal.txt',
                output_csv='organics.csv',
                invalid_csv='inorganics.csv',
                report_subdir='rm_inorganic',
                format_keys={
                    'input': 'rm_inorganic_input',
                    'invalid': 'inorganic',
                    'valid': 'organic'
                },
                index_return_type='values'  # Bug: returns values instead of indices
            ),
            'rm_organometallic': cls(
                step_name='rm_organometallic',
                validation_method='is_organometallic',
                valid_condition='==False',  # FIXED: Non-organometallics are valid (kept)
                invalid_condition='==True', # FIXED: Organometallics are invalid (removed)
                template_file='organometallic_removal.txt',
                output_csv='non_organometallic.csv',
                invalid_csv='organometallic.csv',
                report_subdir='rm_organometallic',
                format_keys={
                    'input': 'rm_organometallic_input',
                    'invalid': 'organometallic',
                    'valid': 'non_organometallic'
                }
            )
        }


@dataclass
class ValidationConfig:
    """General configuration for validation operations."""
    template_dir: Optional[str] = None
    default_n_cpu: Optional[int] = -1
    default_split_factor: int = 1
    
    def __post_init__(self):
        if self.template_dir is None:
            # Get template directory using the shared utility
            from ...utils.io_utils import setup_template_dir
            self.template_dir = setup_template_dir()