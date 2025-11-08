"""Configuration settings for cleaning operations."""
import os
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class CleaningStepConfig:
    """Configuration for individual cleaning steps."""
    step_name: str
    cleaning_method: str          # SMILESCleaner method name
    template_file: str            # Template file name
    output_csv: str               # Output CSV filename
    report_subdir: str            # Report subdirectory name
    format_keys: Dict[str, str]   # Keys for format data
    return_diff: bool = False     # Whether this step returns diff data

    @classmethod
    def get_all_configs(cls) -> Dict[str, 'CleaningStepConfig']:
        """Get all cleaning step configurations."""
        return {
            'cl_salt': cls(
                step_name='cl_salt',
                cleaning_method='clean_salt',
                template_file='salt_cleaning.txt',
                output_csv='post_cleaned_salts.csv',
                report_subdir='cleaning_salts',
                format_keys={
                    'input': 'salt_cleaning_input',
                    'desalted': 'desalted',
                    'unprocessable': 'unprocessable',
                    'output': 'salt_cleaning_output'
                },
                return_diff=True
            ),
            'neutralize': cls(
                step_name='neutralize',
                cleaning_method='neutralize_salt',
                template_file='neutralization.txt',
                output_csv='post_neutralized_smiles.csv',
                report_subdir='neutralization',
                format_keys={
                    'input': 'neutralization_input',
                    'neutralized': 'neutralized',
                    'unprocessable': 'neutralize_unprocessable',
                    'output': 'neutralization_output'
                },
                return_diff=True
            )
        }


@dataclass
class CleaningConfig:
    """General configuration for cleaning operations."""
    template_dir: Optional[str] = None
    default_n_cpu: Optional[int] = -1
    default_split_factor: int = 1
    
    def __post_init__(self):
        if self.template_dir is None:
            # Get template directory using the shared utility
            from ...utils.io_utils import setup_template_dir
            self.template_dir = setup_template_dir()

