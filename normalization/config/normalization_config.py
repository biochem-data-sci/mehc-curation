"""Configuration settings for normalization operations."""
import os
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class NormalizationStepConfig:
    """Configuration for individual normalization steps."""
    step_name: str
    normalization_method: str    # SMILESNormalizer method name
    template_file: str            # Template file name
    output_csv: str               # Output CSV filename
    report_subdir: str            # Report subdirectory name
    format_keys: Dict[str, str]   # Keys for format data
    return_diff: bool = True        # Whether this step returns diff data

    @classmethod
    def get_all_configs(cls) -> Dict[str, 'NormalizationStepConfig']:
        """Get all normalization step configurations."""
        return {
            'detautomerize': cls(
                step_name='detautomerize',
                normalization_method='normalize_tautomer',
                template_file='detautomerization.txt',
                output_csv='post_detautomerized.csv',
                report_subdir='detautomerized',
                format_keys={
                    'input': 'detautomerization_input',
                    'normalized': 'detautomerized',
                    'output': 'detautomerization_output'
                },
                return_diff=True
            ),
            'destereoisomerize': cls(
                step_name='destereoisomerize',
                normalization_method='normalize_stereoisomer',
                template_file='destereoisomerization.txt',
                output_csv='post_stereoisomer_normalized.csv',
                report_subdir='normalize_stereoisomer',
                format_keys={
                    'input': 'destereoisomerization_input',
                    'normalized': 'destereoisomerized',
                    'output': 'destereoisomerization_output'
                },
                return_diff=True
            )
        }


@dataclass
class NormalizationConfig:
    """General configuration for normalization operations."""
    template_dir: Optional[str] = None
    default_n_cpu: Optional[int] = -1
    default_split_factor: int = 1
    
    def __post_init__(self):
        if self.template_dir is None:
            # Get template directory using the shared utility
            from ...utils.io_utils import setup_template_dir
            self.template_dir = setup_template_dir()

