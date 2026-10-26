"""Core validation engine."""
import pandas as pd
from typing import Tuple, List, Dict, Optional

from ..config.validation_config import ValidationStepConfig
from ..utils.validation_utils import (
    apply_smiles_normalization,
    extract_validation_results,
    SMILESValidator  # FIXED: Import from validation utils
)
from ...utils.common import parallel_apply
from ...utils.mol_io import records_for


class ValidationEngine:
    """Core engine for validation operations."""
    
    def __init__(self):
        self.step_configs = ValidationStepConfig.get_all_configs()
    
    def run_validation_step(self, df: pd.DataFrame, step_name: str, 
                           n_cpu: Optional[int] = -1, 
                           split_factor: int = 1,
                           mol_cache: Optional[Dict] = None) -> Tuple[pd.DataFrame, pd.Series, List, Dict, pd.Series]:
        """
        Run a single validation step.
        
        Args:
            df: Input DataFrame
            step_name: Name of validation step
            n_cpu: Number of CPUs for parallel processing
            split_factor: Split factor for parallel processing
            mol_cache: Optional parsed-molecule cache shared between steps
            
        Returns:
            Tuple of (valid_df, invalid_series, invalid_indices, format_data, valid_records)
            The last element is internal, for reuse by later steps.
        """        
        config = self.step_configs[step_name]
        smi_col = df.columns.tolist()
        
        # Apply validation method
        df_copy = df.copy()
        records = records_for(df_copy[smi_col[0]], mol_cache, n_cpu, split_factor)
        df_copy["is_valid"] = parallel_apply(
            records,
            lambda record: getattr(SMILESValidator(record), config.validation_method)(),
            n_cpu,
            split_factor,
        )
        
        # Extract results
        valid_smi, invalid_smi, idx_of_invalid_smi, format_data = \
            extract_validation_results(df_copy, config)
        # positional, so a duplicated index cannot misalign the records
        valid_records = records[
            df_copy["is_valid"].to_numpy() == (config.valid_condition == '==True')
        ]
        if mol_cache is not None:
            mol_cache.update(valid_records.to_dict())
        
        return valid_smi, invalid_smi, idx_of_invalid_smi, format_data, valid_records
    
    def normalize_smiles(self, df: pd.DataFrame, get_isomeric_smi: bool = True,
                        n_cpu: Optional[int] = -1, split_factor: int = 1,
                        records: Optional[pd.Series] = None) -> pd.DataFrame:
        """Apply SMILES normalization."""
        return apply_smiles_normalization(df, get_isomeric_smi, n_cpu, split_factor, records)