"""Core validation engine."""
import pandas as pd
from typing import Tuple, List, Dict, Optional

from ..config.validation_config import ValidationStepConfig
from ..utils.validation_utils import (
    setup_parallel_processing,
    apply_smiles_normalization,
    extract_validation_results,
    SMILESValidator  # FIXED: Import from validation utils
)


class ValidationEngine:
    """Core engine for validation operations."""
    
    def __init__(self):
        self.step_configs = ValidationStepConfig.get_all_configs()
    
    def run_validation_step(self, df: pd.DataFrame, step_name: str, 
                           n_cpu: Optional[int] = -1, 
                           split_factor: int = 1) -> Tuple[pd.DataFrame, pd.Series, List, Dict]:
        """
        Run a single validation step.
        
        Args:
            df: Input DataFrame
            step_name: Name of validation step
            n_cpu: Number of CPUs for parallel processing
            split_factor: Split factor for parallel processing
            
        Returns:
            Tuple of (valid_df, invalid_series, invalid_indices, format_data)
        """        
        setup_parallel_processing(n_cpu, split_factor)
        config = self.step_configs[step_name]
        smi_col = df.columns.tolist()
        
        # Apply validation method
        df_copy = df.copy()
        df_copy["is_valid"] = df_copy[smi_col[0]].p_apply(
            lambda x: getattr(SMILESValidator(x), config.validation_method)()
        )
        
        # Extract results
        return extract_validation_results(df_copy, config)
    
    def normalize_smiles(self, df: pd.DataFrame, get_isomeric_smi: bool = True,
                        n_cpu: Optional[int] = -1, split_factor: int = 1) -> pd.DataFrame:
        """Apply SMILES normalization."""
        setup_parallel_processing(n_cpu, split_factor)
        return apply_smiles_normalization(df, get_isomeric_smi)