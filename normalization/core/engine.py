"""Core normalization engine."""
import pandas as pd
from typing import Tuple, Optional

from ..config.normalization_config import NormalizationStepConfig
from ..utils.normalization_utils import (
    setup_parallel_processing,
    SMILESNormalizer
)


class NormalizationEngine:
    """Core engine for normalization operations."""
    
    def __init__(self):
        self.step_configs = NormalizationStepConfig.get_all_configs()
    
    def run_detautomerize(self, df: pd.DataFrame, 
                         n_cpu: Optional[int] = -1, 
                         split_factor: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Run detautomerization step.
        
        Args:
            df: Input DataFrame
            n_cpu: Number of CPUs for parallel processing
            split_factor: Split factor for parallel processing
            
        Returns:
            Tuple of (normalized_df, diff_df, format_data)
        """
        setup_parallel_processing(n_cpu, split_factor)
        config = self.step_configs['detautomerize']
        smi_col = df.columns.tolist()
        
        # Apply detautomerization
        post_detautomerized = df[smi_col[0]].p_apply(
            lambda x: SMILESNormalizer(x, return_difference=True).normalize_tautomer()
        )
        
        # Extract results
        post_detautomerized_smi_df = pd.DataFrame(
            list(post_detautomerized.p_apply(lambda x: x[0])),
            columns=["post_smiles"],
        )
        diff_after_detautomerized = pd.DataFrame(
            list(post_detautomerized.p_apply(lambda x: x[1])), columns=["diff"]
        )
        
        # Combine results
        post_smi_df = pd.concat(
            [
                post_detautomerized_smi_df.reset_index(drop=True),
                df.reset_index(drop=True),
            ],
            axis=1,
        )
        post_smi_df.drop(columns=post_smi_df.columns[[1]], inplace=True)
        post_smi_df.rename(
            columns={post_smi_df.columns[0]: "smiles"}, inplace=True
        )
        
        # Create format data
        format_data = {
            config.format_keys['input']: len(df),
            config.format_keys['normalized']: len(
                diff_after_detautomerized[diff_after_detautomerized["diff"] == True]
            ),
            config.format_keys['output']: len(post_smi_df),
        }
        
        return post_smi_df, diff_after_detautomerized, format_data
    
    def run_destereoisomerize(self, df: pd.DataFrame,
                              n_cpu: Optional[int] = -1, 
                              split_factor: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Run destereoisomerization step.
        
        Args:
            df: Input DataFrame
            n_cpu: Number of CPUs for parallel processing
            split_factor: Split factor for parallel processing
            
        Returns:
            Tuple of (normalized_df, diff_df, format_data)
        """
        setup_parallel_processing(n_cpu, split_factor)
        config = self.step_configs['destereoisomerize']
        smi_col = df.columns.tolist()
        
        # Apply destereoisomerization
        post_destereoisomerized = df[smi_col[0]].p_apply(
            lambda x: SMILESNormalizer(x, return_difference=True).normalize_stereoisomer()
        )
        
        # Extract results
        post_destereoisomerized_smi_df = pd.DataFrame(
            list(post_destereoisomerized.p_apply(lambda x: x[0])),
            columns=["post_smiles"],
        )
        diff_after_destereoisomerized = pd.DataFrame(
            list(post_destereoisomerized.p_apply(lambda x: x[1])),
            columns=["diff"],
        )
        
        # Combine results
        post_smi_df = pd.concat(
            [
                post_destereoisomerized_smi_df.reset_index(drop=True),
                df.reset_index(drop=True),
            ],
            axis=1,
        )
        post_smi_df.drop(columns=post_smi_df.columns[[1]], inplace=True)
        post_smi_df.rename(
            columns={post_smi_df.columns[0]: "smiles"}, inplace=True
        )
        
        # Create format data
        format_data = {
            config.format_keys['input']: len(df),
            config.format_keys['normalized']: len(
                diff_after_destereoisomerized[diff_after_destereoisomerized["diff"] == True]
            ),
            config.format_keys['output']: len(post_smi_df),
        }
        
        return post_smi_df, diff_after_destereoisomerized, format_data

