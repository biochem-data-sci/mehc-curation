"""Core normalization engine."""
import pandas as pd
from typing import Tuple, Optional

from ..config.normalization_config import NormalizationStepConfig
from ..utils.normalization_utils import SMILESNormalizer
from ...utils.common import parallel_apply
from ...utils.mol_io import records_for


class NormalizationEngine:
    """Core engine for normalization operations."""
    
    def __init__(self):
        self.step_configs = NormalizationStepConfig.get_all_configs()
    
    def run_detautomerize(self, df: pd.DataFrame, 
                         n_cpu: Optional[int] = -1, 
                         split_factor: int = 1,
                         mol_cache: Optional[dict] = None) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Run detautomerization step.
        
        Args:
            df: Input DataFrame
            n_cpu: Number of CPUs for parallel processing
            split_factor: Split factor for parallel processing
            mol_cache: Optional parsed-molecule cache shared between steps
            
        Returns:
            Tuple of (normalized_df, diff_df, format_data)
        """
        config = self.step_configs['detautomerize']
        smi_col = df.columns.tolist()
        
        # Apply detautomerization
        records = records_for(df[smi_col[0]], mol_cache, n_cpu, split_factor)
        post_detautomerized = parallel_apply(
            records,
            lambda x: SMILESNormalizer(x, return_difference=True).normalize_tautomer_record(),
            n_cpu,
            split_factor,
        )
        
        # Extract results
        post_detautomerized_smi_df = pd.DataFrame(
            [record.smi for record, _ in post_detautomerized],
            columns=["post_smiles"],
        )
        diff_after_detautomerized = pd.DataFrame(
            [diff for _, diff in post_detautomerized],
            columns=["diff"]
        )

        post_detautomerized_smi_df.index = post_detautomerized.index
        diff_after_detautomerized.index = post_detautomerized.index

        if mol_cache is not None:
            for idx, (record, _) in zip(post_detautomerized.index, post_detautomerized):
                mol_cache[idx] = record
        
        # Combine results
        # post_smi_df = pd.concat(
        #     [
        #         post_detautomerized_smi_df.reset_index(drop=True),
        #         df.reset_index(drop=True),
        #     ],
        #     axis=1,
        # )
        post_smi_df = pd.concat(
            [
                post_detautomerized_smi_df,
                df,
            ],
            axis=1,
        )
        post_smi_df.drop(columns=post_smi_df.columns[[1]], inplace=True)
        post_smi_df.rename(
            columns={post_smi_df.columns[0]: smi_col[0]}, inplace=True
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
                              split_factor: int = 1,
                              mol_cache: Optional[dict] = None) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Run destereoisomerization step.
        
        Args:
            df: Input DataFrame
            n_cpu: Number of CPUs for parallel processing
            split_factor: Split factor for parallel processing
            mol_cache: Optional parsed-molecule cache shared between steps
            
        Returns:
            Tuple of (normalized_df, diff_df, format_data)
        """
        config = self.step_configs['destereoisomerize']
        smi_col = df.columns.tolist()
        
        # Apply destereoisomerization
        records = records_for(df[smi_col[0]], mol_cache, n_cpu, split_factor)
        post_destereoisomerized = parallel_apply(
            records,
            lambda x: SMILESNormalizer(x, return_difference=True).normalize_stereoisomer_record(),
            n_cpu,
            split_factor,
        )
        
        # Extract results
        post_destereoisomerized_smi_df = pd.DataFrame(
            [record.smi for record, _ in post_destereoisomerized],
            columns=["post_smiles"],
        )
        diff_after_destereoisomerized = pd.DataFrame(
            [diff for _, diff in post_destereoisomerized],
            columns=["diff"],
        )

        post_destereoisomerized_smi_df.index = post_destereoisomerized.index
        diff_after_destereoisomerized.index = post_destereoisomerized.index

        if mol_cache is not None:
            for idx, (record, _) in zip(post_destereoisomerized.index, post_destereoisomerized):
                mol_cache[idx] = record
        
        # Combine results
        # post_smi_df = pd.concat(
        #     [
        #         post_destereoisomerized_smi_df.reset_index(drop=True),
        #         df.reset_index(drop=True),
        #     ],
        #     axis=1,
        # )
        post_smi_df = pd.concat(
            [
                post_destereoisomerized_smi_df,
                df,
            ],
            axis=1,
        )
        post_smi_df.drop(columns=post_smi_df.columns[[1]], inplace=True)
        post_smi_df.rename(
            columns={post_smi_df.columns[0]: smi_col[0]}, inplace=True
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

