"""Core cleaning engine."""
import pandas as pd
from typing import Tuple, Optional

from ..config.cleaning_config import CleaningStepConfig
from ..utils.cleaning_utils import (
    setup_parallel_processing,
    SMILESCleaner
)


class CleaningEngine:
    """Core engine for cleaning operations."""
    
    def __init__(self):
        self.step_configs = CleaningStepConfig.get_all_configs()
    
    def run_salt_cleaning(self, df: pd.DataFrame, 
                         n_cpu: Optional[int] = -1, 
                         split_factor: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
        """
        Run salt cleaning step.
        
        Args:
            df: Input DataFrame
            n_cpu: Number of CPUs for parallel processing
            split_factor: Split factor for parallel processing
            
        Returns:
            Tuple of (cleaned_df, diff_df, missing_df, format_data)
        """
        setup_parallel_processing(n_cpu, split_factor)
        config = self.step_configs['cl_salt']
        smi_col = df.columns.tolist()
        
        # Apply salt cleaning
        salts_cleaned = df[smi_col[0]].p_apply(
            lambda x: SMILESCleaner(x, return_dif=True).clean_salt(
                return_is_null_smi=True
            )
        )
        
        # Extract results
        post_salts_cl_smi_data = pd.DataFrame(
            list(salts_cleaned.p_apply(lambda x: x[0])),
            columns=["post_smiles"],
        )
        diff_after_cl_salt = pd.DataFrame(
            list(salts_cleaned.p_apply(lambda x: x[1])), columns=["diff"]
        )
        is_missing_smi_str = pd.DataFrame(
            list(salts_cleaned.p_apply(lambda x: x[2])), columns=["is_missing"]
        )
        
        # Combine results
        post_smi_df = pd.concat(
            [
                post_salts_cl_smi_data.reset_index(drop=True),
                df.reset_index(drop=True),
                diff_after_cl_salt.reset_index(drop=True),
                is_missing_smi_str.reset_index(drop=True),
            ],
            axis=1,
        )
        
        # Filter out missing SMILES
        post_smi_df = post_smi_df[post_smi_df["is_missing"] == False]
        post_smi_df.drop(
            columns=post_smi_df.columns[[1, -1, -2]], inplace=True
        )
        post_smi_df.rename(
            columns={post_smi_df.columns[0]: "smiles"}, inplace=True
        )
        
        missing_smiles_cnt = len(post_salts_cl_smi_data) - len(post_smi_df)
        
        # Create format data
        format_data = {
            config.format_keys['input']: len(df),
            config.format_keys['desalted']: int(diff_after_cl_salt["diff"].sum()),
            config.format_keys['unprocessable']: missing_smiles_cnt,
            config.format_keys['output']: len(post_smi_df),
        }
        
        return post_smi_df, diff_after_cl_salt, is_missing_smi_str, format_data
    
    def run_neutralization(self, df: pd.DataFrame, method: str = "boyle",
                          n_cpu: Optional[int] = -1, split_factor: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Run neutralization step.
        
        Args:
            df: Input DataFrame
            method: Method to use - "boyle" or "rdkit"
            n_cpu: Number of CPUs for parallel processing
            split_factor: Split factor for parallel processing
            
        Returns:
            Tuple of (cleaned_df, diff_df, format_data)
        """
        setup_parallel_processing(n_cpu, split_factor)
        config = self.step_configs['neutralize']
        smi_col = df.columns.tolist()
        
        # Apply neutralization
        neutralized = df[smi_col[0]].p_apply(
            lambda x: SMILESCleaner(x, return_dif=True).neutralize_salt(method=method)
        )
        
        # Extract results
        post_neutralized_smi_data = pd.DataFrame(
            list(neutralized.p_apply(lambda x: x[0])),
            columns=["neutralized_smiles"],
        )
        diff_after_neutralize = pd.DataFrame(
            list(neutralized.p_apply(lambda x: x[1])), columns=["diff"]
        )
        
        # Combine results
        post_smi_df = pd.concat(
            [
                post_neutralized_smi_data.reset_index(drop=True),
                df.reset_index(drop=True),
                diff_after_neutralize.reset_index(drop=True),
            ],
            axis=1,
        )
        
        # Filter out unprocessable (None values)
        post_smi_df = post_smi_df[pd.notna(post_smi_df["diff"])]
        post_smi_df.drop(columns=post_smi_df.columns[[1, -1]], inplace=True)
        post_smi_df.rename(
            columns={post_smi_df.columns[0]: "smiles"}, inplace=True
        )
        
        unprocessable_cnt = len(df) - len(post_smi_df)
        
        # Create format data
        format_data = {
            config.format_keys['input']: len(df),
            config.format_keys['neutralized']: len(
                diff_after_neutralize[diff_after_neutralize["diff"] == 1]
            ),
            config.format_keys['unprocessable']: unprocessable_cnt,
            config.format_keys['output']: len(post_smi_df),
        }
        
        return post_smi_df, diff_after_neutralize, format_data

