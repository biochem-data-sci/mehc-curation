"""Core cleaning engine."""
import pandas as pd
from typing import Tuple, Optional

from ..config.cleaning_config import CleaningStepConfig
from ..utils.cleaning_utils import SMILESCleaner
from ...utils.common import parallel_apply
from ...utils.mol_io import records_for


class CleaningEngine:
    """Core engine for cleaning operations."""
    
    def __init__(self):
        self.step_configs = CleaningStepConfig.get_all_configs()
    
    def run_salt_cleaning(self, df: pd.DataFrame, 
                         n_cpu: Optional[int] = -1, 
                         split_factor: int = 1,
                         mol_cache: Optional[dict] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
        """
        Run salt cleaning step.
        
        Args:
            df: Input DataFrame
            n_cpu: Number of CPUs for parallel processing
            split_factor: Split factor for parallel processing
            mol_cache: Optional index -> MolRecord cache, updated in place
            
        Returns:
            Tuple of (cleaned_df, diff_df, missing_df, format_data)
        """
        config = self.step_configs['cl_salt']
        smi_col = df.columns.tolist()

        records = records_for(df[smi_col[0]], mol_cache, n_cpu, split_factor)
        salts_cleaned = parallel_apply(
            records,
            lambda r: SMILESCleaner(r, return_dif=True).clean_salt_record(
                return_is_null_smi=True
            ),
            n_cpu,
            split_factor,
        )

        # Extract results
        post_salts_cl_smi_data = pd.DataFrame(
            {"post_smiles": [record.smi for record, _, _ in salts_cleaned]}
        )
        diff_after_cl_salt = pd.DataFrame(
            {"diff": [diff for _, diff, _ in salts_cleaned]}
        )
        is_missing_smi_str = pd.DataFrame(
            {"is_missing": [missing for _, _, missing in salts_cleaned]}
        )

        post_salts_cl_smi_data.index = salts_cleaned.index
        diff_after_cl_salt.index = salts_cleaned.index
        is_missing_smi_str.index = salts_cleaned.index
         
        # Combine results
        post_smi_df = pd.concat(
            [
                post_salts_cl_smi_data,
                df,
                diff_after_cl_salt,
                is_missing_smi_str,
            ],
            axis=1,
        )
        
        # Filter out missing SMILES
        post_smi_df = post_smi_df[post_smi_df["is_missing"] == False]
        post_smi_df.drop(
            columns=post_smi_df.columns[[1, -1, -2]], inplace=True
        )
        post_smi_df.rename(
            columns={post_smi_df.columns[0]: smi_col[0]}, inplace=True
        )
        
        missing_smiles_cnt = len(post_salts_cl_smi_data) - len(post_smi_df)
        
        if mol_cache is not None:
            for idx, (record, _, is_missing) in zip(salts_cleaned.index, salts_cleaned):
                if not is_missing:
                    mol_cache[idx] = record
        
        # Create format data
        format_data = {
            config.format_keys['input']: len(df),
            config.format_keys['desalted']: int(diff_after_cl_salt["diff"].sum()),
            config.format_keys['unprocessable']: missing_smiles_cnt,
            config.format_keys['output']: len(post_smi_df),
        }
        
        return post_smi_df, diff_after_cl_salt, is_missing_smi_str, format_data
    
    def run_neutralization(self, df: pd.DataFrame, neutralizing_method: str = "boyle",
                          n_cpu: Optional[int] = -1, split_factor: int = 1,
                          mol_cache: Optional[dict] = None) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Run neutralization step.
        
        Args:
            df: Input DataFrame
            method: Method to use - "boyle" or "rdkit"
            n_cpu: Number of CPUs for parallel processing
            split_factor: Split factor for parallel processing
            mol_cache: Optional index -> MolRecord cache, updated in place
            
        Returns:
            Tuple of (cleaned_df, diff_df, format_data)
        """
        config = self.step_configs['neutralize']
        smi_col = df.columns.tolist()
        
        # Apply neutralization
        records = records_for(df[smi_col[0]], mol_cache, n_cpu, split_factor)
        neutralized = parallel_apply(
            records,
            lambda r: SMILESCleaner(r, return_dif=True).neutralize_salt_record(neutralizing_method=neutralizing_method),
            n_cpu,
            split_factor,
        )

        # Extract results
        post_neutralized_smi_data = pd.DataFrame(
            {"neutralized_smiles": [record.smi for record, _ in neutralized]}
        )
        diff_after_neutralize = pd.DataFrame(
            {"diff": [diff for _, diff in neutralized]}
        )

        post_neutralized_smi_data.index = neutralized.index
        diff_after_neutralize.index = neutralized.index
        
        # Combine results
        post_smi_df = pd.concat(
            [
                post_neutralized_smi_data,
                df,
                diff_after_neutralize,
            ],
            axis=1,
        )
        
        # Filter out unprocessable (None values)
        post_smi_df = post_smi_df[pd.notna(post_smi_df["diff"])]
        post_smi_df.drop(columns=post_smi_df.columns[[1, -1]], inplace=True)
        post_smi_df.rename(
            columns={post_smi_df.columns[0]: smi_col[0]}, inplace=True
        )
        
        unprocessable_cnt = len(df) - len(post_smi_df)
        
        if mol_cache is not None:
            for idx, (record, diff) in zip(neutralized.index, neutralized):
                if pd.notna(diff):
                    mol_cache[idx] = record
        
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

