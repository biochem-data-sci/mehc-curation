"""Cleaning-specific utility functions."""
import os
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from parallel_pandas import ParallelPandas
from rdkit import Chem
from rdkit.Chem import SaltRemover

# Import rdMolStandardize - will be imported at function level to handle different RDKit versions


class SMILESCleaner:
    """
    A class to clean SMILES strings (salt removal, neutralization).
    
    Attributes:
        smi (str): The SMILES representation of a molecule.
        return_dif (bool): Whether to return difference information.
    """
    
    def __init__(self, smi: str, return_dif: bool = False):
        """
        Initialize the SMILESCleaner class with a SMILES string.
        
        Args:
            smi (str): The SMILES representation of a molecule.
            return_dif (bool): Whether to return difference information.
        """
        self.smi = smi
        self.return_dif = return_dif
    
    def clean_salt(self, return_is_null_smi: bool = False) -> Tuple[str, bool, bool]:
        """
        Remove salts from SMILES string.
        
        Args:
            return_is_null_smi: Whether to return if SMILES is null after cleaning
            
        Returns:
            Tuple of (cleaned_smiles, diff, is_missing)
            - cleaned_smiles: SMILES after salt removal
            - diff: True if salts were removed (changed), False otherwise
            - is_missing: True if SMILES is invalid/missing after cleaning
        """
        try:
            mol = Chem.MolFromSmiles(self.smi)
            if mol is None:
                if return_is_null_smi:
                    return self.smi, False, True
                return self.smi, False, False
            
            # Use RDKit's SaltRemover
            remover = SaltRemover.SaltRemover()
            cleaned_mol = remover.StripMol(mol, dontRemoveEverything=True)
            
            if cleaned_mol is None or cleaned_mol.GetNumAtoms() == 0:
                if return_is_null_smi:
                    return self.smi, False, True
                return self.smi, False, False
            
            cleaned_smi = Chem.MolToSmiles(cleaned_mol)
            diff = (cleaned_smi != self.smi)
            
            if return_is_null_smi:
                is_missing = (cleaned_smi == "" or cleaned_smi is None)
                return cleaned_smi, diff, is_missing
            
            return cleaned_smi, diff, False
            
        except Exception:
            if return_is_null_smi:
                return self.smi, False, True
            return self.smi, False, False
    
    def neutralize_salt(self, method: str = "boyle") -> Tuple[str, Optional[int]]:
        """
        Neutralize SMILES string using specified method.
        
        Args:
            method: Method to use - "boyle" or "rdkit"
            
        Returns:
            Tuple of (neutralized_smiles, diff)
            - neutralized_smiles: SMILES after neutralization
            - diff: 1 if neutralized (changed), 0 if not changed, None if error
        """
        try:
            mol = Chem.MolFromSmiles(self.smi)
            if mol is None:
                return self.smi, None
            
            # Import rdMolStandardize at function level to handle different RDKit versions
            try:
                from rdkit.Chem.MolStandardize import rdMolStandardize
            except ImportError:
                # Try alternative import path
                try:
                    from rdkit.Chem import rdMolStandardize
                except ImportError:
                    raise ImportError(
                        "rdMolStandardize not found. Please ensure RDKit is properly installed. "
                        "Try: conda install -c conda-forge rdkit"
                    )
            
            if method == "boyle":
                # Boyle method: simple charge neutralization
                mol = rdMolStandardize.Normalize(mol)
                neutralized_smi = Chem.MolToSmiles(mol)
            elif method == "rdkit":
                # RDKit standardizer
                normalizer = rdMolStandardize.Normalizer()
                mol = normalizer.normalize(mol)
                neutralized_smi = Chem.MolToSmiles(mol)
            else:
                raise ValueError(f"Unknown method: {method}. Must be 'boyle' or 'rdkit'")
            
            diff = 1 if (neutralized_smi != self.smi) else 0
            return neutralized_smi, diff
            
        except Exception:
            return self.smi, None


def setup_parallel_processing(n_cpu: Optional[int] = -1, split_factor: int = 1) -> None:
    """Initialize parallel processing for cleaning operations."""
    # Get available CPUs
    available_cpus = os.cpu_count() or 1
    
    # Validate and adjust n_cpu
    if n_cpu == -1:
        # Use all available CPUs
        n_cpu = available_cpus
    elif n_cpu == 0:
        # Set to 1 if user specified 0
        n_cpu = 1
    elif n_cpu < 0:
        # Negative numbers (except -1) set to 1
        n_cpu = 1
    elif n_cpu > available_cpus:
        # Cap at maximum available CPUs
        n_cpu = available_cpus
    
    ParallelPandas.initialize(
        n_cpu=n_cpu, split_factor=split_factor, disable_pr_bar=True
    )


def extract_cleaning_results(
    cleaned_data: pd.DataFrame,
    diff_data: pd.DataFrame,
    missing_data: Optional[pd.DataFrame],
    config  # CleaningStepConfig - imported at function level
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Extract cleaning results and format data.
    
    Args:
        cleaned_data: DataFrame with cleaned SMILES
        diff_data: DataFrame with difference flags
        missing_data: Optional DataFrame with missing flags
        config: CleaningStepConfig object
        
    Returns:
        Tuple of (cleaned_df, diff_df, format_data)
    """
    format_data = {
        config.format_keys['input']: len(cleaned_data) if missing_data is None else len(cleaned_data) + len(missing_data[missing_data.iloc[:, 0] == True]),
        config.format_keys['output']: len(cleaned_data)
    }
    
    # Add step-specific format keys
    if config.step_name == 'cl_salt':
        format_data[config.format_keys['desalted']] = int(diff_data.sum().iloc[0]) if len(diff_data) > 0 else 0
        if missing_data is not None:
            format_data[config.format_keys['unprocessable']] = int(missing_data.sum().iloc[0]) if len(missing_data) > 0 else 0
        else:
            format_data[config.format_keys['unprocessable']] = 0
    elif config.step_name == 'neutralize':
        format_data[config.format_keys['neutralized']] = len(diff_data[diff_data.iloc[:, 0] == 1]) if len(diff_data) > 0 else 0
        format_data[config.format_keys['unprocessable']] = len(diff_data[pd.isna(diff_data.iloc[:, 0])]) if len(diff_data) > 0 else 0
    
    return cleaned_data, diff_data, format_data


def format_cleaning_return_values(
    cleaned_smi: pd.DataFrame,
    diff_data: pd.DataFrame,
    format_data: Dict,
    get_diff: bool,
    return_format_data: bool
) -> Any:
    """Format return values based on flags - preserves exact original logic."""
    if get_diff and return_format_data:
        return cleaned_smi, diff_data, format_data
    elif get_diff and not return_format_data:
        return cleaned_smi, diff_data
    elif return_format_data and not get_diff:
        return cleaned_smi, format_data
    elif not get_diff and not return_format_data:
        return cleaned_smi

