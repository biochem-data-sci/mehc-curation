"""Normalization-specific utility functions."""
import os
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from parallel_pandas import ParallelPandas
from rdkit import Chem

# Import rdMolStandardize - will be imported at function level to handle different RDKit versions


class SMILESNormalizer:
    """
    A class to normalize SMILES strings (tautomer and stereoisomer normalization).
    
    Attributes:
        smi (str): The SMILES representation of a molecule.
        return_difference (bool): Whether to return difference information.
    """
    
    def __init__(self, smi: str, return_difference: bool = False):
        """
        Initialize the SMILESNormalizer class with a SMILES string.
        
        Args:
            smi (str): The SMILES representation of a molecule.
            return_difference (bool): Whether to return difference information.
        """
        self.smi = smi
        self.return_difference = return_difference
    
    def normalize_tautomer(self) -> Tuple[str, bool]:
        """
        Normalize tautomers in SMILES string.
        
        Returns:
            Tuple of (normalized_smiles, diff)
            - normalized_smiles: SMILES after tautomer normalization
            - diff: True if normalized (changed), False otherwise
        """
        try:
            mol = Chem.MolFromSmiles(self.smi)
            if mol is None:
                return self.smi, False
            
            # Import rdMolStandardize at function level
            try:
                from rdkit.Chem.MolStandardize import rdMolStandardize
            except ImportError:
                try:
                    from rdkit.Chem import rdMolStandardize
                except ImportError:
                    raise ImportError(
                        "rdMolStandardize not found. Please ensure RDKit is properly installed. "
                        "Try: conda install -c conda-forge rdkit"
                    )
            
            # Use RDKit's TautomerEnumerator
            enumerator = rdMolStandardize.TautomerEnumerator()
            normalized_mol = enumerator.Canonicalize(mol)
            
            if normalized_mol is None:
                return self.smi, False
            
            normalized_smi = Chem.MolToSmiles(normalized_mol)
            diff = (normalized_smi != self.smi)
            
            return normalized_smi, diff
            
        except Exception:
            return self.smi, False
    
    def normalize_stereoisomer(self) -> Tuple[str, bool]:
        """
        Normalize stereoisomers in SMILES string (remove stereochemistry).
        
        Returns:
            Tuple of (normalized_smiles, diff)
            - normalized_smiles: SMILES after stereoisomer normalization
            - diff: True if normalized (changed), False otherwise
        """
        try:
            mol = Chem.MolFromSmiles(self.smi)
            if mol is None:
                return self.smi, False
            
            # Remove stereochemistry by converting to SMILES without isomeric information
            normalized_smi = Chem.MolToSmiles(mol, isomericSmiles=False)
            diff = (normalized_smi != self.smi)
            
            return normalized_smi, diff
            
        except Exception:
            return self.smi, False


def setup_parallel_processing(n_cpu: Optional[int] = -1, split_factor: int = 1) -> None:
    """Initialize parallel processing for normalization operations."""
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


def extract_normalization_results(
    normalized_data: pd.DataFrame,
    diff_data: pd.DataFrame,
    config  # NormalizationStepConfig - imported at function level
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Extract normalization results and format data.
    
    Args:
        normalized_data: DataFrame with normalized SMILES
        diff_data: DataFrame with difference flags
        config: NormalizationStepConfig object
        
    Returns:
        Tuple of (normalized_df, diff_df, format_data)
    """
    format_data = {
        config.format_keys['input']: len(normalized_data) + len(diff_data[diff_data.iloc[:, 0] == False]) if len(diff_data) > 0 else len(normalized_data),
        config.format_keys['normalized']: len(diff_data[diff_data.iloc[:, 0] == True]) if len(diff_data) > 0 else 0,
        config.format_keys['output']: len(normalized_data)
    }
    
    return normalized_data, diff_data, format_data


def format_normalization_return_values(
    normalized_smi: pd.DataFrame,
    diff_data: pd.DataFrame,
    format_data: Dict,
    get_diff: bool,
    return_format_data: bool
) -> Any:
    """Format return values based on flags - preserves exact original logic."""
    if get_diff and return_format_data:
        return normalized_smi, diff_data, format_data
    elif get_diff and not return_format_data:
        return normalized_smi, diff_data
    elif return_format_data and not get_diff:
        return normalized_smi, format_data
    elif not get_diff and not return_format_data:
        return normalized_smi

