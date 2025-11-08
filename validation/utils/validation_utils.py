"""Validation-specific utility functions."""
import os
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from parallel_pandas import ParallelPandas
from rdkit import Chem


class SMILESValidator:
    """
    A class to analyze various properties of a SMILES string.

    Attributes:
        smi (str): The SMILES representation of a molecule.
    """

    def __init__(self, smi: str):
        """
        Initialize the SMILESValidator class with a SMILES string.

        Args:
            smi (str): The SMILES representation of a molecule.
        """
        self.smi = smi

    def is_valid(self):
        """Check if the SMILES string is valid and can be converted to a molecule."""
        try:
            mol = Chem.MolFromSmiles(self.smi)
            return mol is not None
        except Exception:
            return False

    def is_mixture(self):
        """Determine if the SMILES string represents a mixture."""
        if self.smi.find(".") != -1:
            if self.smi.find(".[") == -1:
                return True
            else:
                if self.smi.count(".") != self.smi.count(".["):
                    return True
        return False

    def is_inorganic(self):
        """Check if the molecule represented by the SMILES string is inorganic."""
        mol = Chem.MolFromSmiles(self.smi)
        if mol is None:
            return False
        mol = Chem.rdmolops.AddHs(mol)
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "C":
                return False
        return True

    def is_organometallic(self):
        """Check if the molecule represented by the SMILES string is organometallic."""
        from ...utils.io_utils import setup_data_dir
        
        mol = Chem.MolFromSmiles(self.smi)
        if mol is None:
            return False
        dat_dir = setup_data_dir()
        metals = open(os.path.join(dat_dir, "metals.txt")).read().split(",")
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in metals:
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetSymbol() in ("C", "c"):
                        return True  # FIXED: Metal with carbon neighbor IS organometallic
        return False  # FIXED: No metal-carbon bonds means NOT organometallic


def setup_parallel_processing(n_cpu: Optional[int] = -1, split_factor: int = 1) -> None:
    """Initialize parallel processing for validation operations."""
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


def apply_smiles_normalization(df: pd.DataFrame, get_isomeric_smi: bool = True) -> pd.DataFrame:
    """
    Apply SMILES normalization to the dataframe.
    
    Args:
        df: DataFrame with SMILES data
        get_isomeric_smi: Whether to preserve stereochemistry
        
    Returns:
        DataFrame with normalized SMILES
    """
    smi_col = df.columns.tolist()
    df_copy = df.copy()
    
    if get_isomeric_smi:
        df_copy[smi_col[0]] = df_copy[smi_col[0]].p_apply(
            lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x))
        )
    else:
        df_copy[smi_col[0]] = df_copy[smi_col[0]].p_apply(
            lambda x: Chem.MolToSmiles(
                Chem.MolFromSmiles(x), isomericSmiles=False
            )
        )
    
    return df_copy


def extract_validation_results(df: pd.DataFrame, config) -> Tuple[pd.DataFrame, pd.Series, List, Dict]:
    """
    Extract validation results based on configuration.
    
    Args:
        df: DataFrame with validation results
        config: ValidationStepConfig object
        
    Returns:
        Tuple of (valid_df, invalid_series, invalid_indices, format_data)
    """
    smi_col = df.columns.tolist()
    
    # Extract valid molecules
    if config.valid_condition == '==True':
        valid_smi = df[df["is_valid"] == True].copy()
    elif config.valid_condition == '==False':
        valid_smi = df[df["is_valid"] == False].copy()
    
    valid_smi.drop(columns=["is_valid"], inplace=True)
    
    # Extract invalid molecules
    if config.invalid_condition == '==True':
        invalid_smi = df[df["is_valid"] == True][smi_col[0]]
    elif config.invalid_condition == '==False':
        invalid_smi = df[df["is_valid"] == False][smi_col[0]]
    
    # Extract indices (preserve original bug for rm_inorganic)
    if config.index_return_type == 'values':
        # Bug in rm_inorganic: returns SMILES values instead of indices
        idx_of_invalid_smi = df[df["is_valid"] == False][smi_col[0]].tolist()
    else:
        # Standard behavior: return actual indices  
        # FIXED: Use correct condition based on what we consider "invalid"
        if config.invalid_condition == '==False':
            idx_of_invalid_smi = df[df["is_valid"] == False].index.tolist()
        elif config.invalid_condition == '==True':
            idx_of_invalid_smi = df[df["is_valid"] == True].index.tolist()
    
    # Create format data
    format_data = {
        config.format_keys['input']: len(df),
        config.format_keys['invalid']: len(invalid_smi),
        config.format_keys['valid']: len(valid_smi),
    }
    
    return valid_smi, invalid_smi, idx_of_invalid_smi, format_data


def format_return_values(valid_smi: pd.DataFrame, 
                        idx_of_invalid_smi: List,
                        format_data: Dict,
                        get_invalid_smi_idx: bool,
                        return_format_data: bool) -> Any:
    """Format return values based on flags - preserves exact original logic."""
    if get_invalid_smi_idx and return_format_data:
        return valid_smi, idx_of_invalid_smi, format_data
    elif get_invalid_smi_idx and not return_format_data:
        return valid_smi, idx_of_invalid_smi
    elif not get_invalid_smi_idx and return_format_data:
        return valid_smi, format_data
    elif not get_invalid_smi_idx and not return_format_data:
        return valid_smi