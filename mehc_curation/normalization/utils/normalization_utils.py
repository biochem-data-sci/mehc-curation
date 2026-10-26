"""Normalization-specific utility functions."""
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional

from ...utils.mol_io import MolRecord, parse, to_smiles

# Import rdMolStandardize - will be imported at function level to handle different RDKit versions


class SMILESNormalizer:
    """
    A class to normalize SMILES strings (tautomer and stereoisomer normalization).
    
    Attributes:
        record (MolRecord): The parsed molecule plus the string it was parsed from.
        smi (str): The SMILES representation of a molecule.
        return_difference (bool): Whether to return difference information.
    """
    
    def __init__(self, smi, return_difference: bool = False):
        """
        Initialize the SMILESNormalizer class with a SMILES string or a parsed record.
        
        Args:
            smi (str | MolRecord): The SMILES representation of a molecule, or its record.
            return_difference (bool): Whether to return difference information.
        """
        self.record = smi if isinstance(smi, MolRecord) else parse(smi)
        self.smi = self.record.smi
        self.return_difference = return_difference
    
    def normalize_tautomer_record(self) -> Tuple[MolRecord, bool]:
        """
        Normalize tautomers, keeping the parsed molecule with the new SMILES.
        
        Returns:
            Tuple of (record, diff)
            - record: MolRecord holding the normalized molecule and SMILES
            - diff: True if normalized (changed), False otherwise
        """
        try:
            mol = self.record.mol
            if mol is None:
                return self.record, False
            
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
                return self.record, False
            
            normalized_smi = to_smiles(normalized_mol)
            diff = (normalized_smi != self.record.smi)
            
            return MolRecord(normalized_mol, normalized_smi), diff
            
        except Exception:
            return self.record, False
    
    def normalize_tautomer(self) -> Tuple[str, bool]:
        """
        Normalize tautomers in SMILES string.
        
        Returns:
            Tuple of (normalized_smiles, diff)
            - normalized_smiles: SMILES after tautomer normalization
            - diff: True if normalized (changed), False otherwise
        """
        record, diff = self.normalize_tautomer_record()
        return record.smi, diff
    
    def normalize_stereoisomer_record(self) -> Tuple[MolRecord, bool]:
        """
        Remove stereochemistry, keeping a record whose molecule is also stereo-free.
        
        Returns:
            Tuple of (record, diff)
            - record: MolRecord holding the normalized molecule and SMILES
            - diff: True if normalized (changed), False otherwise
        """
        try:
            mol = self.record.mol
            if mol is None:
                return self.record, False
            
            # Remove stereochemistry by converting to SMILES without isomeric information
            normalized_smi = to_smiles(mol, isomeric=False)
            diff = (normalized_smi != self.record.smi)
            
            # Re-parse so the record's molecule matches the non-isomeric string.
            return MolRecord(parse(normalized_smi).mol, normalized_smi), diff
            
        except Exception:
            return self.record, False
    
    def normalize_stereoisomer(self) -> Tuple[str, bool]:
        """
        Normalize stereoisomers in SMILES string (remove stereochemistry).
        
        Returns:
            Tuple of (normalized_smiles, diff)
            - normalized_smiles: SMILES after stereoisomer normalization
            - diff: True if normalized (changed), False otherwise
        """
        record, diff = self.normalize_stereoisomer_record()
        return record.smi, diff


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

