"""Normalization-specific utility functions."""
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional

from ...utils.mol_io import MolRecord, parse, to_smiles

# Tautomer enumerators are reused within a worker process: building one costs
# ~0.7 ms per row, and one chunk runs per worker at a time.
_ENUMERATORS: Dict[int, Any] = {}
_STANDARDIZE = None


def _standardize():
    """Import rdMolStandardize once per process (handles older RDKit layouts)."""
    global _STANDARDIZE
    if _STANDARDIZE is None:
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
        _STANDARDIZE = rdMolStandardize
    return _STANDARDIZE


def tautomer_enumerator(max_tautomers: Optional[int] = None):
    """Enumerator for a given tautomer cap, built once per worker process.

    ``max_tautomers=None`` keeps RDKit's default (1000). A lower cap bounds
    enumeration time, at the cost of a possibly different canonical tautomer
    for molecules that exceed it.
    """
    key = 1000 if max_tautomers is None else int(max_tautomers)
    enumerator = _ENUMERATORS.get(key)
    if enumerator is None:
        enumerator = _standardize().TautomerEnumerator()
        if max_tautomers is not None:
            enumerator.SetMaxTautomers(key)
        _ENUMERATORS[key] = enumerator
    return enumerator


class SMILESNormalizer:
    """
    A class to normalize SMILES strings (tautomer and stereoisomer normalization).
    
    Attributes:
        record (MolRecord): The parsed molecule plus the string it was parsed from.
        smi (str): The SMILES representation of a molecule.
        return_difference (bool): Whether to return difference information.
    """
    
    def __init__(self, smi, return_difference: bool = False,
                 max_tautomers: Optional[int] = None):
        """
        Initialize the SMILESNormalizer class with a SMILES string or a parsed record.
        
        Args:
            smi (str | MolRecord): The SMILES representation of a molecule, or its record.
            return_difference (bool): Whether to return difference information.
            max_tautomers (int): Cap on enumerated tautomers; None keeps RDKit's
                1000. Lower values can change the result for molecules above the cap.
        """
        if max_tautomers is not None and int(max_tautomers) < 1:
            raise ValueError(f"max_tautomers must be a positive integer, got {max_tautomers}")
        self.record = smi if isinstance(smi, MolRecord) else parse(smi)
        self.smi = self.record.smi
        self.return_difference = return_difference
        self.max_tautomers = max_tautomers
    
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
            
            # Enumerator is reused per worker; the cap bounds its enumeration
            enumerator = tautomer_enumerator(self.max_tautomers)
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

