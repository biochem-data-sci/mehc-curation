"""Cleaning-specific utility functions."""
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from rdkit import Chem
from rdkit.Chem import SaltRemover

from ...utils.mol_io import MolRecord, parse, to_smiles

# Import rdMolStandardize - will be imported at function level to handle different RDKit versions


class SMILESCleaner:
    """
    A class to clean SMILES strings (salt removal, neutralization).
    
    Attributes:
        record (MolRecord): The molecule plus the exact input string.
        smi (str): The SMILES representation of a molecule.
        return_dif (bool): Whether to return difference information.
    """
    
    def __init__(self, smi, return_dif: bool = False):
        """
        Initialize the SMILESCleaner class with a SMILES string or a parsed record.
        
        Args:
            smi (str | MolRecord): The SMILES representation of a molecule.
            return_dif (bool): Whether to return difference information.
        """
        self.record = smi if isinstance(smi, MolRecord) else parse(smi)
        self.smi = self.record.smi
        self.return_dif = return_dif
    
    def clean_salt_record(self, return_is_null_smi: bool = False) -> Tuple[MolRecord, bool, bool]:
        """
        Remove salts, returning the cleaned molecule and the string it serializes to.
        
        Args:
            return_is_null_smi: Whether to return if SMILES is null after cleaning
            
        Returns:
            Tuple of (record, diff, is_missing)
        """
        try:
            mol = self.record.mol
            if mol is None:
                return self.record, False, bool(return_is_null_smi)
            
            # Use RDKit's SaltRemover
            remover = SaltRemover.SaltRemover()
            cleaned_mol = remover.StripMol(mol, dontRemoveEverything=True)
            
            if cleaned_mol is None or cleaned_mol.GetNumAtoms() == 0:
                return self.record, False, bool(return_is_null_smi)
            
            new_smi = to_smiles(cleaned_mol)
            diff = (new_smi != self.record.smi)
            is_missing = (new_smi == "") if return_is_null_smi else False
            return MolRecord(cleaned_mol, new_smi), diff, is_missing
            
        except Exception:
            return self.record, False, bool(return_is_null_smi)
    
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
        record, diff, is_missing = self.clean_salt_record(return_is_null_smi)
        return record.smi, diff, is_missing
    
    def neutralize_salt_record(self, neutralizing_method: str = "boyle") -> Tuple[MolRecord, Optional[int]]:
        """
        Neutralize a molecule, returning the neutralized molecule and its string.
        
        Args:
            neutralizing_method: Method to use - "boyle" or "rdkit"
            
        Returns:
            Tuple of (record, diff); diff is None on error, as before.
        """
        try:
            mol = self.record.mol
            if mol is None:
                return self.record, None
            
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
            
            if neutralizing_method == "boyle":
                # Boyle method: simple charge neutralization (copied: it edits in place)
                new_mol = SMILESCleaner._neutralize_atoms(Chem.Mol(mol))

            elif neutralizing_method == "rdkit":
                # RDKit standardizer
                uncharger = rdMolStandardize.Uncharger()
                new_mol = uncharger.uncharge(mol)
            else:
                raise ValueError(f"Unknown method: {neutralizing_method}. Must be 'boyle' or 'rdkit'")
            
            new_smi = to_smiles(new_mol)
            return MolRecord(new_mol, new_smi), (1 if new_smi != self.record.smi else 0)
            
        except Exception:
            return self.record, None

    def neutralize_salt(self, neutralizing_method: str = "boyle") -> Tuple[str, Optional[int]]:
        """
        Neutralize SMILES string using specified method.
        
        Args:
            method: Method to use - "boyle" or "rdkit"
            
        Returns:
            Tuple of (neutralized_smiles, diff)
            - neutralized_smiles: SMILES after neutralization
            - diff: 1 if neutralized (changed), 0 if not changed, None if error
        """
        record, diff = self.neutralize_salt_record(neutralizing_method)
        return record.smi, diff

    def _neutralize_atoms(mol):
        # Adapted in "No charge - A simple approach to neutralising charged molecules" by Boyle (2019)
        pattern = Chem.MolFromSmarts(
            "[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
        at_matches = mol.GetSubstructMatches(pattern)
        at_matches_list = [y[0] for y in at_matches]
        if len(at_matches_list) > 0:
            for at_idx in at_matches_list:
                atom = mol.GetAtomWithIdx(at_idx)
                chg = atom.GetFormalCharge()
                hcount = atom.GetTotalNumHs()
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(hcount - chg)
                atom.UpdatePropertyCache()
        return mol


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

