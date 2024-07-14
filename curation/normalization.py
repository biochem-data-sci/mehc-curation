import pandas as pd
import numpy as np
from parallel_pandas import ParallelPandas
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.EnumerateStereoisomers import *


def normalize_tautomer(smiles):
    """
    Normalize tautomers for a given SMILES string.

    Parameters:
        smiles (str): SMILES string of the molecule.

    Returns:
        str: Canonical SMILES string of the normalized tautomer.
    """
    # Create a molecule from the SMILES string
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # Normalize tautomers
    enumerator = rdMolStandardize.TautomerEnumerator()
    canonical_mol = enumerator.Canonicalize(mol)

    # Convert the canonical molecule back to a SMILES string
    canonical_smiles = Chem.MolToSmiles(canonical_mol)
    return canonical_smiles