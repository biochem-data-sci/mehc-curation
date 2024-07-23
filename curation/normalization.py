import pandas as pd
import numpy as np
import os
from parallel_pandas import ParallelPandas
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.EnumerateStereoisomers import *


def normalize_tautomer(smiles,
                       return_difference: bool = False):
    """
    Normalize tautomers for a given SMILES string.

    Parameters:
        smiles (str): SMILES string of the molecule.
        return_difference (bool, optional): Whether to return the difference between the two molecules. Defaults to False.

    Returns:
        str: Canonical SMILES string of the normalized tautomer.
    """
    # Create a molecule from the SMILES string
    mol = Chem.MolFromSmiles(smiles)

    # Normalize tautomers
    enumerator = rdMolStandardize.TautomerEnumerator()
    canonical_mol = enumerator.Canonicalize(mol)

    # Convert the canonical molecule back to a SMILES string
    canonical_smiles = Chem.MolToSmiles(canonical_mol)

    # Check if the SMILES string is tautomerized.
    if smiles == canonical_smiles:
        difference = False
    else:
        difference = True

    if return_difference:
        return canonical_smiles, difference
    else:
        return canonical_smiles


def normalize_tautomer_in_dataframe(smiles: pd.DataFrame,
                                    print_log: bool = True):
    """
    Normalize tautomers for a given SMILES dataframe.
    :param smiles:
    :param print_log:
    :return:
    """
    post_tautomer_normalized = smiles['compound'].p_apply(lambda x: normalize_tautomer(x, return_difference=True))
    post_tautomer_normalized_smiles_dataframe = pd.DataFrame(post_tautomer_normalized.p_apply(lambda x: x[0]))
    difference_after_tautomer_normalized = post_tautomer_normalized.p_apply(lambda x: x[1])

    if print_log:
        print(f'Number of SMILES strings before tautomer normalizing: {len(smiles)}')
        print(f'Number of SMILES tautomers are normalizing: {sum(difference_after_tautomer_normalized)}')
        print(f'Number of SMILES strings after tautomer normalizing: {len(post_tautomer_normalized_smiles_dataframe)}')

    return post_tautomer_normalized_smiles_dataframe
