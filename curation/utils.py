import pandas as pd
import numpy as np
from parallel_pandas import ParallelPandas
from rdkit import Chem
from .validate import *


def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def check_validate_smiles(smiles: pd.DataFrame,
                          print_log: bool = False,
                          get_invalid_smiles: bool = False,
                          get_isomeric_smiles: bool = False):

    smiles['is_valid'] = smiles['compound'].p_apply(lambda x: is_valid_smiles(x))
    valid_smiles = smiles[smiles['is_valid'] == True]
    invalid_smiles = smiles[smiles['is_valid'] == False]['compound']

    if get_isomeric_smiles:
        valid_smiles['compound'] = (valid_smiles['compound']
                                    .p_apply(lambda x: Chem.MolFromSmiles(x))
                                    .p_apply(lambda x: Chem.MolToSmiles(x)))
    else:
        valid_smiles['compound'] = (valid_smiles['compound']
                                    .p_apply(lambda x: Chem.MolFromSmiles(x))
                                    .p_apply(lambda x: Chem.MolToSmiles(x, isomericSmiles=True)))

    if get_invalid_smiles:
        return valid_smiles, invalid_smiles
    else:
        return valid_smiles


def neutralize_atoms(mol):
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
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
