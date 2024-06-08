import pandas as pd
import numpy as np
from parallel_pandas import ParallelPandas
from rdkit import Chem

def smiles_validation(smiles: str, print_log: bool=True, get_unvalidate_smiles: bool=False):
    '''
    Validate smiles against chemical formula.
    :param smiles: Input smiles dataframe.
    :param print_log: Print log if True
    :param get_unvalidate_smiles: Get unvalidated smiles dataframe if True
    :return:
    '''
    validate_smiles, unvalidate_smiles = [], []
    # for i in range(len(smiles)):
    #     if Chem.MolFromSmiles(smiles[i]) != None:
    #         validate_smiles.append(smiles[i])
    #     else:
    #         unvalidate_smiles.append(smiles[i])
    # if get_unvalidate_smiles:
    #     return unvalidate_smiles, validate_smiles
    # else:
    #     return validate_smiles
    if Chem.MolFromSmiles(smiles) is None:
        smiles = int(1)
    return smiles


def count_for_failed_smiles(smiles: pd.Series) -> int:
    failed_smiles_number = 0
    for compound in smiles:
        if compound == 1:
            failed_smiles_number += 1
    return failed_smiles_number

