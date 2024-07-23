import pandas as pd
import numpy as np
import os
from parallel_pandas import ParallelPandas
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from .utils import *


def clean_salts(smiles,
                return_difference: bool = False):
    remover = SaltRemover()
    mol = Chem.MolFromSmiles(smiles)
    post_mol = remover.StripMol(mol)
    post_smiles = Chem.MolToSmiles(post_mol)
    if smiles == post_smiles:
        difference = False
    else:
        difference = True
    if return_difference:
        return post_smiles, difference
    else:
        return post_smiles


def neutralize(smiles,
               return_difference: bool = False):
    mol = Chem.MolFromSmiles(smiles)
    post_mol = neutralize_atoms(mol)
    post_smiles = Chem.MolToSmiles(post_mol)
    if smiles == post_smiles:
        difference = False
    else:
        difference = True
    if return_difference:
        return post_smiles, difference
    else:
        return post_smiles


def clean_salts_and_neutralize_smiles_data(smiles: pd.DataFrame,
                                           print_logs: bool = True):
    salts_cleaned = smiles['compound'].p_apply(lambda x: clean_salts(x, return_difference=True))
    post_salts_clean_smiles_data = salts_cleaned.p_apply(lambda x: x[0])
    differ_after_clean_salt = salts_cleaned.p_apply(lambda x: x[1])

    neutralized = post_salts_clean_smiles_data.p_apply(lambda x: neutralize(x, return_difference=True))
    post_neutralized_smiles_data = pd.DataFrame(neutralized.p_apply(lambda x: x[0]))
    differ_after_neutralize = neutralized.p_apply(lambda x: x[1])

    if print_logs:
        print(f'Pre-cleaned smiles data: {len(smiles)}')
        print(f'Number of salts were cleaned: {sum(differ_after_clean_salt)}')
        print(f'Number of substances were neutralized: {sum(differ_after_neutralize)}')
        print(f'Post-cleaned smiles data: {len(post_neutralized_smiles_data)}')
    return post_neutralized_smiles_data
