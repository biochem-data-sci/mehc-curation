import pandas as pd
import numpy as np
from parallel_pandas import ParallelPandas
from rdkit import Chem
from .utils import *


def is_mixtures(smiles):
    if smiles.find('.') != -1:
        if smiles.find('.[') == -1:
            return False
        else:
            if smiles.count('.') != smiles.count('.['):
                return False
    return True


def remove_mixtures(smiles: pd.DataFrame, print_log: bool = False, get_invalid_smiles: bool = False):
    smiles['is_valid'] = smiles['compound'].p_apply(lambda x: is_mixtures(x))
    valid_smiles = smiles[smiles['is_valid'] == True]
    invalid_smiles = smiles[smiles['is_valid'] == False]['compound']
    if get_invalid_smiles:
        return valid_smiles, invalid_smiles
    else:
        return valid_smiles


def is_inorganic_compound(smiles):
    mol = Chem.MolFromSmiles(smiles)
    # Check for the absence of carbon atoms
    has_carbon = any(atom.GetSymbol() == 'C' for atom in mol.GetAtoms())
    if has_carbon is not True:
        return False
    return True


def remove_inorganic_compounds(smiles: pd.DataFrame, print_log: bool = False, get_invalid_smiles: bool = False):
    smiles['is_valid'] = smiles['compound'].p_apply(lambda x: is_inorganic_compound(x))
    valid_smiles = smiles[smiles['is_valid'] == True]
    invalid_smiles = smiles[smiles['is_valid'] == False]['compound']
    if get_invalid_smiles:
        return valid_smiles, invalid_smiles
    else:
        return valid_smiles


def is_organometallic_compound(smiles):
    mol = Chem.MolFromSmiles(smiles)
    metals = ["Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr",
              "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Rb", "Sr", "Y", "Zr",
              "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb",
              "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
              "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os",
              "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Fr", "Ra",
              "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es",
              "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
              "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in metals:
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == "C" or neighbor.GetSymbol() == "c":
                    return False
    return True


def remove_organometallic_compounds(smiles: pd.DataFrame, print_log: bool = False, get_invalid_smiles: bool = False):
    smiles['is_valid'] = smiles['compound'].p_apply(lambda x: is_organometallic_compound(x))
    valid_smiles = smiles[smiles['is_valid'] == True]
    invalid_smiles = smiles[smiles['is_valid'] == False]['compound']
    if get_invalid_smiles:
        return valid_smiles, invalid_smiles
    else:
        return valid_smiles


def remove_unwanted_smiles(smiles: pd.DataFrame, print_log: bool = True):
    smiles_df, unvalidated_smiles_df = check_validate_smiles(smiles, get_invalid_smiles=True)
    smiles_df, mixture_smiles_df = remove_mixtures(smiles_df, get_invalid_smiles=True)
    smiles_df, inorganic_smiles_df = remove_inorganic_compounds(smiles_df, get_invalid_smiles=True)
    smiles_df, organometallic_smiles_df = remove_organometallic_compounds(smiles_df, get_invalid_smiles=True)
    smiles_df.drop(columns=['is_valid'], inplace=True)
    number_of_failed_smiles = len(unvalidated_smiles_df) + len(mixture_smiles_df) + len(inorganic_smiles_df) + len(organometallic_smiles_df)
    if print_log:
        print(f'Number of input smiles for validating: {len(smiles)}')
        print(f'Number of successful smiles: {len(smiles) - number_of_failed_smiles}')
        print(f'Number of failed smiles: {number_of_failed_smiles}')
    return smiles_df