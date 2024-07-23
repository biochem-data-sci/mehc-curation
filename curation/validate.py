import pandas as pd
import numpy as np
import os
from parallel_pandas import ParallelPandas
from rdkit import Chem
from .utils import *


def is_mixtures(smiles: str) -> bool:
    if smiles.find('.') != -1:
        if smiles.find('.[') == -1:
            return False
        else:
            if smiles.count('.') != smiles.count('.['):
                return False
    return True


def remove_mixtures(smiles: pd.DataFrame,
                    reports_dir_path: str = None,
                    get_invalid_smiles: bool = False,
                    print_log: bool = False,
                    get_report_text_file: bool = False):
    smiles['is_valid'] = smiles['compound'].p_apply(lambda x: is_mixtures(x))
    valid_smiles = smiles[smiles['is_valid'] == True]['compound']
    invalid_smiles = smiles[smiles['is_valid'] == False]['compound']

    contents = (f'Number of input SMILES: {len(smiles)}\n'
                f'Number of non-mixture SMILES: {len(valid_smiles)}\n'
                f'Number of mixture SMILES: {len(invalid_smiles)}\n')
    if len(invalid_smiles) > 0:
        contents += f'List of mixture SMILES: \n{invalid_smiles.tolist()}'

    if print_log:
        print(contents)

    if get_report_text_file:
        report_file_path = os.path.join(reports_dir_path, 'remove_mixture_smiles_report.txt')
        csv_file_path = os.path.join(reports_dir_path, 'non_mixture_smiles.csv')
        valid_smiles.to_csv(csv_file_path, index=False, encoding='utf-8')
        get_report(report_file_path, contents)

    if get_invalid_smiles:
        return valid_smiles, invalid_smiles
    else:
        return valid_smiles


def is_inorganic_compound(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    # Check for the absence of carbon atoms
    has_carbon = any(atom.GetSymbol() == 'C' for atom in mol.GetAtoms())
    if has_carbon is not True:
        return False
    return True


def remove_inorganic_compounds(smiles: pd.DataFrame,
                               reports_dir_path: str = None,
                               get_invalid_smiles: bool = False,
                               print_log: bool = False,
                               get_report_text_file: bool = False):
    smiles['is_valid'] = smiles['compound'].p_apply(lambda x: is_inorganic_compound(x))
    valid_smiles = smiles[smiles['is_valid'] == True]
    invalid_smiles = smiles[smiles['is_valid'] == False]['compound']

    contents = (f'Number of input SMILES: {len(smiles)}\n'
                f'Number of organic compounds: {len(valid_smiles)}\n'
                f'Number of inorganic compounds: {len(invalid_smiles)}\n')
    if len(invalid_smiles) > 0:
        contents += f'List of inorganic compounds: \n{invalid_smiles.tolist()}'

    if print_log:
        print(contents)

    if get_report_text_file:
        report_file_path = os.path.join(reports_dir_path, 'remove_inorganic_compounds_report.txt')
        csv_file_path = os.path.join(reports_dir_path, 'organic_smiles.csv')
        valid_smiles.to_csv(csv_file_path, index=False, encoding='utf-8')
        get_report(report_file_path, contents)

    if get_invalid_smiles:
        return valid_smiles, invalid_smiles
    else:
        return valid_smiles


def is_organometallic_compound(smiles: str) -> bool:
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


def remove_organometallic_compounds(smiles: pd.DataFrame,
                                    reports_dir_path: str = None,
                                    get_invalid_smiles: bool = False,
                                    print_log: bool = False,
                                    get_report_text_file: bool = False):
    smiles['is_valid'] = smiles['compound'].p_apply(lambda x: is_organometallic_compound(x))
    valid_smiles = smiles[smiles['is_valid'] == True]
    invalid_smiles = smiles[smiles['is_valid'] == False]['compound']

    contents = (f'Number of input SMILES: {len(smiles)}\n'
                f'Number of organic compounds: {len(valid_smiles)}\n'
                f'Number of organometallic compounds: {len(invalid_smiles)}\n')
    if len(invalid_smiles) > 0:
        contents += f'List of organometallic compounds: \n{invalid_smiles.tolist()}'

    if print_log:
        print(contents)

    if get_report_text_file:
        report_file_path = os.path.join(reports_dir_path, 'remove_organometallic_compounds_report.txt')
        csv_file_path = os.path.join(reports_dir_path, 'organic_smiles.csv')
        valid_smiles.to_csv(csv_file_path, index=False, encoding='utf-8')
        get_report(report_file_path, contents)

    if get_invalid_smiles:
        return valid_smiles, invalid_smiles
    else:
        return valid_smiles


def completely_validate_smiles(smiles: pd.DataFrame,
                               reports_dir_path: str = None,
                               print_log: bool = True,
                               get_report_text_file: bool = False):
    smiles_df_after_validating, invalid_smiles_df = check_validate_smiles_in_dataframe(smiles, get_invalid_smiles=True)
    smiles_df_after_removing_mixture, mixture_smiles_df = remove_mixtures(smiles_df_after_validating, get_invalid_smiles=True)
    smiles_df_after_removing_inorganic_compounds, inorganic_smiles_df = remove_inorganic_compounds(smiles_df_after_removing_mixture, get_invalid_smiles=True)
    smiles_df_after_removing_organometallic_compounds, organometallic_smiles_df = remove_organometallic_compounds(smiles_df_after_removing_inorganic_compounds, get_invalid_smiles=True)
    smiles_df_after_removing_organometallic_compounds.drop(columns=['is_valid'], inplace=True)
    number_of_failed_smiles = len(invalid_smiles_df) + len(mixture_smiles_df) + len(inorganic_smiles_df) + len(organometallic_smiles_df)

    contents = (f'Number of input SMILES: {len(smiles)}\n'
                f'Number of invalid SMILES: {number_of_failed_smiles}\n'
                f'Number of valid SMILES: {len(smiles_df_after_removing_organometallic_compounds)}\n')

    if print_log:
        print(contents)

    if get_report_text_file:
        report_file_path = os.path.join(reports_dir_path, 'remove_unwanted_smiles_report.txt')
        csv_file_path = os.path.join(reports_dir_path, 'output_smiles.csv')
        smiles_df_after_removing_organometallic_compounds.to_csv(csv_file_path, index=False, encoding='utf-8')
        get_report(report_file_path, contents)

    return smiles_df_after_removing_organometallic_compounds
