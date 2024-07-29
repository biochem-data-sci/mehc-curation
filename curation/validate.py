import pandas as pd
import numpy as np
import os
from parallel_pandas import ParallelPandas
from rdkit import Chem

from .utils import *


class RemoveSpecificSMILES:
    def __init__(self, smiles):
        self.smiles = smiles

    def is_mixture(self):
        if self.smiles.find('.') != -1:
            if self.smiles.find('.[') == -1:
                return False
            else:
                if self.smiles.count('.') != self.smiles.count('.['):
                    return False
        return True

    def is_inorganic(self):
        mol = Chem.MolFromSmiles(self.smiles)
        # Check for the absence of carbon atoms
        has_carbon = any(atom.GetSymbol() == 'C' for atom in mol.GetAtoms())
        if has_carbon is not True:
            return False
        return True

    def is_organometallic(self):
        mol = Chem.MolFromSmiles(self.smiles)
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


def remove_mixtures(smiles_df: pd.DataFrame,
                    check_validity: bool = True,
                    reports_dir_path: str = None,
                    get_invalid_smiles: bool = False,
                    print_logs: bool = False,
                    get_report_text_file: bool = False):
    if check_validity:
        smiles_df = check_valid_smiles_in_dataframe(smiles_df)

    smiles_df['is_valid'] = smiles_df['compound'].p_apply(lambda x: RemoveSpecificSMILES(x).is_mixture())
    valid_smiles = smiles_df[smiles_df['is_valid'] == True]
    valid_smiles.drop(columns=['is_valid'], inplace=True)
    invalid_smiles = smiles_df[smiles_df['is_valid'] == False]['compound']

    contents = (f'Number of input SMILES: {len(smiles_df)}\n'
                f'Number of non-mixture SMILES: {len(valid_smiles)}\n'
                f'Number of mixture SMILES: {len(invalid_smiles)}\n')
    if len(invalid_smiles) > 0:
        contents += f'List of mixture SMILES: \n'
        contents += '\n'.join(f"{i + 1}. {smiles}" for i, smiles in enumerate(invalid_smiles.tolist()))

    if print_logs:
        print(contents)

    if get_report_text_file:
        (GetReport(valid_smiles,
                   reports_dir_path,
                   report_subdir_name='remove_mixtures',
                   report_file_name='remove_mixtures.txt',
                   csv_file_name='non_mixture_smiles.csv',
                   content=contents)
         .create_report_and_csv_files())

    if get_invalid_smiles:
        return valid_smiles, invalid_smiles
    else:
        return valid_smiles


def remove_inorganic_compounds(smiles_df: pd.DataFrame,
                               check_validity: bool = True,
                               reports_dir_path: str = None,
                               get_invalid_smiles: bool = False,
                               print_logs: bool = False,
                               get_report_text_file: bool = False):
    if check_validity:
        smiles_df = check_valid_smiles_in_dataframe(smiles_df)

    smiles_df['is_valid'] = smiles_df['compound'].p_apply(lambda x: RemoveSpecificSMILES(x).is_inorganic())
    valid_smiles = smiles_df[smiles_df['is_valid'] == True]
    valid_smiles.drop(columns=['is_valid'], inplace=True)
    invalid_smiles = smiles_df[smiles_df['is_valid'] == False]['compound']

    contents = (f'Number of input SMILES: {len(smiles_df)}\n'
                f'Number of organic compounds: {len(valid_smiles)}\n'
                f'Number of inorganic compounds: {len(invalid_smiles)}\n')
    if len(invalid_smiles) > 0:
        contents += f'List of inorganic compounds: \n'
        contents += '\n'.join(f"{i + 1}. {smiles}" for i, smiles in enumerate(invalid_smiles.tolist()))

    if print_logs:
        print(contents)

    if get_report_text_file:
        (GetReport(valid_smiles,
                   reports_dir_path,
                   report_subdir_name='remove_inorganic_compounds',
                   report_file_name='remove_inorganic_compounds.txt',
                   csv_file_name='organic_compounds.csv',
                   content=contents)
         .create_report_and_csv_files())

    if get_invalid_smiles:
        return valid_smiles, invalid_smiles
    else:
        return valid_smiles


def remove_organometallic_compounds(smiles_df: pd.DataFrame,
                                    check_validity: bool = True,
                                    reports_dir_path: str = None,
                                    get_invalid_smiles: bool = False,
                                    print_logs: bool = False,
                                    get_report_text_file: bool = False):
    if check_validity:
        smiles_df = check_valid_smiles_in_dataframe(smiles_df)

    smiles_df['is_valid'] = smiles_df['compound'].p_apply(lambda x: RemoveSpecificSMILES(x).is_organometallic())
    valid_smiles = smiles_df[smiles_df['is_valid'] == True]
    valid_smiles.drop(columns=['is_valid'], inplace=True)
    invalid_smiles = smiles_df[smiles_df['is_valid'] == False]['compound']

    contents = (f'Number of input SMILES: {len(smiles_df)}\n'
                f'Number of organic compounds: {len(valid_smiles)}\n'
                f'Number of organometallic compounds: {len(invalid_smiles)}\n')
    if len(invalid_smiles) > 0:
        contents += f'List of organometallic compounds: \n'
        contents += '\n'.join(f"{i + 1}. {smiles}" for i, smiles in enumerate(invalid_smiles.tolist()))

    if print_logs:
        print(contents)

    if get_report_text_file:
        (GetReport(valid_smiles,
                   reports_dir_path,
                   report_subdir_name='remove_organometallic_compounds',
                   report_file_name='remove_organometallic_compounds.txt',
                   csv_file_name='organic_compounds.csv',
                   content=contents)
         .create_report_and_csv_files())

    if get_invalid_smiles:
        return valid_smiles, invalid_smiles
    else:
        return valid_smiles


def completely_validate_smiles(smiles_df: pd.DataFrame,
                               check_validity: bool = True,
                               reports_dir_path: str = None,
                               print_logs: bool = True,
                               get_report_text_file: bool = False):
    invalid_smiles_df = pd.DataFrame()
    if check_validity:
        smiles_df, invalid_smiles_df = check_valid_smiles_in_dataframe(smiles_df, get_invalid_smiles=True)

    smiles_df_after_removing_mixture, mixture_smiles_df = remove_mixtures(smiles_df,
                                                                          check_validity=False,
                                                                          get_invalid_smiles=True)
    smiles_df_after_removing_inorganic_compounds, inorganic_smiles_df = remove_inorganic_compounds(
        smiles_df_after_removing_mixture, check_validity=False, get_invalid_smiles=True)
    smiles_df_after_removing_organometallic_compounds, organometallic_smiles_df = remove_organometallic_compounds(
        smiles_df_after_removing_inorganic_compounds, check_validity=False, get_invalid_smiles=True)
    number_of_failed_smiles = len(invalid_smiles_df) + len(mixture_smiles_df) + len(inorganic_smiles_df) + len(
        organometallic_smiles_df)

    contents = (f'Number of input SMILES: {len(smiles_df)}\n'
                f'Number of invalid SMILES: {number_of_failed_smiles}\n'
                f'Number of valid SMILES: {len(smiles_df_after_removing_organometallic_compounds)}\n')

    if print_logs:
        print(contents)

    if get_report_text_file:
        (GetReport(smiles_df_after_removing_organometallic_compounds,
                   reports_dir_path,
                   report_subdir_name='completely_validate_smiles',
                   report_file_name='completely_validate_smiles.txt',
                   csv_file_name='output_smiles.csv',
                   content=contents)
         .create_report_and_csv_files())

    return smiles_df_after_removing_organometallic_compounds
