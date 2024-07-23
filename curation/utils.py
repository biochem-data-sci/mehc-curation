import pandas as pd
import numpy as np
import os
from parallel_pandas import ParallelPandas
from rdkit import Chem
from .validate import *


current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)


def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def get_report(file_path: str, content: str):
    with open(file_path, 'w') as file:
        file.write(content)


def check_validate_smiles_in_dataframe(smiles: pd.DataFrame,
                                       reports_dir_path: str = None,
                                       print_log: bool = False,
                                       get_report_text_file: bool = False,
                                       get_invalid_smiles: bool = False,
                                       get_isomeric_smiles: bool = False):
    """
    Function to validate smiles against a dataframe
    :param reports_dir_path:
    :param get_report_text_file:
    :param smiles: Input dataframe that contains SMILES strings
    :param print_log: Default is False. Print log information
    :param get_invalid_smiles: Default is False. Get invalid SMILES strings
    :param get_isomeric_smiles: Default is False. Get isomeric SMILES strings
    :return:
    """
    smiles['is_valid'] = smiles['compound'].p_apply(lambda x: is_valid_smiles(x))
    valid_smiles = smiles[smiles['is_valid'] == True]['compound']
    invalid_smiles = smiles[smiles['is_valid'] == False]['compound']

    if get_isomeric_smiles:
        valid_smiles['compound'] = (valid_smiles['compound']
                                    .p_apply(lambda x: Chem.MolFromSmiles(x))
                                    .p_apply(lambda x: Chem.MolToSmiles(x)))
    else:
        valid_smiles['compound'] = (valid_smiles['compound']
                                    .p_apply(lambda x: Chem.MolFromSmiles(x))
                                    .p_apply(lambda x: Chem.MolToSmiles(x, isomericSmiles=False)))

    contents = (f'Number of input SMILES: {len(smiles)}\n'
                f'Number of valid SMILES: {len(valid_smiles)}\n'
                f'Number of invalid SMILES: {len(invalid_smiles)}\n')
    if len(invalid_smiles) > 0:
        contents += f'List of invalid SMILES: \n{invalid_smiles.tolist()}'

    if print_log:
        print(contents)

    if get_report_text_file:
        report_file_path = os.path.join(reports_dir_path, 'validate_smiles_data_report.txt')
        csv_file_path = os.path.join(reports_dir_path, 'valid_smiles.csv')
        valid_smiles.to_csv(csv_file_path, index=False, encoding='utf-8')
        get_report(report_file_path, contents)

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


def remove_duplicates_in_dataframe(smiles: pd.DataFrame,
                                   print_log: bool = True,
                                   show_duplicated_smiles_and_index: bool = True):
    duplicated_smiles = smiles[smiles.duplicated(keep=False)]
    duplicated_smiles_include_idx = (((duplicated_smiles
                                     .groupby(duplicated_smiles.columns.tolist()))
                                     .p_apply(lambda x: tuple(x.index)))
                                     .reset_index(name='index'))

    post_removed_duplicates_smiles = smiles.drop_duplicates()

    if print_log:
        print(f'Number of input SMILES strings: {len(smiles)}')
        print(f'Number of unique SMILES strings: {len(post_removed_duplicates_smiles)}')
        print(f'Number of duplicate SMILES strings: {len(smiles) - len(post_removed_duplicates_smiles)}')

    if show_duplicated_smiles_and_index:
        return duplicated_smiles_include_idx, post_removed_duplicates_smiles
    else:
        return post_removed_duplicates_smiles
