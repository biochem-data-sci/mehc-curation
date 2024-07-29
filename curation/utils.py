import pandas as pd
import numpy as np
import os
from parallel_pandas import ParallelPandas
from rdkit import Chem
from .validate import *

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)


class IsValidSMILES:
    def __init__(self, smiles):
        self.smiles = smiles

    def is_valid(self):
        try:
            mol = Chem.MolFromSmiles(self.smiles)
            return mol is not None
        except Exception as e:
            return False


class GetReport:
    def __init__(self,
                 smiles_df: pd.DataFrame,
                 report_dir_path: str,
                 report_subdir_name: str,
                 report_file_name: str,
                 csv_file_name: str,
                 content: str):
        self.smiles_df = smiles_df
        self.report_dir_path = report_dir_path
        self.report_subdir_name = report_subdir_name
        self.report_file_name = report_file_name
        self.csv_file_name = csv_file_name
        self.content = content

    def create_report_and_csv_files(self):
        validate_smiles_dir = os.path.join(self.report_dir_path, self.report_subdir_name)
        if not os.path.exists(validate_smiles_dir):
            os.makedirs(validate_smiles_dir)

        report_file_path = os.path.join(validate_smiles_dir, self.report_file_name)
        with open(report_file_path, 'w') as report_file:
            report_file.write(self.content)

        csv_file_path = os.path.join(validate_smiles_dir, self.csv_file_name)
        self.smiles_df.to_csv(csv_file_path, index=False, encoding='utf-8')


def get_report(file_path: str, content: str):
    with open(file_path, 'w') as file:
        file.write(content)


def check_valid_smiles_in_dataframe(smiles_df: pd.DataFrame,
                                    reports_dir_path: str = None,
                                    print_logs: bool = False,
                                    get_report_text_file: bool = False,
                                    get_invalid_smiles: bool = False,
                                    get_isomeric_smiles: bool = False):
    """
    Function to validate smiles_df against a dataframe
    :param reports_dir_path:
    :param get_report_text_file:
    :param smiles_df: Input dataframe that contains SMILES strings
    :param print_logs: Default is False. Print log information
    :param get_invalid_smiles: Default is False. Get invalid SMILES strings
    :param get_isomeric_smiles: Default is False. Get isomeric SMILES strings
    :return:
    """
    smiles_df['is_valid'] = smiles_df['compound'].p_apply(lambda x: IsValidSMILES(x).is_valid())
    valid_smiles = smiles_df[smiles_df['is_valid'] == True]
    valid_smiles.drop(columns=['is_valid'], inplace=True)
    invalid_smiles = smiles_df[smiles_df['is_valid'] == False]['compound']

    if get_isomeric_smiles:
        valid_smiles['compound'] = (valid_smiles['compound']
                                    .p_apply(lambda x: Chem.MolFromSmiles(x))
                                    .p_apply(lambda x: Chem.MolToSmiles(x)))
    else:
        valid_smiles['compound'] = (valid_smiles['compound']
                                    .p_apply(lambda x: Chem.MolFromSmiles(x))
                                    .p_apply(lambda x: Chem.MolToSmiles(x, isomericSmiles=False)))

    contents = (f'Number of input SMILES: {len(smiles_df)}\n'
                f'Number of valid SMILES: {len(valid_smiles)}\n'
                f'Number of invalid SMILES: {len(invalid_smiles)}\n')
    if len(invalid_smiles) > 0:
        contents += f'List of invalid SMILES: \n{invalid_smiles.tolist()}'

    if print_logs:
        print(contents)

    if get_report_text_file:
        (GetReport(valid_smiles,
                   report_dir_path=reports_dir_path,
                   report_subdir_name='check_valid_smiles',
                   report_file_name='validate_smiles_data_report.txt',
                   csv_file_name='valid_smiles.csv',
                   content=contents)
         .create_report_and_csv_files())

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


def remove_duplicates_in_dataframe(smiles_df: pd.DataFrame,
                                   check_validity: bool = True,
                                   report_dir_path: str = None,
                                   print_logs: bool = True,
                                   get_report_text_file: bool = False,
                                   show_duplicated_smiles_and_index: bool = True):
    if check_validity:
        smiles_df = check_valid_smiles_in_dataframe(smiles_df)

    duplicated_smiles = smiles_df[smiles_df.duplicated(keep=False)]
    duplicated_smiles_include_idx = (((duplicated_smiles
                                       .groupby(duplicated_smiles.columns.tolist()))
                                      .p_apply(lambda x: tuple(x.index)))
                                     .reset_index(name='index'))

    post_removed_duplicates_smiles = smiles_df.drop_duplicates()

    contents = (f'Number of input SMILES strings: {len(smiles_df)}'
                f'Number of unique SMILES strings: {len(post_removed_duplicates_smiles)}'
                f'Number of duplicate SMILES strings: {len(smiles_df) - len(post_removed_duplicates_smiles)}')

    if print_logs:
        print(contents)

    if show_duplicated_smiles_and_index:
        return duplicated_smiles_include_idx, post_removed_duplicates_smiles
    else:
        return post_removed_duplicates_smiles
