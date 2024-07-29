import pandas as pd
import numpy as np
import os
from parallel_pandas import ParallelPandas
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from .utils import *


class CleaningSalts:
    def __init__(self,
                 smiles,
                 return_difference: bool = False,
                 return_is_null_smiles: bool = False):
        self.smiles = smiles
        self.return_difference = return_difference
        self.return_is_null_smiles = return_is_null_smiles

    def cleaning_salts(self):
        remover = SaltRemover()
        mol = Chem.MolFromSmiles(self.smiles)
        post_mol = remover.StripMol(mol)
        post_smiles = Chem.MolToSmiles(post_mol)
        if self.smiles == post_smiles:
            difference = False
        else:
            difference = True
        if post_smiles == '':
            is_null_smiles = True
        else:
            is_null_smiles = False
        if self.return_difference and self.return_is_null_smiles:
            return post_smiles, difference, is_null_smiles
        elif self.return_difference and not self.return_is_null_smiles:
            return post_smiles, difference
        elif self.return_is_null_smiles and not self.return_difference:
            return post_smiles, is_null_smiles
        else:
            return post_smiles


class Neutralizing:
    def __init__(self, smiles, return_difference: bool = False):
        self.smiles = smiles
        self.return_difference = return_difference

    def neutralizing_salts(self):
        mol = Chem.MolFromSmiles(self.smiles)
        post_mol = neutralize_atoms(mol)
        post_smiles = Chem.MolToSmiles(post_mol)
        if self.smiles == post_smiles:
            difference = False
        else:
            difference = True
        if self.return_difference:
            return post_smiles, difference
        else:
            return post_smiles


def clean_salts_smiles_data(smiles_df: pd.DataFrame,
                            check_validity: bool = True,
                            reports_dir_path: str = None,
                            print_logs: bool = True,
                            get_report_text_files: bool = False,
                            get_difference: bool = False):
    if check_validity:
        smiles_df = check_valid_smiles_in_dataframe(smiles_df)

    salts_cleaned = smiles_df['compound'].p_apply(lambda x: CleaningSalts(x,
                                                                          return_difference=True,
                                                                          return_is_null_smiles=True)
                                                  .cleaning_salts())
    post_salts_clean_smiles_data = salts_cleaned.p_apply(lambda x: x[0])
    differ_after_clean_salt = salts_cleaned.p_apply(lambda x: x[1])
    is_missing_smiles_string = salts_cleaned.p_apply(lambda x: x[2])
    post_drop_missing_smiles_string = (post_salts_clean_smiles_data
                                       .drop(is_missing_smiles_string.index[is_missing_smiles_string == True].tolist(),
                                             axis='index')
                                       .reset_index(drop=True))
    missing_smiles_count = len(post_salts_clean_smiles_data) - len(post_drop_missing_smiles_string)
    post_drop_missing_smiles_string = post_drop_missing_smiles_string.to_frame()

    contents = (f'Pre-cleaned smiles data: {len(smiles_df)}\n'
                f'Number of salts were cleaned: {sum(differ_after_clean_salt)}\n'
                f'Number of substance were missing after cleaning salts: {missing_smiles_count}\n'
                f'Post-cleaned smiles data: {len(post_drop_missing_smiles_string)}\n')
    if sum(differ_after_clean_salt) > 0:
        contents += 'List of salts were cleaned:\n'
        contents += '\n'.join(f'{i + 1}. {smiles}' for i, smiles in enumerate(smiles_df['compound']
                                                                              [differ_after_clean_salt == True]
                                                                              .tolist()))

    if print_logs:
        print(contents)

    if get_report_text_files:
        (GetReport(post_drop_missing_smiles_string,
                   reports_dir_path,
                   report_subdir_name='cleaning_salts',
                   report_file_name='cleaning_salts_report.txt',
                   csv_file_name='post_cleaned_salts.csv',
                   content=contents)
         .create_report_and_csv_files())

    if get_difference:
        return post_drop_missing_smiles_string, differ_after_clean_salt
    else:
        return post_drop_missing_smiles_string


def neutralizing_smiles_data(smiles_df: pd.DataFrame,
                             check_validity: bool = True,
                             reports_dir_path: str = None,
                             print_logs: bool = True,
                             get_report_text_files: bool = False,
                             get_difference: bool = False):
    if check_validity:
        smiles_df = check_valid_smiles_in_dataframe(smiles_df)

    neutralized = smiles_df['compound'].p_apply(lambda x: Neutralizing(x, return_difference=True)
                                                .neutralizing_salts())
    post_neutralized_smiles_data = neutralized.p_apply(lambda x: x[0])
    differ_after_neutralize = neutralized.p_apply(lambda x: x[1])

    contents = (f'Pre-cleaned smiles data: {len(smiles_df)}\n'
                f'Number of salts were cleaned: {sum(differ_after_neutralize)}\n'
                f'Post-cleaned smiles data: {len(post_neutralized_smiles_data)}\n')
    if sum(differ_after_neutralize) > 0:
        contents += 'List of salts were cleaned:\n'
        contents += '\n'.join(f'{i + 1}. {smiles}' for i, smiles in enumerate(post_neutralized_smiles_data
                                                                              [differ_after_neutralize == True]
                                                                              .tolist()))

    post_neutralized_smiles_data = post_neutralized_smiles_data.to_frame()

    if print_logs:
        print(contents)

    if get_report_text_files:
        (GetReport(post_neutralized_smiles_data,
                   reports_dir_path,
                   report_subdir_name='neutralizing_smiles',
                   report_file_name='neutralizing_smiles_report.txt',
                   csv_file_name='post_neutralized_smiles.csv',
                   content=contents)
         .create_report_and_csv_files())

    if get_difference:
        return post_neutralized_smiles_data, differ_after_neutralize
    else:
        return post_neutralized_smiles_data


def clean_salts_and_neutralize_smiles_data(smiles_df: pd.DataFrame,
                                           check_validity: bool = True,
                                           reports_dir_path: str = None,
                                           print_logs: bool = True,
                                           get_report_text_file: bool = False):
    if check_validity:
        smiles_df = check_valid_smiles_in_dataframe(smiles_df)

    post_salts_cleaned_smiles_data, differ_after_cleaning_salt = clean_salts_smiles_data(smiles_df,
                                                                                         check_validity=False,
                                                                                         print_logs=False,
                                                                                         get_difference=True)

    post_neutralized_smiles_data, differ_after_neutralizing = neutralizing_smiles_data(post_salts_cleaned_smiles_data,
                                                                                       check_validity=False,
                                                                                       print_logs=False,
                                                                                       get_difference=True)

    contents = (f'Pre-cleaned smiles data: {len(smiles_df)}\n'
                f'Number of salts were cleaned: {sum(differ_after_cleaning_salt)}\n'
                f'Number of substances were neutralized: {sum(differ_after_neutralizing)}\n'
                f'Post-cleaned smiles data: {len(post_neutralized_smiles_data)}\n')

    if print_logs:
        print(contents)

    if get_report_text_file:
        GetReport(post_neutralized_smiles_data,
                  reports_dir_path,
                  report_subdir_name='cleaning_salts_and_neutralizing_smiles',
                  report_file_name='cleaning_salts_and_neutralizing_smiles_report.txt',
                  csv_file_name='post_cleaned_and_neutralized_smiles.csv',
                  content=contents)

    return post_neutralized_smiles_data
