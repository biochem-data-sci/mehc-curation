import pandas as pd
import numpy as np
import os
from parallel_pandas import ParallelPandas
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from .utils import *


class CleaningSalts:
    def __init__(self, smiles, return_difference: bool = False):
        self.smiles = smiles
        self.return_difference = return_difference

    def cleaning_salts(self):
        remover = SaltRemover()
        mol = Chem.MolFromSmiles(self.smiles)
        post_mol = remover.StripMol(mol)
        post_smiles = Chem.MolToSmiles(post_mol)
        if self.smiles == post_smiles:
            difference = False
        else:
            difference = True
        if self.return_difference:
            return post_smiles, difference
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


# def clean_salts_smiles_data(smiles_df: pd.DataFrame,
#                             check_validity: bool = True,
#                             report_dir_path: str = None,
#                             print_log: bool = True,
#                             get_report_text_files: bool = False):
#     if check_validity:
#         smiles_df = clean_salts_smiles_data(smiles_df)
#
#     salts_cleaned = smiles_df['compound'].p_apply(lambda x: CleaningSalts(x, return_difference=True)
#                                                   .cleaning_salts())
#     post_salts_clean_smiles_data = salts_cleaned.p_apply(lambda x: x[0])
#     differ_after_clean_salt = salts_cleaned.p_apply(lambda x: x[1])


def clean_salts_and_neutralize_smiles_data(smiles_df: pd.DataFrame,
                                           check_validity: bool = True,
                                           reports_dir_path: str = None,
                                           print_logs: bool = True,
                                           get_report_text_file: bool = False):
    if check_validity:
        smiles_df = check_valid_smiles_in_dataframe(smiles_df)

    salts_cleaned = smiles_df['compound'].p_apply(lambda x: CleaningSalts(x, return_difference=True)
                                                  .cleaning_salts())
    post_salts_clean_smiles_data = salts_cleaned.p_apply(lambda x: x[0])
    differ_after_clean_salt = salts_cleaned.p_apply(lambda x: x[1])

    neutralized = post_salts_clean_smiles_data.p_apply(lambda x: Neutralizing(x, return_difference=True)
                                                       .neutralizing_salts())
    post_neutralized_smiles_data = pd.DataFrame(neutralized.p_apply(lambda x: x[0]))
    differ_after_neutralize = neutralized.p_apply(lambda x: x[1])

    contents = (f'Pre-cleaned smiles data: {len(smiles_df)}\n'
                f'Number of salts were cleaned: {sum(differ_after_clean_salt)}'
                f'Number of substances were neutralized: {sum(differ_after_neutralize)}'
                f'Post-cleaned smiles data: {len(post_neutralized_smiles_data)}')

    if print_logs:
        print(contents)

    if get_report_text_file:
        report_file_path = os.path.join(reports_dir_path, 'cleaning_salts_and_neutralizing.txt')
        csv_file_path = os.path.join(reports_dir_path, 'output_smiles.csv')
        post_neutralized_smiles_data.to_csv(csv_file_path, index=False, encoding='utf-8')
        get_report(report_file_path, contents)

    return post_neutralized_smiles_data
