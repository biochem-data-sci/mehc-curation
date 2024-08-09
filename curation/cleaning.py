import pandas as pd
import numpy as np
import os
from parallel_pandas import ParallelPandas
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from curation.validate import ValidationStage
from curation.utils import GetReport, neutralize_atoms


class CleaningSMILES:
    def __init__(self, smiles: str, return_difference: bool = False):
        self.smiles = smiles
        self.return_difference = return_difference

    def cleaning_salts(self, return_is_null_smiles: bool = False):
        remover = SaltRemover()
        mol = Chem.MolFromSmiles(self.smiles)
        post_mol = remover.StripMol(mol)
        post_smiles = Chem.MolToSmiles(post_mol)
        difference = self.smiles != post_smiles
        is_null_smiles = post_smiles == ''
        if self.return_difference and return_is_null_smiles:
            return post_smiles, difference, is_null_smiles
        elif self.return_difference and not return_is_null_smiles:
            return post_smiles, difference
        elif return_is_null_smiles and not self.return_difference:
            return post_smiles, is_null_smiles
        else:
            return post_smiles

    def neutralizing_salts(self):
        mol = Chem.MolFromSmiles(self.smiles)
        post_mol = neutralize_atoms(mol)
        post_smiles = Chem.MolToSmiles(post_mol)
        difference = self.smiles != post_smiles
        if self.return_difference:
            return post_smiles, difference
        else:
            return post_smiles


class CleaningStage:
    def __init__(self, smiles_df: pd.DataFrame):
        self.smiles_df = smiles_df

    def clean_salts(self, check_validity: bool = True, output_dir_path: str = None,
                    print_logs: bool = True, get_report: bool = False,
                    get_csv: bool = True, get_difference: bool = False):
        if check_validity:
            self.smiles_df = ValidationStage(self.smiles_df).check_valid_smiles(get_csv=False)

        salts_cleaned = self.smiles_df['compound'].p_apply(lambda x: CleaningSMILES(x, return_difference=True)
                                                           .cleaning_salts(return_is_null_smiles=True))
        post_salts_clean_smiles_data = salts_cleaned.p_apply(lambda x: x[0])
        differ_after_clean_salt = salts_cleaned.p_apply(lambda x: x[1])
        is_missing_smiles_string = salts_cleaned.p_apply(lambda x: x[2])
        index_of_salts = differ_after_clean_salt[differ_after_clean_salt == True].index.tolist()
        salts_in_data = pd.Series(self.smiles_df['compound'].iloc[index_of_salts]).tolist()
        post_drop_missing_smiles_string = (post_salts_clean_smiles_data
                                           .drop(is_missing_smiles_string.index[is_missing_smiles_string == True].tolist(),
                                                 axis='index'))
        missing_smiles_count = len(post_salts_clean_smiles_data) - len(post_drop_missing_smiles_string)
        post_drop_missing_smiles_string = post_drop_missing_smiles_string.to_frame()

        contents = (f'Pre-cleaned smiles data: {len(self.smiles_df)}\n'
                    f'Number of salts were cleaned: {sum(differ_after_clean_salt)}\n'
                    f'Number of substance were missing after cleaning salts: {missing_smiles_count}\n'
                    f'Post-cleaned smiles data: {len(post_drop_missing_smiles_string)}\n')
        if sum(differ_after_clean_salt) > 0:
            contents += f'List of salts were cleaned: {index_of_salts}\n'
            contents += '\n'.join(f'{i}. {smiles}' for i, smiles in zip(index_of_salts, salts_in_data))

        if print_logs:
            print(contents)

        if get_csv:
            (GetReport(post_drop_missing_smiles_string,
                       output_dir_path=output_dir_path,
                       report_subdir_name='cleaning_salts')
             .create_csv_file(csv_file_name='post_cleaned_salts.csv'))

        if get_report:
            (GetReport(post_drop_missing_smiles_string,
                       output_dir_path=output_dir_path,
                       report_subdir_name='cleaning_salts')
             .create_report_file(report_file_name='cleaning_salts_report.txt',
                                 content=contents))

        if get_difference:
            return post_drop_missing_smiles_string, differ_after_clean_salt
        else:
            return post_drop_missing_smiles_string

    def neutralize(self, check_validity: bool = True, output_dir_path: str = None,
                   print_logs: bool = True, get_report: bool = False,
                   get_csv: bool = True, get_difference: bool = False):
        if check_validity:
            self.smiles_df = ValidationStage(self.smiles_df).check_valid_smiles(get_csv=False)

        neutralized = self.smiles_df['compound'].p_apply(lambda x: CleaningSMILES(x, return_difference=True)
                                                         .neutralizing_salts())
        post_neutralized_smiles_data = neutralized.p_apply(lambda x: x[0])
        differ_after_neutralize = neutralized.p_apply(lambda x: x[1])

        contents = (f'Pre-cleaned smiles data: {len(self.smiles_df)}\n'
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

        if get_csv:
            (GetReport(post_neutralized_smiles_data,
                       output_dir_path=output_dir_path,
                       report_subdir_name='neutralizing_smiles')
             .create_csv_file(csv_file_name='post_neutralized_smiles.csv'))

        if get_report:
            (GetReport(post_neutralized_smiles_data,
                       output_dir_path=output_dir_path,
                       report_subdir_name='neutralizing_smiles')
             .create_report_file(report_file_name='neutralizing_smiles_report.txt',
                                 content=contents))

        if get_difference:
            return post_neutralized_smiles_data, differ_after_neutralize
        else:
            return post_neutralized_smiles_data

    def clean_and_neutralize(self, check_validity: bool = True, output_dir_path: str = None,
                             print_logs: bool = True, get_report: bool = False, get_csv: bool = True):
        if check_validity:
            self.smiles_df = ValidationStage(self.smiles_df).check_valid_smiles(get_csv=False)

        post_salts_cleaned_smiles_data, differ_after_cleaning_salt = self.clean_salts(check_validity=False,
                                                                                      print_logs=False,
                                                                                      get_difference=True,
                                                                                      get_csv=False)

        post_neutralized_smiles_data, differ_after_neutralizing = (CleaningStage(post_salts_cleaned_smiles_data)
                                                                   .neutralize(check_validity=False,
                                                                               print_logs=False,
                                                                               get_difference=True,
                                                                               get_csv=False))

        contents = (f'Pre-cleaned SMILES data: {len(self.smiles_df)}\n'
                    f'Number of salts were cleaned: {sum(differ_after_cleaning_salt)}\n'
                    f'Number of substances were neutralized: {sum(differ_after_neutralizing)}\n'
                    f'Post-cleaned SMILES data: {len(post_neutralized_smiles_data)}\n')

        if print_logs:
            print(contents)

        if get_csv:
            (GetReport(post_neutralized_smiles_data,
                       output_dir_path=output_dir_path,
                       report_subdir_name='cleaning_salts_and_neutralizing_smiles')
             .create_csv_file(csv_file_name='post_cleaned_and_neutralized_smiles.csv'))

        if get_report:
            (GetReport(post_neutralized_smiles_data,
                       output_dir_path=output_dir_path,
                       report_subdir_name='cleaning_salts_and_neutralizing_smiles')
             .create_report_file(report_file_name='cleaning_salts_and_neutralizing_smiles_report.txt',
                                 content=contents))

        return post_neutralized_smiles_data
