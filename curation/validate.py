import pandas as pd
from parallel_pandas import ParallelPandas
from rdkit import Chem
from curation.utils import GetReport, RemoveSpecificSMILES

ParallelPandas.initialize(n_cpu=16, split_factor=4, disable_pr_bar=True)


class ValidationStage:
    def __init__(self, smiles_df: pd.DataFrame):
        self.smiles_df = smiles_df

    def check_valid_smiles(self,
                           output_dir_path: str = None,
                           print_logs: bool = True,
                           get_report: bool = False,
                           get_csv: bool = True,
                           get_invalid_smile_indexes: bool = False,
                           get_isomeric_smiles: bool = True,
                           return_contents: bool = False):
        smiles_col = self.smiles_df.columns.tolist()
        self.smiles_df['is_valid'] = self.smiles_df[smiles_col[0]].p_apply(lambda x: RemoveSpecificSMILES(x).is_valid())
        valid_smiles = self.smiles_df[self.smiles_df['is_valid'] == True].copy()
        valid_smiles.drop(columns=['is_valid'], inplace=True)
        invalid_smiles = self.smiles_df[self.smiles_df['is_valid'] == False][smiles_col[0]]
        index_of_invalid_smiles = self.smiles_df[self.smiles_df['is_valid'] == False].index.tolist()

        if get_isomeric_smiles:
            valid_smiles[smiles_col[0]] = valid_smiles[smiles_col[0]].p_apply(
                lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
        else:
            valid_smiles[smiles_col[0]] = valid_smiles[smiles_col[0]].p_apply(
                lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x), isomericSmiles=False))

        contents = (f'Number of input SMILES: {len(self.smiles_df)}\n'
                    f'Number of valid SMILES: {len(valid_smiles)}\n'
                    f'Number of invalid SMILES: {len(invalid_smiles)}\n')
        if len(invalid_smiles) > 0:
            contents += f'List of invalid SMILES indexes: \n{index_of_invalid_smiles}\n'
            contents += '\n'.join(f'{i}. {smiles}' for i, smiles in zip(index_of_invalid_smiles, invalid_smiles))
            contents += '\n'

        if print_logs:
            print(contents)

        if get_csv:
            (GetReport(valid_smiles,
                       output_dir_path=output_dir_path,
                       report_subdir_name='check_valid_smiles')
             .create_csv_file(csv_file_name='valid_smiles.csv'))

        if get_report:
            (GetReport(valid_smiles,
                       output_dir_path=output_dir_path,
                       report_subdir_name='check_valid_smiles')
             .create_report_file(report_file_name='validate_smiles_data_report.txt',
                                 content=contents))

        if get_invalid_smile_indexes and return_contents:
            return valid_smiles, index_of_invalid_smiles, contents
        elif get_invalid_smile_indexes and not return_contents:
            return valid_smiles, index_of_invalid_smiles
        elif not get_invalid_smile_indexes and return_contents:
            return valid_smiles, contents
        elif not get_invalid_smile_indexes and not return_contents:
            return valid_smiles

    def remove_mixtures(self,
                        check_validity: bool = True,
                        output_dir_path: str = None,
                        get_invalid_smile_indexes: bool = False,
                        print_logs: bool = True,
                        get_report: bool = False,
                        get_csv: bool = True,
                        return_contents: bool = False):
        smiles_col = self.smiles_df.columns.tolist()
        if check_validity:
            self.smiles_df = self.check_valid_smiles(get_csv=False,
                                                     print_logs=False)

        self.smiles_df['is_valid'] = self.smiles_df[smiles_col[0]].p_apply(lambda x: RemoveSpecificSMILES(x).is_mixture())
        valid_smiles = self.smiles_df[self.smiles_df['is_valid'] == False].copy()
        valid_smiles.drop(columns=['is_valid'], inplace=True)
        invalid_smiles = self.smiles_df[self.smiles_df['is_valid'] == True][smiles_col[0]]
        index_of_invalid_smiles = self.smiles_df[self.smiles_df['is_valid'] == True].index.tolist()

        contents = (f'Number of input SMILES: {len(self.smiles_df)}\n'
                    f'Number of non-mixture SMILES: {len(valid_smiles)}\n'
                    f'Number of mixture SMILES: {len(invalid_smiles)}\n')
        if len(invalid_smiles) > 0:
            contents += f'List of mixture indexes: \n{index_of_invalid_smiles}\n'
            contents += '\n'.join(f'{i}. {smiles}' for i, smiles in zip(index_of_invalid_smiles, invalid_smiles))
            contents += '\n'

        if print_logs:
            print(contents)

        if get_csv:
            (GetReport(valid_smiles,
                       output_dir_path=output_dir_path,
                       report_subdir_name='remove_mixtures')
             .create_csv_file(csv_file_name='non_mixture_smiles.csv'))

        if get_report:
            (GetReport(valid_smiles,
                       output_dir_path=output_dir_path,
                       report_subdir_name='remove_mixtures')
             .create_report_file(report_file_name='remove_mixtures.txt',
                                 content=contents))

        if get_invalid_smile_indexes and return_contents:
            return valid_smiles, index_of_invalid_smiles, contents
        elif get_invalid_smile_indexes and not return_contents:
            return valid_smiles, index_of_invalid_smiles
        elif return_contents and not get_invalid_smile_indexes:
            return valid_smiles, contents
        elif not get_invalid_smile_indexes and not return_contents:
            return valid_smiles

    def remove_inorganics(self,
                          check_validity: bool = True,
                          output_dir_path: str = None,
                          get_invalid_smile_indexes: bool = False,
                          print_logs: bool = True,
                          get_report: bool = False,
                          get_csv: bool = True,
                          return_contents: bool = False):
        smiles_col = self.smiles_df.columns.tolist()
        if check_validity:
            self.smiles_df = self.check_valid_smiles(get_csv=False,
                                                     print_logs=False)

        self.smiles_df['is_valid'] = self.smiles_df[smiles_col[0]].p_apply(
            lambda x: RemoveSpecificSMILES(x).is_inorganic())
        valid_smiles = self.smiles_df[self.smiles_df['is_valid'] == False].copy()
        valid_smiles.drop(columns=['is_valid'], inplace=True)
        invalid_smiles = self.smiles_df[self.smiles_df['is_valid'] == True][smiles_col[0]]
        index_of_invalid_smiles = self.smiles_df[self.smiles_df['is_valid'] == True].index.tolist()

        contents = (f'Number of input SMILES: {len(self.smiles_df)}\n'
                    f'Number of organic compounds: {len(valid_smiles)}\n'
                    f'Number of inorganic compounds: {len(invalid_smiles)}\n')
        if len(invalid_smiles) > 0:
            contents += f'List of inorganic indexes: \n{index_of_invalid_smiles}\n'
            contents += '\n'.join(f'{i}. {smiles}' for i, smiles in zip(index_of_invalid_smiles, invalid_smiles))
            contents += '\n'

        if print_logs:
            print(contents)

        if get_csv:
            (GetReport(valid_smiles,
                       output_dir_path=output_dir_path,
                       report_subdir_name='remove_inorganic_compounds')
             .create_csv_file(csv_file_name='organic_compounds.csv'))

        if get_report:
            (GetReport(valid_smiles,
                       output_dir_path=output_dir_path,
                       report_subdir_name='remove_inorganic_compounds')
             .create_report_file(report_file_name='remove_inorganic_compounds.txt',
                                 content=contents))

        if get_invalid_smile_indexes and return_contents:
            return valid_smiles, index_of_invalid_smiles, contents
        elif get_invalid_smile_indexes and not return_contents:
            return valid_smiles, index_of_invalid_smiles
        elif return_contents and not get_invalid_smile_indexes:
            return valid_smiles, contents
        elif not get_invalid_smile_indexes and not return_contents:
            return valid_smiles

    def remove_organometallics(self,
                               check_validity: bool = True,
                               output_dir_path: str = None,
                               get_invalid_smile_indexes: bool = False,
                               print_logs: bool = True,
                               get_report: bool = False,
                               get_csv: bool = True,
                               return_contents: bool = False):
        smiles_col = self.smiles_df.columns.tolist()
        if check_validity:
            self.smiles_df = self.check_valid_smiles(get_csv=False,
                                                     print_logs=False)

        self.smiles_df['is_valid'] = self.smiles_df[smiles_col[0]].p_apply(
            lambda x: RemoveSpecificSMILES(x).is_organometallic())
        valid_smiles = self.smiles_df[self.smiles_df['is_valid'] == True].copy()
        valid_smiles.drop(columns=['is_valid'], inplace=True)
        invalid_smiles = self.smiles_df[self.smiles_df['is_valid'] == False][smiles_col[0]]
        index_of_invalid_smiles = self.smiles_df[self.smiles_df['is_valid'] == False].index.tolist()

        contents = (f'Number of input SMILES: {len(self.smiles_df)}\n'
                    f'Number of organic compounds: {len(valid_smiles)}\n'
                    f'Number of organometallic compounds: {len(invalid_smiles)}\n')
        if len(invalid_smiles) > 0:
            contents += f'List of organometallic indexes: \n{index_of_invalid_smiles}\n'
            contents += '\n'.join(f'{i}. {smiles}' for i, smiles in zip(index_of_invalid_smiles, invalid_smiles))
            contents += '\n'

        if print_logs:
            print(contents)

        if get_csv:
            (GetReport(valid_smiles,
                       output_dir_path=output_dir_path,
                       report_subdir_name='remove_organometallic_compounds')
             .create_csv_file(csv_file_name='organic_compounds.csv'))

        if get_report:
            (GetReport(valid_smiles,
                       output_dir_path=output_dir_path,
                       report_subdir_name='remove_organometallic_compounds')
             .create_report_file(report_file_name='remove_organometallic_compounds.txt',
                                 content=contents))

        if get_invalid_smile_indexes and return_contents:
            return valid_smiles, index_of_invalid_smiles, contents
        elif get_invalid_smile_indexes and not return_contents:
            return valid_smiles, index_of_invalid_smiles
        elif return_contents and not get_invalid_smile_indexes:
            return valid_smiles, contents
        elif not get_invalid_smile_indexes and not return_contents:
            return valid_smiles

    def wholly_validate_smiles(self,
                               check_validity: bool = True,
                               output_dir_path: str = None,
                               print_logs: bool = True,
                               get_report: bool = False,
                               get_csv: bool = True):
        invalid_smiles_indexes = []
        validation_step_contents = ''
        smiles_count_before = len(self.smiles_df)

        if check_validity:
            self.smiles_df, invalid_smiles_indexes, validation_step_contents = self.check_valid_smiles(
                print_logs=False,
                get_invalid_smile_indexes=True,
                get_csv=False,
                return_contents=True)

        self.smiles_df, mixture_smiles_indexes, remove_mixtures_contents = self.remove_mixtures(check_validity=False,
                                                                                                print_logs=False,
                                                                                                get_invalid_smile_indexes=True,
                                                                                                get_csv=False,
                                                                                                return_contents=True)
        self.smiles_df, inorganic_smiles_indexes, remove_inorganics_contents = self.remove_inorganics(
            check_validity=False,
            print_logs=False,
            get_invalid_smile_indexes=True,
            get_csv=False,
            return_contents=True)
        self.smiles_df, organometallic_smiles_indexes, remove_organometallics_contents = self.remove_organometallics(
            check_validity=False,
            print_logs=False,
            get_invalid_smile_indexes=True,
            get_csv=False,
            return_contents=True)

        failed_smiles_indexes = (invalid_smiles_indexes + mixture_smiles_indexes
                                 + inorganic_smiles_indexes + organometallic_smiles_indexes)
        number_of_failed_smiles = len(failed_smiles_indexes)

        summary = (f'Number of input SMILES: {smiles_count_before}\n'
                   f'Number of invalid SMILES: {number_of_failed_smiles}\n'
                   f'Number of valid SMILES: {len(self.smiles_df)}\n')
        if number_of_failed_smiles > 0:
            summary += f'List of invalid SMILES indexes: \n{failed_smiles_indexes}\n'

        contents = "\n".join([
            "VALIDATION STEP:",
            validation_step_contents,
            "----------",
            "MIXTURES REMOVING STEP:",
            remove_mixtures_contents,
            "----------",
            "INORGANICS REMOVING STEP:",
            remove_inorganics_contents,
            "----------",
            "ORGANOMETALLIC REMOVING STEP:",
            remove_organometallics_contents,
            "----------",
            "VALIDATE COMPLETE!",
            "SUMMARY:",
            summary
        ])

        if print_logs:
            print(contents)

        if get_csv:
            (GetReport(self.smiles_df,
                       output_dir_path=output_dir_path,
                       report_subdir_name='wholly_validate_smiles')
             .create_csv_file(csv_file_name='output_smiles.csv'))

        if get_report:
            (GetReport(self.smiles_df,
                       output_dir_path=output_dir_path,
                       report_subdir_name='wholly_validate_smiles')
             .create_report_file(report_file_name='wholly_validate_smiles.txt',
                                 content=contents))

        return self.smiles_df
