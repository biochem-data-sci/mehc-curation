import pandas as pd
import textwrap
from parallel_pandas import ParallelPandas
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.EnumerateStereoisomers import *
from curation.utils import GetReport
from curation.validate import ValidationStage

ParallelPandas.initialize(n_cpu=16, split_factor=4, disable_pr_bar=True)


class NormalizeSMILES:
    def __init__(self, smiles: str, return_difference: bool = False):
        self.smiles = smiles
        self.return_difference = return_difference

    def normalize_tautomer(self):
        # Create a molecule from the SMILES string
        mol = Chem.MolFromSmiles(self.smiles)

        try:
            # Normalize tautomers
            canonical_mol = rdMolStandardize.CanonicalTautomer(mol)

            # Convert the canonical molecule back to a SMILES string
            canonical_smiles = Chem.MolToSmiles(canonical_mol)
        except RuntimeError:
            canonical_smiles = self.smiles

        # Check if the SMILES string is tautomerized.
        difference = self.smiles != canonical_smiles

        if self.return_difference:
            return canonical_smiles, difference
        else:
            return canonical_smiles

    def normalize_stereoisomer(self):
        # Create a molecule from the SMILES string
        mol = Chem.MolFromSmiles(self.smiles)

        # Normalize stereochemistry by converting to isomeric SMILES
        stereo_normalized_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)

        # Check if the SMILES string has been altered
        difference = self.smiles != stereo_normalized_smiles

        if self.return_difference:
            return stereo_normalized_smiles, difference
        else:
            return stereo_normalized_smiles


class NormalizingStage:
    def __init__(self, smiles_df: pd.DataFrame):
        self.smiles_df = smiles_df

    def normalize_tautomer(self, check_validity: bool = True, output_dir_path: str = None,
                           print_logs: bool = True, get_report: bool = False, get_csv: bool = True,
                           get_difference: bool = False, return_contents: bool = False):
        """
        Normalize tautomers for the given SMILES dataframe.
        """
        if check_validity:
            self.smiles_df = ValidationStage(self.smiles_df).check_valid_smiles(get_csv=False,
                                                                                get_isomeric_smiles=True,
                                                                                print_logs=False)

        post_tautomer_normalized = self.smiles_df['compound'].p_apply(
            lambda x: NormalizeSMILES(x, return_difference=True)
            .normalize_tautomer())
        post_tautomer_normalized_smiles_dataframe = post_tautomer_normalized.p_apply(lambda x: x[0])
        difference_after_tautomer_normalized = post_tautomer_normalized.p_apply(lambda x: x[1])

        contents = (f'Number of SMILES strings before tautomer normalizing: {len(self.smiles_df)}\n'
                    f'Number of SMILES tautomers normalized: {sum(difference_after_tautomer_normalized)}\n'
                    f'Number of SMILES strings after tautomer normalizing: {len(post_tautomer_normalized_smiles_dataframe)}\n')

        post_tautomer_normalized_smiles_dataframe = post_tautomer_normalized_smiles_dataframe.to_frame()

        if print_logs:
            print(contents)

        if get_csv:
            (GetReport(post_tautomer_normalized_smiles_dataframe,
                       output_dir_path=output_dir_path,
                       report_subdir_name='normalize_tautomers')
             .create_csv_file(csv_file_name='post_tautomers_normalized.csv'))

        if get_report:
            (GetReport(post_tautomer_normalized_smiles_dataframe,
                       output_dir_path=output_dir_path,
                       report_subdir_name='normalize_tautomers')
             .create_report_file(report_file_name='normalize_tautomers_report.txt',
                                 content=contents))
        if get_difference and return_contents:
            return post_tautomer_normalized_smiles_dataframe, difference_after_tautomer_normalized, contents
        elif get_difference and not return_contents:
            return post_tautomer_normalized_smiles_dataframe, difference_after_tautomer_normalized
        elif return_contents and not get_difference:
            return post_tautomer_normalized_smiles_dataframe, contents
        else:
            return post_tautomer_normalized_smiles_dataframe

    def normalize_stereoisomer(self, check_validity: bool = True, output_dir_path: str = None,
                               print_logs: bool = True, get_report: bool = False, get_csv: bool = True,
                               get_difference: bool = False, return_contents: bool = False):
        """
        Normalize stereoisomers for the given SMILES dataframe.
        """
        if check_validity:
            self.smiles_df = ValidationStage(self.smiles_df).check_valid_smiles(get_csv=False,
                                                                                get_isomeric_smiles=True,
                                                                                print_logs=False)

        post_stereoisomer_normalized = self.smiles_df['compound'].p_apply(
            lambda x: NormalizeSMILES(x, return_difference=True)
            .normalize_stereoisomer())
        post_stereoisomer_normalized_smiles_dataframe = post_stereoisomer_normalized.p_apply(lambda x: x[0])
        difference_after_stereoisomer_normalized = post_stereoisomer_normalized.p_apply(lambda x: x[1])

        contents = (f'Number of SMILES strings before stereoisomer normalizing: {len(self.smiles_df)}\n'
                    f'Number of SMILES stereoisomers normalized: {sum(difference_after_stereoisomer_normalized)}\n'
                    f'Number of SMILES strings after stereoisomer normalizing: {len(post_stereoisomer_normalized_smiles_dataframe)}\n')

        post_stereoisomer_normalized_smiles_dataframe = post_stereoisomer_normalized_smiles_dataframe.to_frame()

        if print_logs:
            print(contents)

        if get_csv:
            (GetReport(post_stereoisomer_normalized_smiles_dataframe,
                       output_dir_path=output_dir_path,
                       report_subdir_name='normalize_stereoisomer')
             .create_csv_file(csv_file_name='post_stereoisomer_normalized.csv'))

        if get_report:
            (GetReport(post_stereoisomer_normalized_smiles_dataframe,
                       output_dir_path=output_dir_path,
                       report_subdir_name='normalize_stereoisomer')
             .create_report_file(report_file_name='normalize_stereoisomer_report.txt',
                                 content=contents))
        if get_difference and return_contents:
            return post_stereoisomer_normalized_smiles_dataframe, difference_after_stereoisomer_normalized, contents
        elif get_difference and not return_contents:
            return post_stereoisomer_normalized_smiles_dataframe, difference_after_stereoisomer_normalized
        elif return_contents and not get_difference:
            return post_stereoisomer_normalized_smiles_dataframe, contents
        else:
            return post_stereoisomer_normalized_smiles_dataframe

    def normalize_tautomer_and_stereoisomer(self, check_validity: bool = True, output_dir_path: str = None,
                                            print_logs: bool = True, get_report: bool = False, get_csv: bool = True):
        """
            Normalize both tautomers and stereoisomers for the given SMILES dataframe.
        """
        # Validation step
        validation_step_contents = ''
        if check_validity:
            self.smiles_df, validation_step_contents = ValidationStage(self.smiles_df).check_valid_smiles(
                get_csv=False, print_logs=False, get_isomeric_smiles=True, return_contents=True)

        # Normalize stereoisomers
        post_stereoisomer_normalized_smiles_data, differ_after_stereoisomer_normalizing, stereoisomer_step_contents = (
            NormalizingStage(self.smiles_df)
            .normalize_stereoisomer(check_validity=False,  # Already validated
                                    print_logs=False,
                                    get_csv=False,
                                    get_report=False,
                                    get_difference=True,
                                    return_contents=True))

        # Normalize tautomers
        post_tautomer_normalized_smiles_data, differ_after_tautomer_normalizing, tautomer_step_contents = (
            NormalizingStage(post_stereoisomer_normalized_smiles_data)
            .normalize_tautomer(check_validity=False,  # Already validated
                                print_logs=False,
                                get_csv=False,
                                get_report=False,
                                get_difference=True,
                                return_contents=True))

        # Summary of normalization steps
        summary = (f'Pre-normalized SMILES data: {len(self.smiles_df)}\n'
                   f'Number of stereoisomers normalized: {sum(differ_after_stereoisomer_normalizing)}\n'
                   f'Number of tautomers normalized: {sum(differ_after_tautomer_normalizing)}\n'
                   f'Post-normalized SMILES data: {len(post_stereoisomer_normalized_smiles_data)}\n')

        contents = '\n'.join([
            'VALIDATION STEP:',
            validation_step_contents,
            '----------',
            'STEREOISOMER NORMALIZATION STEP:',
            stereoisomer_step_contents,
            '----------',
            'TAUTOMER NORMALIZATION STEP:',
            tautomer_step_contents,
            '----------',
            'NORMALIZATION COMPLETE!',
            'SUMMARY:',
            summary
        ])

        # Print logs if requested
        if print_logs:
            print(contents)

        # Generate CSV file if requested
        if get_csv:
            (GetReport(post_stereoisomer_normalized_smiles_data,
                       output_dir_path=output_dir_path,
                       report_subdir_name='tautomer_and_stereoisomer_normalization')
             .create_csv_file(csv_file_name='post_normalized_smiles.csv'))

        # Generate report file if requested
        if get_report:
            (GetReport(post_stereoisomer_normalized_smiles_data,
                       output_dir_path=output_dir_path,
                       report_subdir_name='tautomer_and_stereoisomer_normalization')
             .create_report_file(report_file_name='tautomer_and_stereoisomer_normalization_report.txt',
                                 content=contents))

        return post_stereoisomer_normalized_smiles_data
