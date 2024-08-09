import pandas as pd
from parallel_pandas import ParallelPandas
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.EnumerateStereoisomers import *
from curation.utils import *


class NormalizeTautomer:
    def __init__(self, smiles: str, return_difference: bool = False):
        self.smiles = smiles
        self.return_difference = return_difference

    def normalize_tautomer(self):
        # Create a molecule from the SMILES string
        mol = Chem.MolFromSmiles(self.smiles)

        # Normalize tautomers
        enumerator = rdMolStandardize.TautomerEnumerator()
        canonical_mol = enumerator.Canonicalize(mol)

        # Convert the canonical molecule back to a SMILES string
        canonical_smiles = Chem.MolToSmiles(canonical_mol)

        # Check if the SMILES string is tautomerized.
        if self.smiles == canonical_smiles:
            difference = False
        else:
            difference = True

        if self.return_difference:
            return canonical_smiles, difference
        else:
            return canonical_smiles


def normalize_tautomer_in_dataframe(smiles_df: pd.DataFrame,
                                    check_validity: bool = True,
                                    report_dir_path: str = None,
                                    print_logs: bool = True,
                                    get_report_text_file: bool = False):
    """
    Normalize tautomers for a given SMILES dataframe.
    :param report_dir_path:
    :param get_report_text_file:
    :param check_validity:
    :param smiles_df:
    :param print_logs:
    :return:
    """
    if check_validity:
        smiles_df = ValidationStage(smiles_df).check_valid_smiles()

    post_tautomer_normalized = smiles_df['compound'].p_apply(lambda x: NormalizeTautomer(x, return_difference=True)
                                                             .normalize_tautomer())
    post_tautomer_normalized_smiles_dataframe = post_tautomer_normalized.p_apply(lambda x: x[0])
    difference_after_tautomer_normalized = post_tautomer_normalized.p_apply(lambda x: x[1])

    contents = (f'Number of SMILES strings before tautomer normalizing: {len(smiles_df)}\n'
                f'Number of SMILES tautomers are normalizing: {sum(difference_after_tautomer_normalized)}\n'
                f'Number of SMILES strings after tautomer normalizing: {len(post_tautomer_normalized_smiles_dataframe)}\n')
    post_tautomer_normalized_smiles_dataframe = post_tautomer_normalized_smiles_dataframe.to_frame()

    if print_logs:
        print(contents)

    if get_report_text_file:
        (GetReport(post_tautomer_normalized_smiles_dataframe,
                   output_dir_path=report_dir_path,
                   report_subdir_name='normalize_tautomers',
                   report_file_name='normalize_tautomers_report.txt',
                   csv_file_name='post_tautomers_normalized.csv',
                   content=contents)
         .create_report_and_csv_files())

    return post_tautomer_normalized_smiles_dataframe
