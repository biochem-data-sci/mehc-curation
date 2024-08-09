import pandas as pd
import numpy as np
import os
from parallel_pandas import ParallelPandas
from rdkit import Chem
from .validate import *


class GetReport:
    def __init__(self,
                 smiles_df: pd.DataFrame,
                 output_dir_path: str,
                 report_subdir_name: str):
        self.smiles_df = smiles_df
        self.output_dir_path = output_dir_path
        self.report_subdir_name = report_subdir_name

    def create_report_file(self,
                           report_file_name: str,
                           content: str):
        validate_smiles_dir = os.path.join(self.output_dir_path, self.report_subdir_name)
        if not os.path.exists(validate_smiles_dir):
            os.makedirs(validate_smiles_dir)

        report_file_path = os.path.join(validate_smiles_dir, report_file_name)
        with open(report_file_path, 'w') as report_file:
            report_file.write(content)

    def create_csv_file(self,
                        csv_file_name: str):
        validate_smiles_dir = os.path.join(self.output_dir_path, self.report_subdir_name)
        if not os.path.exists(validate_smiles_dir):
            os.makedirs(validate_smiles_dir)

        csv_file_path = os.path.join(validate_smiles_dir, csv_file_name)
        self.smiles_df.to_csv(csv_file_path, index=False, encoding='utf-8')


class RemoveSpecificSMILES:
    def __init__(self, smiles: str):
        self.smiles = smiles

    def is_valid(self):
        try:
            mol = Chem.MolFromSmiles(self.smiles)
            return mol is not None
        except Exception as e:
            return False

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
        smiles_df = ValidationStage(smiles_df).check_valid_smiles()

    duplicated_smiles = smiles_df[smiles_df.duplicated(keep=False)]
    duplicated_smiles_include_idx = (((duplicated_smiles
                                       .groupby(duplicated_smiles.columns.tolist()))
                                      .p_apply(lambda x: tuple(x.index)))
                                     .reset_index(name='index'))

    post_duplicates_removed_smiles = smiles_df.drop_duplicates()

    contents = (f'Number of input SMILES strings: {len(smiles_df)}\n'
                f'Number of unique SMILES strings: {len(post_duplicates_removed_smiles)}\n'
                f'Number of duplicate SMILES strings: {len(smiles_df) - len(post_duplicates_removed_smiles)}\n')

    if print_logs:
        print(contents)

    if get_report_text_file:
        (GetReport(post_duplicates_removed_smiles,
                   output_dir_path=report_dir_path,
                   report_subdir_name='remove_duplicates',
                   report_file_name='remove_duplicates_report.txt',
                   csv_file_name='post_duplicates_removed.csv',
                   content=contents)
         .create_report_and_csv_files())

    if show_duplicated_smiles_and_index:
        return duplicated_smiles_include_idx, post_duplicates_removed_smiles
    else:
        return post_duplicates_removed_smiles
