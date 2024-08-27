import pandas as pd
import os
from parallel_pandas import ParallelPandas
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import Chem

ParallelPandas.initialize(n_cpu=16, split_factor=4, disable_pr_bar=True)


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
        # inorganic_patterns = [
        #     Chem.MolFromSmarts("O=C=O"),  # Carbon dioxide
        #     Chem.MolFromSmarts("C(=O)[O-]"),  # Carbonate ion
        #     Chem.MolFromSmarts("C#N"),  # Cyanide ion
        #     Chem.MolFromSmarts("[C]#[O]"),  # Carbon monoxide
        #     Chem.MolFromSmarts("C(=O)[OH]"),  # Carboxyl group (inorganic context)
        # ]
        mol = Chem.MolFromSmiles(self.smiles)
        # Check for the absence of carbon atoms
        has_carbon = any(atom.GetSymbol() == 'C' for atom in mol.GetAtoms())
        if has_carbon is not True:
            return False
        # for pattern in inorganic_patterns:
        #     if mol.HasSubstructMatch(pattern):
        #         return False
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


def remove_duplicates(smiles_df: pd.DataFrame,
                      check_validity: bool = True,
                      output_dir_path: str = None,
                      print_logs: bool = True,
                      get_report: bool = False,
                      get_csv: bool = True,
                      show_duplicated_smiles_and_index: bool = True):
    from curation.validate import ValidationStage
    if check_validity:
        smiles_df = ValidationStage(smiles_df).check_valid_smiles(get_csv=False)

    duplicated_smiles = smiles_df[smiles_df.duplicated(keep=False)]
    duplicated_smiles_include_idx = (((duplicated_smiles
                                       .groupby(duplicated_smiles.columns.tolist(), sort=False))
                                      .p_apply(lambda x: tuple(x.index)))
                                     .reset_index(name='index'))
    post_smiles_df = smiles_df.merge(duplicated_smiles_include_idx, on='compound', how='left')

    post_duplicates_removed_smiles = smiles_df.drop_duplicates()

    contents = (f'Number of input SMILES strings: {len(smiles_df)}\n'
                f'Number of unique SMILES strings: {len(post_duplicates_removed_smiles)}\n'
                f'Number of duplicate SMILES strings: {len(smiles_df) - len(post_duplicates_removed_smiles)}\n')

    if print_logs:
        print(contents)

    if get_csv:
        (GetReport(post_duplicates_removed_smiles,
                   output_dir_path=output_dir_path,
                   report_subdir_name='remove_duplicates')
         .create_csv_file(csv_file_name='post_duplicates_removed.csv'))

    if get_report:
        (GetReport(post_duplicates_removed_smiles,
                   output_dir_path=output_dir_path,
                   report_subdir_name='remove_duplicates')
         .create_report_file(report_file_name='remove_duplicates_report.txt',
                             content=contents))
        (GetReport(post_smiles_df,
                   output_dir_path=output_dir_path,
                   report_subdir_name='remove_duplicates')
         .create_csv_file(csv_file_name='duplicated_smiles_include_idx.csv'))

    if show_duplicated_smiles_and_index:
        return post_duplicates_removed_smiles, post_smiles_df
    else:
        return post_duplicates_removed_smiles
