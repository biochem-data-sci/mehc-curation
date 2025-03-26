import pandas as pd
import os
import sys
from parallel_pandas import ParallelPandas
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import Chem

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
template_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "template_report"
)
dat_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dat")


class GetReport:
    def __init__(self, output_dir: str, report_subdir_name: str):
        self.output_dir = output_dir
        self.report_subdir_name = report_subdir_name

    def create_report_file(self, report_file_name: str, content: str):
        validate_smiles_dir = os.path.join(self.output_dir,
                                           self.report_subdir_name)
        if not os.path.exists(validate_smiles_dir):
            os.makedirs(validate_smiles_dir)

        report_file_path = os.path.join(validate_smiles_dir,
                                        report_file_name)
        with open(report_file_path, "w") as report_file:
            report_file.write(content)

    def create_csv_file(self, smiles_df: pd.DataFrame, csv_file_name: str):
        validate_smiles_dir = os.path.join(self.output_dir,
                                           self.report_subdir_name)
        if not os.path.exists(validate_smiles_dir):
            os.makedirs(validate_smiles_dir)

        csv_file_path = os.path.join(validate_smiles_dir, csv_file_name)
        smiles_df.to_csv(csv_file_path, index=False, encoding="utf-8")


class RemoveSpecificSMILES:
    def __init__(self, smi: str):
        self.smi = smi

    def is_valid(self):
        try:
            mol = Chem.MolFromSmiles(self.smi)
            return mol is not None
        except Exception:
            return False

    def is_mixture(self):
        if self.smi.find(".") != -1:
            if self.smi.find(".[") == -1:
                return True
            else:
                if self.smi.count(".") != self.smi.count(".["):
                    return True
        return False

    def is_inorganic(self):
        mol = Chem.rdmolops.AddHs(Chem.MolFromSmiles(self.smi))
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "C":
                return False
        return True

    def is_organometallic(self):
        mol = Chem.MolFromSmiles(self.smi)
        metals = open(os.path.join(dat_dir, "metals.txt")).read().split(",")
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in metals:
                for neighbor in atom.GetNeighbors():
                    if (neighbor.GetSymbol() == "C"
                            or neighbor.GetSymbol() == "c"):
                        return False
        return True


class CleaningSMILES:
    def __init__(self, smi: str, return_dif: bool = False):
        self.smi = smi
        self.return_dif = return_dif

    def cleaning_salts(self, return_is_null_smi: bool = False):
        remover = SaltRemover()
        mol = Chem.MolFromSmiles(self.smi)
        post_mol = remover.StripMol(mol, dontRemoveEverything=True)
        post_smi = Chem.MolToSmiles(post_mol)
        dif = self.smi != post_smi
        is_null_smi = post_smi == ""
        if self.return_dif and return_is_null_smi:
            return post_smi, dif, is_null_smi
        elif self.return_dif and not return_is_null_smi:
            return post_smi, dif
        elif return_is_null_smi and not self.return_dif:
            return post_smi, is_null_smi
        else:
            return post_smi

    def neutralizing_salts(self, method: str = "boyle"):
        mol = Chem.MolFromSmiles(self.smi)
        try:
            if method == "boyle":
                post_mol = neutralize_atoms(mol)
                post_smi = Chem.MolToSmiles(post_mol)
            elif method == "rdkit":
                post_mol = rdMolStandardize.Uncharger().uncharge(mol)
                post_smi = Chem.MolToSmiles(post_mol)
            dif = self.smi != post_smi
            if self.return_dif:
                return post_smi, dif
            else:
                return post_smi
        except Chem.rdchem.AtomValenceException:
            post_smi = self.smi
            dif = False
            if self.return_dif:
                return post_smi, dif
            else:
                return post_smi


class NormalizeSMILES:
    def __init__(self, smiles: str, return_difference: bool = False):
        self.smiles = smiles
        self.return_difference = return_difference

    def normalize_tautomer(self):
        mol = Chem.MolFromSmiles(self.smiles)
        try:
            canonical_mol = rdMolStandardize.CanonicalTautomer(mol)
            canonical_smiles = Chem.MolToSmiles(canonical_mol)
        except RuntimeError:
            canonical_smiles = self.smiles
        difference = self.smiles != canonical_smiles
        if self.return_difference:
            return canonical_smiles, difference
        else:
            return canonical_smiles

    def normalize_stereoisomer(self):
        mol = Chem.MolFromSmiles(self.smiles)
        stereo_normalized_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        difference = self.smiles != stereo_normalized_smiles
        if self.return_difference:
            return stereo_normalized_smiles, difference
        else:
            return stereo_normalized_smiles


def neutralize_atoms(mol):
    pattern = Chem.MolFromSmarts(
        "[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
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


def deduplicate(
    smi_df: pd.DataFrame,
    validate: bool = False,
    output_dir: str = None,
    print_logs: bool = True,
    get_report: bool = False,
    get_output: bool = True,
    return_format_data: bool = False,
    show_dup_smi_and_idx: bool = False,
    n_cpu: int = 1,
    split_factor: int = 1,
):
    ParallelPandas.initialize(
        n_cpu=n_cpu, split_factor=split_factor, disable_pr_bar=True
    )
    from curation.validate import ValidationStage

    if validate:
        smi_df, validate_format_data = ValidationStage(smi_df).validate_smi(
            get_output=False,
            print_logs=False,
            param_deduplicate=False,
            return_format_data=True,
            n_cpu=n_cpu,
            split_factor=split_factor,
        )

    smi_col = smi_df.columns.tolist()

    perfect_dups_mask = smi_df.duplicated(keep="first")
    perfect_dups = smi_df[perfect_dups_mask]

    grouped = smi_df.groupby(smi_col[0]).nunique()
    partial_dups_smiles = grouped[grouped.gt(1).any(axis=1)].index
    partial_dups = smi_df[smi_df[smi_col[0]].isin(partial_dups_smiles)]

    dups_info = (
        pd.concat([perfect_dups, partial_dups], ignore_index=True)
        .drop_duplicates()[[smi_col[0]]]
        .copy()
    )
    dups_info["is_perfect_dups"] = (dups_info[smi_col[0]]
                                    .isin(perfect_dups[smi_col[0]]))
    index_mapping = (
        smi_df.groupby(smi_col[0])
        .p_apply(lambda x: list(x.index))
        .reset_index(name="indexes")
    )
    dups_info = dups_info.merge(
        index_mapping, on=smi_col[0], how="left"
    ).drop_duplicates(subset=[smi_col[0]])

    cleaned_df = smi_df[
        ~smi_df[smi_col[0]].isin(partial_dups_smiles)
    ]
    cleaned_df = cleaned_df.drop_duplicates()

    format_data = {
        "duplicate_validation_input": len(smi_df),
        "perfect_dups": len(dups_info[dups_info["is_perfect_dups"]]),
        "partial_dups": len(dups_info[dups_info["is_perfect_dups"] == False]),
        "validation_unique": len(cleaned_df),
    }

    template_report = ""
    if validate:
        format_data.update(validate_format_data)
        with open(
            os.path.join(template_dir, "validity_check.txt"), "r"
        ) as validity_check:
            template_report += validity_check.read()

    with open(os.path.join(template_dir, "deduplicate.txt"), "r") as dedup_file:
        template_report += dedup_file.read()

    with open(os.path.join(template_dir, "end.txt"), "r") as end:
        template_report += end.read()

    formatted_report = template_report.format(**format_data)

    if print_logs:
        print(formatted_report)

    if get_output:
        (
            GetReport(
                output_dir=output_dir, report_subdir_name="deduplicate"
            ).create_csv_file(cleaned_df,
                              csv_file_name="post_duplicates_removed.csv")
        )

    if get_report:
        (
            GetReport(
                output_dir=output_dir, report_subdir_name="deduplicate"
            ).create_report_file(
                report_file_name="deduplicate.txt", content=formatted_report
            )
        )
        (
            GetReport(
                output_dir=output_dir, report_subdir_name="deduplicate"
            ).create_csv_file(
                dups_info, csv_file_name="duplicated_smiles_include_idx.csv"
            )
        )

    if show_dup_smi_and_idx and return_format_data:
        return cleaned_df, dups_info, format_data
    elif show_dup_smi_and_idx and not return_format_data:
        return cleaned_df, dups_info
    elif return_format_data and not show_dup_smi_and_idx:
        return cleaned_df, format_data
    else:
        return cleaned_df
