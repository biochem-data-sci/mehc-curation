import os
import sys
import argparse
import pandas as pd
from parallel_pandas import ParallelPandas

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
template_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "template_report"
)


class ValidationStage:
    def __init__(self, smi_df: pd.DataFrame):
        self.smi_df = smi_df

    def validate_smi(
        self,
        output_dir: str = None,
        print_logs: bool = True,
        get_report: bool = False,
        get_output: bool = True,
        get_invalid_smi_idx: bool = False,
        get_isomeric_smi: bool = True,
        param_deduplicate: bool = False,
        return_format_data: bool = False,
        n_cpu: int = 1,
        split_factor: int = 1,
    ):
        from curation.utils import GetReport, RemoveSpecificSMILES, deduplicate
        from rdkit import Chem

        ParallelPandas.initialize(
            n_cpu=n_cpu, split_factor=split_factor, disable_pr_bar=True
        )
        smi_col = self.smi_df.columns.tolist()
        self.smi_df["is_valid"] = self.smi_df[smi_col[0]].p_apply(
            lambda x: RemoveSpecificSMILES(x).is_valid()
        )
        valid_smi = self.smi_df[self.smi_df["is_valid"]].copy()
        valid_smi.drop(columns=["is_valid"], inplace=True)
        invalid_smi = self.smi_df[self.smi_df["is_valid"] == False][smi_col[0]]
        idx_of_invalid_smi = self.smi_df[
            self.smi_df["is_valid"] == False
        ].index.tolist()

        if get_isomeric_smi:
            valid_smi[smi_col[0]] = valid_smi[smi_col[0]].p_apply(
                lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x))
            )
        else:
            valid_smi[smi_col[0]] = valid_smi[smi_col[0]].p_apply(
                lambda x: Chem.MolToSmiles(
                    Chem.MolFromSmiles(x), isomericSmiles=False
                )
            )

        format_data = {
            "validity_input": len(self.smi_df),
            "invalid": len(invalid_smi),
            "valid": len(valid_smi),
        }

        with open(
            os.path.join(template_dir, "validation_title.txt"), "r"
        ) as validation_title:
            template_report = validation_title.read()

        with open(
            os.path.join(template_dir, "validity_check.txt"), "r"
        ) as validity_check:
            template_report += validity_check.read()

        dup_idx_data = pd.DataFrame()
        if param_deduplicate:
            valid_smi, dup_idx_data, rm_dup_format_data = deduplicate(
                valid_smi,
                validate=False,
                get_output=False,
                print_logs=False,
                show_dup_smi_and_idx=True,
                return_format_data=True,
                n_cpu=n_cpu,
                split_factor=split_factor,
            )
            format_data.update(rm_dup_format_data)
            with open(
                os.path.join(template_dir, "deduplicate.txt"), "r"
            ) as deduplicate_txt:
                template_report += deduplicate_txt.read()
            valid_smi = valid_smi.reset_index(drop=True)

        with open(os.path.join(template_dir, "end.txt"), "r") as end:
            template_report += end.read()

        formatted_report = template_report.format(**format_data)

        if print_logs:
            print(formatted_report)

        if get_output:
            (
                GetReport(
                    output_dir=output_dir,
                    report_subdir_name="check_valid_smiles",
                ).create_csv_file(valid_smi, csv_file_name="valid_smiles.csv")
            )

        if get_report:
            (
                GetReport(
                    output_dir=output_dir,
                    report_subdir_name="check_valid_smiles",
                ).create_report_file(
                    report_file_name="validity_check.txt",
                    content=formatted_report,
                )
            )
            (
                GetReport(
                    output_dir=output_dir,
                    report_subdir_name="check_valid_smiles",
                ).create_csv_file(
                    invalid_smi, csv_file_name="invalid_smiles.csv"
                )
            )
            if param_deduplicate:
                (
                    GetReport(
                        output_dir=output_dir,
                        report_subdir_name="check_valid_smiles",
                    ).create_csv_file(
                        dup_idx_data, csv_file_name="duplicate_index_data.csv"
                    )
                )

        if get_invalid_smi_idx and return_format_data:
            return valid_smi, idx_of_invalid_smi, format_data
        elif get_invalid_smi_idx and not return_format_data:
            return valid_smi, idx_of_invalid_smi
        elif not get_invalid_smi_idx and return_format_data:
            return valid_smi, format_data
        elif not get_invalid_smi_idx and not return_format_data:
            return valid_smi

    def rm_mixtures(
        self,
        validate: bool = True,
        output_dir: str = None,
        get_invalid_smi_idx: bool = False,
        print_logs: bool = True,
        get_report: bool = False,
        get_output: bool = True,
        param_deduplicate: bool = False,
        return_format_data: bool = False,
        n_cpu: int = 1,
        split_factor: int = 1,
    ):
        from curation.utils import GetReport, RemoveSpecificSMILES, deduplicate

        ParallelPandas.initialize(
            n_cpu=n_cpu, split_factor=split_factor, disable_pr_bar=True
        )
        smi_col = self.smi_df.columns.tolist()
        if validate:
            self.smi_df, validate_format_data = self.validate_smi(
                get_output=False,
                print_logs=False,
                param_deduplicate=False,
                return_format_data=True,
                n_cpu=n_cpu,
                split_factor=split_factor,
            )

        self.smi_df["is_valid"] = self.smi_df[smi_col[0]].p_apply(
            lambda x: RemoveSpecificSMILES(x).is_mixture()
        )
        valid_smi = self.smi_df[self.smi_df["is_valid"] == False].copy()
        valid_smi.drop(columns=["is_valid"], inplace=True)
        invalid_smi = self.smi_df[self.smi_df["is_valid"]][smi_col[0]]
        idx_of_invalid_smi = self.smi_df[
            self.smi_df["is_valid"] == False
        ].index.tolist()

        format_data = {
            "rm_mixtures_input": len(self.smi_df),
            "mixture": len(invalid_smi),
            "non_mixture": len(valid_smi),
        }

        with open(
            os.path.join(template_dir, "validation_title.txt"), "r"
        ) as validation_title:
            template_report = validation_title.read()

        if validate:
            format_data.update(validate_format_data)
            with open(
                os.path.join(template_dir, "validity_check.txt"), "r"
            ) as validity_check:
                template_report += validity_check.read()

        with open(
            os.path.join(template_dir, "mixture_removal.txt"), "r"
        ) as mixture_removal:
            template_report += mixture_removal.read()

        dup_idx_data = pd.DataFrame()
        if param_deduplicate:
            valid_smi, dup_idx_data, deduplicate_format_data = deduplicate(
                valid_smi,
                validate=False,
                get_output=False,
                print_logs=False,
                show_dup_smi_and_idx=True,
                return_format_data=True,
                n_cpu=n_cpu,
                split_factor=split_factor,
            )
            format_data.update(deduplicate_format_data)
            with open(
                os.path.join(template_dir, "deduplicate.txt"), "r"
            ) as deduplicate_txt:
                template_report += deduplicate_txt.read()
            valid_smi = valid_smi.reset_index(drop=True)

        with open(os.path.join(template_dir, "end.txt"), "r") as end:
            template_report += end.read()

        formatted_report = template_report.format(**format_data)

        if print_logs:
            print(formatted_report)

        if get_output:
            (
                GetReport(
                    output_dir=output_dir, report_subdir_name="rm_mixtures"
                ).create_csv_file(valid_smi, csv_file_name="non_mixtures.csv")
            )

        if get_report:
            (
                GetReport(
                    output_dir=output_dir, report_subdir_name="rm_mixtures"
                ).create_report_file(
                    report_file_name="rm_mixtures.txt",
                    content=formatted_report,
                )
            )
            (
                GetReport(
                    output_dir=output_dir, report_subdir_name="rm_mixtures"
                ).create_csv_file(invalid_smi, csv_file_name="mixtures.csv")
            )
            if param_deduplicate:
                (
                    GetReport(
                        output_dir=output_dir,
                        report_subdir_name="check_valid_smiles",
                    ).create_csv_file(
                        dup_idx_data, csv_file_name="duplicate_index_data.csv"
                    )
                )

        if get_invalid_smi_idx and return_format_data:
            return valid_smi, idx_of_invalid_smi, format_data
        elif get_invalid_smi_idx and not return_format_data:
            return valid_smi, idx_of_invalid_smi
        elif return_format_data and not get_invalid_smi_idx:
            return valid_smi, format_data
        elif not get_invalid_smi_idx and not return_format_data:
            return valid_smi

    def rm_inorganics(
        self,
        validate: bool = True,
        output_dir: str = None,
        get_invalid_smi_idx: bool = False,
        print_logs: bool = True,
        get_report: bool = False,
        get_output: bool = True,
        param_deduplicate: bool = False,
        return_format_data: bool = False,
        n_cpu: int = 1,
        split_factor: int = 1,
    ):
        from curation.utils import GetReport, RemoveSpecificSMILES, deduplicate

        ParallelPandas.initialize(
            n_cpu=n_cpu, split_factor=split_factor, disable_pr_bar=True
        )
        smi_col = self.smi_df.columns.tolist()
        if validate:
            self.smi_df, validate_format_data = self.validate_smi(
                get_output=False,
                print_logs=False,
                param_deduplicate=False,
                return_format_data=True,
                n_cpu=n_cpu,
                split_factor=split_factor,
            )

        self.smi_df["is_valid"] = self.smi_df[smi_col[0]].p_apply(
            lambda x: RemoveSpecificSMILES(x).is_inorganic()
        )
        valid_smi = self.smi_df[self.smi_df["is_valid"] == False].copy()
        valid_smi.drop(columns=["is_valid"], inplace=True)
        invalid_smi = self.smi_df[self.smi_df["is_valid"]][smi_col[0]]
        idx_of_invalid_smi = self.smi_df[
            self.smi_df["is_valid"] == False
        ].index.tolist()

        format_data = {
            "rm_inorganic_input": len(self.smi_df),
            "inorganic": len(invalid_smi),
            "organic": len(valid_smi),
        }

        with open(
            os.path.join(template_dir, "validation_title.txt"), "r"
        ) as validation_title:
            template_report = validation_title.read()

        if validate:
            format_data.update(validate_format_data)
            with open(
                os.path.join(template_dir, "validity_check.txt"), "r"
            ) as validity_check:
                template_report += validity_check.read()

        with open(
            os.path.join(template_dir, "inorganic_removal.txt"), "r"
        ) as inorganic_removal:
            template_report += inorganic_removal.read()

        dup_idx_data = pd.DataFrame()
        if param_deduplicate:
            valid_smi, dup_idx_data, deduplicate_format_data = deduplicate(
                valid_smi,
                validate=False,
                get_output=False,
                print_logs=False,
                show_dup_smi_and_idx=True,
                return_format_data=True,
                n_cpu=n_cpu,
                split_factor=split_factor,
            )
            format_data.update(deduplicate_format_data)
            with open(
                os.path.join(template_dir, "deduplicate.txt"), "r"
            ) as deduplicate_txt:
                template_report += deduplicate_txt.read()
            valid_smi = valid_smi.reset_index(drop=True)

        with open(os.path.join(template_dir, "end.txt"), "r") as end:
            template_report += end.read()

        formatted_report = template_report.format(**format_data)

        if print_logs:
            print(formatted_report)

        if get_output:
            (
                GetReport(
                    output_dir=output_dir, report_subdir_name="rm_inorganics"
                ).create_csv_file(valid_smi, csv_file_name="organics.csv")
            )

        if get_report:
            (
                GetReport(
                    output_dir=output_dir, report_subdir_name="rm_inorganics"
                ).create_report_file(
                    report_file_name="rm_inorganics.txt",
                    content=formatted_report,
                )
            )
            (
                GetReport(
                    output_dir=output_dir, report_subdir_name="rm_inorganics"
                ).create_csv_file(invalid_smi, csv_file_name="inorganics.csv")
            )
            if param_deduplicate:
                (
                    GetReport(
                        output_dir=output_dir,
                        report_subdir_name="rm_inorganics",
                    ).create_csv_file(
                        dup_idx_data, csv_file_name="duplicate_index_data.csv"
                    )
                )

        if get_invalid_smi_idx and return_format_data:
            return valid_smi, idx_of_invalid_smi, format_data
        elif get_invalid_smi_idx and not return_format_data:
            return valid_smi, idx_of_invalid_smi
        elif return_format_data and not get_invalid_smi_idx:
            return valid_smi, format_data
        elif not get_invalid_smi_idx and not return_format_data:
            return valid_smi

    def rm_organometallics(
        self,
        validate: bool = True,
        output_dir: str = None,
        get_invalid_smi_idx: bool = False,
        print_logs: bool = True,
        get_report: bool = False,
        get_output: bool = True,
        param_deduplicate: bool = False,
        return_format_data: bool = False,
        n_cpu: int = 1,
        split_factor: int = 1,
    ):
        from curation.utils import GetReport, RemoveSpecificSMILES, deduplicate

        ParallelPandas.initialize(
            n_cpu=n_cpu, split_factor=split_factor, disable_pr_bar=True
        )
        smi_col = self.smi_df.columns.tolist()
        if validate:
            self.smi_df, validate_format_data = self.validate_smi(
                get_output=False,
                print_logs=False,
                param_deduplicate=False,
                return_format_data=True,
                n_cpu=n_cpu,
                split_factor=split_factor,
            )

        self.smi_df["is_valid"] = self.smi_df[smi_col[0]].p_apply(
            lambda x: RemoveSpecificSMILES(x).is_organometallic()
        )
        valid_smi = self.smi_df[self.smi_df["is_valid"]].copy()
        valid_smi.drop(columns=["is_valid"], inplace=True)
        invalid_smi = self.smi_df[self.smi_df["is_valid"] == False][smi_col[0]]
        idx_of_invalid_smi = self.smi_df[
            self.smi_df["is_valid"] == False
        ].index.tolist()

        format_data = {
            "rm_organometallic_input": len(self.smi_df),
            "organometallic": len(invalid_smi),
            "non_organometallic": len(valid_smi),
        }

        with open(
            os.path.join(template_dir, "validation_title.txt"), "r"
        ) as validation_title:
            template_report = validation_title.read()

        if validate:
            format_data.update(validate_format_data)
            with open(
                os.path.join(template_dir, "validity_check.txt"), "r"
            ) as validity_check:
                template_report += validity_check.read()

        with open(
            os.path.join(template_dir, "organometallic_removal.txt"), "r"
        ) as organometallic_removal:
            template_report += organometallic_removal.read()

        dup_idx_data = pd.DataFrame()
        if param_deduplicate:
            valid_smi, dup_idx_data, deduplicate_format_data = deduplicate(
                valid_smi,
                validate=False,
                get_output=False,
                print_logs=False,
                show_dup_smi_and_idx=True,
                return_format_data=True,
                n_cpu=n_cpu,
                split_factor=split_factor,
            )
            format_data.update(deduplicate_format_data)
            with open(
                os.path.join(template_dir, "deduplicate.txt"), "r"
            ) as deduplicate_txt:
                template_report += deduplicate_txt.read()
            valid_smi = valid_smi.reset_index(drop=True)

        with open(os.path.join(template_dir, "end.txt"), "r") as end:
            template_report += end.read()

        formatted_report = template_report.format(**format_data)

        if print_logs:
            print(formatted_report)

        if get_output:
            (
                GetReport(
                    output_dir=output_dir,
                    report_subdir_name="rm_organometallics",
                ).create_csv_file(
                    valid_smi, csv_file_name="non_organometallic.csv"
                )
            )

        if get_report:
            (
                GetReport(
                    output_dir=output_dir,
                    report_subdir_name="rm_organometallics",
                ).create_report_file(
                    report_file_name="rm_organometallics.txt",
                    content=formatted_report,
                )
            )
            (
                GetReport(
                    output_dir=output_dir,
                    report_subdir_name="rm_organometallics",
                ).create_csv_file(
                    invalid_smi, csv_file_name="organometallic.csv"
                )
            )
            if param_deduplicate:
                (
                    GetReport(
                        output_dir=output_dir,
                        report_subdir_name="rm_organometallics",
                    ).create_csv_file(
                        dup_idx_data, csv_file_name="duplicate_index_data.csv"
                    )
                )

        if get_invalid_smi_idx and return_format_data:
            return valid_smi, idx_of_invalid_smi, format_data
        elif get_invalid_smi_idx and not return_format_data:
            return valid_smi, idx_of_invalid_smi
        elif return_format_data and not get_invalid_smi_idx:
            return valid_smi, format_data
        elif not get_invalid_smi_idx and not return_format_data:
            return valid_smi

    def complete_validation(
        self,
        validate: bool = True,
        output_dir: str = None,
        print_logs: bool = True,
        get_report: bool = False,
        get_output: bool = True,
        param_deduplicate: bool = True,
        n_cpu: int = 1,
        split_factor: int = 1,
    ):
        from curation.utils import GetReport, deduplicate

        format_data = {}

        with open(
            os.path.join(template_dir, "validation_title.txt"), "r"
        ) as validation_title:
            template_report = validation_title.read()

        if validate:
            self.smi_df, invalid_smi_idx, validation_format_data = (
                self.validate_smi(
                    print_logs=False,
                    get_invalid_smi_idx=True,
                    get_output=False,
                    param_deduplicate=False,
                    return_format_data=True,
                    n_cpu=n_cpu,
                    split_factor=split_factor,
                )
            )
            format_data.update(validation_format_data)
            with open(
                os.path.join(template_dir, "validity_check.txt"), "r"
            ) as validity_check:
                template_report += validity_check.read()

        self.smi_df, mixture_idx, rm_mixtures_format_data = self.rm_mixtures(
            validate=False,
            print_logs=False,
            get_invalid_smi_idx=True,
            get_output=False,
            param_deduplicate=False,
            return_format_data=True,
            n_cpu=n_cpu,
            split_factor=split_factor,
        )
        format_data.update(rm_mixtures_format_data)
        with open(
            os.path.join(template_dir, "mixture_removal.txt"), "r"
        ) as mixture_removal:
            template_report += mixture_removal.read()

        self.smi_df, inorganic_idx, rm_inorganics_format_data = (
            self.rm_inorganics(
                validate=False,
                print_logs=False,
                get_invalid_smi_idx=True,
                get_output=False,
                param_deduplicate=False,
                return_format_data=True,
                n_cpu=n_cpu,
                split_factor=split_factor,
            )
        )
        format_data.update(rm_inorganics_format_data)
        with open(
            os.path.join(template_dir, "inorganic_removal.txt"), "r"
        ) as inorganic_removal:
            template_report += inorganic_removal.read()

        self.smi_df, organometallic_idx, rm_organometallics_format_data = (
            self.rm_organometallics(
                validate=False,
                print_logs=False,
                get_invalid_smi_idx=True,
                get_output=False,
                param_deduplicate=False,
                return_format_data=True,
                n_cpu=n_cpu,
                split_factor=split_factor,
            )
        )
        format_data.update(rm_organometallics_format_data)
        with open(
            os.path.join(template_dir, "organometallic_removal.txt"), "r"
        ) as organometallic_removal:
            template_report += organometallic_removal.read()

        dup_idx_data = pd.DataFrame()
        if param_deduplicate:
            self.smi_df, dup_idx_data, deduplicate_format_data = deduplicate(
                self.smi_df,
                validate=False,
                get_output=False,
                print_logs=False,
                show_dup_smi_and_idx=True,
                return_format_data=True,
                n_cpu=n_cpu,
                split_factor=split_factor,
            )
            format_data.update(deduplicate_format_data)
            with open(
                os.path.join(template_dir, "deduplicate.txt"), "r"
            ) as deduplicate_txt:
                template_report += deduplicate_txt.read()
            self.smi_df = self.smi_df.reset_index(drop=True)

        with open(os.path.join(template_dir, "end.txt"), "r") as end:
            template_report += end.read()

        formatted_report = template_report.format(**format_data)

        if print_logs:
            print(formatted_report)

        if get_output:
            (
                GetReport(
                    output_dir=output_dir,
                    report_subdir_name="complete_validation",
                ).create_csv_file(
                    self.smi_df, csv_file_name="output_smiles.csv"
                )
            )

        if get_report:
            (
                GetReport(
                    output_dir=output_dir,
                    report_subdir_name="complete_validation",
                ).create_report_file(
                    report_file_name="complete_validation.txt",
                    content=formatted_report,
                )
            )
            if param_deduplicate:
                (
                    GetReport(
                        output_dir=output_dir,
                        report_subdir_name="complete_validation",
                    ).create_csv_file(
                        dup_idx_data, csv_file_name="duplicate_index_data.csv"
                    )
                )

        return self.smi_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This module is used to validate_beginning SMILES data",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", required=True, type=str, help="Input SMILES csv file"
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="Output folder, include csv and report (if have any)",
    )
    parser.add_argument(
        "-c",
        "--choice",
        required=True,
        type=int,
        choices=range(1, 6),
        help="Choose the action:\n"
        "1. Check the validity of the SMILES data\n"
        "2. Remove mixtures in the SMILES data\n"
        "3. Remove inorganics in the SMILES data\n"
        "4. Remove organometallics in the SMILES data\n"
        "5. Do all the validation stage\n",
    )
    parser.add_argument(
        "--validate_beginning",
        required=False,
        action="store_true",
        default=False,
        help="Validate SMILES data before doing anything "
        "(except action 1, optional, default is false)",
    )
    parser.add_argument(
        "--deduplicate",
        required=False,
        action="store_true",
        default=False,
        help="Remove duplicate SMILES after your action "
        "(optional, default is false)",
    )
    parser.add_argument(
        "-p",
        "--print_logs",
        required=False,
        action="store_false",
        help="Print logs (optional, default is true)",
    )
    parser.add_argument(
        "--get_report",
        required=False,
        action="store_true",
        help="Get report "
        "(include report file and supported csv for more information) "
        "(optional, default is False)",
    )
    parser.add_argument(
        "--n_cpu",
        required=False,
        type=int,
        default=1,
        help="Number of CPUs to use (optional)",
    )
    parser.add_argument(
        "--split_factor",
        required=False,
        type=int,
        default=1,
        help="Split factor (optional)",
    )
    args: argparse.Namespace = parser.parse_args()

    smi_df = pd.read_csv(args.input)
    validation = ValidationStage(smi_df)

    param_dict = {
        "output_dir": args.output,
        "print_logs": args.print_logs,
        "get_report": args.get_report,
        "validate": args.validate_beginning,
        "param_deduplicate": args.deduplicate,
        "n_cpu": args.n_cpu,
        "split_factor": args.split_factor,
    }

    if args.choice == 1:
        del param_dict["validate_beginning"]

    match args.choice:
        case 1:
            output = validation.validate_smi(**param_dict)
        case 2:
            output = validation.rm_mixtures(**param_dict)
        case 3:
            output = validation.rm_inorganics(**param_dict)
        case 4:
            output = validation.rm_organometallics(**param_dict)
        case 5:
            output = validation.complete_validation(**param_dict)
    print("Your action is done!")
