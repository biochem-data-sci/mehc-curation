import pandas as pd
import os
import sys
import argparse
from parallel_pandas import ParallelPandas

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
template_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "template_report"
)


class CleaningStage:
    def __init__(self, smi_df: pd.DataFrame):
        self.smi_df = smi_df

    def cl_salt(
        self,
        validate: bool = True,
        output_dir: str = None,
        print_logs: bool = True,
        get_report: bool = False,
        # get_output: bool = True,
        get_diff: bool = False,
        param_deduplicate: bool = False,
        return_format_data: bool = False,
        n_cpu: int = None,
        split_factor: int = 1,
        partial_dup_cols: list = None
    ):
        from curation.utils import GetReport, CleaningSMILES, deduplicate
        from curation.validate import ValidationStage

        if output_dir is None:
            get_output = False
        else:
            get_output = True

        ParallelPandas.initialize(
            n_cpu=n_cpu, split_factor=split_factor, disable_pr_bar=True
        )
        smi_col = self.smi_df.columns.tolist()
        if validate:
            self.smi_df, validation_format_data = ValidationStage(
                self.smi_df
            ).validate_smi(
                print_logs=False,
                param_deduplicate=False,
                return_format_data=True,
                n_cpu=n_cpu,
                split_factor=split_factor,
            )

        salts_cleaned = self.smi_df[smi_col[0]].p_apply(
            lambda x: CleaningSMILES(x, return_dif=True).cleaning_salts(
                return_is_null_smi=True
            )
        )
        post_salts_cl_smi_data = pd.DataFrame(
            list(salts_cleaned.p_apply(lambda x: x[0])),
            columns=["post_smiles"],
        )
        diff_after_cl_salt = pd.DataFrame(
            list(salts_cleaned.p_apply(lambda x: x[1])), columns=["diff"]
        )
        is_missing_smi_str = pd.DataFrame(
            list(salts_cleaned.p_apply(lambda x: x[2])), columns=["is_missing"]
        )
        post_smi_df = pd.concat(
            [
                post_salts_cl_smi_data.reset_index(drop=True),
                self.smi_df.reset_index(drop=True),
                diff_after_cl_salt.reset_index(drop=True),
                is_missing_smi_str.reset_index(drop=True),
            ],
            axis=1,
        )
        post_smi_df = post_smi_df[post_smi_df["is_missing"] == False]
        post_smi_df.drop(
            columns=post_smi_df.columns[[1, -1, -2]], inplace=True
        )
        post_smi_df.rename(
            columns={post_smi_df.columns[0]: "smiles"}, inplace=True
        )

        # post_salts_cl_smi_data = salts_cleaned.p_apply(lambda x: x[0])
        # diff_after_cl_salt = salts_cleaned.p_apply(lambda x: x[1])
        # is_missing_smi_str = salts_cleaned.p_apply(lambda x: x[2])

        # post_drop_missing_smi_str = (post_salts_cl_smi_data.drop(
        #     is_missing_smi_str.index[is_missing_smi_str == True].tolist(),
        #     axis='index'))
        missing_smiles_cnt = len(post_salts_cl_smi_data) - len(post_smi_df)
        # post_drop_missing_smi_str = post_drop_missing_smi_str.to_frame()

        format_data = {
            "salt_cleaning_input": len(self.smi_df),
            "desalted": sum(diff_after_cl_salt["diff"]),
            "unprocessable": missing_smiles_cnt,
            "salt_cleaning_output": len(post_smi_df),
        }

        with open(
            os.path.join(template_dir, "cleaning_title.txt"), "r"
        ) as cleaning_title:
            template_report = cleaning_title.read()

        if validate:
            format_data.update(validation_format_data)
            with open(
                os.path.join(template_dir, "validity_check.txt"), "r"
            ) as validity_check:
                template_report += validity_check.read()

        with open(
            os.path.join(template_dir, "salt_cleaning.txt"), "r"
        ) as salt_cleaning:
            template_report += salt_cleaning.read()

        dup_idx_data = pd.DataFrame()
        if param_deduplicate:
            post_smi_df, dup_idx_data, deduplicate_format_data = deduplicate(
                post_smi_df,
                validate=False,
                print_logs=False,
                show_dup_smi_and_idx=True,
                return_format_data=True,
                n_cpu=n_cpu,
                split_factor=split_factor,
                partial_dup_cols=partial_dup_cols,
            )
            format_data.update(deduplicate_format_data)
            with open(
                os.path.join(template_dir, "deduplicate.txt"), "r"
            ) as deduplicate_txt:
                template_report += deduplicate_txt.read()
            post_smi_df = post_smi_df.reset_index(drop=True)

        with open(os.path.join(template_dir, "end.txt"), "r") as end:
            template_report += end.read()

        formatted_report = template_report.format(**format_data)

        if print_logs:
            print(formatted_report)

        if get_output:
            (
                GetReport(
                    output_dir=output_dir, report_subdir_name="cleaning_salts"
                ).create_csv_file(
                    post_smi_df, csv_file_name="post_cleaned_salts.csv"
                )
            )

        if get_report:
            (
                GetReport(
                    output_dir=output_dir, report_subdir_name="cleaning_salts"
                ).create_report_file(
                    report_file_name="cleaning_salts_report.txt",
                    content=formatted_report,
                )
            )
            if param_deduplicate:
                (
                    GetReport(
                        output_dir=output_dir,
                        report_subdir_name="cleaning_salts",
                    ).create_csv_file(
                        dup_idx_data, csv_file_name="duplicate_index_data.csv"
                    )
                )

        if get_diff and return_format_data:
            return post_smi_df, diff_after_cl_salt, format_data
        elif get_diff and not return_format_data:
            return post_smi_df, diff_after_cl_salt
        elif return_format_data and not get_diff:
            return post_smi_df, format_data
        elif not get_diff and not return_format_data:
            return post_smi_df

    def neutralize(
        self,
        validate: bool = True,
        method: str = "boyle",
        output_dir: str = None,
        print_logs: bool = True,
        get_report: bool = False,
        # get_output: bool = True,
        get_diff: bool = False,
        param_deduplicate: bool = False,
        return_format_data: bool = False,
        n_cpu: int = None,
        split_factor: int = 1,
        partial_dup_cols: list = None
    ):
        from curation.utils import GetReport, CleaningSMILES, deduplicate
        from curation.validate import ValidationStage

        if output_dir is None:
            get_output = False
        else:
            get_output = True

        ParallelPandas.initialize(
            n_cpu=n_cpu, split_factor=split_factor, disable_pr_bar=True
        )
        smi_col = self.smi_df.columns.tolist()
        if validate:
            self.smi_df, validation_format_data = ValidationStage(
                self.smi_df
            ).validate_smi(
                print_logs=False,
                param_deduplicate=False,
                return_format_data=True,
                n_cpu=n_cpu,
                split_factor=split_factor,
            )

        if method == "boyle":
            neutralized = self.smi_df[smi_col[0]].p_apply(
                lambda x: CleaningSMILES(
                    x, return_dif=True
                ).neutralizing_salts(method="boyle")
            )
        elif method == "rdkit":
            neutralized = self.smi_df[smi_col[0]].p_apply(
                lambda x: CleaningSMILES(
                    x, return_dif=True
                ).neutralizing_salts(method="rdkit")
            )
        else:
            raise ValueError(
                "Neutralizing method must be either 'boyle' or 'rdkit'"
            )
        post_neutralized_smi_data = pd.DataFrame(
            list(neutralized.p_apply(lambda x: x[0])),
            columns=["neutralized_smiles"],
        )
        diff_after_neutralize = pd.DataFrame(
            list(neutralized.p_apply(lambda x: x[1])), columns=["diff"]
        )
        post_smi_df = pd.concat(
            [
                post_neutralized_smi_data.reset_index(drop=True),
                self.smi_df.reset_index(drop=True),
                diff_after_neutralize.reset_index(drop=True),
            ],
            axis=1,
        )
        post_smi_df = post_smi_df[pd.notna(post_smi_df["diff"])]
        post_smi_df.drop(columns=post_smi_df.columns[[1, -1]], inplace=True)
        post_smi_df.rename(
            columns={post_smi_df.columns[0]: "smiles"}, inplace=True
        )

        unprocessable_cnt = len(self.smi_df) - len(post_smi_df)

        format_data = {
            "neutralization_input": len(self.smi_df),
            "neutralized": len(
                diff_after_neutralize[diff_after_neutralize["diff"] == 1]
            ),
            "neutralize_unprocessable": unprocessable_cnt,
            "neutralization_output": len(post_smi_df),
        }

        with open(
            os.path.join(template_dir, "cleaning_title.txt"), "r"
        ) as cleaning_title:
            template_report = cleaning_title.read()

        if validate:
            format_data.update(validation_format_data)
            with open(
                os.path.join(template_dir, "validity_check.txt"), "r"
            ) as validity_check:
                template_report += validity_check.read()

        with open(
            os.path.join(template_dir, "neutralization.txt"), "r"
        ) as neutralization:
            template_report += neutralization.read()

        dup_idx_data = pd.DataFrame()
        if param_deduplicate:
            post_smi_df, dup_idx_data, deduplicate_format_data = deduplicate(
                post_smi_df,
                print_logs=False,
                show_dup_smi_and_idx=True,
                return_format_data=True,
                n_cpu=n_cpu,
                split_factor=split_factor,
                partial_dup_cols=partial_dup_cols,
            )
            format_data.update(deduplicate_format_data)
            with open(
                os.path.join(template_dir, "deduplicate.txt"), "r"
            ) as deduplicate_txt:
                template_report += deduplicate_txt.read()
            post_smi_df = post_smi_df.reset_index(drop=True)

        with open(os.path.join(template_dir, "end.txt"), "r") as end:
            template_report += end.read()

        formatted_report = template_report.format(**format_data)

        if print_logs:
            print(formatted_report)

        if get_output:
            (
                GetReport(
                    output_dir=output_dir, report_subdir_name="neutralization"
                ).create_csv_file(
                    post_smi_df, csv_file_name="post_neutralized_smiles.csv"
                )
            )

        if get_report:
            (
                GetReport(
                    output_dir=output_dir, report_subdir_name="neutralization"
                ).create_report_file(
                    report_file_name="neutralizing_smiles_report.txt",
                    content=formatted_report,
                )
            )
            if param_deduplicate:
                (
                    GetReport(
                        output_dir=output_dir,
                        report_subdir_name="neutralization",
                    ).create_csv_file(
                        dup_idx_data, csv_file_name="duplicate_index_data.csv"
                    )
                )

        if get_diff and return_format_data:
            return post_smi_df, diff_after_neutralize, format_data
        elif get_diff and not return_format_data:
            return post_smi_df, diff_after_neutralize
        elif return_format_data and not get_diff:
            return post_smi_df, format_data
        elif not get_diff and not return_format_data:
            return post_smi_df

    def complete_cleaning(
        self,
        validate: bool = True,
        neutralizing_method: str = "boyle",
        output_dir: str = None,
        print_logs: bool = True,
        get_report: bool = False,
        # get_output: bool = True,
        param_deduplicate: bool = False,
        n_cpu: int = None,
        split_factor: int = 1,
        partial_dup_cols: list = None
    ):
        from curation.utils import GetReport, deduplicate
        from curation.validate import ValidationStage

        if output_dir is None:
            get_output = False
        else:
            get_output = True

        format_data = {}

        with open(
            os.path.join(template_dir, "cleaning_title.txt"), "r"
        ) as cleaning_title:
            template_report = cleaning_title.read()

        if validate:
            self.smi_df, validation_format_data = ValidationStage(
                self.smi_df
            ).validate_smi(
                print_logs=False,
                return_format_data=True,
                n_cpu=n_cpu,
                split_factor=split_factor,
            )
            format_data.update(validation_format_data)
            with open(
                os.path.join(template_dir, "validity_check.txt"), "r"
            ) as validity_check:
                template_report += validity_check.read()

        post_salts_cl_smi_data, diff_after_cl_salt, cl_salt_format_data = (
            self.cl_salt(
                validate=False,
                print_logs=False,
                get_diff=True,
                return_format_data=True,
                n_cpu=n_cpu,
                split_factor=split_factor,
            )
        )
        format_data.update(cl_salt_format_data)
        with open(
            os.path.join(template_dir, "salt_cleaning.txt"), "r"
        ) as salt_cleaning:
            template_report += salt_cleaning.read()

        (
            post_neutralized_smi_data,
            diff_after_neutralizing,
            neutralizing_format_data,
        ) = CleaningStage(post_salts_cl_smi_data).neutralize(
            validate=False,
            method=neutralizing_method,
            print_logs=False,
            get_diff=True,
            return_format_data=True,
            n_cpu=n_cpu,
            split_factor=split_factor,
        )
        format_data.update(neutralizing_format_data)
        with open(
            os.path.join(template_dir, "neutralization.txt"), "r"
        ) as neutralization:
            template_report += neutralization.read()

        dup_idx_data = pd.DataFrame()
        if param_deduplicate:
            (
                post_neutralized_smi_data,
                dup_idx_data,
                deduplicate_format_data,
            ) = deduplicate(
                post_neutralized_smi_data,
                validate=False,
                print_logs=False,
                show_dup_smi_and_idx=True,
                return_format_data=True,
                n_cpu=n_cpu,
                split_factor=split_factor,
                partial_dup_cols=partial_dup_cols,
            )
            format_data.update(deduplicate_format_data)
            with open(
                os.path.join(template_dir, "deduplicate.txt"), "r"
            ) as deduplicate_txt:
                template_report += deduplicate_txt.read()
            post_neutralized_smi_data = post_neutralized_smi_data.reset_index(
                drop=True
            )

        with open(os.path.join(template_dir, "end.txt"), "r") as end:
            template_report += end.read()

        formatted_report = template_report.format(**format_data)

        if print_logs:
            print(formatted_report)

        if get_output:
            (
                GetReport(
                    output_dir=output_dir,
                    report_subdir_name="cleaning_salts_and_neutralizing_smiles",
                ).create_csv_file(
                    post_neutralized_smi_data,
                    csv_file_name="post_cleaned_and_neutralized_smiles.csv",
                )
            )

        if get_report:
            (
                GetReport(
                    output_dir=output_dir,
                    report_subdir_name="cleaning_salts_and_neutralizing_smiles",
                ).create_report_file(
                    report_file_name="cleaning_salts_and_neutralizing_smiles_report.txt",
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

        return post_neutralized_smi_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This module is used to clean salts and neutralize SMILES data",
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
        choices=range(1, 4),
        help="Choose the action:\n"
        "1. Only clean salts in data\n"
        "2. Only neutralize SMILES in data\n"
        "3. Clean and neutralize SMILES in data\n",
    )
    parser.add_argument(
        "--validate_beginning",
        required=False,
        action="store_true",
        default=False,
        help="Validate SMILES data before doing anything "
             "(optional, default is false)",
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
        default=None,
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
    cleaning = CleaningStage(smi_df)

    param_dict = {
        "output_dir": args.output,
        "print_logs": args.print_logs,
        "get_report": args.get_report,
        "validate": args.validate_beginning,
        "param_deduplicate": args.deduplicate,
        "n_cpu": args.n_cpu,
        "split_factor": args.split_factor,
    }

    match args.choice:
        case 1:
            output = cleaning.cl_salt(**param_dict)
        case 2:
            output = cleaning.neutralize(**param_dict)
        case 3:
            output = cleaning.complete_cleaning(**param_dict)
    print("Your action is done!")

