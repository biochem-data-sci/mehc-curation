import os
import sys
import argparse
import pandas as pd
from parallel_pandas import ParallelPandas

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class NormalizingStage:
    def __init__(self, smi_df: pd.DataFrame):
        self.smi_df = smi_df

    def detautomerize(self,
                      validate: bool = True,
                      output_dir: str = None,
                      print_logs: bool = True,
                      get_report: bool = False,
                      get_output: bool = True,
                      get_diff: bool = False,
                      param_deduplicate: bool = False,
                      return_format_data: bool = False,
                      n_cpu: int = 1,
                      split_factor: int = 1):
        """
        Normalize tautomers for the given SMILES dataframe.
        """
        from curation.utils import GetReport, NormalizeSMILES, deduplicate
        from curation.validate import ValidationStage

        ParallelPandas.initialize(n_cpu=n_cpu, split_factor=split_factor, disable_pr_bar=True)
        smi_col = self.smi_df.columns.tolist()
        if validate:
            self.smi_df, validation_format_data = ValidationStage(self.smi_df).validate_smi(get_output=False,
                                                                                            print_logs=False,
                                                                                            return_format_data=True,
                                                                                            n_cpu=n_cpu,
                                                                                            split_factor=split_factor)

        post_detautomerized = self.smi_df[smi_col[0]].p_apply(
            lambda x: NormalizeSMILES(x, return_difference=True)
            .normalize_tautomer())
        post_detautomerized_smi_df = post_detautomerized.p_apply(lambda x: x[0])
        diff_after_detautomerized = post_detautomerized.p_apply(lambda x: x[1])
        post_detautomerized_smi_df = post_detautomerized_smi_df.to_frame()

        format_data = {
            'detautomerization_input': len(self.smi_df),
            'detautomerized': sum(diff_after_detautomerized),
            'detautomerization_output': len(post_detautomerized_smi_df),
        }

        with open("./curation/template_report/normalization_title.txt", 'r') as normalization_title:
            template_report = normalization_title.read()

        if validate:
            format_data.update(validation_format_data)
            with open("./curation/template_report/validity_check.txt", "r") as validity_check:
                template_report += validity_check.read()

        with open("./curation/template_report/detautomerization.txt", "r") as detautomerization:
            template_report += detautomerization.read()

        dup_idx_data = pd.DataFrame()
        if param_deduplicate:
            post_detautomerized_smi_df, dup_idx_data, deduplicate_format_data = (
                deduplicate(post_detautomerized_smi_df,
                            validate=False,
                            get_output=False,
                            print_logs=False,
                            show_dup_smi_and_idx=True,
                            return_format_data=True,
                            n_cpu=n_cpu,
                            split_factor=split_factor))
            format_data.update(deduplicate_format_data)
            with open("./curation/template_report/deduplicate.txt", "r") as deduplicate_txt:
                template_report += deduplicate_txt.read()
            post_detautomerized_smi_df = post_detautomerized_smi_df.reset_index(drop=True)

        with open("./curation/template_report/end.txt", "r") as end:
            template_report += end.read()

        formatted_report = template_report.format(**format_data)

        if print_logs:
            print(formatted_report)

        if get_output:
            (GetReport(output_dir=output_dir,
                       report_subdir_name='detautomerized')
             .create_csv_file(post_detautomerized_smi_df,
                              csv_file_name='post_detautomerized.csv'))

        if get_report:
            (GetReport(output_dir=output_dir,
                       report_subdir_name='detautomerized')
             .create_report_file(report_file_name='detautomerize_report.txt',
                                 content=formatted_report))
            if param_deduplicate:
                (GetReport(output_dir=output_dir,
                           report_subdir_name='cleaning_salts')
                 .create_csv_file(dup_idx_data,
                                  csv_file_name='duplicate_index_data.csv'))

        if get_diff and return_format_data:
            return post_detautomerized_smi_df, diff_after_detautomerized, format_data
        elif get_diff and not return_format_data:
            return post_detautomerized_smi_df, diff_after_detautomerized
        elif return_format_data and not get_diff:
            return post_detautomerized_smi_df, format_data
        else:
            return post_detautomerized_smi_df

    def destereoisomerize(self,
                          validate: bool = True,
                          output_dir: str = None,
                          print_logs: bool = True,
                          get_report: bool = False,
                          get_output: bool = True,
                          get_diff: bool = False,
                          param_deduplicate: bool = False,
                          return_format_data: bool = False,
                          n_cpu: int = 1,
                          split_factor: int = 1):
        """
        Normalize stereoisomers for the given SMILES dataframe.
        """
        from curation.utils import GetReport, NormalizeSMILES, deduplicate
        from curation.validate import ValidationStage

        ParallelPandas.initialize(n_cpu=n_cpu, split_factor=split_factor, disable_pr_bar=True)
        smi_col = self.smi_df.columns.tolist()
        if validate:
            self.smi_df, validation_format_data = ValidationStage(self.smi_df).validate_smi(get_output=False,
                                                                                            print_logs=False,
                                                                                            return_format_data=True,
                                                                                            n_cpu=n_cpu,
                                                                                            split_factor=split_factor)

        post_destereoisomerized = self.smi_df[smi_col[0]].p_apply(
            lambda x: NormalizeSMILES(x, return_difference=True)
            .normalize_stereoisomer())
        post_destereoisomerized_smi_df = post_destereoisomerized.p_apply(lambda x: x[0])
        diff_after_destereoisomerized = post_destereoisomerized.p_apply(lambda x: x[1])
        post_destereoisomerized_smi_df = post_destereoisomerized_smi_df.to_frame()

        format_data = {
            'destereoisomerization_input': len(self.smi_df),
            'destereoisomerized': sum(diff_after_destereoisomerized),
            'destereoisomerization_output': len(post_destereoisomerized_smi_df),
        }

        with open("./curation/template_report/normalization_title.txt", 'r') as normalization_title:
            template_report = normalization_title.read()

        if validate:
            format_data.update(validation_format_data)
            with open("./curation/template_report/validity_check.txt", "r") as validity_check:
                template_report += validity_check.read()

        with open("./curation/template_report/destereoisomerization.txt", "r") as destereoisomerization:
            template_report += destereoisomerization.read()

        dup_idx_data = pd.DataFrame()
        if param_deduplicate:
            post_destereoisomerized_smi_df, dup_idx_data, deduplicate_format_data = (
                deduplicate(post_destereoisomerized_smi_df,
                            validate=False,
                            get_output=False,
                            print_logs=False,
                            show_dup_smi_and_idx=True,
                            return_format_data=True,
                            n_cpu=n_cpu,
                            split_factor=split_factor))
            format_data.update(deduplicate_format_data)
            with open("./curation/template_report/deduplicate.txt", "r") as deduplicate_txt:
                template_report += deduplicate_txt.read()
            post_destereoisomerized_smi_df = post_destereoisomerized_smi_df.reset_index(drop=True)

        with open("./curation/template_report/end.txt", "r") as end:
            template_report += end.read()

        formatted_report = template_report.format(**format_data)

        if print_logs:
            print(formatted_report)

        if get_output:
            (GetReport(output_dir=output_dir,
                       report_subdir_name='normalize_stereoisomer')
             .create_csv_file(post_destereoisomerized_smi_df,
                              csv_file_name='post_stereoisomer_normalized.csv'))

        if get_report:
            (GetReport(output_dir=output_dir,
                       report_subdir_name='normalize_stereoisomer')
             .create_report_file(report_file_name='normalize_stereoisomer_report.txt',
                                 content=formatted_report))
        if get_diff and return_format_data:
            return post_destereoisomerized_smi_df, diff_after_destereoisomerized, format_data
        elif get_diff and not return_format_data:
            return post_destereoisomerized_smi_df, diff_after_destereoisomerized
        elif return_format_data and not get_diff:
            return post_destereoisomerized_smi_df, format_data
        else:
            return post_destereoisomerized_smi_df

    def complete_normalization(self,
                               validate: bool = True,
                               output_dir: str = None,
                               print_logs: bool = True,
                               get_report: bool = False,
                               get_output: bool = True,
                               param_deduplicate: bool = False,
                               n_cpu: int = 1,
                               split_factor: int = 1):
        """
        Normalize both tautomers and stereoisomers for the given SMILES dataframe.
        """
        from curation.utils import GetReport, deduplicate
        from curation.validate import ValidationStage

        format_data = {}

        with open("./curation/template_report/normalization_title.txt", 'r') as normalization_title:
            template_report = normalization_title.read()

        if validate:
            self.smi_df, validation_format_data = ValidationStage(self.smi_df).validate_smi(
                print_logs=False,
                get_output=False,
                return_format_data=True,
                n_cpu=n_cpu,
                split_factor=split_factor)
            format_data.update(validation_format_data)
            with open("./curation/template_report/validity_check.txt", "r") as validity_check:
                template_report += validity_check.read()

        post_destereoisomerized_smi_data, diff_after_destereoisomerized, destereoisomerized_format_data = (
            NormalizingStage(self.smi_df)
            .destereoisomerize(validate=False,
                               print_logs=False,
                               get_output=False,
                               get_diff=True,
                               return_format_data=True,
                               n_cpu=n_cpu,
                               split_factor=split_factor))
        format_data.update(destereoisomerized_format_data)
        with open("./curation/template_report/destereoisomerization.txt", 'r') as destereoisomerization:
            template_report += destereoisomerization.read()

        post_detautomerized_smi_data, diff_after_detautomerized, detautomerized_format_data = (
            NormalizingStage(post_destereoisomerized_smi_data)
            .detautomerize(validate=False,
                           print_logs=False,
                           get_output=False,
                           get_report=False,
                           get_diff=True,
                           return_format_data=True,
                           n_cpu=n_cpu,
                           split_factor=split_factor))
        format_data.update(detautomerized_format_data)
        with open("./curation/template_report/detautomerization.txt", "r") as detautomerization:
            template_report += detautomerization.read()

        dup_idx_data = pd.DataFrame()
        if param_deduplicate:
            post_detautomerized_smi_data, dup_idx_data, deduplicate_format_data = (
                deduplicate(post_detautomerized_smi_data,
                            validate=False,
                            get_output=False,
                            print_logs=False,
                            show_dup_smi_and_idx=True,
                            return_format_data=True,
                            n_cpu=n_cpu,
                            split_factor=split_factor))
            format_data.update(deduplicate_format_data)
            with open("./curation/template_report/deduplicate.txt", "r") as deduplicate_txt:
                template_report += deduplicate_txt.read()
            post_detautomerized_smi_data = post_detautomerized_smi_data.reset_index(drop=True)

        with open("./curation/template_report/end.txt", "r") as end:
            template_report += end.read()

        formatted_report = template_report.format(**format_data)

        if print_logs:
            print(formatted_report)

        if get_output:
            (GetReport(output_dir=output_dir,
                       report_subdir_name='tautomer_and_stereoisomer_normalization')
             .create_csv_file(post_detautomerized_smi_data,
                              csv_file_name='post_normalized_smiles.csv'))

        if get_report:
            (GetReport(output_dir=output_dir,
                       report_subdir_name='tautomer_and_stereoisomer_normalization')
             .create_report_file(report_file_name='tautomer_and_stereoisomer_normalization_report.txt',
                                 content=formatted_report))
            if param_deduplicate:
                (GetReport(output_dir=output_dir,
                           report_subdir_name='complete_validation')
                 .create_csv_file(dup_idx_data,
                                  csv_file_name='duplicate_index_data.csv'))

        return post_detautomerized_smi_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This module is used to normalize SMILES data')
    parser.add_argument('-i', '--input', required=True, type=str, help='Input SMILES csv file')
    parser.add_argument('-o', '--output', required=True, type=str,
                        help='Output folder, include csv and report (if have any)')
    parser.add_argument('-c', '--choice', required=True, type=int, choices=range(1, 4),
                        help='Choose the action:\n'
                             '1. Only normalize stereoisomers\n'
                             '2. Only normalize tautomers\n'
                             '3. Do all the normalization stage\n')
    parser.add_argument('--validate', required=False, action='store_true', default=False,
                        help='Validate SMILES data before doing anything (optional, default is false)')
    parser.add_argument('--deduplicate', required=False, action='store_true', default=False,
                        help='Remove duplicate SMILES after your action (optional, default is false)')
    parser.add_argument('-p', '--print_logs', required=False, action='store_false',
                        help='Print logs (optional, default is true)')
    parser.add_argument('--get_report', required=False, action='store_true',
                        help='Get report (include report file and supported csv for more information) (optional, default is False)')
    parser.add_argument('--n_cpu', required=False, type=int, default=1,
                        help='Number of CPUs to use (optional)')
    parser.add_argument('--split_factor', required=False, type=int, default=1,
                        help='Split factor (optional)')
    args: argparse.Namespace = parser.parse_args()

    smi_df = pd.read_csv(args.input)
    normalizing = NormalizingStage(smi_df)

    param_dict = {
        'output_dir': args.output,
        'print_logs': args.print_logs,
        'get_report': args.get_report,
        'validate': args.validate,
        'param_deduplicate': args.deduplicate,
        'n_cpu': args.n_cpu,
        'split_factor': args.split_factor
    }

    match args.choice:
        case 1:
            output = normalizing.destereoisomerize(**param_dict)
        case 2:
            print('WARNING: You should destereoisomerize SMILES before detautomerize for sure for the good output.')
            output = normalizing.detautomerize(**param_dict)
        case 3:
            output = normalizing.complete_normalization(**param_dict)
    print("Your action is done!")
