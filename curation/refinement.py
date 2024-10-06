import os
import sys
import pandas as pd
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class Refinement:
    def __init__(self, smi_df: pd.DataFrame):
        self.smi_df = smi_df

    def refine_smiles(self,
                      output_dir: str = None,
                      validate: bool = True,
                      rm_mixtures: bool = True,
                      rm_inorganics: bool = True,
                      rm_organometallics: bool = True,
                      cl_salts: bool = True,
                      neutralize: bool = True,
                      destereoisomerize: bool = True,
                      detautomerize: bool = True,
                      rm_dup_between_stages: bool = True,
                      print_logs: bool = True,
                      get_report: bool = False,
                      get_output: bool = True,
                      n_cpu: int = 1,
                      split_factor: int = 1):
        """
        Note: 1st stage = Validation stage
              2nd stage = Cleaning stage
              3rd stage = Normalizing stage
        """
        from curation.utils import GetReport, deduplicate
        from curation.validate import ValidationStage
        from curation.cleaning import CleaningStage
        from curation.normalization import NormalizingStage

        pre_refined_smiles = self.smi_df.copy()
        rm_dup_after_1st_stage: bool = (validate or rm_mixtures
                                        or rm_inorganics or rm_organometallics)
        rm_dup_after_2nd_stage: bool = (cl_salts or neutralize)
        rm_dup_after_3rd_stage: bool = (destereoisomerize or detautomerize)
        data_post_1st_stage_rm_dup = pd.DataFrame()
        data_post_2nd_stage_rm_dup = pd.DataFrame()
        data_post_3rd_stage_rm_dup = pd.DataFrame()
        format_data = {}

        with open("./curation/template_report/validation_title.txt", "r") as validation_title:
            template_report = validation_title.read()

        if validate:
            self.smi_df, validate_format_data = (ValidationStage(self.smi_df)
                                                 .validate_smi(get_output=False,
                                                               return_format_data=True,
                                                               print_logs=False,
                                                               param_deduplicate=False,
                                                               n_cpu=n_cpu,
                                                               split_factor=split_factor))
            format_data.update(validate_format_data)
            with open("./curation/template_report/validity_check.txt", "r") as validity_check:
                template_report += validity_check.read()

        if rm_mixtures:
            self.smi_df, rm_mixtures_format_data = (ValidationStage(self.smi_df)
                                                    .rm_mixtures(validate=False,
                                                                 get_output=False,
                                                                 print_logs=False,
                                                                 return_format_data=True,
                                                                 param_deduplicate=False,
                                                                 n_cpu=n_cpu,
                                                                 split_factor=split_factor))
            format_data.update(rm_mixtures_format_data)
            with open("./curation/template_report/mixture_removal.txt", "r") as mixture_removal:
                template_report += mixture_removal.read()

        if rm_inorganics:
            self.smi_df, rm_inorganics_format_data = (ValidationStage(self.smi_df)
                                                      .rm_inorganics(validate=False,
                                                                     get_output=False,
                                                                     print_logs=False,
                                                                     return_format_data=True,
                                                                     param_deduplicate=False,
                                                                     n_cpu=n_cpu,
                                                                     split_factor=split_factor))
            format_data.update(rm_inorganics_format_data)
            with open("./curation/template_report/inorganic_removal.txt", "r") as inorganic_removal:
                template_report += inorganic_removal.read()

        if rm_organometallics:
            self.smi_df, rm_organometallics_format_data = (ValidationStage(self.smi_df)
                                                           .rm_organometallics(validate=False,
                                                                               get_output=False,
                                                                               print_logs=False,
                                                                               return_format_data=True,
                                                                               param_deduplicate=False,
                                                                               n_cpu=n_cpu,
                                                                               split_factor=split_factor))
            format_data.update(rm_organometallics_format_data)
            with open("./curation/template_report/organometallic_removal.txt", "r") as organometallic_removal:
                template_report += organometallic_removal.read()

        if rm_dup_between_stages and rm_dup_after_1st_stage:
            self.smi_df, data_post_1st_stage_rm_dup, rm_dup_1st_format_data = (
                deduplicate(self.smi_df,
                            validate=False,
                            get_output=False,
                            print_logs=False,
                            show_dup_smi_and_idx=True,
                            return_format_data=True,
                            n_cpu=n_cpu,
                            split_factor=split_factor))
            with open("./curation/template_report/deduplicate.txt", "r") as deduplicate_txt:
                first_dedup_template_report = deduplicate_txt.read()
            third_dedup_formatted_report = first_dedup_template_report.format(**rm_dup_1st_format_data)
            template_report += third_dedup_formatted_report
            self.smi_df = self.smi_df.reset_index(drop=True)

        with open("./curation/template_report/cleaning_title.txt", "r") as cleaning_title:
            template_report += cleaning_title.read()

        if cl_salts:
            self.smi_df, cl_salt_format_data = CleaningStage(self.smi_df).cl_salts(validate=False,
                                                                                   print_logs=False,
                                                                                   get_output=False,
                                                                                   return_format_data=True,
                                                                                   n_cpu=n_cpu,
                                                                                   split_factor=split_factor)
            format_data.update(cl_salt_format_data)
            with open("./curation/template_report/salt_cleaning.txt", "r") as salt_cleaning:
                template_report += salt_cleaning.read()

        if neutralize:
            self.smi_df, neutralizing_format_data = CleaningStage(self.smi_df).neutralize(validate=False,
                                                                                          print_logs=False,
                                                                                          get_output=False,
                                                                                          return_format_data=True,
                                                                                          n_cpu=n_cpu,
                                                                                          split_factor=split_factor)
            format_data.update(neutralizing_format_data)
            with open("./curation/template_report/neutralization.txt", "r") as neutralization:
                template_report += neutralization.read()
            # self.smi_df = self.smi_df.iloc[:, 0]

        if rm_dup_between_stages and rm_dup_after_2nd_stage:
            self.smi_df, data_post_2nd_stage_rm_dup, rm_dup_2nd_format_data = (
                deduplicate(self.smi_df,
                            validate=False,
                            get_output=False,
                            print_logs=False,
                            show_dup_smi_and_idx=True,
                            return_format_data=True,
                            n_cpu=n_cpu,
                            split_factor=split_factor))
            with open("./curation/template_report/deduplicate.txt", "r") as deduplicate_txt:
                third_dedup_template_report = deduplicate_txt.read()
            third_dedup_formatted_report = third_dedup_template_report.format(**rm_dup_2nd_format_data)
            template_report += third_dedup_formatted_report
            self.smi_df = self.smi_df.reset_index(drop=True)

        with open("./curation/template_report/normalization_title.txt", "r") as normalization_title:
            template_report += normalization_title.read()

        if destereoisomerize:
            self.smi_df, destereoisomerized_format_data = (NormalizingStage(self.smi_df)
                                                           .destereoisomerize(validate=False,
                                                                              get_output=False,
                                                                              print_logs=False,
                                                                              return_format_data=True,
                                                                              n_cpu=n_cpu,
                                                                              split_factor=split_factor))
            format_data.update(destereoisomerized_format_data)
            with open("./curation/template_report/destereoisomerization.txt", 'r') as destereoisomerization:
                template_report += destereoisomerization.read()

        if detautomerize:
            self.smi_df, detautomerized_format_data = (NormalizingStage(self.smi_df)
                                                       .detautomerize(validate=False,
                                                                      get_output=False,
                                                                      print_logs=False,
                                                                      return_format_data=True,
                                                                      n_cpu=n_cpu,
                                                                      split_factor=split_factor))
            format_data.update(detautomerized_format_data)
            with open("./curation/template_report/detautomerization.txt", "r") as detautomerization:
                template_report += detautomerization.read()
            # self.smi_df = self.smi_df.iloc[:, 0]

        if rm_dup_between_stages and rm_dup_after_3rd_stage:
            self.smi_df, data_post_3rd_stage_rm_dup, rm_dup_3rd_format_data = (
                deduplicate(self.smi_df,
                            validate=False,
                            get_output=False,
                            print_logs=False,
                            show_dup_smi_and_idx=True,
                            return_format_data=True,
                            n_cpu=n_cpu,
                            split_factor=split_factor))
            with open("./curation/template_report/deduplicate.txt", "r") as deduplicate_txt:
                third_dedup_template_report = deduplicate_txt.read()
            third_dedup_formatted_report = third_dedup_template_report.format(**rm_dup_3rd_format_data)
            template_report += third_dedup_formatted_report
            self.smi_df = self.smi_df.reset_index(drop=True)

        with open("./curation/template_report/end.txt", "r") as end:
            template_report += end.read()

        formatted_report = template_report.format(**format_data)

        if print_logs:
            print(formatted_report)

        if get_output:
            (GetReport(output_dir=output_dir,
                       report_subdir_name='refinement')
             .create_csv_file(self.smi_df,
                              csv_file_name='post_refined_smiles.csv'))
        if get_report:
            (GetReport(output_dir=output_dir,
                       report_subdir_name='refinement')
             .create_report_file(report_file_name='refinement_report.txt',
                                 content=formatted_report))
            (GetReport(output_dir=output_dir,
                       report_subdir_name='refinement')
             .create_csv_file(data_post_1st_stage_rm_dup,
                              csv_file_name='remove_dupls_1st_data.csv'))
            (GetReport(output_dir=output_dir,
                       report_subdir_name='refinement')
             .create_csv_file(data_post_2nd_stage_rm_dup,
                              csv_file_name='remove_dupls_2nd_data.csv'))
            (GetReport(output_dir=output_dir,
                       report_subdir_name='refinement')
             .create_csv_file(data_post_3rd_stage_rm_dup,
                              csv_file_name='remove_dupls_3rd_data.csv'))
        return self.smi_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This module is used to normalize SMILES data')
    parser.add_argument('-i', '--input', required=True, type=str, help='Input SMILES csv file')
    parser.add_argument('-o', '--output', required=True, type=str,
                        help='Output folder, include csv and report (if have any)')
    parser.add_argument('--validate', required=False, action='store_true', default=True,
                        help='Validate SMILES data before doing anything (optional, default is false)')
    parser.add_argument('--rm_mixtures', required=False, action='store_false', default=True,
                        help='Remove mixtures from SMILES data (optional, default is true)')
    parser.add_argument('--rm_inorganics', required=False, action='store_false', default=True,
                        help='Remove inorganic molecules from SMILES data (optional, default is true)')
    parser.add_argument('--rm_organometallics', required=False, action='store_false', default=True,
                        help='Remove organometallics from SMILES data (optional, default is true)')
    parser.add_argument('--cl_salts', required=False, action='store_false', default=True,
                        help='Remove salts from SMILES data (optional, default is true)')
    parser.add_argument('--neutralize', required=False, action='store_false', default=True,
                        help='Neutralize charged molecules from SMILES data (optional, default is true)')
    parser.add_argument('--destereoisomerize', required=False, action='store_false', default=True,
                        help='Destereoisomerization from SMILES data (optional, default is true)')
    parser.add_argument('--detautomerize', required=False, action='store_false', default=True,
                        help='Detautomerization from SMILES data (optional, default is true)')
    parser.add_argument('--deduplicate', required=False, action='store_false', default=True,
                        help='Remove duplicate SMILES after each stage (optional, default is True)')
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
    refining = Refinement(smi_df)

    param_dict = {
        'validate': args.validate,
        'rm_mixtures': args.rm_mixtures,
        'rm_inorganics': args.rm_inorganics,
        'rm_organometallics': args.rm_organometallics,
        'cl_salts': args.cl_salts,
        'neutralize': args.neutralize,
        'destereoisomerize': args.destereoisomerize,
        'detautomerize': args.detautomerize,
        'rm_dup_between_stages': args.deduplicate,
        'output_dir': args.output,
        'print_logs': args.print_logs,
        'get_report': args.get_report,
        'n_cpu': args.n_cpu,
        'split_factor': args.split_factor
    }

    output = refining.refine_smiles(**param_dict)
    print('Your action is done!')
