import pandas as pd
from curation.utils import GetReport, remove_duplicates
from curation.validate import ValidationStage
from curation.cleaning import CleaningStage
from curation.normalization import NormalizingStage


class Refinement:
    def __init__(self, smiles_df: pd.DataFrame):
        self.smiles_df = smiles_df

    def refine_smiles(self,
                      output_dir: str = None,
                      check_validity: bool = True,
                      remove_mixtures: bool = True,
                      remove_inorganics: bool = True,
                      remove_organometallics: bool = True,
                      clean_salts: bool = True,
                      neutralize: bool = True,
                      normalize_stereoisomers: bool = True,
                      normalize_tautomers: bool = True,
                      remove_dupls_between_stages: bool = True,
                      print_logs: bool = True,
                      get_report: bool = True,
                      get_csv: bool = True):
        pre_refined_smiles = self.smiles_df.copy()
        remove_dupls_after_1st_stage: bool = (check_validity or remove_mixtures
                                              or remove_inorganics or remove_organometallics)
        remove_dupls_after_2nd_stage: bool = (clean_salts or neutralize)
        remove_dupls_after_3rd_stage: bool = (normalize_stereoisomers or normalize_tautomers)
        contents: str = ''
        if check_validity:
            self.smiles_df, check_valid_contents = (ValidationStage(self.smiles_df)
                                                    .check_valid_smiles(get_csv=False,
                                                                        return_contents=True,
                                                                        print_logs=False))
            check_valid_contents = '\n'.join([
                'VALIDATION STEP:',
                check_valid_contents,
                '----------'
            ])
            if print_logs:
                print(check_valid_contents)
            contents += check_valid_contents
        if remove_mixtures:
            self.smiles_df, remove_mixtures_contents = (ValidationStage(self.smiles_df)
                                                        .remove_mixtures(check_validity=False,
                                                                         get_csv=False,
                                                                         print_logs=False,
                                                                         return_contents=True))
            remove_mixtures_contents = '\n'.join([
                'MIXTURES REMOVING STEP:',
                remove_mixtures_contents,
                '----------'
            ])
            if print_logs:
                print(remove_mixtures_contents)
            contents += remove_mixtures_contents
        if remove_inorganics:
            self.smiles_df, remove_inorganics_contents = (ValidationStage(self.smiles_df)
                                                          .remove_inorganics(check_validity=False,
                                                                             get_csv=False,
                                                                             print_logs=False,
                                                                             return_contents=True))
            remove_inorganics_contents = '\n'.join([
                'INORGANICS REMOVING STEP:',
                remove_inorganics_contents,
                '----------'
            ])
            if print_logs:
                print(remove_inorganics_contents)
            contents += remove_inorganics_contents
        if remove_organometallics:
            self.smiles_df, remove_organometallics_contents = (ValidationStage(self.smiles_df)
                                                               .remove_organometallics(check_validity=False,
                                                                                       get_csv=False,
                                                                                       print_logs=False,
                                                                                       return_contents=True))
            remove_organometallics_contents = '\n'.join([
                'ORGANOMETALLICS REMOVING STEP:',
                remove_organometallics_contents,
                '----------'
            ])
            if print_logs:
                print(remove_organometallics_contents)
            contents += remove_organometallics_contents
        if remove_dupls_between_stages and remove_dupls_after_1st_stage:
            self.smiles_df, remove_dupls_1st_data, remove_dupls_1st_contents = (
                remove_duplicates(self.smiles_df,
                                  check_validity=False,
                                  get_csv=False,
                                  print_logs=False,
                                  show_duplicated_smiles_and_index=True,
                                  return_contents=True))
            remove_dupls_1st_contents = '\n'.join([
                'DUPLICATES REMOVING 1ST TIME (after validation stage):',
                remove_dupls_1st_contents,
                '----------'
            ])
            if print_logs:
                print(remove_dupls_1st_contents)
            contents += remove_dupls_1st_contents
        # self.smiles_df = self.smiles_df.reset_index(drop=True)
        if clean_salts:
            self.smiles_df, clean_salts_contents = CleaningStage(self.smiles_df).clean_salts(check_validity=False,
                                                                                             print_logs=False,
                                                                                             get_csv=False,
                                                                                             return_contents=True)
            clean_salts_contents = '\n'.join([
                'SALTS CLEANING STEP:',
                clean_salts_contents,
                '----------'
            ])
            if print_logs:
                print(clean_salts_contents)
            contents += clean_salts_contents
        if neutralize:
            self.smiles_df, neutralize_contents = CleaningStage(self.smiles_df).neutralize(check_validity=False,
                                                                                           print_logs=False,
                                                                                           get_csv=False,
                                                                                           return_contents=True)
            neutralize_contents = '\n'.join([
                'NEUTRALIZE STEP:',
                neutralize_contents,
                '----------'
            ])
            if print_logs:
                print(neutralize_contents)
            contents += neutralize_contents
        if remove_dupls_between_stages and remove_dupls_after_2nd_stage:
            self.smiles_df, remove_dupls_2nd_data, remove_dupls_2nd_contents = (
                remove_duplicates(self.smiles_df,
                                  check_validity=False,
                                  get_csv=False,
                                  print_logs=False,
                                  show_duplicated_smiles_and_index=True,
                                  return_contents=True))
            remove_dupls_2nd_contents = '\n'.join([
                'DUPLICATES REMOVING 2ND TIME (after cleaning stage):',
                remove_dupls_2nd_contents,
                '----------'
            ])
            if print_logs:
                print(remove_dupls_2nd_contents)
            contents += remove_dupls_2nd_contents
        # self.smiles_df = self.smiles_df.reset_index(drop=True)
        if normalize_stereoisomers:
            self.smiles_df, normalize_stereoisomers_contents = (NormalizingStage(self.smiles_df)
                                                                .normalize_stereoisomer(check_validity=False,
                                                                                        get_csv=False,
                                                                                        print_logs=False,
                                                                                        return_contents=True))
            normalize_stereoisomers_contents = '\n'.join([
                'NORMALIZE STEREOISOMERS STEP:',
                normalize_stereoisomers_contents,
                '----------'
            ])
            if print_logs:
                print(normalize_stereoisomers_contents)
            contents += normalize_stereoisomers_contents
        if normalize_tautomers:
            self.smiles_df, normalize_tautomers_contents = (NormalizingStage(self.smiles_df)
                                                            .normalize_tautomer(check_validity=False,
                                                                                get_csv=False,
                                                                                print_logs=False,
                                                                                return_contents=True))
            normalize_tautomers_contents = '\n'.join([
                'NORMALIZE TAUTOMERS STEP:',
                normalize_tautomers_contents,
                '----------'
            ])
            if print_logs:
                print(normalize_tautomers_contents)
            contents += normalize_tautomers_contents
        if remove_dupls_between_stages and remove_dupls_after_3rd_stage:
            self.smiles_df, remove_dupls_3rd_data, remove_dupls_3rd_contents = (
                remove_duplicates(self.smiles_df,
                                  check_validity=False,
                                  get_csv=False,
                                  print_logs=False,
                                  show_duplicated_smiles_and_index=True,
                                  return_contents=True))
            remove_dupls_3rd_contents = '\n'.join([
                'DUPLICATES REMOVING 3RD TIME (after normalizing stage):',
                remove_dupls_3rd_contents,
                '----------'
            ])
            if print_logs:
                print(remove_dupls_3rd_contents)
            contents += remove_dupls_3rd_contents
        # self.smiles_df = self.smiles_df.reset_index(drop=True)
        return self.smiles_df
