"""Refinement pipeline - orchestrates validation, cleaning, and normalization stages."""
import pandas as pd

from .template_manager import RefinementTemplateManager
from .output_manager import RefinementOutputManager
from ..config.refinement_config import RefinementConfig


class RefinementPipeline:
    """
    Main refinement pipeline - orchestrates all three stages (validation, cleaning, normalization).
    
    Note: 1st stage = Validation stage
          2nd stage = Cleaning stage
          3rd stage = Normalization stage
    """
    
    def __init__(self, smi_df: pd.DataFrame):
        self.smi_df = smi_df
        self.config = RefinementConfig()
        self.template_manager = RefinementTemplateManager(self.config.template_dir)
    
    def complete_refinement(
        self,
        output_dir: str = None,
        validate: bool = True,
        rm_mixture: bool = True,
        rm_inorganic: bool = True,
        rm_organometallic: bool = True,
        cl_salt: bool = True,
        neutralize: bool = True,
        neutralizing_method: str = "boyle",
        validate_post_neutr: bool = True,
        destereoisomerize: bool = True,
        detautomerize: bool = True,
        rm_dup_between_stages: bool = True,
        print_logs: bool = True,
        get_report: bool = False,
        n_cpu: int = -1,
        split_factor: int = 1,
        partial_dup_cols: list = None
    ) -> pd.DataFrame:
        """
        Complete refinement pipeline - orchestrates validation, cleaning, and normalization stages.
        
        Args:
            output_dir: Directory for output files
            validate: Validate SMILES at the beginning
            rm_mixture: Remove mixtures
            rm_inorganic: Remove inorganics
            rm_organometallic: Remove organometallics
            cl_salt: Clean salts
            neutralize: Neutralize charged molecules
            neutralizing_method: Method for neutralization ("boyle" or "rdkit")
            validate_post_neutr: Validate after neutralization
            destereoisomerize: Remove stereoisomers
            detautomerize: Normalize tautomers
            rm_dup_between_stages: Remove duplicates between stages
            print_logs: Print logs to console
            get_report: Generate detailed reports
            n_cpu: Number of CPUs for parallel processing
            split_factor: Split factor for parallel processing
            partial_dup_cols: Columns to consider for partial duplicates
            
        Returns:
            pd.DataFrame: Refined SMILES DataFrame
        """
        from ...utils import deduplicate
        from ...validation import ValidationStage
        from ...cleaning import CleaningStage
        from ...normalization import NormalizationStage
        
        # Determine when to deduplicate
        rm_dup_after_1st_stage: bool = (
            validate
            or rm_mixture
            or rm_inorganic
            or rm_organometallic
        )
        rm_dup_after_2nd_stage: bool = cl_salt or neutralize
        rm_dup_after_3rd_stage: bool = destereoisomerize or detautomerize
        
        # Initialize data containers
        data_post_1st_stage_rm_dup = pd.DataFrame()
        data_post_2nd_stage_rm_dup = pd.DataFrame()
        data_post_3rd_stage_rm_dup = pd.DataFrame()
        format_data = {}
        
        # Build template report
        template_report = self.template_manager.build_refinement_template(
            validate=validate,
            rm_mixture=rm_mixture,
            rm_inorganic=rm_inorganic,
            rm_organometallic=rm_organometallic,
            cl_salt=cl_salt,
            neutralize=neutralize,
            destereoisomerize=destereoisomerize,
            detautomerize=detautomerize,
            rm_dup_after_1st=rm_dup_between_stages and rm_dup_after_1st_stage,
            rm_dup_after_2nd=rm_dup_between_stages and rm_dup_after_2nd_stage,
            rm_dup_after_3rd=rm_dup_between_stages and rm_dup_after_3rd_stage,
        )
        
        # ========== STAGE 1: VALIDATION ==========
        if validate:
            self.smi_df, validate_beginning_format_data = ValidationStage(
                self.smi_df
            ).validate_smi(
                return_format_data=True,
                print_logs=False,
                param_deduplicate=False,
                n_cpu=n_cpu,
                split_factor=split_factor,
            )
            format_data.update(validate_beginning_format_data)
        
        if rm_mixture:
            self.smi_df, rm_mixture_format_data = ValidationStage(
                self.smi_df
            ).rm_mixture(
                validate=False,
                print_logs=False,
                return_format_data=True,
                param_deduplicate=False,
                n_cpu=n_cpu,
                split_factor=split_factor,
            )
            format_data.update(rm_mixture_format_data)
        
        if rm_inorganic:
            self.smi_df, rm_inorganic_format_data = ValidationStage(
                self.smi_df
            ).rm_inorganic(
                validate=False,
                print_logs=False,
                return_format_data=True,
                param_deduplicate=False,
                n_cpu=n_cpu,
                split_factor=split_factor,
            )
            format_data.update(rm_inorganic_format_data)
        
        if rm_organometallic:
            self.smi_df, rm_organometallic_format_data = ValidationStage(
                self.smi_df
            ).rm_organometallic(
                validate=False,
                print_logs=False,
                return_format_data=True,
                param_deduplicate=False,
                n_cpu=n_cpu,
                split_factor=split_factor,
            )
            format_data.update(rm_organometallic_format_data)
        
        # Deduplicate after stage 1
        if rm_dup_between_stages and rm_dup_after_1st_stage:
            self.smi_df, data_post_1st_stage_rm_dup, rm_dup_1st_format_data = (
                deduplicate(
                    self.smi_df,
                    validate=False,
                    print_logs=False,
                    show_dup_smi_and_idx=True,
                    return_format_data=True,
                    n_cpu=n_cpu,
                    split_factor=split_factor,
                    partial_dup_cols=partial_dup_cols,
                )
            )
            format_data.update(rm_dup_1st_format_data)
            self.smi_df = self.smi_df.reset_index(drop=True)
        
        # ========== STAGE 2: CLEANING ==========
        if cl_salt:
            self.smi_df, cl_salt_format_data = CleaningStage(
                self.smi_df
            ).cl_salt(
                validate=False,
                print_logs=False,
                return_format_data=True,
                n_cpu=n_cpu,
                split_factor=split_factor,
            )
            format_data.update(cl_salt_format_data)
        
        if neutralize:
            self.smi_df, neutralizing_format_data = CleaningStage(
                self.smi_df
            ).neutralize(
                validate=False,
                method=neutralizing_method,
                print_logs=False,
                return_format_data=True,
                n_cpu=n_cpu,
                split_factor=split_factor,
            )
            format_data.update(neutralizing_format_data)
        
        # Deduplicate after stage 2
        if rm_dup_between_stages and rm_dup_after_2nd_stage:
            self.smi_df, data_post_2nd_stage_rm_dup, rm_dup_2nd_format_data = (
                deduplicate(
                    self.smi_df,
                    validate=False,
                    print_logs=False,
                    show_dup_smi_and_idx=True,
                    return_format_data=True,
                    n_cpu=n_cpu,
                    split_factor=split_factor,
                    partial_dup_cols=partial_dup_cols,
                )
            )
            format_data.update(rm_dup_2nd_format_data)
            self.smi_df = self.smi_df.reset_index(drop=True)
        
        # ========== STAGE 3: NORMALIZATION ==========
        if validate_post_neutr:
            self.smi_df, _ = ValidationStage(
                self.smi_df
            ).validate_smi(
                return_format_data=True,
                print_logs=False,
                param_deduplicate=False,
                n_cpu=n_cpu,
                split_factor=split_factor,
            )
            # Note: format_data from post-neutralization validation is intentionally
            # not added to format_data as it's not part of the template
        
        if destereoisomerize:
            self.smi_df, destereoisomerized_format_data = NormalizationStage(
                self.smi_df
            ).destereoisomerize(
                validate=False,
                print_logs=False,
                return_format_data=True,
                n_cpu=n_cpu,
                split_factor=split_factor,
            )
            format_data.update(destereoisomerized_format_data)
        
        if detautomerize:
            self.smi_df, detautomerized_format_data = NormalizationStage(
                self.smi_df
            ).detautomerize(
                validate=False,
                print_logs=False,
                return_format_data=True,
                n_cpu=n_cpu,
                split_factor=split_factor,
            )
            format_data.update(detautomerized_format_data)
        
        # Deduplicate after stage 3
        if rm_dup_between_stages and rm_dup_after_3rd_stage:
            self.smi_df, data_post_3rd_stage_rm_dup, rm_dup_3rd_format_data = (
                deduplicate(
                    self.smi_df,
                    validate=False,
                    print_logs=False,
                    show_dup_smi_and_idx=True,
                    return_format_data=True,
                    n_cpu=n_cpu,
                    split_factor=split_factor,
                    partial_dup_cols=partial_dup_cols,
                )
            )
            format_data.update(rm_dup_3rd_format_data)
            self.smi_df = self.smi_df.reset_index(drop=True)
        
        # Format the final report
        formatted_report = self.template_manager.format_template(
            template_report, format_data
        )
        
        # Print logs if requested
        if print_logs:
            print(formatted_report)
        
        # Save outputs
        output_manager = RefinementOutputManager(output_dir, self.config)
        output_manager.save_refinement_outputs(
            self.smi_df,
            formatted_report,
            data_post_1st_stage_rm_dup,
            data_post_2nd_stage_rm_dup,
            data_post_3rd_stage_rm_dup,
            get_report,
        )
        
        return self.smi_df

