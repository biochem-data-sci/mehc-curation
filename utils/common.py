"""Common functions used by multiple modules."""
import os
import pandas as pd
from typing import Dict, Any, Optional, List
from parallel_pandas import ParallelPandas


def deduplicate(
    smi_df: pd.DataFrame,
    validate: bool = False,
    output_dir: str = None,
    print_logs: bool = True,
    get_report: bool = False,
    return_format_data: bool = False,
    show_dup_smi_and_idx: bool = False,
    n_cpu: int = -1,
    split_factor: int = 1,
    partial_dup_cols: list = None
):
    """
    Deduplicate SMILES entries in a DataFrame based on various criteria.
    
    This function is shared across validation, cleaning, and normalization modules.

    Args:
        smi_df (pd.DataFrame): DataFrame containing SMILES strings to deduplicate.
        validate (bool): If True, the SMILES will be validated before deduplication.
        output_dir (str): Directory where reports and output files will be saved.
        print_logs (bool): If True, log messages will be printed to console.
        get_report (bool): If True, generates a report of the deduplication process.
        return_format_data (bool): If True, returns the format data statistics.
        show_dup_smi_and_idx (bool): If True, returns the duplicated SMILES and their indices.
        n_cpu (int): Number of CPUs to use for parallel processing (-1 for all CPUs).
        split_factor (int): The factor for splitting DataFrame during parallel processing.
        partial_dup_cols (list): Specific columns to consider for partial duplicates.

    Returns:
        pd.DataFrame: Cleaned DataFrame with duplicates removed, and optionally other outputs based on flags.
    """
    
    # Initialize parallel processing
    # Get available CPUs
    available_cpus = os.cpu_count() or 1
    
    # Validate and adjust n_cpu
    if n_cpu == -1:
        # Use all available CPUs
        n_cpu = available_cpus
    elif n_cpu == 0:
        # Set to 1 if user specified 0
        n_cpu = 1
    elif n_cpu < 0:
        # Negative numbers (except -1) set to 1
        n_cpu = 1
    elif n_cpu > available_cpus:
        # Cap at maximum available CPUs
        n_cpu = available_cpus
    
    ParallelPandas.initialize(n_cpu=n_cpu, split_factor=split_factor, disable_pr_bar=True)
    
    get_output = output_dir is not None
    
    # Validate if requested (avoid circular import)
    validate_format_data = {}
    if validate:
        try:
            # Import inside function to avoid circular dependency

            from ..validation import ValidationStage
            smi_df, validate_format_data = ValidationStage(smi_df).validate_smi(
                get_output=False,
                print_logs=False,
                param_deduplicate=False,
                return_format_data=True,
                n_cpu=n_cpu,
                split_factor=split_factor,
            )
        except ImportError:
            if print_logs:
                print("Warning: Validation module not available, skipping validation.")
    
    # Get the column names from the DataFrame
    smi_col = smi_df.columns.tolist()
    
    # Identify perfect duplicates
    perfect_dups_mask = smi_df.duplicated(keep="first")
    perfect_dups = smi_df[perfect_dups_mask]
    
    # Use specified columns for detecting partial duplicates, defaulting if necessary
    if partial_dup_cols is None:
        partial_dup_cols = [smi_col[0]]  # Default to the first column
    
    # Group by specified columns to identify partial duplicates
    grouped = smi_df.groupby(partial_dup_cols).nunique()
    partial_dups_mask = grouped.gt(1).any(axis=1)
    partial_dups_index = partial_dups_mask[partial_dups_mask].index
    
    # Handle single vs multiple columns for partial duplicates
    if len(partial_dup_cols) == 1:
        # Single column: index contains the duplicate values, use directly with .isin()
        partial_dups = smi_df[smi_df[partial_dup_cols[0]].isin(partial_dups_index)]
        # Remove duplicates from the original DataFrame
        cleaned_df = smi_df[~smi_df[partial_dup_cols[0]].isin(partial_dups_index)].drop_duplicates()
    else:
        # Multiple columns: index is MultiIndex (tuples)
        # Create tuples from the DataFrame columns for comparison
        row_tuples = smi_df[partial_dup_cols].apply(tuple, axis=1)
        partial_dups = smi_df[row_tuples.isin(partial_dups_index)]
        # Remove duplicates: need to check if tuple of columns is in the index
        cleaned_df = smi_df[~row_tuples.isin(partial_dups_index)].drop_duplicates()
    
    # Compile duplicate information
    dups_info = (
        pd.concat([perfect_dups, partial_dups], ignore_index=True)
        .drop_duplicates(subset=[smi_col[0]])
        .copy()
    )
    dups_info["is_perfect_dups"] = dups_info[smi_col[0]].isin(perfect_dups[smi_col[0]])
    
    # Create index mapping for duplicates
    index_mapping = (
        smi_df.groupby(smi_col[0])
        .p_apply(lambda x: list(x.index))
        .reset_index(name="indexes")
    )
    # Merge and drop duplicates - only consider SMILES column, not the indexes column (which contains lists)
    dups_info = dups_info.merge(index_mapping, on=smi_col[0], how="left").drop_duplicates(subset=[smi_col[0]])
    
    # Prepare format data for reporting
    format_data = {
        "duplicate_validation_input": len(smi_df),
        "perfect_dups": len(dups_info[dups_info["is_perfect_dups"]]),
        "partial_dups": len(dups_info[dups_info["is_perfect_dups"] == False]),
        "validation_unique": len(cleaned_df),
    }
    
    # Generate reports
    template_report = ""
    if validate and validate_format_data:
        format_data.update(validate_format_data)
        
    try:
        from .io_utils import setup_template_dir
        template_dir = setup_template_dir()
        
        if validate:
            try:
                with open(os.path.join(template_dir, "validity_check.txt"), "r") as f:
                    template_report += f.read()
            except FileNotFoundError:
                pass
                
        with open(os.path.join(template_dir, "deduplicate.txt"), "r") as f:
            template_report += f.read()
        with open(os.path.join(template_dir, "end.txt"), "r") as f:
            template_report += f.read()
    except (ImportError, FileNotFoundError):
        # Fallback report
        template_report = f"""
Deduplication Report
===================
Input SMILES: {format_data['duplicate_validation_input']}
Perfect duplicates: {format_data['perfect_dups']}
Partial duplicates: {format_data['partial_dups']}
Unique SMILES: {format_data['validation_unique']}
"""
    
    formatted_report = template_report.format(**format_data)
    
    if print_logs:
        print(formatted_report)
    
    # Save outputs
    if get_output:
        from .report_utils import GetReport
        report_generator = GetReport(output_dir=output_dir, report_subdir_name="deduplicate")
        report_generator.create_csv_file(cleaned_df, csv_file_name="post_duplicates_removed.csv")
    
    if get_report:
        from .report_utils import GetReport
        report_generator = GetReport(output_dir=output_dir, report_subdir_name="deduplicate")
        report_generator.create_report_file(report_file_name="deduplicate.txt", content=formatted_report)
        report_generator.create_csv_file(dups_info, csv_file_name="duplicated_smiles_include_idx.csv")
    
    # Return output based on flags
    if show_dup_smi_and_idx and return_format_data:
        return cleaned_df, dups_info, format_data
    elif show_dup_smi_and_idx:
        return cleaned_df, dups_info
    elif return_format_data:
        return cleaned_df, format_data
    else:
        return cleaned_df