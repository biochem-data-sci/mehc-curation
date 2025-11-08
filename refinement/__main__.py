"""Command line interface for refinement module - allows 'python -m refinement'."""
import argparse
import pandas as pd
import sys
import os
from .core.pipeline import RefinementPipeline


def main():
    """Main CLI function - preserves exact original CLI behavior."""
    parser = argparse.ArgumentParser(
        description="This module is used to refine SMILES data",
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
        "--validate_beginning",
        required=False,
        action="store_true",
        default=True,
        help="Validate SMILES data before doing anything "
             "(optional, default is true)",
    )
    parser.add_argument(
        "--rm_mixture",
        required=False,
        action="store_false",
        default=True,
        help="Remove mixtures from SMILES data (optional, default is true)",
    )
    parser.add_argument(
        "--rm_inorganic",
        required=False,
        action="store_false",
        default=True,
        help="Remove inorganic molecules from SMILES data "
             "(optional, default is true)",
    )
    parser.add_argument(
        "--rm_organometallic",
        required=False,
        action="store_false",
        default=True,
        help="Remove organometallics from SMILES data "
             "(optional, default is true)",
    )
    parser.add_argument(
        "--cl_salt",
        required=False,
        action="store_false",
        default=True,
        help="Remove salts from SMILES data (optional, default is true)",
    )
    parser.add_argument(
        "--neutralize",
        required=False,
        action="store_false",
        default=True,
        help="Neutralize charged molecules from SMILES data "
             "(optional, default is true)",
    )
    parser.add_argument(
        "--destereoisomerize",
        required=False,
        action="store_false",
        default=True,
        help="Destereoisomerization from SMILES data "
             "(optional, default is true)",
    )
    parser.add_argument(
        "--detautomerize",
        required=False,
        action="store_false",
        default=True,
        help="Detautomerization from SMILES data "
             "(optional, default is true)",
    )
    parser.add_argument(
        "--deduplicate",
        required=False,
        action="store_false",
        default=True,
        help="Remove duplicate SMILES after each stage "
             "(optional, default is True)",
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
        default=-1,
        help="Number of CPUs to use (-1 for all CPUs, optional)",
    )
    parser.add_argument(
        "--split_factor",
        required=False,
        type=int,
        default=1,
        help="Split factor (optional)",
    )
    
    try:
        args = parser.parse_args()

        # Validate input file exists
        if not os.path.exists(args.input):
            print(f"Error: Input file '{args.input}' does not exist.")
            sys.exit(1)

        # Create output directory if it doesn't exist
        os.makedirs(args.output, exist_ok=True)

        # Load data
        try:
            smi_df = pd.read_csv(args.input)
            if smi_df.empty:
                print(f"Error: Input file '{args.input}' is empty.")
                sys.exit(1)
        except Exception as e:
            print(f"Error reading input file '{args.input}': {e}")
            sys.exit(1)

        refining = RefinementPipeline(smi_df)

        param_dict = {
            "validate": args.validate_beginning,
            "rm_mixture": args.rm_mixture,
            "rm_inorganic": args.rm_inorganic,
            "rm_organometallic": args.rm_organometallic,
            "cl_salt": args.cl_salt,
            "neutralize": args.neutralize,
            "destereoisomerize": args.destereoisomerize,
            "detautomerize": args.detautomerize,
            "rm_dup_between_stages": args.deduplicate,
            "output_dir": args.output,
            "print_logs": args.print_logs,
            "get_report": args.get_report,
            "n_cpu": args.n_cpu,
            "split_factor": args.split_factor,
        }

        output = refining.complete_refinement(**param_dict)
        print("Your action is done!")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

