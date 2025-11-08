"""Command line interface for cleaning module - allows 'python -m cleaning'."""
import argparse
import pandas as pd
import sys
import os
from .core.pipeline import CleaningPipeline


def main():
    """Main CLI function - preserves exact original CLI behavior."""
    parser = argparse.ArgumentParser(
        description="This module is used to clean salts and neutralize SMILES data",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", required=True, type=str, help="Input SMILES csv file"
    )
    parser.add_argument(
        "-o", "--output", required=True, type=str,
        help="Output folder, include csv and report (if have any)"
    )
    parser.add_argument(
        "-c", "--choice", required=True, type=int, choices=range(1, 4),
        help="Choose the action:\n"
        "1. Only clean salts in data\n"
        "2. Only neutralize SMILES in data\n"
        "3. Clean and neutralize SMILES in data\n",
    )
    parser.add_argument(
        "--validate_beginning", required=False, action="store_true", default=False,
        help="Validate SMILES data before doing anything "
        "(optional, default is false)",
    )
    parser.add_argument(
        "--deduplicate", required=False, action="store_true", default=False,
        help="Remove duplicate SMILES after your action "
        "(optional, default is false)",
    )
    parser.add_argument(
        "-p", "--print_logs", required=False, action="store_false",
        help="Print logs (optional, default is true)",
    )
    parser.add_argument(
        "--get_report", required=False, action="store_true",
        help="Get report "
        "(include report file and supported csv for more information) "
        "(optional, default is False)",
    )
    parser.add_argument(
        "--n_cpu", required=False, type=int, default=-1,
        help="Number of CPUs to use (-1 for all CPUs, optional)",
    )
    parser.add_argument(
        "--split_factor", required=False, type=int, default=1,
        help="Split factor (optional)",
    )
    parser.add_argument(
        "--method", required=False, type=str, default="boyle",
        choices=["boyle", "rdkit"],
        help="Neutralization method (only for choice 2 or 3, default is 'boyle')",
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

        cleaning = CleaningPipeline(smi_df)

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
                param_dict["method"] = args.method
                output = cleaning.neutralize(**param_dict)
            case 3:
                param_dict["neutralizing_method"] = args.method
                output = cleaning.complete_cleaning(**param_dict)
        
        print("Your action is done!")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

