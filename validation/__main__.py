"""Command line interface for validation module - allows 'python -m validation'."""
import argparse
import pandas as pd
import sys
import os
from .core.pipeline import ValidationPipeline


def main():
    """Main CLI function - preserves exact original CLI behavior."""
    parser = argparse.ArgumentParser(
        description="This module is used to validate SMILES data",
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
        "-c", "--choice", required=True, type=int, choices=range(1, 6),
        help="Choose the action:\n"
        "1. Check the validity of the SMILES data\n"
        "2. Remove mixtures in the SMILES data\n"
        "3. Remove inorganics in the SMILES data\n"
        "4. Remove organometallics in the SMILES data\n"
        "5. Do all the validation stage\n",
    )
    parser.add_argument(
        "--validate_beginning", required=False, action="store_true", default=False,
        help="Validate SMILES data before doing anything "
        "(except action 1, optional, default is false)",
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

        validation = ValidationPipeline(smi_df)

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
            del param_dict["validate"]

        match args.choice:
            case 1:
                output = validation.validate_smi(**param_dict)
            case 2:
                output = validation.rm_mixture(**param_dict)
            case 3:
                output = validation.rm_inorganic(**param_dict)
            case 4:
                output = validation.rm_organometallic(**param_dict)
            case 5:
                output = validation.complete_validation(**param_dict)
        
        print("Your action is done!")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()