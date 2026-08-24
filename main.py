"""
CLI entry point for the molecular toxicity prediction pipeline.

Usage:
    python main.py "SMILES_STRING"
    python main.py "SMILES1" "SMILES2" "SMILES3"
    python main.py --file input.txt
"""

import argparse
import sys
from pathlib import Path

from pipeline import predict_and_explain, predict_batch, get_batch_summary


def main():
    parser = argparse.ArgumentParser(
        description="Predict molecular toxicity (NR-AhR) from SMILES strings"
    )
    parser.add_argument(
        'smiles',
        nargs='*',
        help="One or more SMILES strings to predict"
    )
    parser.add_argument(
        '--file', '-f',
        type=str,
        help="Path to file containing SMILES strings (one per line)"
    )
    parser.add_argument(
        '--summary', '-s',
        action='store_true',
        help="Show summary table only (for batch predictions)"
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Show progress during batch processing"
    )
    
    args = parser.parse_args()
    
    # Collect SMILES from arguments and/or file
    smiles_list = list(args.smiles) if args.smiles else []
    
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    smiles_list.append(line)
    
    if not smiles_list:
        parser.print_help()
        print("\nError: No SMILES strings provided", file=sys.stderr)
        sys.exit(1)
    
    # Single molecule - show full explanation
    if len(smiles_list) == 1 and not args.summary:
        result = predict_and_explain(smiles_list[0])
        
        if result.error:
            print(f"Error: {result.error}", file=sys.stderr)
            sys.exit(1)
        
        print(result.explanation)
    
    # Multiple molecules - show batch results
    else:
        results = predict_batch(smiles_list, verbose=args.verbose)
        
        if args.summary:
            print(get_batch_summary(results))
        else:
            # Print full explanation for each
            for result in results:
                if result.error:
                    print(f"\nError for '{result.smiles}': {result.error}")
                else:
                    print(result.explanation)
                    print()


if __name__ == "__main__":
    main()
