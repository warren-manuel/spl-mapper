# Contents of /rxnorm-term-getter/main.py

import sys
from src.rxnorm_term_getter import identify_missing_rxnorm_terms

def main():
    vo_path_or_url = None  # You can modify this to specify a path or URL if needed
    if len(sys.argv) > 1:
        vo_path_or_url = sys.argv[1]

    missing_terms = identify_missing_rxnorm_terms(vo_path_or_url)
    
    # Define output CSV filename
    output_file = "missing_rxnorm_terms.csv"
    
    # Save to CSV file
    missing_terms.to_csv(output_file, index=False)
    
    # Print confirmation and preview
    print(f"Missing RxNorm terms saved to: {output_file}")
    print("\nPreview of missing terms:")
    print(missing_terms.head())

if __name__ == "__main__":
    main()