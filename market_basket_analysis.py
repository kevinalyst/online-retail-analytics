#!/usr/bin/env python3
"""
Market Basket Analysis Script

Performs association rule mining on retail transaction data using the Apriori
algorithm from mlxtend. Implements a complete pipeline including data loading,
cleaning, basket matrix construction, frequent itemset mining, association rule
generation, description enrichment, and comprehensive reporting.

Usage:
    python market_basket_analysis.py [options]

Example:
    python market_basket_analysis.py --input-path online_retail_clean.csv --min-support 0.02
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def load_data(path: str) -> pd.DataFrame:
    """
    Load transaction data from CSV file.
    
    Args:
        path: Path to the CSV file containing transaction data.
        
    Returns:
        DataFrame with raw transaction data.
        
    Raises:
        FileNotFoundError: If the input file doesn't exist.
        ValueError: If the file is empty or cannot be parsed.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    
    try:
        df = pd.read_csv(path)
        print(f"Loaded {len(df):,} rows from {path}")
        return df
    except pd.errors.EmptyDataError:
        raise ValueError(f"Input file is empty: {path}")
    except pd.errors.ParserError:
        raise ValueError(f"Failed to parse CSV file: {path}")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and filter transaction data for market basket analysis.
    
    Cleaning steps:
    1. Drop rows with missing invoice_number or item_code
    2. Filter out returns (keep only is_return == 0)
    3. Filter out anonymous customers (keep only is_anonymous == 0)
    4. Keep only positive quantities
    5. Remove duplicate (invoice_number, item_code) pairs
    
    Args:
        df: Raw transaction DataFrame with required columns.
            
    Returns:
        Cleaned DataFrame ready for basket matrix construction.
    """
    print("\n--- Data Preprocessing ---")
    initial_rows = len(df)
    
    # Step 1: Drop rows with missing critical fields
    df_clean = df.dropna(subset=['invoice_number', 'item_code'])
    print(f"After removing missing invoice/item: {len(df_clean):,} rows")
    
    # Step 2: Filter out returns
    df_clean = df_clean[df_clean['is_return'] == 0]
    print(f"After removing returns: {len(df_clean):,} rows")
    
    # Step 3: Filter out anonymous customers
    df_clean = df_clean[df_clean['is_anonymous'] == 0]
    print(f"After removing anonymous customers: {len(df_clean):,} rows")
    
    # Step 4: Keep only positive quantities
    df_clean = df_clean[df_clean['quantity'] > 0]
    print(f"After filtering positive quantities: {len(df_clean):,} rows")
    
    # Step 5: Remove duplicate (invoice_number, item_code) pairs
    # For basket analysis, we only care about presence, not quantity
    df_clean = df_clean.drop_duplicates(subset=['invoice_number', 'item_code'])
    print(f"After deduplication: {len(df_clean):,} rows")
    
    rows_removed = initial_rows - len(df_clean)
    print(f"Total rows removed: {rows_removed:,} ({rows_removed/initial_rows*100:.1f}%)")
    
    return df_clean


def build_basket_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct transaction-product binary matrix for market basket analysis.
    
    Creates a matrix where:
    - Rows: unique invoice_number (transactions)
    - Columns: unique item_code (products)
    - Values: 1 if item appears in transaction, 0 otherwise
    
    Args:
        df: Cleaned transaction DataFrame with invoice_number and item_code.
        
    Returns:
        Binary matrix (DataFrame) with invoice_number as index and 
        item_code as columns. Values are boolean (True/False) indicating
        item presence in each transaction.
    """
    print("\n--- Building Basket Matrix ---")
    
    # Create a binary indicator column
    df_basket = df.copy()
    df_basket['purchased'] = 1
    
    # Pivot to create transaction-product matrix
    basket = df_basket.pivot_table(
        index='invoice_number',
        columns='item_code',
        values='purchased',
        aggfunc='max',
        fill_value=0
    )
    
    # Convert to boolean for mlxtend
    basket = basket.astype(bool)
    
    print(f"Basket matrix shape: {basket.shape[0]:,} transactions × {basket.shape[1]:,} products")
    
    return basket


def run_apriori(
    basket: pd.DataFrame, 
    min_support: float, 
    max_len: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate frequent itemsets using the Apriori algorithm.
    
    Args:
        basket: Binary transaction-product matrix (bool values).
        min_support: Minimum support threshold (e.g., 0.01 = 1% of transactions).
        max_len: Maximum length of itemsets to consider (None = no limit).
        
    Returns:
        DataFrame with columns: support, itemsets (frozenset of item codes).
        
    Raises:
        ValueError: If no frequent itemsets found with given parameters.
    """
    print("\n--- Running Apriori Algorithm ---")
    print(f"Min support: {min_support} ({min_support*100:.1f}%)")
    if max_len:
        print(f"Max itemset length: {max_len}")
    
    # Run Apriori algorithm
    frequent_itemsets = apriori(
        basket, 
        min_support=min_support, 
        use_colnames=True,
        max_len=max_len
    )
    
    if frequent_itemsets.empty:
        raise ValueError(
            f"No frequent itemsets found with min_support={min_support}. "
            "Try lowering the support threshold."
        )
    
    print(f"Found {len(frequent_itemsets):,} frequent itemsets")
    
    return frequent_itemsets


def generate_rules(
    itemsets: pd.DataFrame, 
    min_confidence: float
) -> pd.DataFrame:
    """
    Generate association rules from frequent itemsets.
    
    Args:
        itemsets: Frequent itemsets DataFrame from apriori().
        min_confidence: Minimum confidence threshold for rules (e.g., 0.2 = 20%).
        
    Returns:
        DataFrame with association rules containing:
        - antecedents, consequents (frozensets)
        - support, confidence, lift, leverage, conviction
        - antecedent_len, consequent_len
        
    Raises:
        ValueError: If no rules meet the confidence threshold.
    """
    print("\n--- Generating Association Rules ---")
    print(f"Min confidence: {min_confidence} ({min_confidence*100:.1f}%)")
    
    # Generate rules with lift metric
    rules = association_rules(
        itemsets, 
        metric="lift", 
        min_threshold=1.0  # We'll filter by confidence separately
    )
    
    if rules.empty:
        raise ValueError("No association rules could be generated from the itemsets.")
    
    # Filter by minimum confidence
    rules = rules[rules['confidence'] >= min_confidence]
    
    if rules.empty:
        raise ValueError(
            f"No rules meet min_confidence={min_confidence}. "
            "Try lowering the confidence threshold."
        )
    
    # Add antecedent and consequent lengths
    rules['antecedent_len'] = rules['antecedents'].apply(len)
    rules['consequent_len'] = rules['consequents'].apply(len)
    
    print(f"Generated {len(rules):,} rules")
    
    return rules


def attach_descriptions(rules: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach human-readable item descriptions to association rules.
    
    Creates a mapping from item_code to the most frequent item_description,
    then adds readable columns for antecedents and consequents.
    
    Args:
        rules: Association rules DataFrame with antecedents/consequents as frozensets.
        df: Original transaction DataFrame with item_code and item_description.
        
    Returns:
        Rules DataFrame with additional columns:
        - antecedent_codes, consequent_codes (comma-separated strings)
        - antecedent_descriptions, consequent_descriptions (comma-separated strings)
    """
    print("\n--- Attaching Item Descriptions ---")
    
    # Build item_code -> most frequent description mapping
    item_desc_map = (
        df.groupby('item_code')['item_description']
        .agg(lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0])
        .to_dict()
    )
    
    def frozenset_to_codes(fs):
        """Convert frozenset to sorted comma-separated string."""
        return ', '.join(sorted(str(item) for item in fs))
    
    def frozenset_to_descriptions(fs):
        """Convert frozenset of codes to comma-separated descriptions."""
        descriptions = [item_desc_map.get(item, f"Unknown ({item})") for item in sorted(fs)]
        return ', '.join(descriptions)
    
    # Create readable columns
    rules_with_desc = rules.copy()
    rules_with_desc['antecedent_codes'] = rules['antecedents'].apply(frozenset_to_codes)
    rules_with_desc['consequent_codes'] = rules['consequents'].apply(frozenset_to_codes)
    rules_with_desc['antecedent_descriptions'] = rules['antecedents'].apply(frozenset_to_descriptions)
    rules_with_desc['consequent_descriptions'] = rules['consequents'].apply(frozenset_to_descriptions)
    
    return rules_with_desc


def save_rules(rules: pd.DataFrame, output_path: str) -> None:
    """
    Save association rules to CSV file.
    
    Sorts rules by lift (descending) then confidence (descending) before saving.
    
    Args:
        rules: Association rules DataFrame with descriptions attached.
        output_path: Path to output CSV file.
    """
    print("\n--- Saving Results ---")
    
    # Select and order columns for output
    output_columns = [
        'antecedent_codes',
        'consequent_codes',
        'antecedent_descriptions',
        'consequent_descriptions',
        'support',
        'confidence',
        'lift',
        'leverage',
        'conviction',
        'antecedent_len',
        'consequent_len'
    ]
    
    # Sort by lift (desc), then confidence (desc)
    rules_sorted = rules.sort_values(
        by=['lift', 'confidence'], 
        ascending=[False, False]
    )
    
    # Save to CSV
    rules_sorted[output_columns].to_csv(output_path, index=False)
    print(f"Saved {len(rules_sorted):,} rules to {output_path}")


def print_summary(rules: pd.DataFrame, n_transactions: int, n_products: int, n_itemsets: int) -> None:
    """
    Print summary statistics and top rules to console.
    
    Args:
        rules: Association rules DataFrame sorted by lift.
        n_transactions: Number of transactions in basket matrix.
        n_products: Number of unique products in basket matrix.
        n_itemsets: Number of frequent itemsets found.
    """
    print("\n" + "="*80)
    print("MARKET BASKET ANALYSIS SUMMARY")
    print("="*80)
    print(f"Transactions analyzed: {n_transactions:,}")
    print(f"Unique products: {n_products:,}")
    print(f"Frequent itemsets found: {n_itemsets:,}")
    print(f"Association rules generated: {len(rules):,}")
    
    # Sort by lift for display
    top_rules = rules.sort_values(by=['lift', 'confidence'], ascending=[False, False]).head(5)
    
    print("\n" + "-"*80)
    print("TOP 5 RULES BY LIFT")
    print("-"*80)
    
    for idx, (_, rule) in enumerate(top_rules.iterrows(), 1):
        print(f"\nRule {idx}:")
        print(f"  IF: {rule['antecedent_descriptions']}")
        print(f"  THEN: {rule['consequent_descriptions']}")
        print(f"  Support: {rule['support']:.4f} | Confidence: {rule['confidence']:.4f} | Lift: {rule['lift']:.2f}")
    
    print("\n" + "="*80)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Namespace object with parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Perform market basket analysis on retail transaction data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python market_basket_analysis.py
  python market_basket_analysis.py --input-path data.csv --min-support 0.02
  python market_basket_analysis.py --output-path rules.csv --min-confidence 0.3
        """
    )
    
    parser.add_argument(
        '--input-path',
        type=str,
        default='online_retail_clean.csv',
        help='Path to input CSV file (default: online_retail_clean.csv)'
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        default='market_basket_rules.csv',
        help='Path to output CSV file (default: market_basket_rules.csv)'
    )
    
    parser.add_argument(
        '--min-support',
        type=float,
        default=0.01,
        help='Minimum support threshold (default: 0.01 = 1%%)'
    )
    
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.2,
        help='Minimum confidence threshold (default: 0.2 = 20%%)'
    )
    
    parser.add_argument(
        '--max-len',
        type=int,
        default=None,
        help='Maximum itemset length (default: None = no limit)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.min_support <= 0 or args.min_support > 1:
        parser.error("--min-support must be between 0 and 1")
    
    if args.min_confidence <= 0 or args.min_confidence > 1:
        parser.error("--min-confidence must be between 0 and 1")
    
    if args.max_len is not None and args.max_len < 1:
        parser.error("--max-len must be a positive integer")
    
    return args


def main() -> None:
    """
    Main execution function for market basket analysis.
    
    Orchestrates the complete pipeline:
    1. Load data
    2. Preprocess/clean
    3. Build basket matrix
    4. Run Apriori algorithm
    5. Generate association rules
    6. Attach descriptions
    7. Save results and print summary
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    print("="*80)
    print("MARKET BASKET ANALYSIS")
    print("="*80)
    print(f"Input: {args.input_path}")
    print(f"Output: {args.output_path}")
    print(f"Parameters: min_support={args.min_support}, min_confidence={args.min_confidence}")
    if args.max_len:
        print(f"Max itemset length: {args.max_len}")
    
    try:
        # Step 1: Load data
        df = load_data(args.input_path)
        
        # Step 2: Preprocess data
        df_clean = preprocess_data(df)
        
        if df_clean.empty:
            print("\nERROR: No data remaining after preprocessing.")
            sys.exit(1)
        
        # Step 3: Build basket matrix
        basket = build_basket_matrix(df_clean)
        
        # Step 4: Run Apriori algorithm
        frequent_itemsets = run_apriori(basket, args.min_support, args.max_len)
        
        # Step 5: Generate association rules
        rules = generate_rules(frequent_itemsets, args.min_confidence)
        
        # Step 6: Attach descriptions
        rules_with_desc = attach_descriptions(rules, df_clean)
        
        # Step 7: Save results
        save_rules(rules_with_desc, args.output_path)
        
        # Print summary
        print_summary(
            rules_with_desc,
            n_transactions=basket.shape[0],
            n_products=basket.shape[1],
            n_itemsets=len(frequent_itemsets)
        )
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
