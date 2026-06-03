import os
import time
import json
import argparse
import pandas as pd
import numpy as np

def fit_normalize(
    input_csv: str,           # Input dataset name
    target_column: str,       # column name of target
    normalized_csv: str,      # Name of output normalized data set
    outInitalRes_json: str,   # Name of output statistics and data file
    minPercValid: float = 0.05,  # Minimum % of valid non-zero data for a column
):
    """
    Reads the input CSV, filters out sparse features, standardizes the remaining features,
    and writes the results to CSV and statistics to JSON.
    """
    print(f"Starting preprocessing for {input_csv}...")
    
    # 1. Load dataset & measure input time
    start_input = time.time()
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")
    
    df = pd.read_csv(input_csv)
    dataset_input_time = time.time() - start_input
    print(f"Loaded dataset in {dataset_input_time:.4f}s. Shape: {df.shape}")
    
    # 2. Process dataset & measure processing time
    start_proc = time.time()
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset columns.")
    
    total_rows = len(df)
    if total_rows == 0:
        raise ValueError("Input dataset is empty.")
    
    # Separate features and target
    features_df = df.drop(columns=[target_column])
    target_series = df[target_column]
    
    n_input_features = features_df.shape[1]
    
    # Filter columns: valid is non-null and non-zero
    # Count of valid values in each column
    valid_mask = features_df.notna() & (features_df != 0) & (features_df != 0.0)
    valid_counts = valid_mask.sum(axis=0)
    valid_ratios = valid_counts / total_rows
    
    kept_features_mask = valid_ratios >= minPercValid
    kept_feature_names = list(features_df.columns[kept_features_mask])
    dropped_feature_names = list(features_df.columns[~kept_features_mask])
    
    print(f"Kept features: {len(kept_feature_names)} / {n_input_features}")
    print(f"Dropped features: {len(dropped_feature_names)}")
    
    filtered_features = features_df[kept_feature_names].copy()
    
    # Normalize features using z-score: (x - mean) / std (ddof=0)
    means = filtered_features.mean(axis=0)
    stds = filtered_features.std(axis=0, ddof=0)
    
    for col in kept_feature_names:
        mean_val = means[col]
        std_val = stds[col]
        
        # Handle cases with constant values or all NaNs
        if pd.isna(mean_val):
            mean_val = 0.0
        if pd.isna(std_val) or std_val == 0.0:
            filtered_features[col] = 0.0
        else:
            filtered_features[col] = (filtered_features[col] - mean_val) / std_val
            # Impute remaining NaN values (if any) with the mean (which is 0.0 after z-score)
            filtered_features[col] = filtered_features[col].fillna(0.0)

    # Recombine features and target keeping the original column order
    output_columns = []
    for col in df.columns:
        if col == target_column:
            output_columns.append(col)
        elif col in kept_feature_names:
            output_columns.append(col)
            
    normalized_columns = {}
    for col in output_columns:
        if col == target_column:
            normalized_columns[col] = target_series
        else:
            normalized_columns[col] = filtered_features[col]
    normalized_df = pd.DataFrame(normalized_columns, index=df.index)
            
    # Measure processing time (excluding loading and saving)
    dataset_processing_time = time.time() - start_proc
    
    # Create parent directories for output files if they do not exist
    os.makedirs(os.path.dirname(os.path.abspath(normalized_csv)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(outInitalRes_json)), exist_ok=True)
    
    # Write normalized CSV
    normalized_df.to_csv(normalized_csv, index=False)
    print(f"Saved normalized dataset to {normalized_csv}")
    
    # Prepare statistics JSON
    stats = {
        "n_input_features": int(n_input_features),
        "n_kept_features": int(len(kept_feature_names)),
        "dataset_size": int(total_rows),
        "dataset_input_time": float(round(dataset_input_time, 4)),
        "dataset_processing_time": float(round(dataset_processing_time, 4)),
        "dropped_feature_names": dropped_feature_names
    }
    
    with open(outInitalRes_json, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {outInitalRes_json}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Phase 1: Preprocessing for QUBO Feature Selection.")
    parser.add_argument('--input', required=True, help="Input CSV dataset path")
    parser.add_argument('--target', required=True, help="Name of the target column")
    parser.add_argument('--out-data', required=True, help="Path for the output normalized CSV dataset")
    parser.add_argument('--out-json', required=True, help="Path for the output statistics JSON file")
    parser.add_argument('--min-perc-valid', type=float, default=0.05, help="Minimum percentage of valid non-zero/non-null data (0.0 to 1.0)")
    
    args = parser.parse_args()
    
    fit_normalize(
        input_csv=args.input,
        target_column=args.target,
        normalized_csv=args.out_data,
        outInitalRes_json=args.out_json,
        minPercValid=args.min_perc_valid
    )
    
    # Print the paths of the generated files to standard output as expected by the spec
    print(args.out_data)
    print(args.out_json)
