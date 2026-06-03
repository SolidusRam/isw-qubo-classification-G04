import os
import json
import shutil
import tempfile
import subprocess
import pandas as pd
import numpy as np
import pytest
from qubo_project.preprocessing import fit_normalize

@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)

def test_fit_normalize_basic(temp_dir):
    # Prepare input and output paths
    input_csv = "data/sample_test_dataset.csv"
    normalized_csv = os.path.join(temp_dir, "normalized.csv")
    stats_json = os.path.join(temp_dir, "stats.json")
    
    # Run fit_normalize
    fit_normalize(
        input_csv=input_csv,
        target_column="target",
        normalized_csv=normalized_csv,
        outInitalRes_json=stats_json,
        minPercValid=0.05
    )
    
    # Assert output files exist
    assert os.path.exists(normalized_csv)
    assert os.path.exists(stats_json)
    
    # Assert CSV contents
    df_norm = pd.read_csv(normalized_csv)
    assert "target" in df_norm.columns
    # Check that all features are numeric
    for col in df_norm.columns:
        assert pd.api.types.is_numeric_dtype(df_norm[col])
        
    # Assert JSON statistics format and types
    with open(stats_json, "r") as f:
        stats = json.load(f)
        
    expected_keys = {
        "n_input_features", "n_kept_features", "dataset_size",
        "dataset_input_time", "dataset_processing_time", "dropped_feature_names"
    }
    assert expected_keys.issubset(stats.keys())
    assert isinstance(stats["n_input_features"], int)
    assert isinstance(stats["n_kept_features"], int)
    assert isinstance(stats["dataset_size"], int)
    assert isinstance(stats["dataset_input_time"], float)
    assert isinstance(stats["dataset_processing_time"], float)
    assert isinstance(stats["dropped_feature_names"], list)
    
    # Kept + dropped should equal input features
    assert stats["n_kept_features"] + len(stats["dropped_feature_names"]) == stats["n_input_features"]

def test_missing_and_zero_values_filter(temp_dir):
    # Construct a mock dataset with specific null/zero ratios
    # 10 rows:
    # col_keep: 10 non-zero, non-null values (100% valid) -> keep
    # col_drop_zero: 9 zeros, 1 non-zero value (10% valid) -> drop if minPercValid=0.20
    # col_drop_nan: 9 NaNs, 1 non-zero value (10% valid) -> drop if minPercValid=0.20
    # target: all zeros (should be kept anyway)
    data = {
        "col_keep": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "col_drop_zero": [0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
        "col_drop_nan": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 5],
        "target": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    }
    mock_df = pd.DataFrame(data)
    mock_input_csv = os.path.join(temp_dir, "mock_input.csv")
    mock_df.to_csv(mock_input_csv, index=False)
    
    normalized_csv = os.path.join(temp_dir, "normalized.csv")
    stats_json = os.path.join(temp_dir, "stats.json")
    
    fit_normalize(
        input_csv=mock_input_csv,
        target_column="target",
        normalized_csv=normalized_csv,
        outInitalRes_json=stats_json,
        minPercValid=0.20
    )
    
    df_norm = pd.read_csv(normalized_csv)
    # col_keep and target should exist, others should be dropped
    assert "col_keep" in df_norm.columns
    assert "target" in df_norm.columns
    assert "col_drop_zero" not in df_norm.columns
    assert "col_drop_nan" not in df_norm.columns
    
    with open(stats_json, "r") as f:
        stats = json.load(f)
    assert stats["n_input_features"] == 3
    assert stats["n_kept_features"] == 1
    assert set(stats["dropped_feature_names"]) == {"col_drop_zero", "col_drop_nan"}

def test_z_score_normalization(temp_dir):
    # col_normal: normal features (mean=5, std=2) -> should become mean ~0, std ~1
    # col_constant: constant values -> should become all 0.0
    # target: target variable (should not be standardized)
    data = {
        "col_normal": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "col_constant": [5.0] * 10,
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    }
    mock_df = pd.DataFrame(data)
    mock_input_csv = os.path.join(temp_dir, "mock_input.csv")
    mock_df.to_csv(mock_input_csv, index=False)
    
    normalized_csv = os.path.join(temp_dir, "normalized.csv")
    stats_json = os.path.join(temp_dir, "stats.json")
    
    fit_normalize(
        input_csv=mock_input_csv,
        target_column="target",
        normalized_csv=normalized_csv,
        outInitalRes_json=stats_json,
        minPercValid=0.05
    )
    
    df_norm = pd.read_csv(normalized_csv)
    
    # Assert col_normal z-score: mean=0, std=1 (or very close to it)
    assert np.isclose(df_norm["col_normal"].mean(), 0.0)
    assert np.isclose(df_norm["col_normal"].std(ddof=0), 1.0)
    
    # Assert col_constant has been filled with 0.0
    assert (df_norm["col_constant"] == 0.0).all()
    
    # Assert target column is unchanged
    assert (df_norm["target"] == data["target"]).all()

def test_cli(temp_dir):
    input_csv = "data/sample_test_dataset.csv"
    normalized_csv = os.path.join(temp_dir, "normalized.csv")
    stats_json = os.path.join(temp_dir, "stats.json")
    
    # Execute preprocessing.py as a script
    cmd = [
        ".venv/bin/python",
        "src/qubo_project/preprocessing.py",
        "--input", input_csv,
        "--target", "target",
        "--out-data", normalized_csv,
        "--out-json", stats_json,
        "--min-perc-valid", "0.05"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    # Assert CLI exited with 0
    assert result.returncode == 0
    
    # Assert output files were generated
    assert os.path.exists(normalized_csv)
    assert os.path.exists(stats_json)
    
    # Assert stdout contains the output paths as requested
    assert normalized_csv in result.stdout
    assert stats_json in result.stdout
