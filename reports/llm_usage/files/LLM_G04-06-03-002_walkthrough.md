# Walkthrough - Phase 1: Preprocessing Completed

We have fully implemented and verified Phase 1 (Preprocessing) of the ISW QUBO Classification project.

## Changes Made

### 1. Preprocessing Module
- Created [preprocessing.py](file:///home/para/Work/isw-qubo-classification-G04/src/qubo_project/preprocessing.py):
  - Function `fit_normalize(...)` which loads dataset, separates target column, filters features with sparse data (less than `minPercValid` non-zero/non-null ratio), standardizes kept features using Z-score (with safety fallback for zero-variance features), and saves both a normalized CSV and stats JSON.
  - Typo in parameter name handled exactly as requested: `outInitalRes_json`.
  - Command-line interface (CLI) to execute preprocessing directly from the terminal, printing outputs to stdout.
  - Refactored column recombination to prevent DataFrame fragmentation warnings and improve memory/CPU speed on large datasets (1.5M+ rows).

### 2. Requirements & Dependencies
- Modified [requirements.txt](file:///home/para/Work/isw-qubo-classification-G04/requirements.txt) to include `pandas`, `numpy`, and `pytest`.
- Created a `.venv` virtual environment in the workspace and successfully installed all dependencies.

### 3. Testing
- Extracted a representative 100-row sample dataset (at least 20% target=1) to [sample_test_dataset.csv](file:///home/para/Work/isw-qubo-classification-G04/data/sample_test_dataset.csv).
- Created unit tests in [test_preprocessing.py](file:///home/para/Work/isw-qubo-classification-G04/tests/test_preprocessing.py) verifying:
  - Basic function logic and JSON outputs.
  - Correct filtering of sparse/zero-value columns.
  - Standard scaling values correctness and division-by-zero robustness.
  - CLI script execution and stdout format.

### 4. Logging
- Documented our implementation process as `Interaction G04-06-03-002` in [LOG-G04-01.md](file:///home/para/Work/isw-qubo-classification-G04/reports/llm_usage/LOG-G04-01.md).

---

## Verification Results

### 1. Unit Tests
All 4 unit tests run and pass successfully:
```
============================= test session starts ==============================
platform linux -- Python 3.14.5, pytest-9.0.3, pluggy-1.6.0
rootdir: /home/para/Work/isw-qubo-classification-G04
collecting ...
collected 4 items
tests/test_preprocessing.py ....                                         [100%]
============================== 4 passed in 0.52s ===============================
```

### 2. CLI Run on Trial Dataset
Running the CLI on `data/trial_dataset_ISW.csv` works correctly, dropping 46 features and keeping 100, generating the outputs without warnings:
```bash
.venv/bin/python src/qubo_project/preprocessing.py \
  --input data/trial_dataset_ISW.csv \
  --target target \
  --out-data outputs/normalized.csv \
  --out-json outputs/preprocessing_result.json \
  --min-perc-valid 0.05
```
Output:
```
Starting preprocessing for data/trial_dataset_ISW.csv...
Loaded dataset in 0.1118s. Shape: (20000, 147)
Kept features: 100 / 146
Dropped features: 46
Saved normalized dataset to outputs/normalized.csv
Saved statistics to outputs/preprocessing_result.json
outputs/normalized.csv
outputs/preprocessing_result.json
```
The output stats JSON `preprocessing_result.json` contains:
```json
{
  "n_input_features": 146,
  "n_kept_features": 100,
  "dataset_size": 20000,
  "dataset_input_time": 0.1118,
  "dataset_processing_time": 0.0703,
  "dropped_feature_names": [ ... ]
}
```
All criteria have been fully satisfied.
