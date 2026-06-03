# Prompt: Implementation of Phase 1 (Preprocessing)

You can use the following detailed prompt to guide an LLM (like Claude or GPT) in generating the code for **Phase 1: Preprocessing**.

***

```markdown
Role: Senior Python Developer & Software Architect
Task: Implement the Preprocessing Module (Phase 1) for a Credit Risk Binary Classification System using QUBO Feature Selection.

### 1. Context & Objectives
We are building a machine learning pipeline that classifies bank clients as reliable ("0") or at risk ("1") from tabular datasets containing up to 1.5 million rows and over 100 features. Before performing feature selection via Quadratic Unconstrained Binary Optimization (QUBO) and training classifiers, we must preprocess the data.

You need to implement the preprocessing logic in the file `src/qubo_project/preprocessing.py`.
This module must:
1. Support execution both as an importable module (exposing a specific API function) and as a Command-Line Interface (CLI).
2. Handle large datasets efficiently (up to 1.5M rows) without running out of memory.
3. Remove features that contain too many zero or missing values.
4. Normalize the remaining features using z-score standardization while keeping the target column unchanged.
5. Export the normalized dataset to a CSV file and output key performance and data statistics to a JSON file.

---

### 2. Required Function Interface
Implement the following function in `src/qubo_project/preprocessing.py`:

```python
def fit_normalize(
    input_csv: str,           # Input dataset name/path
    target_column: str,       # Column name of the target variable
    normalized_csv: str,      # Path for the output normalized CSV file
    outInitalRes_json: str,   # Path for the output statistics JSON file
    minPercValid: float = 0.05,  # Minimum % of valid (non-zero/non-null) data to keep a column
):
    """
    Reads the input CSV, filters out sparse features, standardizes the remaining features,
    and writes the results to CSV and statistics to JSON.
    """
```
*Note: Make sure to name the parameter `outInitalRes_json` exactly as shown (with the typo "Inital" instead of "Initial") to match the grading system's specifications.*

#### Implementation Details & Formulas:
- **Time Measurement**:
  - Track the exact time spent reading the dataset from disk. Let this be `dataset_input_time` (in seconds).
  - Track the time spent processing the dataset (filtering columns, computing z-scores, and preparing statistics, excluding I/O time). Let this be `dataset_processing_time` (in seconds).
- **Target Separation**:
  - The target column must be excluded from feature filtering and normalization. It should be written back to the output dataset exactly as it was.
- **Sparse Feature Removal**:
  - A feature column is dropped if the ratio of its **valid** values is strictly less than `minPercValid` (e.g., if `minPercValid=0.05`, keep columns with at least 5% valid values).
  - A value is considered **invalid** if it is null (NaN/None) or exactly zero (`0` or `0.0`).
  - Formally, a column is kept if:
    $$\frac{\text{Count of non-null and non-zero elements}}{\text{Total number of rows}} \ge \text{minPercValid}$$
  - The target column must never be dropped under any circumstance.
- **Z-score Standardization**:
  - Apply standard scaling to all remaining features:
    $$z = \frac{x - \mu}{\sigma}$$
    where $\mu$ is the mean of the column and $\sigma$ is its standard deviation.
  - **Edge Case (Zero Variance)**: If $\sigma = 0$ (all values in the column are identical), set all standardized values for that column to `0.0` to avoid division by zero errors and NaN propagation.
- **Output CSV**:
  - Save the normalized features along with the unmodified target column to `normalized_csv`.
  - Maintain the original column headers.
- **Output JSON (`outInitalRes_json`)**:
  - Export the metadata statistics in JSON format matching this exact structure:
    ```json
    {
      "n_input_features": 140,
      "n_kept_features": 120,
      "dataset_size": 20000,
      "dataset_input_time": 2.34,
      "dataset_processing_time": 3.02,
      "dropped_feature_names": ["feature_1", "feature_20"]
    }
    ```
    *(Note: `n_input_features` and `n_kept_features` count features ONLY, excluding the target column).*

---

### 3. Command Line Interface (CLI)
When `src/qubo_project/preprocessing.py` is executed directly from the terminal, it must parse arguments using Python's `argparse` module and call `fit_normalize`.

#### CLI Syntax:
```bash
python preprocessing.py \
  --input <path_to_input_csv> \
  --target <target_column_name> \
  --out-data <path_to_output_normalized_csv> \
  --out-json <path_to_output_stats_json> \
  --min-perc-valid <float_threshold>
```

#### Example Command:
```bash
python preprocessing.py \
  --input dati_credito.csv \
  --target target \
  --out-data normalized.csv \
  --out-json preprocessing_result.json \
  --min-perc-valid 0.06
```

#### CLI Outputs:
Upon successful completion, the CLI must print the paths of the generated files to standard output:
```
outputs/normalized.csv
outputs/preprocessing_result.json
```
*(Ensure parent directories of these output files are automatically created if they do not exist).*

---

### 4. Non-Functional & Reproducibility Requirements
- **Python Compatibility**: The code must be compatible with Python 3.11 or higher.
- **Memory Efficiency**: Ensure that operations on large datasets (1.5M rows) use pandas or numpy memory-saving techniques (e.g. processing or downcasting types where appropriate, avoiding copying large DataFrames unnecessarily).
- **Paths**: Use relative paths. Ensure all paths specified in CLI args can resolve dynamically.
- **Robustness**: Include error handling for missing files, missing target columns, division by zero, and invalid data types. Add logging to describe progress.
- **Clean Code**: Follow PEP 8 style guidelines. Document the function with clear docstrings and type hints.

Please generate the complete, self-contained implementation of `src/qubo_project/preprocessing.py`.
```
