# README information

## Preprocessing note

The current preprocessing implementation works correctly on numeric columns, but it does not yet handle non-numeric feature columns robustly. If a CSV contains text or categorical values, the preprocessing step may fail during z-score computation unless those columns are excluded or converted first.

## Step 3 Algorithm Selection

To maximize performance on the test dataset (which includes over 1.5M samples) and particularly optimize the F1-score for the positive class (`target=1`), we implemented three classifiers:

1. **Random Forest (Mandatory)**: Chosen as the required baseline algorithm. We use `class_weight='balanced'` and `n_jobs=-1` to efficiently handle imbalanced targets across multiple CPU cores.
2. **Hist Gradient Boosting**: We upgraded the standard Gradient Boosting to scikit-learn's `HistGradientBoostingClassifier` because it is specifically optimized for large datasets (like the 1.5M+ rows test set). It runs blazingly fast natively without external dependencies.
3. **Logistic Regression**: Selected as a robust, fast linear baseline. This scales effortlessly to 1.5 million rows and natively supports `class_weight='balanced'`.

## GUI

To start the graphical user interface for this project, simply activate your environment and run the provided launcher script:

```bash
python start_gui.py
```

Alternatively, you can manually run: `streamlit run src/qubo_project/gui.py`

## CLI Execution

The project can also be entirely executed via command line, as per specifications. For example:

```bash
# 1. Preprocessing
python src/qubo_project/preprocessing.py --input data/trial_dataset_ISW.csv --target target --out-data outputs/normalized.csv --out-json outputs/preprocessing_result.json

# 2. QUBO Feature Selection
python src/qubo_project/feature_selection.py --in-normalized outputs/normalized.csv --out-train outputs/training_reduced.csv --out-test outputs/test_reduced.csv --out-optimizations outputs/optimizations.csv --out-json outputs/feature_selection_result.json --target target

# 3. Model Training
python src/qubo_project/model.py train --classifier gradient_boosting --in-reduced outputs/training_reduced.csv --target target --out-model outputs/model.joblib --out-metrics outputs/training_metrics.json

# 4. Model Prediction
python src/qubo_project/model.py predict --input-testset outputs/test_reduced.csv --target target --model outputs/model.joblib --out-predictions outputs/predictions.csv --out-stats outputs/classification_stats.json
```
