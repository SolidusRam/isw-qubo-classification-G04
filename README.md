# README information

## Preprocessing note

The current preprocessing implementation works correctly on numeric columns, but it does not yet handle non-numeric feature columns robustly. If a CSV contains text or categorical values, the preprocessing step may fail during z-score computation unless those columns are excluded or converted first.

## Step 3 Algorithm Selection

To maximize performance on the test dataset (which includes over 1.5M samples) and particularly optimize the F1-score for the positive class (`target=1`), we implemented three classifiers:

1. **Random Forest (Mandatory)**: Chosen as the required baseline algorithm. We use `class_weight='balanced'` and `n_jobs=-1` to efficiently handle imbalanced targets across multiple CPU cores.
2. **LightGBM**: This algorithm was selected specifically for its unmatched training speed and low memory footprint on very large tabular datasets. We heavily utilize its `is_unbalance=True` feature to inherently handle the minority positive class and secure a high F1-score.
3. **Logistic Regression**: Selected as a robust, fast linear baseline. This scales effortlessly to 1.5 million rows and natively supports `class_weight='balanced'`.
