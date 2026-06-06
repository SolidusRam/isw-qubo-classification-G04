import os
import json
import shutil
import tempfile
import pandas as pd
import numpy as np
import pytest
import joblib
from qubo_project.model import train, predict

@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)

def test_train_and_predict(temp_dir):
    # Mock training and test data
    np.random.seed(42)
    m = 5
    n_samples_train = 100
    n_samples_test = 30
    
    train_data = {f"feature_{i}": np.random.randn(n_samples_train) for i in range(m)}
    train_data["target"] = np.random.randint(0, 2, size=n_samples_train)
    df_train = pd.DataFrame(train_data)
    
    test_data = {f"feature_{i}": np.random.randn(n_samples_test) for i in range(m)}
    test_data["target"] = np.random.randint(0, 2, size=n_samples_test)
    df_test = pd.DataFrame(test_data)
    
    train_csv = os.path.join(temp_dir, "train_reduced.csv")
    test_csv = os.path.join(temp_dir, "test_reduced.csv")
    
    df_train.to_csv(train_csv, index=False)
    df_test.to_csv(test_csv, index=False)
    
    model_path = os.path.join(temp_dir, "model.joblib")
    metrics_json = os.path.join(temp_dir, "metrics.json")
    
    # 1. Test training (Point 6: training produces a saved model)
    train(
        classifier="random_forest",
        reducedTrain_csv=train_csv,
        target_column="target",
        model_path=model_path,
        metrics_json=metrics_json,
        seed=42
    )
    
    assert os.path.exists(model_path)
    assert os.path.exists(metrics_json)
    
    # Check if the model can be loaded
    loaded_model = joblib.load(model_path)
    assert loaded_model is not None
    
    predictions_csv = os.path.join(temp_dir, "predictions.csv")
    stats_json = os.path.join(temp_dir, "stats.json")
    
    # 2. Test prediction (Point 7: prediction produces a CSV with required columns)
    predict(
        reduced_Test_csv=test_csv,
        target_column="target",
        model_path=model_path,
        predictions_csv=predictions_csv,
        classif_stats_json=stats_json
    )
    
    assert os.path.exists(predictions_csv)
    assert os.path.exists(stats_json)
    
    df_preds = pd.read_csv(predictions_csv)
    expected_cols = {"row_n", "target", "prediction", "score"}
    assert expected_cols.issubset(set(df_preds.columns))
    assert len(df_preds) == n_samples_test

def test_classifiers_support(temp_dir):
    # Verify that the different requested classifiers run without error
    np.random.seed(42)
    m = 2
    n_samples = 50
    data = {f"feature_{i}": np.random.randn(n_samples) for i in range(m)}
    data["target"] = np.random.randint(0, 2, size=n_samples)
    df = pd.DataFrame(data)
    
    train_csv = os.path.join(temp_dir, "train.csv")
    df.to_csv(train_csv, index=False)
    
    classifiers = ["random_forest", "gradient_boosting", "logistic_regression"]
    
    for clf in classifiers:
        model_path = os.path.join(temp_dir, f"model_{clf}.joblib")
        metrics_json = os.path.join(temp_dir, f"metrics_{clf}.json")
        train(
            classifier=clf,
            reducedTrain_csv=train_csv,
            target_column="target",
            model_path=model_path,
            metrics_json=metrics_json,
            seed=42
        )
        assert os.path.exists(model_path)
