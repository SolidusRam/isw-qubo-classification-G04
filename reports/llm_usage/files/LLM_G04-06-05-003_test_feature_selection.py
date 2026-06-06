import os
import json
import shutil
import tempfile
import pandas as pd
import numpy as np
import pytest
from qubo_project.feature_selection import select_features

@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)

def test_select_features_basic(temp_dir):
    # Creiamo un dataset mock normalizzato con 10 features e 100 sample
    np.random.seed(42)
    m = 10
    n_samples = 100
    
    data = {}
    for i in range(m):
        data[f"feature_{i}"] = np.random.randn(n_samples)
    
    # Creiamo un target correlato artificialmente ad alcune feature
    data["target"] = (data["feature_0"] + data["feature_1"] > 0).astype(int)
    
    df = pd.DataFrame(data)
    normalized_csv = os.path.join(temp_dir, "normalized.csv")
    df.to_csv(normalized_csv, index=False)
    
    reducedTrain_csv = os.path.join(temp_dir, "reducedTrain.csv")
    reducedTest_csv = os.path.join(temp_dir, "reducedTest.csv")
    output_ottim_csv = os.path.join(temp_dir, "optimizations.csv")
    output_json = os.path.join(temp_dir, "stats.json")
    
    # Lanciamo la feature selection con percentuale desiderata 0.20 (20%)
    select_features(
        normalized_csv=normalized_csv,
        reducedTrain_csv=reducedTrain_csv,
        reducedTest_csv=reducedTest_csv,
        output_ottim_csv=output_ottim_csv,
        output_json=output_json,
        target_column="target",
        percTest=0.30,
        percSelected=0.20,
        allowance=1,
        seed=42,
        alpha_computations=15
    )
    
    # 1. Verifica file in output
    assert os.path.exists(reducedTrain_csv)
    assert os.path.exists(reducedTest_csv)
    assert os.path.exists(output_ottim_csv)
    assert os.path.exists(output_json)
    
    # Lettura delle stats
    with open(output_json, "r") as f:
        stats = json.load(f)
        
    vector = stats["selected_vector"]
    
    # 2. Requisito 4: che la feature selection produca un vettore binario
    for v in vector:
        assert v in [0, 1]
    
    # La lunghezza del vettore deve essere uguale al numero di feature in input
    assert len(vector) == m
    
    # 3. Requisito 5: che il numero di feature selezionate sia circa il 20%
    n_selected = sum(vector)
    target_k = int(round(0.20 * m)) # 2 feature attese
    assert abs(n_selected - target_k) <= 1 # tolleranza 1
    
    assert stats["n_selected"] == n_selected
    
    # Verifica che il numero di colonne dei file estratti sia corretto
    df_train = pd.read_csv(reducedTrain_csv)
    df_test = pd.read_csv(reducedTest_csv)
    
    # Numero di colonne = feature selezionate + colonna target
    assert len(df_train.columns) == n_selected + 1
    assert len(df_test.columns) == n_selected + 1
    
    assert "target" in df_train.columns
    assert "target" in df_test.columns
    
    # Verifica dello split: 100 righe totali, 30% test -> M = 70
    assert len(df_train) == 70
    assert len(df_test) == 30
