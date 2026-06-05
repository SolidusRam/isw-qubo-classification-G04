import argparse
import time
import json
import pandas as pd
import numpy as np
import neal
from sklearn.model_selection import train_test_split

def select_features(
    normalized_csv: str,
    reducedTrain_csv: str,
    reducedTest_csv: str,
    output_ottim_csv: str,
    output_json: str,
    target_column: str,
    percTest: float = 0.30,
    percSelected: float = 0.20,
    allowance: int = 1,
    seed: int = 42,
    alpha_computations: int = 100
):
    np.random.seed(seed)
    
    # 1. Lettura dataset
    df = pd.read_csv(normalized_csv)
    features = [c for c in df.columns if c != target_column]
    m = len(features)
    target_k = int(round(percSelected * m))
    
    # 2. Calcolo matrice di correlazione di Spearman
    t0_corr = time.time()
    corr_matrix = df.corr(method='spearman').abs()
    
    rho_V = corr_matrix.loc[features, target_column].values
    rho_U = corr_matrix.loc[features, features].values
    q_matrix_creation_time = time.time() - t0_corr
    
    # Funzione per costruire la matrice QUBO per un dato alpha
    def build_qubo(alpha):
        Q = {}
        for j in range(m):
            Q[(j, j)] = -alpha * rho_V[j]
            for k in range(j + 1, m):
                Q[(j, k)] = 2 * (1.0 - alpha) * rho_U[j, k]
        return Q

    sampler = neal.SimulatedAnnealingSampler()
    
    alpha_low = 0.0
    alpha_high = 1.0
    
    optimizations = []
    opt_times = []
    
    best_vector = None
    best_n_selected = -1
    best_alpha = -1
    best_cost = None
    
    # 3. Ottimizzazione QUBO al variare di alpha
    for i in range(alpha_computations):
        alpha = (alpha_low + alpha_high) / 2.0
        
        Q = build_qubo(alpha)
        
        t0_opt = time.time()
        # num_reads è impostato a 100 per avere una buona stabilità nell'annealing
        response = sampler.sample_qubo(Q, num_reads=100, seed=seed+i)
        t_opt = time.time() - t0_opt
        opt_times.append(t_opt)
        
        best_sample = response.first.sample
        best_energy = response.first.energy
        
        vector = [int(best_sample[j]) for j in range(m)]
        n_selected = sum(vector)
        
        optimizations.append({
            'alpha': alpha,
            'time': t_opt,
            'n_features': n_selected,
            'cost': best_energy
        })
        
        if best_vector is None or abs(n_selected - target_k) < abs(best_n_selected - target_k):
            best_vector = vector
            best_n_selected = n_selected
            best_alpha = alpha
            best_cost = best_energy
            
        if abs(n_selected - target_k) <= allowance:
            break
            
        if n_selected > target_k:
            alpha_high = alpha
        else:
            alpha_low = alpha
            
    # 4. Salvataggio risultati e dati
    optimizations_df = pd.DataFrame(optimizations)
    optimizations_df = optimizations_df.sort_values(by='alpha')
    optimizations_df.to_csv(output_ottim_csv, index=False)
    
    selected_feature_names = [features[j] for j in range(m) if best_vector[j] == 1]
    
    columns_to_keep = selected_feature_names + [target_column]
    df_reduced = df[columns_to_keep]
    
    # Suddivisione dataset mantenendo l'ordine (primi M per training)
    M = int(len(df_reduced) * (1 - percTest))
    train_df = df_reduced.iloc[:M]
    test_df = df_reduced.iloc[M:]
    
    train_df.to_csv(reducedTrain_csv, index=False)
    test_df.to_csv(reducedTest_csv, index=False)
    
    stats = {
        "n_features": m,
        "target_ratio": percSelected,
        "target_k": target_k,
        "allowance": allowance,
        "n_selected": best_n_selected,
        "alpha": best_alpha,
        "selected_vector": best_vector,
        "selected_feature_names": selected_feature_names,
        "algorithm": "simulated_annealing",
        "seed": seed,
        "alpha_computations": len(optimizations),
        "percTest": percTest,
        "training_dataset_size": len(train_df),
        "test_dataset_size": len(test_df),
        "q_matrix_creation_time": q_matrix_creation_time,
        "mean_optimization_time": float(np.mean(opt_times)) if opt_times else 0.0,
        "std_dev_optimization_time": float(np.std(opt_times)) if len(opt_times) > 1 else 0.0
    }
    
    with open(output_json, 'w') as f:
        json.dump(stats, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature selection QUBO")
    parser.add_argument("--in-normalized", required=True)
    parser.add_argument("--out-train", required=True)
    parser.add_argument("--out-test", required=True)
    parser.add_argument("--out-optimizations", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--perc-selected", type=float, default=0.20)
    parser.add_argument("--allowance", type=int, default=1)
    parser.add_argument("--perc-test", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha-computations", type=int, default=100)
    
    args = parser.parse_args()
    
    select_features(
        normalized_csv=args.in_normalized,
        reducedTrain_csv=args.out_train,
        reducedTest_csv=args.out_test,
        output_ottim_csv=args.out_optimizations,
        output_json=args.out_json,
        target_column=args.target,
        percTest=args.perc_test,
        percSelected=args.perc_selected,
        allowance=args.allowance,
        seed=args.seed,
        alpha_computations=args.alpha_computations
    )
