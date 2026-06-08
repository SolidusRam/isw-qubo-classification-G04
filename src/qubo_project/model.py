import argparse
import json
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import joblib

def train(
    classifier: str,
    reducedTrain_csv: str,
    target_column: str,
    model_path: str,
    metrics_json: str,
    seed: int = 42,
):
    start_time_io = time.time()
    df = pd.read_csv(reducedTrain_csv)
    dataset_input_time = time.time() - start_time_io
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
        
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    n_samples, n_features = X.shape
    target_1_percentage = float((y == 1).mean() * 100)
    
    if classifier == "random_forest":
        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=seed, n_jobs=-1)
    elif classifier == "gradient_boosting":
        model = HistGradientBoostingClassifier(random_state=seed)
    elif classifier == "logistic_regression":
        model = LogisticRegression(class_weight='balanced', random_state=seed, max_iter=1000, n_jobs=-1)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
        
    start_time_train = time.time()
    model.fit(X, y)
    training_time = time.time() - start_time_train
    
    joblib.dump(model, model_path)
    
    metrics = {
        "classifier": classifier,
        "seed": seed,
        "training_dataset": reducedTrain_csv,
        "target_column": target_column,
        "model_path": model_path,
        "n_samples": n_samples,
        "n_features": n_features,
        "target_1_percentage": round(target_1_percentage, 2),
        "dataset_input_time": round(dataset_input_time, 2),
        "training_time": round(training_time, 2)
    }
    
    with open(metrics_json, 'w') as f:
        json.dump(metrics, f, indent=4)

def predict(
    reduced_Test_csv: str,
    target_column: str,
    model_path: str,
    predictions_csv: str,
    classif_stats_json: str,
):
    df = pd.read_csv(reduced_Test_csv)
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
        
    X = df.drop(columns=[target_column])
    y_true = df[target_column]
    
    model = joblib.load(model_path)
    
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    else:
        y_score = y_pred

    out_df = pd.DataFrame({
        'row_n': df.index,
        'target': y_true,
        'prediction': y_pred,
        'score': y_score
    })
    
    out_df.to_csv(predictions_csv, index=False)
    
    n_samples = len(y_true)
    target_1_count = int((y_true == 1).sum())
    target_1_percentage = (target_1_count / n_samples) * 100
    
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = 0.0
        
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    stats = {
        "classifier": type(model).__name__,
        "n_samples": n_samples,
        "target_1_count": target_1_count,
        "target_1_percentage": round(target_1_percentage, 2),
        "accuracy": acc,
        "class_0": {
            "precision": prec[0],
            "recall": rec[0],
            "f1": f1[0],
            "support": int(support[0])
        },
        "class_1": {
            "precision": prec[1],
            "recall": rec[1],
            "f1": f1[1],
            "support": int(support[1])
        },
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": cm.tolist()
        }
    }
    
    with open(classif_stats_json, 'w') as f:
        json.dump(stats, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training and prediction")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--classifier", type=str, required=True, choices=["random_forest", "gradient_boosting", "logistic_regression"])
    train_parser.add_argument("--in-reduced", type=str, required=True)
    train_parser.add_argument("--target", type=str, required=True)
    train_parser.add_argument("--out-model", type=str, required=True)
    train_parser.add_argument("--out-metrics", type=str, required=True)
    train_parser.add_argument("--seed", type=int, default=42)
    
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--input-testset", type=str, required=True)
    predict_parser.add_argument("--target", type=str, required=True)
    predict_parser.add_argument("--model", type=str, required=True)
    predict_parser.add_argument("--out-predictions", type=str, required=True)
    predict_parser.add_argument("--out-stats", type=str, required=True)
    
    args = parser.parse_args()
    
    if args.command == "train":
        train(
            classifier=args.classifier,
            reducedTrain_csv=args.in_reduced,
            target_column=args.target,
            model_path=args.out_model,
            metrics_json=args.out_metrics,
            seed=args.seed
        )
    elif args.command == "predict":
        predict(
            reduced_Test_csv=args.input_testset,
            target_column=args.target,
            model_path=args.model,
            predictions_csv=args.out_predictions,
            classif_stats_json=args.out_stats
        )
