"""Evaluate model and output DVC metrics + plots."""
import os
import json
import yaml
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)["evaluate"]

def evaluate():
    params = load_params()
    with open("models/random_forest.pkl", "rb") as f:
        model = pickle.load(f)
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    metrics = {"mae": round(mae, 4), "rmse": round(rmse, 4),
               "r2_score": round(r2, 4), "test_samples": len(y_test)}
    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    if params.get("generate_plots", True):
        os.makedirs("results/plots", exist_ok=True)
        imp_df = pd.DataFrame({"feature": X_test.columns, "importance": np.round(model.feature_importances_, 4)})
        imp_df.sort_values("importance", ascending=False).to_csv("results/plots/feature_importance.csv", index=False)
        print("Plots saved")

if __name__ == "__main__":
    evaluate()
