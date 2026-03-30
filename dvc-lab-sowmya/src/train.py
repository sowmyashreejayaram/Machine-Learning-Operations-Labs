"""Train Random Forest Regressor on Wine Quality data."""
import os
import yaml
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)["train"]

def train():
    params = load_params()
    print(f"Training parameters: {params}")
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    print(f"Training on {len(X_train)} samples, {X_train.shape[1]} features")
    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        random_state=params["random_state"],
        n_jobs=-1)
    model.fit(X_train, y_train)
    print(f"Training R2: {model.score(X_train, y_train):.4f}")
    os.makedirs("models", exist_ok=True)
    with open("models/random_forest.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved to models/random_forest.pkl")

if __name__ == "__main__":
    train()
