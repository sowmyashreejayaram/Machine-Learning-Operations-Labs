"""Preprocess Wine Quality dataset."""
import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)["preprocess"]

def preprocess():
    params = load_params()
    print(f"Preprocessing parameters: {params}")
    df = pd.read_csv("data/wine_quality.csv")
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    df = df.dropna()
    if "wine_type" in df.columns:
        df["is_red"] = (df["wine_type"] == "red").astype(int)
        df = df.drop(columns=["wine_type"])
    X = df.drop(columns=["quality"])
    y = df["quality"]
    print(f"Target range: {y.min()}-{y.max()}, mean: {y.mean():.2f}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=params["random_state"])
    if params.get("scale_features", True):
        scaler = StandardScaler()
        cols = X_train.columns
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=cols)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=cols)
        print("Applied StandardScaler")
    os.makedirs("data/processed", exist_ok=True)
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

if __name__ == "__main__":
    preprocess()
