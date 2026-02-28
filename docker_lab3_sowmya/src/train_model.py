"""
train_model.py
--------------
Trains a Random Forest Classifier on the Breast Cancer Wisconsin dataset.

Modifications from the original Docker Labs (which use Iris / Decision Tree):
  - Dataset  : Breast Cancer Wisconsin (30 features, binary classification)
  - Model    : RandomForestClassifier inside a StandardScaler pipeline
  - Logging  : Saves training metrics (accuracy, precision, recall, f1)
               to model/metrics.json so the Flask dashboard can display them
  - Artifacts: Exports model pipeline (.pkl) + feature names + class names
"""

import os
import json
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


def train_and_save():
    """Train the model, evaluate it, and persist artifacts."""

    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)
    target_names = list(data.target_names)

    print(f"Dataset : Breast Cancer Wisconsin")
    print(f"Samples : {X.shape[0]}  |  Features: {X.shape[1]}")
    print(f"Classes : {target_names}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average="weighted"), 4),
        "recall": round(recall_score(y_test, y_pred, average="weighted"), 4),
        "f1_score": round(f1_score(y_test, y_pred, average="weighted"), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X.shape[1]),
        "model_type": "RandomForestClassifier",
        "dataset": "Breast Cancer Wisconsin",
    }

    print("Evaluation Metrics")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v}")
    print()
    print(classification_report(y_test, y_pred, target_names=target_names))

    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "breast_cancer_model.pkl")
    joblib.dump(pipeline, model_path)
    print(f"Model saved   -> {model_path}")

    meta = {
        "feature_names": feature_names,
        "target_names": target_names,
    }
    joblib.dump(meta, os.path.join(model_dir, "metadata.pkl"))

    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved -> {os.path.join(model_dir, 'metrics.json')}")

    return pipeline, metrics


if __name__ == "__main__":
    train_and_save()
