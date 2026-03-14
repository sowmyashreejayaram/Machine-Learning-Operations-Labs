"""
Train multiple ML models with MLflow experiment tracking.

This script demonstrates:
- Creating MLflow experiments
- Logging parameters, metrics, and artifacts
- Training Logistic Regression, Random Forest, and XGBoost
- Saving confusion matrix plots as artifacts
"""

import os
import sys
import warnings

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import load_data, preprocess

warnings.filterwarnings("ignore")

# ─── MLflow Configuration ───
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "wine-quality-classification"


def plot_confusion_matrix(y_true, y_pred, model_name: str, save_dir: str) -> str:
    """Generate and save a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()

    filepath = os.path.join(save_dir, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    return filepath


def train_logistic_regression(X_train, X_test, y_train, y_test, feature_names):
    """Train Logistic Regression with MLflow tracking."""
    with mlflow.start_run(run_name="LogisticRegression"):
        # Hyperparameters
        params = {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
            "multi_class": "multinomial",
            "random_state": 42,
        }

        # Log parameters
        mlflow.log_params(params)
        mlflow.set_tag("model_type", "LogisticRegression")
        mlflow.set_tag("author", "sowmya")

        # Train
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        # Predict & evaluate
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "precision_weighted": precision_score(y_test, y_pred, average="weighted"),
            "recall_weighted": recall_score(y_test, y_pred, average="weighted"),
        }

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log confusion matrix as artifact
        os.makedirs("temp_artifacts", exist_ok=True)
        cm_path = plot_confusion_matrix(y_test, y_pred, "Logistic Regression", "temp_artifacts")
        mlflow.log_artifact(cm_path)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(f"\n{'='*50}")
        print(f"Logistic Regression Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  F1 Score:  {metrics['f1_weighted']:.4f}")
        print(f"  Precision: {metrics['precision_weighted']:.4f}")
        print(f"  Recall:    {metrics['recall_weighted']:.4f}")
        print(f"{'='*50}")

        return metrics


def train_random_forest(X_train, X_test, y_train, y_test, feature_names):
    """Train Random Forest with MLflow tracking."""
    with mlflow.start_run(run_name="RandomForest"):
        # Hyperparameters
        params = {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1,
        }

        # Log parameters
        mlflow.log_params(params)
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("author", "sowmya")

        # Train
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Predict & evaluate
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "precision_weighted": precision_score(y_test, y_pred, average="weighted"),
            "recall_weighted": recall_score(y_test, y_pred, average="weighted"),
        }

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log feature importance as artifact
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[sorted_idx], align="center")
        plt.xticks(range(len(importances)), [feature_names[i] for i in sorted_idx], rotation=45, ha="right")
        plt.title("Random Forest — Feature Importance")
        plt.ylabel("Importance")
        plt.tight_layout()
        fi_path = "temp_artifacts/feature_importance_rf.png"
        plt.savefig(fi_path, dpi=150)
        plt.close()
        mlflow.log_artifact(fi_path)

        # Log confusion matrix
        cm_path = plot_confusion_matrix(y_test, y_pred, "Random Forest", "temp_artifacts")
        mlflow.log_artifact(cm_path)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(f"\n{'='*50}")
        print(f"Random Forest Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  F1 Score:  {metrics['f1_weighted']:.4f}")
        print(f"  Precision: {metrics['precision_weighted']:.4f}")
        print(f"  Recall:    {metrics['recall_weighted']:.4f}")
        print(f"{'='*50}")

        return metrics


def train_xgboost(X_train, X_test, y_train, y_test, feature_names):
    """Train XGBoost with MLflow tracking."""
    with mlflow.start_run(run_name="XGBoost"):
        # Hyperparameters
        params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": "mlogloss",
        }

        # Log parameters
        mlflow.log_params(params)
        mlflow.set_tag("model_type", "XGBoost")
        mlflow.set_tag("author", "sowmya")

        # Train
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        # Predict & evaluate
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "precision_weighted": precision_score(y_test, y_pred, average="weighted"),
            "recall_weighted": recall_score(y_test, y_pred, average="weighted"),
        }

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log feature importance
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[sorted_idx], align="center", color="darkorange")
        plt.xticks(range(len(importances)), [feature_names[i] for i in sorted_idx], rotation=45, ha="right")
        plt.title("XGBoost — Feature Importance")
        plt.ylabel("Importance")
        plt.tight_layout()
        fi_path = "temp_artifacts/feature_importance_xgb.png"
        plt.savefig(fi_path, dpi=150)
        plt.close()
        mlflow.log_artifact(fi_path)

        # Log confusion matrix
        cm_path = plot_confusion_matrix(y_test, y_pred, "XGBoost", "temp_artifacts")
        mlflow.log_artifact(cm_path)

        # Log model
        mlflow.xgboost.log_model(model, "model")

        print(f"\n{'='*50}")
        print(f"XGBoost Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  F1 Score:  {metrics['f1_weighted']:.4f}")
        print(f"  Precision: {metrics['precision_weighted']:.4f}")
        print(f"  Recall:    {metrics['recall_weighted']:.4f}")
        print(f"{'='*50}")

        return metrics


def main():
    """Run all experiments."""
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(TRACKING_URI)

    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
        print(f"Created experiment '{EXPERIMENT_NAME}' with ID: {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment '{EXPERIMENT_NAME}' (ID: {experiment_id})")

    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load and preprocess data
    print("\n--- Loading and preprocessing data ---")
    df = load_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df)

    # Train models
    print("\n--- Training Logistic Regression ---")
    lr_metrics = train_logistic_regression(X_train, X_test, y_train, y_test, feature_names)

    print("\n--- Training Random Forest ---")
    rf_metrics = train_random_forest(X_train, X_test, y_train, y_test, feature_names)

    print("\n--- Training XGBoost ---")
    xgb_metrics = train_xgboost(X_train, X_test, y_train, y_test, feature_names)

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    all_results = {
        "Logistic Regression": lr_metrics,
        "Random Forest": rf_metrics,
        "XGBoost": xgb_metrics,
    }
    best_model = max(all_results, key=lambda k: all_results[k]["accuracy"])
    for name, metrics in all_results.items():
        marker = " <-- BEST" if name == best_model else ""
        print(f"  {name:25s} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1_weighted']:.4f}{marker}")
    print("=" * 60)
    print(f"\nView results at: http://127.0.0.1:5000")
    print("Run `mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 127.0.0.1 --port 5000`")


if __name__ == "__main__":
    main()
