"""
Evaluate and compare MLflow experiment runs.

This script demonstrates:
- Querying runs from MLflow programmatically
- Comparing metrics across runs
- Identifying the best-performing model
"""

import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "wine-quality-classification"


def compare_runs():
    """Retrieve and compare all runs in the experiment."""
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    # Get experiment
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print(f"Experiment '{EXPERIMENT_NAME}' not found. Run train.py first.")
        return

    # Search all runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
    )

    if not runs:
        print("No runs found. Run train.py first.")
        return

    # Display comparison table
    print("\n" + "=" * 80)
    print(f"{'Run Name':25s} | {'Accuracy':>10s} | {'F1':>10s} | {'Precision':>10s} | {'Recall':>10s}")
    print("-" * 80)

    best_run = None
    best_accuracy = 0.0

    for run in runs:
        name = run.info.run_name or "unnamed"
        acc = run.data.metrics.get("accuracy", 0)
        f1 = run.data.metrics.get("f1_weighted", 0)
        prec = run.data.metrics.get("precision_weighted", 0)
        rec = run.data.metrics.get("recall_weighted", 0)

        print(f"  {name:25s} | {acc:>10.4f} | {f1:>10.4f} | {prec:>10.4f} | {rec:>10.4f}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_run = run

    print("=" * 80)

    if best_run:
        print(f"\nBest model: {best_run.info.run_name}")
        print(f"  Run ID:   {best_run.info.run_id}")
        print(f"  Accuracy: {best_accuracy:.4f}")
        print(f"  Model URI: runs:/{best_run.info.run_id}/model")

    return best_run


if __name__ == "__main__":
    compare_runs()
