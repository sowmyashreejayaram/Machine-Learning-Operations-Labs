"""
Register the best model to the MLflow Model Registry.

This script demonstrates:
- Finding the best run programmatically
- Registering a model in the MLflow Model Registry
- Transitioning model stages (None -> Staging -> Production)
- Loading a registered model for inference
"""

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "wine-quality-classification"
MODEL_NAME = "wine-quality-best-model"


def find_best_run(client: MlflowClient) -> dict:
    """Find the best run by accuracy."""
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError("No runs found.")

    return runs[0]


def register_best_model():
    """Register the best model to the MLflow Model Registry."""
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    # Find best run
    best_run = find_best_run(client)
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"

    print(f"Best run: {best_run.info.run_name} (ID: {run_id})")
    print(f"Accuracy: {best_run.data.metrics.get('accuracy', 0):.4f}")
    print(f"Model URI: {model_uri}")

    # Register model
    print(f"\nRegistering model as '{MODEL_NAME}'...")
    result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    print(f"Registered version: {result.version}")

    # Transition to Staging
    print(f"Transitioning version {result.version} to 'Staging'...")
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=result.version,
        stage="Staging",
    )

    # Transition to Production
    print(f"Promoting version {result.version} to 'Production'...")
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=result.version,
        stage="Production",
    )

    print(f"\nModel '{MODEL_NAME}' v{result.version} is now in Production!")
    return result


def load_and_predict():
    """Load the production model and make a sample prediction."""
    mlflow.set_tracking_uri(TRACKING_URI)

    # Load model from registry
    model_uri = f"models:/{MODEL_NAME}/Production"
    print(f"\nLoading model from: {model_uri}")

    model = mlflow.pyfunc.load_model(model_uri)

    # Create a sample input (11 features for wine quality)
    sample_input = np.array([[
        7.4,   # fixed acidity
        0.70,  # volatile acidity
        0.00,  # citric acid
        1.9,   # residual sugar
        0.076, # chlorides
        11.0,  # free sulfur dioxide
        34.0,  # total sulfur dioxide
        0.9978,# density
        3.51,  # pH
        0.56,  # sulphates
        9.4,   # alcohol
    ]])

    prediction = model.predict(sample_input)
    print(f"Sample prediction: Wine quality = {prediction[0]}")
    print("Model loaded and serving successfully!")


if __name__ == "__main__":
    result = register_best_model()
    load_and_predict()
