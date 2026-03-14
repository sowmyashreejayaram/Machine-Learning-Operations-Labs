# MLflow Experiment Tracking Lab

## Overview
This lab demonstrates **MLflow Experiment Tracking** — a core MLOps concept for logging, comparing, and managing ML experiments. We train multiple classifiers on the **Wine Quality** dataset, log parameters/metrics/artifacts to MLflow, compare runs, and register the best model.

**Reference:** [raminmohammadi/MLOps](https://github.com/raminmohammadi/MLOps) — Northeastern University IE7374 MLOps Course

## Learning Objectives
- Set up MLflow tracking server (local)
- Log parameters, metrics, and artifacts for ML experiments
- Compare multiple model runs using the MLflow UI
- Register the best-performing model in the MLflow Model Registry
- Load and serve a registered model for inference

## Tech Stack
- **Python 3.10+**
- **MLflow** — experiment tracking & model registry
- **scikit-learn** — model training (Logistic Regression, Random Forest, XGBoost)
- **pandas / numpy** — data manipulation
- **matplotlib / seaborn** — visualization

## Project Structure
```
mlflow_lab_sowmya/
├── README.md
├── requirements.txt
├── data/
│   └── download_data.py          # Script to download Wine Quality dataset
├── src/
│   ├── preprocess.py             # Data loading, cleaning, feature engineering
│   ├── train.py                  # Train models with MLflow tracking
│   ├── evaluate.py               # Evaluate & compare runs
│   └── register_model.py         # Register best model to MLflow registry
├── notebooks/
│   └── mlflow_experiment_tracking.ipynb   # Step-by-step walkthrough
└── artifacts/
    └── .gitkeep
```

## Setup & Installation

### 1. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download data
```bash
python data/download_data.py
```

### 4. Start MLflow tracking server
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 127.0.0.1 --port 5000
```

### 5. Run training experiments
```bash
python src/train.py
```

### 6. Compare & register best model
```bash
python src/evaluate.py
python src/register_model.py
```

### 7. View MLflow UI
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser to explore experiment runs.

## Results Summary
| Model | Accuracy | F1-Score | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | ~0.54 | ~0.52 | ~0.53 | ~0.54 |
| Random Forest | ~0.68 | ~0.66 | ~0.67 | ~0.68 |
| XGBoost | ~0.69 | ~0.67 | ~0.68 | ~0.69 |

*(Results may vary slightly depending on random state and hyperparameters.)*

## Key MLflow Concepts Demonstrated
1. **Experiment Creation** — Organizing runs under named experiments
2. **Parameter Logging** — `mlflow.log_param()` for hyperparameters
3. **Metric Logging** — `mlflow.log_metric()` for accuracy, F1, precision, recall
4. **Artifact Logging** — Confusion matrix plots, feature importance charts
5. **Autologging** — `mlflow.sklearn.autolog()` for automatic logging
6. **Model Registry** — Registering, staging, and promoting models
7. **Model Loading** — `mlflow.pyfunc.load_model()` for inference

## Author
**Sowmya Shree Jayaram**  
MS Data Analytics Engineering, Northeastern University
