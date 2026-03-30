# DVC Lab - Data Version Control for ML Pipeline

**Student:** Sowmyashree Jayaram
**Course:** IE7374 - Machine Learning Operations
**Lab Assignment:** 5

## Overview

This lab demonstrates **Data Version Control (DVC)** for managing datasets and ML model artifacts. Modifications from the original lab:

1. Uses the **Wine Quality dataset** (UCI ML Repository) instead of CC_GENERAL
2. Implements a **full DVC pipeline** with `dvc.yaml` (preprocess -> train -> evaluate)
3. Uses **local remote storage** instead of Google Cloud Storage
4. Adds **DVC metrics and plots** for experiment tracking
5. Includes **params.yaml** for hyperparameter management

## Project Structure
```
dvc-lab-sowmya/
├── README.md
├── requirements.txt
├── params.yaml              # Hyperparameters tracked by DVC
├── dvc.yaml                 # DVC pipeline definition
├── dvc.lock                 # Auto-generated pipeline lock file
├── data/
│   └── wine_quality.csv.dvc # DVC metafile for dataset
├── src/
│   ├── download_data.py     # Downloads Wine Quality dataset
│   ├── preprocess.py        # Data cleaning & feature engineering
│   ├── train.py             # Random Forest model training
│   └── evaluate.py          # Model evaluation with metrics
├── models/
│   └── random_forest.pkl    # Trained model artifact
└── results/
    ├── metrics.json         # DVC metrics output
    └── plots/               # Feature importance CSV
```

## Setup & Reproduction
```bash
cd dvc-lab-sowmya
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
dvc init --no-scm
mkdir -p /tmp/dvc-remote-storage
dvc remote add -d localremote /tmp/dvc-remote-storage
python src/download_data.py
dvc add data/wine_quality.csv
dvc repro
dvc metrics show
```

## Results

- **MAE:** 0.4434
- **RMSE:** 0.5635
- **R2 Score:** 0.5141
- **Training R2:** 0.8253
- **Samples:** 1279 train / 320 test

## Key Modifications from Original Lab

| Feature | Original Lab | This Lab |
|---------|-------------|----------|
| Dataset | CC_GENERAL (credit card) | Wine Quality (UCI) |
| Remote Storage | Google Cloud Storage | Local remote |
| Pipeline | Manual DVC add only | Full dvc.yaml pipeline |
| Metrics | None | dvc metrics show with JSON |
| Params | None | params.yaml hyperparameters |
| Model | None | Random Forest Regressor |

## Technologies

Python, DVC 3.x, scikit-learn, pandas, numpy, matplotlib
