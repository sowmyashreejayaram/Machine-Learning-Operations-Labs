# Docker Lab 3 — Breast Cancer Prediction API

**MLOps Course | Northeastern University**
**Student:** Sowmyashree Jayaram
**Based on:** [raminmohammadi/MLOps/Labs/Docker_Labs](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Docker_Labs)

---

## Overview

This lab containerizes a **Breast Cancer classification model** using Docker. A trained Random Forest model is served through a Flask API with prediction, health-check, and dashboard endpoints.

## Modifications from the Original Lab

| Aspect | Original Docker Labs | This Lab (Lab 3) |
|---|---|---|
| **Dataset** | Iris (150 samples, 4 features) | Breast Cancer Wisconsin (569 samples, 30 features) |
| **Model** | Decision Tree Classifier | Random Forest + StandardScaler Pipeline |
| **Endpoints** | Basic `/predict` only | `/predict`, `/dashboard`, `/health`, `/` |
| **Dashboard** | None | Visual metrics dashboard with bar charts |
| **Dockerfile** | Single-stage build | Multi-stage build (smaller final image) |
| **Health Check** | None | Built-in HEALTHCHECK instruction |
| **Docker Compose** | Not included | docker-compose.yml for one-command startup |
| **Tests** | Not included | Pytest suite with 7 test cases |
| **Metrics Logging** | None | Exports accuracy/precision/recall/F1 to JSON |

## Project Structure

    docker_lab3_sowmya/
    ├── app.py
    ├── Dockerfile
    ├── docker-compose.yml
    ├── requirements.txt
    ├── .dockerignore
    ├── .gitignore
    ├── README.md
    ├── src/
    │   ├── __init__.py
    │   └── train_model.py
    ├── model/
    │   ├── breast_cancer_model.pkl
    │   ├── metadata.pkl
    │   └── metrics.json
    ├── templates/
    │   ├── index.html
    │   └── dashboard.html
    └── tests/
        ├── __init__.py
        └── test_app.py

## Prerequisites

- Docker Desktop installed and running
- Git
- Python 3.11+ (for local development)

## Step-by-Step Instructions

### Step 1 — Train the Model

    python -m venv docker_lab3_env
    source docker_lab3_env/bin/activate
    pip install -r requirements.txt
    python src/train_model.py

### Step 2 — Build the Docker Image

    docker build -t breast-cancer-predictor .

### Step 3 — Run the Container

    docker run -d -p 5000:5000 --name bc-predictor breast-cancer-predictor

Or using Docker Compose:

    docker-compose up -d

### Step 4 — Test the Application

- Landing page: http://localhost:5000
- Dashboard: http://localhost:5000/dashboard
- Health check: curl http://localhost:5000/health
- Prediction:

      curl -X POST http://localhost:5000/predict \
        -H "Content-Type: application/json" \
        -d '{"features": [17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]}'

### Step 5 — Run Tests

    pytest tests/ -v

### Step 6 — Stop the Container

    docker stop bc-predictor && docker rm bc-predictor

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Landing page with API docs |
| POST | /predict | Send 30 features, get prediction |
| GET | /dashboard | Model performance dashboard |
| GET | /health | Health-check for orchestration |

## Technologies Used

- Python 3.11, Flask 3.1, scikit-learn 1.6
- Docker (multi-stage build), Docker Compose, pytest
