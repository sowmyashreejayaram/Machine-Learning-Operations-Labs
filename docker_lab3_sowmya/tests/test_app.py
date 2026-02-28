"""
test_app.py - Unit tests for the Flask prediction API.
"""

import json
import pytest
import numpy as np
from sklearn.datasets import load_breast_cancer


@pytest.fixture
def client():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from app import app
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_features():
    data = load_breast_cancer()
    return data.data[0].tolist()


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = json.loads(resp.data)
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True


def test_home_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Breast Cancer Prediction" in resp.data


def test_dashboard_page(client):
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert b"Model Performance Dashboard" in resp.data


def test_predict_valid_input(client, sample_features):
    resp = client.post(
        "/predict",
        data=json.dumps({"features": sample_features}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    body = json.loads(resp.data)
    assert "prediction" in body
    assert body["class_name"] in ["malignant", "benign"]
    assert "probabilities" in body


def test_predict_missing_features(client):
    resp = client.post(
        "/predict",
        data=json.dumps({"data": [1, 2, 3]}),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_predict_wrong_feature_count(client):
    resp = client.post(
        "/predict",
        data=json.dumps({"features": [1.0, 2.0, 3.0]}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    body = json.loads(resp.data)
    assert "Expected 30 features" in body["error"]


def test_predict_returns_probabilities(client, sample_features):
    resp = client.post(
        "/predict",
        data=json.dumps({"features": sample_features}),
        content_type="application/json",
    )
    body = json.loads(resp.data)
    prob_sum = sum(body["probabilities"].values())
    assert abs(prob_sum - 1.0) < 0.01
