"""
app.py
------
Flask application that serves the trained Breast Cancer classification model.
"""

import os
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

model = joblib.load(os.path.join(MODEL_DIR, "breast_cancer_model.pkl"))
meta = joblib.load(os.path.join(MODEL_DIR, "metadata.pkl"))

FEATURE_NAMES = meta["feature_names"]
TARGET_NAMES = meta["target_names"]

METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
with open(METRICS_PATH) as f:
    METRICS = json.load(f)


@app.route("/")
def home():
    return render_template("index.html", features=FEATURE_NAMES)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        if "features" not in data:
            return jsonify({"error": "Missing 'features' key in request body"}), 400

        features = np.array(data["features"]).reshape(1, -1)

        if features.shape[1] != len(FEATURE_NAMES):
            return (
                jsonify(
                    {
                        "error": f"Expected {len(FEATURE_NAMES)} features, "
                        f"got {features.shape[1]}"
                    }
                ),
                400,
            )

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        result = {
            "prediction": int(prediction),
            "class_name": TARGET_NAMES[prediction],
            "probabilities": {
                TARGET_NAMES[i]: round(float(p), 4)
                for i, p in enumerate(probabilities)
            },
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", metrics=METRICS, features=FEATURE_NAMES)


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
