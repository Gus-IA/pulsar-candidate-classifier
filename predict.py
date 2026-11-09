"""Serve the pulsar classifier as a web service."""

import pickle

import pandas as pd
from flask import Flask, jsonify, request

MODEL_PATH = "model.pkl"

with open(MODEL_PATH, "rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]

app = Flask("pulsar-classifier")


@app.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "service": "pulsar-classifier",
            "usage": "POST a JSON with the 8 features to /predict",
            "features": features,
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    candidate = request.get_json(silent=True)
    if not isinstance(candidate, dict):
        return jsonify({"error": "Send a JSON object with the 8 features"}), 400

    missing = [f for f in features if f not in candidate]
    if missing:
        return jsonify({"error": f"Missing features: {missing}"}), 400

    try:
        X = pd.DataFrame([{f: float(candidate[f]) for f in features}])
    except (TypeError, ValueError):
        return jsonify({"error": "All features must be numbers"}), 400

    proba = float(model.predict_proba(X)[0, 1])
    return jsonify(
        {
            "pulsar_probability": round(proba, 4),
            "is_pulsar": proba >= threshold,
            "threshold": threshold,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696)
