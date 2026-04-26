"""
Flask Web Application - Iris Classification API
Serves ML model predictions via REST API and interactive web UI.
"""

import os
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)

# ── Load Model & Metadata ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH    = os.path.join(BASE_DIR, "model", "model.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "model", "metadata.json")

with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)

with open(METADATA_PATH) as f:
    META = json.load(f)

FEATURES     = META["feature_names"]
CLASS_NAMES  = META["class_names"]
FEAT_RANGES  = META["feature_ranges"]

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", metadata=META)


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "model": META["model_name"]})


@app.route("/api/info")
def model_info():
    return jsonify({
        "model_name":    META["model_name"],
        "best_params":   META["best_params"],
        "features":      FEATURES,
        "classes":       CLASS_NAMES,
        "feature_ranges": FEAT_RANGES,
        "metrics": {
            "accuracy": META["metrics"]["accuracy"],
            "f1_score": META["metrics"]["f1_score"],
        }
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    # Validate
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing features: {missing}"}), 400

    try:
        values = [float(data[f]) for f in FEATURES]
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid value: {e}"}), 400

    X = np.array(values).reshape(1, -1)

    pred_idx   = int(MODEL.predict(X)[0])
    pred_class = CLASS_NAMES[pred_idx]
    proba      = MODEL.predict_proba(X)[0].tolist()

    confidence = {CLASS_NAMES[i]: round(float(p) * 100, 2) for i, p in enumerate(proba)}
    sorted_conf = dict(sorted(confidence.items(), key=lambda x: x[1], reverse=True))

    return jsonify({
        "prediction":  pred_class,
        "class_index": pred_idx,
        "confidence":  sorted_conf,
        "input":       dict(zip(FEATURES, values)),
    })


@app.route("/api/batch_predict", methods=["POST"])
def batch_predict():
    data = request.get_json(force=True)
    if not isinstance(data, list):
        return jsonify({"error": "Expected a JSON array of samples"}), 400

    results = []
    for i, sample in enumerate(data):
        try:
            values = [float(sample[f]) for f in FEATURES]
            X = np.array(values).reshape(1, -1)
            pred_idx   = int(MODEL.predict(X)[0])
            pred_class = CLASS_NAMES[pred_idx]
            proba      = MODEL.predict_proba(X)[0].tolist()
            confidence = {CLASS_NAMES[j]: round(float(p) * 100, 2) for j, p in enumerate(proba)}
            results.append({"index": i, "prediction": pred_class, "confidence": confidence})
        except Exception as e:
            results.append({"index": i, "error": str(e)})

    return jsonify({"count": len(results), "results": results})


@app.route("/api/metrics")
def metrics():
    return jsonify(META["metrics"])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
