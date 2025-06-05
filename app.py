from flask import Flask, request, jsonify
import joblib
import numpy as np
from huggingface_hub import hf_hub_download

app = Flask(__name__)

# Load model from Hugging Face Hub
MODEL_REPO = "reza-academia/paper-metrics"
MODEL_FILENAME = "model.joblib"

model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
model = joblib.load(model_path)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json.get("input")
        if not data or len(data) != 30:
            return jsonify(error="Input must be an array of 30 floats."), 400

        X = np.array(data).reshape(1, -1)
        prob = model.predict_proba(X)[0][1]
        return jsonify(probability=round(float(prob), 4))
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route("/")
def root():
    return "Paper Metrics API is live!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
