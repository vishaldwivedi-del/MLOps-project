from flask import Flask, request, jsonify   # Flask = web framework to build the API
import joblib                               # loads the saved model (.pkl)
import os                                   # OS helpers (not used much here)
from pathlib import Path                    # clean file path handling

app = Flask(__name__)                       # create the web app
MODEL_PATH = Path("artifacts/model.pkl")    # where train.py saved the model
FLOWER_NAMES = ["setosa", "versicolor", "virginica"]  # model output 0/1/2 -> flower name

if not MODEL_PATH.exists():
    # convenience: train if model missing
    import train as _train
    _train.main()

model = joblib.load(MODEL_PATH)             # load model once when the app starts

@app.route("/health", methods=["GET"])      # health check: is the API alive?
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])    # send flower measurements, get flower name
def predict():
    data = request.get_json()               # read JSON body
    if not data or "features" not in data:
        return jsonify({"error": "send JSON with key 'features'"}), 400
    features = data["features"]             # [sepal_len, sepal_wid, petal_len, petal_wid]
    try:
        pred = int(model.predict([features])[0])    # model gives 0, 1 or 2
        return jsonify({"prediction": pred, "flower": FLOWER_NAMES[pred]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)      # run API on port 5001
