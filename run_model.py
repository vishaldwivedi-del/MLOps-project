#!/usr/bin/env python3
"""
Usage:
    python run_model.py --input "[5.1, 3.5, 1.4, 0.2]"
"""

import argparse                 # reads command line arguments (--input)
import json                     # parses input list and prints JSON output
from pathlib import Path        # clean file path handling
import numpy as np              # shapes input into the array the model expects
import joblib                   # loads the saved model (.pkl)

MODEL_PATH = Path("artifacts/model.pkl")              # model saved by train.py
FLOWER_NAMES = ["setosa", "versicolor", "virginica"]  # model output 0/1/2 -> flower name

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)   # <-- correct binary loading

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Feature list as JSON string. Example: \"[5.1,3.5,1.4,0.2]\"")
    args = parser.parse_args()

    # Parse input
    try:
        features = json.loads(args.input)
    except json.JSONDecodeError:
        raise ValueError("Invalid input. Use JSON list, e.g. --input \"[5.1,3.5,1.4,0.2]\"")

    X = np.array(features).reshape(1, -1)    # 1 flower, 4 features

    model = load_model()
    pred = int(model.predict(X)[0])          # 0, 1 or 2

    print(json.dumps({"prediction": pred, "flower": FLOWER_NAMES[pred]}))

if __name__ == "__main__":
    main()
