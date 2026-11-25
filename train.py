"""
Simple training script:
- loads iris dataset from sklearn
- trains a LogisticRegression
- saves model to model.pkl
"""

from sklearn.datasets import load_iris                  # ready-made flower dataset (150 rows, 4 features)
from sklearn.linear_model import LogisticRegression     # the ML algorithm we train
from sklearn.model_selection import train_test_split    # splits data into train / test sets
import joblib                                           # saves / loads the trained model (.pkl)
import os                                               # folder + file path helpers
import json                                             # writes metrics as JSON

def main():
    iris = load_iris()                                   # load the dataset
    X, y = iris.data, iris.target                        # X = flower measurements, y = flower type (0/1/2)
    names = iris.target_names                            # 0=setosa, 1=versicolor, 2=virginica

    # 80% data to train, 20% data to test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=200)             # choose the algorithm
    model.fit(X_train, y_train)                          # train -> model learns the pattern

    # Save model
    os.makedirs("artifacts", exist_ok=True)              # create artifacts/ folder if missing
    model_path = os.path.join("artifacts", "model.pkl")
    joblib.dump(model, model_path)                       # package the model as .pkl

    # Save a tiny metrics file
    acc = model.score(X_test, y_test)                    # accuracy on the 20% test data
    metrics = {"accuracy": float(acc)}
    with open(os.path.join("artifacts", "metrics.json"), "w") as f:
        json.dump(metrics, f)

    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")

    # Show a few test predictions with real flower names
    preds = model.predict(X_test)
    print("\nSample test predictions:")
    print(f"{'features':<24} {'actual':<12} {'predicted':<12} ok")
    for features, actual, pred in list(zip(X_test, y_test, preds))[:5]:
        ok = "yes" if actual == pred else "NO"
        print(f"{str(list(features)):<24} {names[actual]:<12} {names[pred]:<12} {ok}")

    print(f"\nSaved model to {model_path}")
    print(f"Test accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
