import os
import pickle
import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix


def load_test_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def load_model(model_path: str):
    with open(model_path, "rb") as file:
         return pickle.load(file)
    

def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
  
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    return metrics
    

def save_metrics(metrics: dict, output_path: str):
    with open(output_path, "w") as file:
        json.dump(metrics, file, indent=5)


def main():
    test_data_path = "./data/processed/test_processed.csv"
    model_path = "model.pkl"
    metrics_path = "metrics.json"

    test_data = load_test_data(test_data_path)
    
    X_test = test_data.drop(columns=['HeartDisease'])
    y_test = test_data['HeartDisease']

    model = load_model(model_path)
    metrics = evaluate_model(model, X_test, y_test)
    save_metrics(metrics, metrics_path)


if __name__ == "__main__":
    main()
